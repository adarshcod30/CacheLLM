"""Application state and its lifecycle.

Two deliberate choices live here.

**Storage is pluggable, and the default asks for nothing.** With
``CACHELLM_BACKEND=auto`` the proxy uses Redis when it can reach it and a numpy
matrix in this process when it cannot, so a bare ``pip install`` works with no
server to run. Redis earns its place once several workers share a cache or the
entry count passes roughly 100k; below that the in-process scan is faster than
the round trip to reach Redis at all.

**The cache fails open.** If the chosen backend cannot start, the proxy keeps
serving by forwarding every request upstream and reports itself degraded. A
cache that takes an application down when it breaks is worse than no cache.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import structlog
from fastapi import Request

from cachellm.cache.coalesce import SingleFlight
from cachellm.cache.memory import MemoryAnalytics, MemoryExactStore, MemoryVectorStore
from cachellm.cache.service import CacheService
from cachellm.embeddings import build_embedder
from cachellm.embeddings.base import Embedder
from cachellm.observability.metrics import Metrics, get_metrics
from cachellm.providers import detect
from cachellm.providers.registry import ProviderRegistry
from cachellm.settings import Settings

if TYPE_CHECKING:  # redis modules are imported only when that backend is chosen
    from cachellm.cache.analytics import Analytics
    from cachellm.cache.exact_store import ExactStore
    from cachellm.cache.vector_store import VectorStore

log = structlog.get_logger(__name__)

REDIS_MISSING = (
    "The redis backend needs the optional extra. Install it with "
    "`pip install 'cachellm-proxy[redis]'`, or leave CACHELLM_BACKEND at its "
    "default and the cache runs in memory with nothing to install."
)


class RedisUnusableError(RuntimeError):
    """Redis answered, but it cannot hold this cache.

    Different in kind from "no Redis here". Nobody runs a Redis server by
    accident, so when one answers and still cannot be used, the operator almost
    certainly meant to use it and deserves to hear why it was passed over.
    """


def _where(url: str) -> str:
    """Host and port only. The URL may carry a password, and this gets logged."""
    parsed = urlparse(url)
    if parsed.hostname:
        return f"{parsed.hostname}:{parsed.port or 6379}"
    return "the configured address"


async def _require_search(redis: Any, settings: Settings) -> None:
    """Fail with a plain explanation before RedisVL fails with a cryptic one."""
    where = _where(settings.redis_url)
    db = int(redis.connection_pool.connection_kwargs.get("db", 0) or 0)
    if db != 0:
        raise RedisUnusableError(
            f"Redis at {where} answered, but CACHELLM_REDIS_URL selects database {db}. "
            "Redis search can only index database 0, so end the URL with /0."
        )
    try:
        await redis.execute_command("FT._LIST")
    except Exception as exc:
        if "unknown command" not in str(exc).lower():
            raise
        raise RedisUnusableError(
            f"Redis at {where} answered, but it has no search module, so it cannot "
            "store vectors. Redis 8 from Homebrew or the official Docker image includes "
            "it. Many Linux distribution packages do not."
        ) from exc


@dataclass
class AppState:
    settings: Settings
    metrics: Metrics
    providers: ProviderRegistry
    embedder: Embedder
    singleflight: SingleFlight = field(default_factory=SingleFlight)
    redis: Any = None
    vectors: VectorStore | MemoryVectorStore | None = None
    exact: ExactStore | MemoryExactStore | None = None
    analytics: Analytics | MemoryAnalytics | None = None
    cache: CacheService | None = None
    cache_available: bool = False
    degraded_reason: str = ""
    #: Which backend actually started: "redis" or "memory".
    backend: str = "none"
    #: Why a reachable Redis was passed over, shown by `cachellm stats`.
    backend_note: str = ""

    @property
    def caching_on(self) -> bool:
        return self.cache_available and self.cache is not None and self.settings.enabled


async def build_state(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    providers: ProviderRegistry | None = None,
) -> AppState:
    if providers is None:
        route_plan = detect.plan(settings)
        if route_plan.source == "detected":
            detect.apply_plan(settings, route_plan)
            default = route_plan.routes[route_plan.default]
            log.info(
                "provider_autodetected",
                using=detect.describe_route(default),
                also=[route_plan.routes[k].name for k in route_plan.enabled[1:]],
                hint="set CACHELLM_HOSTS to choose hosts, or CACHELLM_OPENAI_BASE_URL "
                "for one endpoint; run `cachellm providers` to see every option",
            )
        providers = ProviderRegistry(settings, plan=route_plan)

    state = AppState(
        settings=settings,
        metrics=get_metrics(),
        providers=providers,
        embedder=embedder or build_embedder(settings),
    )
    if settings.discover_models:
        # Fetched before serving, so the very first request already routes by
        # what each host really offers. Bounded by discover_timeout per host.
        await providers.discover()
    redis_error = ""
    if settings.backend in ("auto", "redis"):
        try:
            await _attach_redis(state, settings)
        except Exception as exc:
            redis_error = f"{type(exc).__name__}: {exc}"
            if settings.backend == "redis":
                unusable = isinstance(exc, RedisUnusableError)
                state.degraded_reason = str(exc) if unusable else redis_error
                log.error("redis_unavailable_failing_open", error=state.degraded_reason)
            elif isinstance(exc, RedisUnusableError):
                state.backend_note = f"{exc} Using the in-process cache instead."
                log.warning("redis_unusable_using_memory", reason=str(exc))
            else:
                log.info(
                    "redis_unavailable_using_memory",
                    error=redis_error[:160],
                    hint="set CACHELLM_BACKEND=redis to make this a hard failure",
                )

    if not state.cache_available and settings.backend in ("auto", "memory"):
        try:
            await _attach_memory(state, settings)
        except Exception as exc:
            state.degraded_reason = f"{type(exc).__name__}: {exc}"
            log.error("cache_unavailable_failing_open", error=state.degraded_reason)

    if not state.cache_available and not state.degraded_reason:
        state.degraded_reason = redis_error or "no cache backend could be started"

    try:
        await state.embedder.warmup()
    except Exception as exc:  # embeddings are optional for pass-through
        log.error("embedder_warmup_failed", error=str(exc)[:300])
        state.cache_available = False
        state.degraded_reason = f"embedder unavailable: {exc}"

    return state


async def _attach_redis(state: AppState, settings: Settings) -> None:
    try:
        from cachellm.cache.analytics import Analytics
        from cachellm.cache.exact_store import ExactStore
        from cachellm.cache.redis_client import build_redis
        from cachellm.cache.vector_store import VectorStore
    except ImportError as exc:
        raise ImportError(REDIS_MISSING) from exc

    redis = build_redis(settings)
    try:
        await redis.ping()
        await _require_search(redis, settings)
        vectors = VectorStore(redis, settings)
        await vectors.connect()
    except Exception:
        with contextlib.suppress(Exception):
            await redis.aclose()
        raise
    state.redis = redis
    _wire(
        state, settings, vectors, ExactStore(redis, settings), Analytics(redis, settings), "redis"
    )


async def _attach_memory(state: AppState, settings: Settings) -> None:
    vectors = MemoryVectorStore(settings)
    await vectors.connect()
    _wire(state, settings, vectors, MemoryExactStore(settings), MemoryAnalytics(settings), "memory")


def _wire(
    state: AppState, settings: Settings, vectors: Any, exact: Any, analytics: Any, backend: str
) -> None:
    state.vectors = vectors
    state.exact = exact
    state.analytics = analytics
    state.backend = backend
    state.cache = CacheService(
        settings=settings,
        embedder=state.embedder,
        vectors=vectors,
        exact=exact,
        analytics=analytics,
    )
    state.cache_available = True
    state.degraded_reason = ""


async def shutdown_state(state: AppState) -> None:
    await state.providers.close()
    if isinstance(state.vectors, MemoryVectorStore):
        await state.vectors.close()
    if state.redis is not None:
        await state.redis.aclose()


def get_state(request: Request) -> AppState:
    return request.app.state.cachellm  # type: ignore[no-any-return]
