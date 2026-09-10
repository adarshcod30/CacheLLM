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

from dataclasses import dataclass, field
from typing import Any

import structlog
from fastapi import Request

from cachellm.cache.analytics import Analytics
from cachellm.cache.coalesce import SingleFlight
from cachellm.cache.exact_store import ExactStore
from cachellm.cache.memory import MemoryAnalytics, MemoryExactStore, MemoryVectorStore
from cachellm.cache.redis_client import build_redis
from cachellm.cache.service import CacheService
from cachellm.cache.vector_store import VectorStore
from cachellm.embeddings import build_embedder
from cachellm.embeddings.base import Embedder
from cachellm.observability.metrics import Metrics, get_metrics
from cachellm.providers.registry import ProviderRegistry
from cachellm.settings import Settings

log = structlog.get_logger(__name__)


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

    @property
    def caching_on(self) -> bool:
        return self.cache_available and self.cache is not None and self.settings.enabled


async def build_state(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    providers: ProviderRegistry | None = None,
) -> AppState:
    state = AppState(
        settings=settings,
        metrics=get_metrics(),
        providers=providers or ProviderRegistry(settings),
        embedder=embedder or build_embedder(settings),
    )
    redis_error = ""
    if settings.backend in ("auto", "redis"):
        try:
            await _attach_redis(state, settings)
        except Exception as exc:
            redis_error = f"{type(exc).__name__}: {exc}"
            if settings.backend == "redis":
                state.degraded_reason = redis_error
                log.error("redis_unavailable_failing_open", error=redis_error)
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
    redis = build_redis(settings)
    await redis.ping()
    vectors = VectorStore(redis, settings)
    await vectors.connect()
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
