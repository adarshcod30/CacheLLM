"""Application state and its lifecycle.

One deliberate design choice here: the cache **fails open**. If Redis is
unreachable or the index cannot be built, the proxy keeps serving by forwarding
every request upstream and reports itself degraded. A cache that takes an
application down when it breaks is worse than no cache, and this is the first
thing a reviewer looks for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog
from fastapi import Request

from cachellm.cache.analytics import Analytics
from cachellm.cache.coalesce import SingleFlight
from cachellm.cache.exact_store import ExactStore
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
    vectors: VectorStore | None = None
    exact: ExactStore | None = None
    analytics: Analytics | None = None
    cache: CacheService | None = None
    cache_available: bool = False
    degraded_reason: str = ""

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
    try:
        redis = build_redis(settings)
        await redis.ping()
        vectors = VectorStore(redis, settings)
        await vectors.connect()
        exact = ExactStore(redis, settings)
        analytics = Analytics(redis, settings)
        state.redis = redis
        state.vectors = vectors
        state.exact = exact
        state.analytics = analytics
        state.cache = CacheService(
            settings=settings,
            embedder=state.embedder,
            vectors=vectors,
            exact=exact,
            analytics=analytics,
        )
        state.cache_available = True
    except Exception as exc:  # degrade, never crash the proxy
        state.cache_available = False
        state.degraded_reason = f"{type(exc).__name__}: {exc}"
        log.error("cache_unavailable_failing_open", error=state.degraded_reason)

    try:
        await state.embedder.warmup()
    except Exception as exc:  # embeddings are optional for pass-through
        log.error("embedder_warmup_failed", error=str(exc)[:300])
        state.cache_available = False
        state.degraded_reason = f"embedder unavailable: {exc}"

    return state


async def shutdown_state(state: AppState) -> None:
    await state.providers.close()
    if state.redis is not None:
        await state.redis.aclose()


def get_state(request: Request) -> AppState:
    return request.app.state.cachellm  # type: ignore[no-any-return]
