"""The cache orchestrator: one lookup path, one store path, one invalidate path.

Everything the proxy knows about caching lives behind this class, so the HTTP
layer stays thin and the same logic can be reused by an in-process client
wrapper without a server.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import structlog

from cachellm.cache.analytics import Analytics, NearMiss
from cachellm.cache.entry import CacheEntry
from cachellm.cache.exact_store import ExactStore
from cachellm.cache.keys import (
    embedding_text,
    exact_hash,
    namespace_for,
    prompt_fingerprint,
)
from cachellm.cache.keys import (
    entry_id as make_entry_id,
)
from cachellm.cache.policy import PolicyDecision, decide
from cachellm.cache.vector_store import VectorStore
from cachellm.embeddings.base import Embedder
from cachellm.models import ChatCompletionRequest
from cachellm.pricing import estimate_cost
from cachellm.settings import Settings

log = structlog.get_logger(__name__)

LookupStatus = Literal["hit", "miss", "bypass", "shadow_hit"]


@dataclass
class LookupResult:
    status: LookupStatus
    decision: PolicyDecision
    namespace: str = ""
    similarity: float = 0.0
    tier: str = ""  # "exact" | "semantic"
    entry: CacheEntry | None = None
    embedding: np.ndarray | None = None
    exact: str = ""
    cache_text: str = ""
    lookup_ms: float = 0.0
    neighbours: list[tuple[str, float]] = field(default_factory=list)

    @property
    def served_from_cache(self) -> bool:
        return self.status == "hit" and self.entry is not None

    @property
    def category(self) -> str:
        return self.decision.category

    def saved_usd(self, model: str) -> float:
        if self.entry is None:
            return 0.0
        return estimate_cost(model, self.entry.prompt_tokens, self.entry.completion_tokens)


class CacheService:
    def __init__(
        self,
        *,
        settings: Settings,
        embedder: Embedder,
        vectors: VectorStore,
        exact: ExactStore,
        analytics: Analytics,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.vectors = vectors
        self.exact = exact
        self.analytics = analytics

    # ------------------------------------------------------------------ lookup
    async def lookup(
        self,
        request: ChatCompletionRequest,
        provider_name: str,
        *,
        cache_control: str = "",
    ) -> LookupResult:
        started = time.perf_counter()
        decision = decide(request, self.settings, cache_control=cache_control)
        if not decision.cacheable:
            return LookupResult(
                status="bypass", decision=decision, lookup_ms=(time.perf_counter() - started) * 1000
            )

        namespace = namespace_for(request, provider_name)
        raw_text = request.cache_text(include_history=self.settings.cache_multi_turn)
        exact_key = exact_hash(raw_text)

        # --- level 1: exact match, no embedding needed -----------------------
        entry_id = await self.exact.get(namespace, exact_key)
        if entry_id:
            entry = await self.vectors.get(entry_id)
            if entry is not None:
                status: LookupStatus = "shadow_hit" if self.settings.shadow_mode else "hit"
                return LookupResult(
                    status=status,
                    decision=decision,
                    namespace=namespace,
                    similarity=1.0,
                    tier="exact",
                    entry=entry,
                    exact=exact_key,
                    cache_text=raw_text,
                    lookup_ms=(time.perf_counter() - started) * 1000,
                )
            # Pointer outlived its entry (entry TTL shorter, or manual delete).
            await self.exact.delete(namespace, exact_key)

        # --- level 2: semantic nearest neighbour -----------------------------
        text_to_embed = embedding_text(raw_text, strip_fillers=self.settings.strip_filler_words)
        vector = await self.embedder.embed(text_to_embed)
        matches = await self.vectors.search(namespace, vector, self.settings.top_k)
        neighbours = [(m.entry_id, round(sim, 4)) for m, sim in matches]

        if matches:
            best_entry, best_sim = matches[0]
            if best_sim >= decision.threshold:
                status = "shadow_hit" if self.settings.shadow_mode else "hit"
                return LookupResult(
                    status=status,
                    decision=decision,
                    namespace=namespace,
                    similarity=best_sim,
                    tier="semantic",
                    entry=best_entry,
                    embedding=vector,
                    exact=exact_key,
                    cache_text=raw_text,
                    neighbours=neighbours,
                    lookup_ms=(time.perf_counter() - started) * 1000,
                )
            if best_sim >= decision.threshold - self.settings.near_miss_margin:
                await self.analytics.record_near_miss(
                    NearMiss(
                        prompt=raw_text,
                        matched_prompt=best_entry.prompt,
                        similarity=round(best_sim, 4),
                        threshold=decision.threshold,
                        category=decision.category,
                        namespace=namespace,
                        model=request.model,
                        at=time.time(),
                    )
                )
                await self.analytics.incr("near_misses")

        return LookupResult(
            status="miss",
            decision=decision,
            namespace=namespace,
            similarity=matches[0][1] if matches else 0.0,
            embedding=vector,
            exact=exact_key,
            cache_text=raw_text,
            neighbours=neighbours,
            lookup_ms=(time.perf_counter() - started) * 1000,
        )

    # ------------------------------------------------------------------- store
    async def store(
        self,
        *,
        lookup: LookupResult,
        request: ChatCompletionRequest,
        provider_name: str,
        response_text: str,
        prompt_tokens: int,
        completion_tokens: int,
        finish_reason: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> str | None:
        """Persist a completed answer. Only clean, finished responses are stored."""
        if not lookup.decision.cacheable or lookup.status == "bypass":
            return None
        if finish_reason not in ("stop", "end_turn", "eos", "stop_sequence", "complete"):
            log.debug("skip_store_incomplete", finish_reason=finish_reason)
            return None
        if not response_text.strip():
            return None

        vector = lookup.embedding
        if vector is None:
            text_to_embed = embedding_text(
                lookup.cache_text
                or request.cache_text(include_history=self.settings.cache_multi_turn),
                strip_fillers=self.settings.strip_filler_words,
            )
            vector = await self.embedder.embed(text_to_embed)

        namespace = lookup.namespace or namespace_for(request, provider_name)
        exact_key = lookup.exact or exact_hash(lookup.cache_text)
        eid = make_entry_id(namespace, exact_key)
        entry = CacheEntry(
            entry_id=eid,
            namespace=namespace,
            category=lookup.decision.category,
            model=request.model,
            provider=provider_name,
            prompt=lookup.cache_text or request.last_user_text(),
            response_text=response_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            ttl=lookup.decision.ttl,
            exact_hash=exact_key,
            embedding=vector,
            tool_calls=tool_calls,
        )
        await self.vectors.put(entry)
        await self.exact.put(namespace, exact_key, eid, lookup.decision.ttl)
        await self.analytics.incr("entries_written")
        log.debug(
            "cache_store",
            entry_id=eid,
            category=entry.category,
            fingerprint=prompt_fingerprint(entry.prompt),
            ttl=entry.ttl,
        )
        return eid

    # -------------------------------------------------------------- maintenance
    async def register_hit(self, lookup: LookupResult, model: str) -> float:
        """Bump hit bookkeeping and return the modelled dollars saved."""
        if lookup.entry is None:
            return 0.0
        await self.vectors.touch_hit(lookup.entry.entry_id)
        saved = lookup.saved_usd(model)
        await self.analytics.bulk(
            {
                "hits": 1,
                f"hits_{lookup.tier}": 1,
                "tokens_saved_prompt": lookup.entry.prompt_tokens,
                "tokens_saved_completion": lookup.entry.completion_tokens,
            },
            {"usd_saved": saved},
        )
        return saved

    async def invalidate(
        self, *, namespace: str | None = None, model: str | None = None, drop_all: bool = False
    ) -> int:
        removed = await self.vectors.invalidate(namespace=namespace, model=model, drop_all=drop_all)
        await self.analytics.incr("invalidations")
        log.info(
            "cache_invalidate", namespace=namespace, model=model, drop_all=drop_all, removed=removed
        )
        return removed

    async def stats(self) -> dict[str, Any]:
        counters = await self.analytics.counters()
        requests = counters.get("requests", 0.0)
        hits = counters.get("hits", 0.0)
        cacheable = counters.get("cacheable_requests", 0.0)
        return {
            "entries": await self.vectors.count(),
            "requests": int(requests),
            "cacheable_requests": int(cacheable),
            "hits": int(hits),
            "hits_exact": int(counters.get("hits_exact", 0.0)),
            "hits_semantic": int(counters.get("hits_semantic", 0.0)),
            "misses": int(counters.get("misses", 0.0)),
            "bypassed": int(counters.get("bypass", 0.0)),
            "shadow_hits": int(counters.get("shadow_hits", 0.0)),
            "near_misses": int(counters.get("near_misses", 0.0)),
            "coalesced": int(counters.get("coalesced", 0.0)),
            "errors": int(counters.get("provider_errors", 0.0)),
            "hit_rate": round(hits / requests, 4) if requests else 0.0,
            "hit_rate_of_cacheable": round(hits / cacheable, 4) if cacheable else 0.0,
            "usd_saved": round(counters.get("usd_saved", 0.0), 6),
            "usd_spent": round(counters.get("usd_spent", 0.0), 6),
            "tokens_saved": int(
                counters.get("tokens_saved_prompt", 0.0)
                + counters.get("tokens_saved_completion", 0.0)
            ),
            "latency_ms": {
                "hit": await self.analytics.latency_percentiles("hit"),
                "miss": await self.analytics.latency_percentiles("miss"),
            },
        }
