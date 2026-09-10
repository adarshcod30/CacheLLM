"""Operator endpoints: stats, invalidation, near-miss analysis, threshold tuning.

These are what turn the cache from a black box into something a team is willing
to leave switched on. Every one of them answers a question an operator actually
asks: what is it doing, what would a different threshold do, and how do I make
it forget something right now.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog
from fastapi import APIRouter, Body, Query, Request
from pydantic import BaseModel, Field

from cachellm.api.auth import verify
from cachellm.api.deps import get_state
from cachellm.errors import CacheLLMError
from cachellm.providers.catalog import (
    BEDROCK_VENDORS,
    HOSTS,
    OPENAI_FAMILIES,
    ROUTING_PREFIXES,
    looks_like_bedrock,
    looks_like_openai,
)

log = structlog.get_logger(__name__)
router = APIRouter()

DEFAULT_SWEEP = [0.80, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99]


class InvalidateRequest(BaseModel):
    namespace: str | None = None
    model: str | None = None
    all: bool = False


class LabelledPair(BaseModel):
    a: str
    b: str
    duplicate: bool


class SweepRequest(BaseModel):
    pairs: list[LabelledPair] = Field(min_length=1)
    thresholds: list[float] | None = None


def _require_cache(request: Request) -> Any:
    state = get_state(request)
    if not state.cache_available or state.cache is None:
        raise CacheLLMError(
            503,
            f"Cache backend unavailable: {state.degraded_reason or 'not connected'}",
            "api_error",
            code="cache_unavailable",
        )
    return state


@router.get("/stats")
async def stats(request: Request) -> dict[str, Any]:
    state = get_state(request)
    if state.settings.require_auth_for_admin:
        verify(request, state.settings.client_keys)
    if not state.cache_available or state.cache is None:
        return {
            "cache_available": False,
            "caching_enabled": state.settings.enabled,
            "degraded_reason": state.degraded_reason or "backend not connected",
        }
    data = await state.cache.stats()
    data["cache_available"] = True
    data["backend"] = state.backend
    if state.backend_note:
        data["backend_note"] = state.backend_note
    data["caching_enabled"] = state.settings.enabled
    data["shadow_mode"] = state.settings.shadow_mode
    data["in_flight"] = state.singleflight.in_flight
    state.metrics.entries.set(data.get("entries", 0))
    return data


@router.get("/config")
async def config(request: Request) -> dict[str, Any]:
    state = get_state(request)
    if state.settings.require_auth_for_admin:
        verify(request, state.settings.client_keys)
    s = state.settings
    return {
        "enabled": s.enabled,
        "shadow_mode": s.shadow_mode,
        "backend": {
            "requested": s.backend,
            "active": state.backend,
            "max_entries": s.memory_max_entries if state.backend == "memory" else None,
        },
        "embedding": {
            "backend": s.embedding_backend,
            "model": s.embedding_model,
            "dim": s.expected_dim(),
        },
        "thresholds": {
            c: s.threshold_for(c)
            for c in (
                "factual",
                "classification",
                "creative",
                "volatile",
                "conversational",
                "default",
            )
        },
        "ttl_seconds": {
            c: s.ttl_for(c)
            for c in (
                "factual",
                "classification",
                "creative",
                "volatile",
                "conversational",
                "default",
            )
        },
        "rules": {
            "max_cacheable_temperature": s.max_cacheable_temperature,
            "cache_multi_turn": s.cache_multi_turn,
            "cache_json_mode": s.cache_json_mode,
            "cache_tool_calls": s.cache_tool_calls,
            "pii_guard": s.pii_guard,
            "strip_filler_words": s.strip_filler_words,
            "max_prompt_chars": s.max_prompt_chars,
        },
        "default_provider": s.default_provider,
        "auth_enabled": s.auth_enabled,
    }


@router.get("/providers")
async def providers(request: Request) -> dict[str, Any]:
    """Where model names go, and what to type for each host.

    Answers the question the router answers, but for a human: which names route
    where, what base URL each hosting provider needs, and which need a pip extra.
    """
    state = get_state(request)
    if state.settings.require_auth_for_admin:
        verify(request, state.settings.client_keys)
    return {
        "default_provider": state.settings.default_provider,
        "routing": {
            "1_explicit_prefix": list(ROUTING_PREFIXES),
            "2_bedrock_vendor_prefixes": list(BEDROCK_VENDORS),
            "2_openai_families": list(OPENAI_FAMILIES),
            "3_everything_else": state.settings.default_provider,
        },
        "hosts": [
            {
                "name": h.name,
                "adapter": h.provider,
                "base_url": h.base_url,
                "pip_extra": h.extra,
                "example_models": list(h.examples),
                "note": h.note,
            }
            for h in HOSTS
        ],
    }


@router.get("/route/{model:path}")
async def route(request: Request, model: str) -> dict[str, Any]:
    """Explain where one model name would go, and why."""
    state = get_state(request)
    if state.settings.require_auth_for_admin:
        verify(request, state.settings.client_keys)
    provider, name = state.providers.resolve(model)
    head, sep, _ = model.partition("/")
    if sep and head.lower() in ROUTING_PREFIXES:
        why = "explicit routing prefix"
    elif looks_like_bedrock(model):
        why = "Bedrock vendor naming convention"
    elif looks_like_openai(model):
        why = "OpenAI model family"
    else:
        why = f"no vendor could be identified, so the default ({name}) applies"
    return {
        "model": model,
        "provider": name,
        "forwarded_as": provider.resolve_model(model),
        "reason": why,
    }


@router.post("/invalidate")
async def invalidate(request: Request, body: InvalidateRequest) -> dict[str, Any]:
    state = _require_cache(request)
    if state.settings.require_auth_for_admin:
        verify(request, state.settings.client_keys)
    if not (body.namespace or body.model or body.all):
        raise CacheLLMError(400, "Pass one of `namespace`, `model` or `all`.", param="namespace")
    removed = await state.cache.invalidate(
        namespace=body.namespace, model=body.model, drop_all=body.all
    )
    return {
        "removed_keys": removed,
        "namespace": body.namespace,
        "model": body.model,
        "all": body.all,
    }


@router.get("/requests")
async def requests_log(request: Request, limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
    """The recent request log: what the cache did with each one.

    This is what `cachellm stats` reads. It has to come over HTTP because with
    the in-memory backend the cache lives inside the serving process, and a CLI
    building its own state would report on an empty cache of its own.
    """
    state = _require_cache(request)
    if state.settings.require_auth_for_admin:
        verify(request, state.settings.client_keys)
    rows = await state.analytics.recent_requests(limit=limit)
    return {"count": len(rows), "requests": rows}


@router.get("/near-misses")
async def near_misses(request: Request, limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
    state = _require_cache(request)
    if state.settings.require_auth_for_admin:
        verify(request, state.settings.client_keys)
    rows = await state.analytics.near_misses(limit=limit)
    return {"count": len(rows), "near_misses": rows}


@router.get("/near-miss-histogram")
async def near_miss_histogram(
    request: Request, buckets: int = Query(20, ge=5, le=100)
) -> dict[str, Any]:
    """Cumulative view: how many more requests each lower threshold would serve."""
    state = _require_cache(request)
    if state.settings.require_auth_for_admin:
        verify(request, state.settings.client_keys)
    return {"histogram": await state.analytics.near_miss_histogram(buckets=buckets)}


@router.get("/entries")
async def entries(request: Request, limit: int = Query(20, ge=1, le=200)) -> dict[str, Any]:
    state = _require_cache(request)
    if state.settings.require_auth_for_admin:
        verify(request, state.settings.client_keys)
    keys = await state.vectors.scan_keys()
    rows = []
    for key in keys[:limit]:
        entry_id = key.split(state.settings.entry_prefix, 1)[-1]
        entry = await state.vectors.get(entry_id)
        if entry is None:
            continue
        summary = entry.summary()
        if not state.settings.log_prompts:
            summary["prompt"] = summary["prompt"][:120]
            summary["response_text"] = summary["response_text"][:160]
        rows.append(summary)
    return {"total_keys": len(keys), "returned": len(rows), "entries": rows}


@router.post("/reset-stats")
async def reset_stats(request: Request) -> dict[str, str]:
    state = _require_cache(request)
    if state.settings.require_auth_for_admin:
        verify(request, state.settings.client_keys)
    await state.analytics.reset()
    return {"status": "counters reset"}


@router.post("/threshold-sweep")
async def threshold_sweep(request: Request, body: SweepRequest = Body(...)) -> dict[str, Any]:
    """Score labelled prompt pairs at several thresholds.

    This is the answer to the only question that really matters when tuning a
    semantic cache: at threshold X, how many extra requests do I serve, and how
    many of those are actually different questions wearing similar words?

    ``precision`` is the share of would-be cache hits that were genuine
    duplicates. ``false_positive_rate`` is the share of *non*-duplicate pairs
    that would wrongly hit. Read them together: hit rate you can only spend if
    precision holds.
    """
    state = get_state(request)
    if state.settings.require_auth_for_admin:
        verify(request, state.settings.client_keys)

    thresholds = sorted(body.thresholds or DEFAULT_SWEEP)
    texts_a = [p.a for p in body.pairs]
    texts_b = [p.b for p in body.pairs]
    labels = np.array([p.duplicate for p in body.pairs], dtype=bool)

    vectors_a = await state.embedder.embed_batch(texts_a)
    vectors_b = await state.embedder.embed_batch(texts_b)
    sims = np.array(
        [float(np.dot(a, b)) for a, b in zip(vectors_a, vectors_b, strict=True)],
        dtype=np.float32,
    )

    positives = int(labels.sum())
    negatives = int((~labels).sum())
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        predicted = sims >= threshold
        tp = int((predicted & labels).sum())
        fp = int((predicted & ~labels).sum())
        fn = int((~predicted & labels).sum())
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / positives if positives else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "threshold": round(threshold, 4),
                "would_hit": tp + fp,
                "hit_rate": round((tp + fp) / len(body.pairs), 4),
                "true_hits": tp,
                "false_hits": fp,
                "missed_duplicates": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "false_positive_rate": round(fp / negatives, 4) if negatives else 0.0,
            }
        )

    best = max(rows, key=lambda r: r["f1"]) if rows else None
    safe = [r for r in rows if r["false_positive_rate"] <= 0.01]
    return {
        "pairs": len(body.pairs),
        "duplicates": positives,
        "non_duplicates": negatives,
        "embedding_model": state.embedder.name,
        "similarity": {
            "duplicates_mean": round(float(sims[labels].mean()), 4) if positives else None,
            "non_duplicates_mean": round(float(sims[~labels].mean()), 4) if negatives else None,
        },
        "sweep": rows,
        "best_f1": best,
        "lowest_threshold_under_1pct_false_positives": (
            min(safe, key=lambda r: r["threshold"]) if safe else None
        ),
    }
