"""The drop-in endpoint: POST /v1/chat/completions.

Same request shape as OpenAI, same response shape, same error envelope, plus a
set of ``X-Cache-*`` headers describing what happened. An application adopts
CacheLLM by changing one base URL and nothing else.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from cachellm.api import sse
from cachellm.api.auth import verify
from cachellm.api.deps import AppState, get_state
from cachellm.cache.analytics import RequestRecord
from cachellm.cache.entry import CacheEntry
from cachellm.cache.keys import prompt_fingerprint
from cachellm.cache.policy import PolicyDecision
from cachellm.cache.service import LookupResult
from cachellm.errors import CacheMissError, UpstreamError
from cachellm.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelCard,
    ModelList,
)
from cachellm.observability import span
from cachellm.pricing import estimate_cost
from cachellm.providers.base import Provider

log = structlog.get_logger(__name__)
router = APIRouter()

BYPASS_LOOKUP = LookupResult(
    status="bypass",
    decision=PolicyDecision(False, "default", 1.0, 0, "cache_unavailable"),
)


def cache_headers(
    lookup: LookupResult, *, total_ms: float, saved_usd: float = 0.0, coalesced: bool = False
) -> dict[str, str]:
    status = {
        "hit": "HIT",
        "miss": "MISS",
        "bypass": "BYPASS",
        "shadow_hit": "SHADOW",
    }[lookup.status]
    headers = {
        "X-Cache": status,
        "X-Cache-Category": lookup.category,
        "X-Cache-Lookup-Ms": f"{lookup.lookup_ms:.2f}",
        "X-Cache-Latency-Ms": f"{total_ms:.2f}",
        "X-Cache-Threshold": f"{lookup.decision.threshold:.3f}",
    }
    if lookup.namespace:
        headers["X-Cache-Namespace"] = lookup.namespace
    if lookup.tier:
        headers["X-Cache-Tier"] = lookup.tier
    if lookup.status in ("hit", "shadow_hit", "miss"):
        headers["X-Cache-Similarity"] = f"{lookup.similarity:.4f}"
    if lookup.entry is not None:
        headers["X-Cache-Entry-Id"] = lookup.entry.entry_id
        headers["X-Cache-Age-Seconds"] = f"{lookup.entry.age_seconds():.0f}"
    if lookup.decision.bypass_reason:
        headers["X-Cache-Bypass-Reason"] = lookup.decision.bypass_reason
    if saved_usd:
        headers["X-Cache-Saved-USD"] = f"{saved_usd:.8f}"
    if coalesced:
        headers["X-Cache-Coalesced"] = "true"
    return headers


def _debug_block(lookup: LookupResult, saved_usd: float, coalesced: bool) -> dict[str, Any]:
    """Non-standard extra field. SDKs ignore unknown keys; humans find it useful."""
    return {
        "status": lookup.status,
        "tier": lookup.tier or None,
        "similarity": round(lookup.similarity, 4)
        if lookup.tier or lookup.status == "miss"
        else None,
        "threshold": lookup.decision.threshold,
        "category": lookup.category,
        "namespace": lookup.namespace or None,
        "bypass_reason": lookup.decision.bypass_reason or None,
        "saved_usd": round(saved_usd, 8) or None,
        "coalesced": coalesced or None,
        "lookup_ms": round(lookup.lookup_ms, 2),
    }


async def _record_outcome(
    state: AppState,
    lookup: LookupResult,
    model: str,
    provider: str,
    total_ms: float,
    saved_usd: float = 0.0,
    spent_usd: float = 0.0,
    prompt: str = "",
) -> None:
    metrics = state.metrics
    result = lookup.status
    metrics.requests.labels(
        result=result, category=lookup.category, provider=provider, model=model
    ).inc()
    metrics.request_duration.labels(result=result).observe(total_ms / 1000.0)
    if lookup.tier or result == "miss":
        metrics.lookup_duration.labels(tier=lookup.tier or "semantic").observe(
            lookup.lookup_ms / 1000.0
        )
        metrics.similarity.labels(result=result).observe(max(0.0, min(1.0, lookup.similarity)))
    if state.analytics is not None:
        counters = {"requests": 1}
        if lookup.decision.cacheable:
            counters["cacheable_requests"] = 1
        if result == "miss":
            counters["misses"] = 1
        elif result == "bypass":
            counters["bypass"] = 1
        elif result == "shadow_hit":
            counters["shadow_hits"] = 1
        await state.analytics.bulk(counters)
        await state.analytics.record_latency("hit" if result == "hit" else "miss", total_ms)
        # The request log is what `cachellm stats` and `cachellm watch` read, so
        # a person can see which of their own requests hit without a dashboard.
        await state.analytics.record_request(
            RequestRecord(
                at=time.time(),
                status={"hit": "HIT", "miss": "MISS", "bypass": "BYPASS", "shadow_hit": "SHADOW"}[
                    result
                ],
                tier=lookup.tier,
                similarity=round(lookup.similarity, 4),
                category=lookup.category,
                model=model,
                latency_ms=round(total_ms, 2),
                saved_usd=saved_usd,
                spent_usd=spent_usd,
                # A bypassed request never reaches the point where cache_text is
                # set, so the route passes the prompt in; without it the log
                # would show blank rows for exactly the requests you most want
                # to understand.
                prompt=lookup.cache_text or prompt,
                reason=lookup.decision.bypass_reason,
            )
        )


@router.get("/models", response_model=ModelList)
async def list_models(request: Request) -> ModelList:
    state = get_state(request)
    verify(request, state.settings.client_keys)
    return ModelList(data=[ModelCard(id=m) for m in state.providers.models()])


@router.post("/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest) -> Any:
    state = get_state(request)
    verify(request, state.settings.client_keys)
    started = time.perf_counter()

    cache_control = request.headers.get("x-cache-control", "")
    only_if_cached = "only-if-cached" in cache_control.lower()
    provider, provider_name = state.providers.resolve(body.model)

    with span(
        "cachellm.request",
        **{
            "gen_ai.system": provider_name,
            "gen_ai.request.model": body.model,
            "gen_ai.request.temperature": body.effective_temperature,
            "cachellm.prompt_fingerprint": prompt_fingerprint(body.last_user_text()),
        },
    ) as current:
        # ---------------------------------------------------------- lookup
        if state.caching_on and state.cache is not None:
            lookup = await state.cache.lookup(body, provider_name, cache_control=cache_control)
        else:
            lookup = BYPASS_LOOKUP

        if current is not None:
            current.set_attribute("cachellm.status", lookup.status)
            current.set_attribute("cachellm.category", lookup.category)
            current.set_attribute("cachellm.similarity", round(lookup.similarity, 4))

        if lookup.status == "shadow_hit" and state.analytics is not None:
            state.metrics.shadow_hits.labels(category=lookup.category).inc()
            log.info(
                "shadow_hit",
                similarity=round(lookup.similarity, 4),
                tier=lookup.tier,
                category=lookup.category,
                fingerprint=prompt_fingerprint(lookup.cache_text),
            )

        # ------------------------------------------------------- cache hit
        if lookup.served_from_cache and state.cache is not None and lookup.entry is not None:
            entry = lookup.entry
            saved = await state.cache.register_hit(lookup, body.model)
            state.metrics.observe_cost(body.model, saved, "saved")
            state.metrics.observe_tokens(
                body.model, entry.prompt_tokens, entry.completion_tokens, "saved"
            )
            total_ms = (time.perf_counter() - started) * 1000
            await _record_outcome(
                state,
                lookup,
                body.model,
                provider_name,
                total_ms,
                saved_usd=saved,
                prompt=body.last_user_text(),
            )
            headers = cache_headers(lookup, total_ms=total_ms, saved_usd=saved)
            if body.stream:
                return _replay_stream(body, entry, headers)
            payload = ChatCompletionResponse.from_text(
                model=body.model,
                text=entry.response_text,
                prompt_tokens=entry.prompt_tokens,
                completion_tokens=entry.completion_tokens,
                finish_reason=entry.finish_reason,
            ).model_dump()
            payload["cachellm"] = _debug_block(lookup, saved, False)
            return JSONResponse(payload, headers=headers)

        if only_if_cached:
            raise CacheMissError()

        # ----------------------------------------------------------- miss
        if body.stream:
            return await _proxy_stream(state, body, lookup, provider, provider_name, started)
        return await _proxy_once(state, body, lookup, provider, provider_name, started)


async def _proxy_once(
    state: AppState,
    body: ChatCompletionRequest,
    lookup: LookupResult,
    provider: Provider,
    provider_name: str,
    started: float,
) -> JSONResponse:
    """Non-streaming miss: one upstream call, shared by identical concurrent misses."""
    coalesce_key = f"{lookup.namespace}:{lookup.exact}" if lookup.namespace else ""

    async def call() -> Any:
        with span(
            "cachellm.provider",
            **{"gen_ai.system": provider_name, "gen_ai.request.model": body.model},
        ):
            return await provider.complete(body)

    try:
        if coalesce_key and state.caching_on:
            result, coalesced = await state.singleflight.do(coalesce_key, call)
        else:
            result, coalesced = await call(), False
    except UpstreamError:
        state.metrics.provider_errors.labels(provider=provider_name).inc()
        if state.analytics is not None:
            await state.analytics.incr("provider_errors")
        raise

    if coalesced:
        state.metrics.coalesced.inc()
        if state.analytics is not None:
            await state.analytics.incr("coalesced")

    if state.caching_on and state.cache is not None and not coalesced:
        await state.cache.store(
            lookup=lookup,
            request=body,
            provider_name=provider_name,
            response_text=result.text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            finish_reason=result.finish_reason,
            tool_calls=result.tool_calls,
        )

    spent = estimate_cost(body.model, result.prompt_tokens, result.completion_tokens)
    state.metrics.observe_cost(body.model, spent, "spent")
    state.metrics.observe_tokens(
        body.model, result.prompt_tokens, result.completion_tokens, "spent"
    )
    if state.analytics is not None:
        await state.analytics.incr_float("usd_spent", spent)

    total_ms = (time.perf_counter() - started) * 1000
    await _record_outcome(
        state,
        lookup,
        body.model,
        provider_name,
        total_ms,
        spent_usd=spent,
        prompt=body.last_user_text(),
    )

    payload = ChatCompletionResponse.from_text(
        model=body.model,
        text=result.text,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        finish_reason=result.finish_reason,
    ).model_dump()
    payload["cachellm"] = _debug_block(lookup, 0.0, coalesced)
    return JSONResponse(
        payload, headers=cache_headers(lookup, total_ms=total_ms, coalesced=coalesced)
    )


def _replay_stream(
    body: ChatCompletionRequest, entry: CacheEntry, headers: dict[str, str]
) -> StreamingResponse:
    """Serve a cache hit to a client that asked for a stream."""
    stream_id = sse.new_stream_id()
    created = int(time.time())

    async def generate() -> AsyncIterator[str]:
        yield sse.role_chunk(stream_id, body.model, created)
        for piece in sse.split_for_replay(entry.response_text):
            yield sse.text_chunk(stream_id, body.model, created, piece)
        yield sse.final_chunk(
            stream_id,
            body.model,
            created,
            entry.finish_reason,
            {
                "prompt_tokens": entry.prompt_tokens,
                "completion_tokens": entry.completion_tokens,
                "total_tokens": entry.prompt_tokens + entry.completion_tokens,
            },
        )
        yield sse.DONE

    return StreamingResponse(generate(), media_type="text/event-stream", headers=headers)


async def _proxy_stream(
    state: AppState,
    body: ChatCompletionRequest,
    lookup: LookupResult,
    provider: Provider,
    provider_name: str,
    started: float,
) -> StreamingResponse:
    """Streaming miss: forward chunks live while buffering for the cache.

    The buffer is only committed once the upstream reports a clean finish. A
    disconnect halfway through leaves nothing behind, which is exactly what you
    want: a truncated answer served from cache forever would be a silent bug.
    """
    stream_id = sse.new_stream_id()
    created = int(time.time())

    async def generate() -> AsyncIterator[str]:
        buffer: list[str] = []
        finish_reason: str | None = None
        prompt_tokens = 0
        completion_tokens = 0
        yield sse.role_chunk(stream_id, body.model, created)
        try:
            async for event in provider.stream(body):
                if event.delta:
                    buffer.append(event.delta)
                    yield sse.text_chunk(stream_id, body.model, created, event.delta)
                if event.finish_reason:
                    finish_reason = event.finish_reason
                if event.prompt_tokens or event.completion_tokens:
                    prompt_tokens = event.prompt_tokens or prompt_tokens
                    completion_tokens = event.completion_tokens or completion_tokens
        except UpstreamError as exc:
            state.metrics.provider_errors.labels(provider=provider_name).inc()
            if state.analytics is not None:
                await state.analytics.incr("provider_errors")
            log.warning("stream_failed", error=str(exc)[:200])
            yield sse.final_chunk(stream_id, body.model, created, "error", None)
            yield sse.DONE
            return

        text = "".join(buffer)
        if not completion_tokens and text:
            completion_tokens = provider.approx_tokens(text)
        if not prompt_tokens:
            prompt_tokens = provider.approx_tokens(body.last_user_text())

        yield sse.final_chunk(
            stream_id,
            body.model,
            created,
            finish_reason or "stop",
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )
        yield sse.DONE

        if state.caching_on and state.cache is not None and state.settings.cache_streaming:
            await state.cache.store(
                lookup=lookup,
                request=body,
                provider_name=provider_name,
                response_text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason or "stop",
            )
        spent = estimate_cost(body.model, prompt_tokens, completion_tokens)
        state.metrics.observe_cost(body.model, spent, "spent")
        state.metrics.observe_tokens(body.model, prompt_tokens, completion_tokens, "spent")
        if state.analytics is not None:
            await state.analytics.incr_float("usd_spent", spent)
        total_ms = (time.perf_counter() - started) * 1000
        await _record_outcome(
            state,
            lookup,
            body.model,
            provider_name,
            total_ms,
            spent_usd=spent,
            prompt=body.last_user_text(),
        )

    total_ms = (time.perf_counter() - started) * 1000
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=cache_headers(lookup, total_ms=total_ms),
    )
