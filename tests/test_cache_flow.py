"""End-to-end cache behaviour against a real Redis.

These are the tests that would catch the failures that actually matter: an
answer served across a system-prompt boundary, a truncated response cached
forever, a personal question stored where another user can reach it.
"""

from __future__ import annotations

import asyncio

import pytest

from cachellm.models import ChatCompletionRequest
from tests.conftest import requires_redis

pytestmark = [pytest.mark.integration, requires_redis]


def make_request(text: str, **kwargs) -> ChatCompletionRequest:
    messages = kwargs.pop("messages", None) or [{"role": "user", "content": text}]
    kwargs.setdefault("temperature", 0.0)
    return ChatCompletionRequest(model="fake/echo", messages=messages, **kwargs)


async def seed(state, text: str, answer: str = "stored answer", **kwargs) -> str:
    """Run a lookup then store, the way the route does on a miss."""
    request = make_request(text, **kwargs)
    lookup = await state.cache.lookup(request, "fake")
    entry_id = await state.cache.store(
        lookup=lookup,
        request=request,
        provider_name="fake",
        response_text=answer,
        prompt_tokens=10,
        completion_tokens=20,
        finish_reason="stop",
    )
    return entry_id


async def test_exact_repeat_hits_without_embedding(state) -> None:
    await seed(state, "What is the Python programming language?")
    result = await state.cache.lookup(
        make_request("what is the python programming language"), "fake"
    )
    assert result.status == "hit"
    assert result.tier == "exact"
    assert result.similarity == 1.0
    assert result.entry.response_text == "stored answer"


async def test_paraphrase_hits_semantically(state) -> None:
    state.settings.threshold_factual = 0.90
    await seed(state, "what is the python programming language")
    result = await state.cache.lookup(make_request("what is python programming language"), "fake")
    assert result.status == "hit"
    assert result.tier == "semantic"
    assert 0.90 <= result.similarity < 1.0


async def test_similar_wording_different_meaning_does_not_hit(state) -> None:
    """The failure mode that matters: France and Finland look alike, are not."""
    state.settings.threshold_factual = 0.90
    await seed(state, "what is the capital of france", answer="Paris")
    result = await state.cache.lookup(make_request("what is the capital of finland"), "fake")
    assert result.status == "miss"
    assert result.similarity < 0.90


async def test_a_high_threshold_rejects_a_paraphrase(state) -> None:
    state.settings.threshold_factual = 0.99
    await seed(state, "what is the python programming language")
    result = await state.cache.lookup(make_request("what is python programming language"), "fake")
    assert result.status == "miss"


async def test_system_prompts_are_isolated(state) -> None:
    doctor = [
        {"role": "system", "content": "You are a doctor."},
        {"role": "user", "content": "What is a virus?"},
    ]
    poet = [
        {"role": "system", "content": "You are a poet."},
        {"role": "user", "content": "What is a virus?"},
    ]
    await seed(state, "", messages=doctor, answer="A pathogen.")
    result = await state.cache.lookup(make_request("", messages=poet), "fake")
    assert result.status == "miss"


async def test_models_are_isolated(state) -> None:
    await seed(state, "what is python")
    other = ChatCompletionRequest(
        model="fake/other-model",
        temperature=0.0,
        messages=[{"role": "user", "content": "what is python"}],
    )
    assert (await state.cache.lookup(other, "fake")).status == "miss"


async def test_uncacheable_requests_store_nothing(state) -> None:
    entry_id = await seed(state, "What is Python?", temperature=0.9)
    assert entry_id is None
    assert await state.vectors.count() == 0


async def test_personal_questions_are_never_stored(state) -> None:
    entry_id = await seed(state, "what is my order 123456789012 status")
    assert entry_id is None
    assert await state.vectors.count() == 0


async def test_truncated_responses_are_not_stored(state) -> None:
    request = make_request("what is python")
    lookup = await state.cache.lookup(request, "fake")
    entry_id = await state.cache.store(
        lookup=lookup,
        request=request,
        provider_name="fake",
        response_text="half an ans",
        prompt_tokens=5,
        completion_tokens=5,
        finish_reason="length",
    )
    assert entry_id is None
    assert await state.vectors.count() == 0


async def test_truncation_skips_are_counted_and_explained(state) -> None:
    """A low hit rate caused by max_tokens must be diagnosable, not mysterious."""
    for i in range(5):
        request = make_request(f"question number {i}")
        lookup = await state.cache.lookup(request, "fake")
        await state.cache.store(
            lookup=lookup,
            request=request,
            provider_name="fake",
            response_text="a truncated ans",
            prompt_tokens=5,
            completion_tokens=120,
            finish_reason="length",
        )
        await state.analytics.incr("misses")
    stats = await state.cache.stats()
    assert stats["stores_skipped_truncated"] == 5
    assert any("max_tokens" in note for note in stats["diagnostics"])


async def test_truncated_responses_can_be_cached_on_purpose(state) -> None:
    state.settings.cache_truncated = True
    request = make_request("what is python")
    lookup = await state.cache.lookup(request, "fake")
    entry_id = await state.cache.store(
        lookup=lookup,
        request=request,
        provider_name="fake",
        response_text="half an ans",
        prompt_tokens=5,
        completion_tokens=5,
        finish_reason="length",
    )
    assert entry_id is not None
    assert await state.vectors.count() == 1


async def test_empty_responses_are_not_stored(state) -> None:
    request = make_request("what is python")
    lookup = await state.cache.lookup(request, "fake")
    assert (
        await state.cache.store(
            lookup=lookup,
            request=request,
            provider_name="fake",
            response_text="   ",
            prompt_tokens=1,
            completion_tokens=0,
            finish_reason="stop",
        )
        is None
    )


async def test_ttl_is_applied_from_the_category(state) -> None:
    entry_id = await seed(state, "what is the weather today in jaipur")  # volatile
    ttl = await state.vectors.ttl(entry_id)
    assert 0 < ttl <= state.settings.ttl_volatile


async def test_factual_entries_live_longer_than_volatile_ones(state) -> None:
    entry_id = await seed(state, "what is the capital of france")
    assert await state.vectors.ttl(entry_id) > state.settings.ttl_volatile


async def test_ttl_reports_missing_entries_the_same_way_on_both_backends(state) -> None:
    assert await state.vectors.ttl("no-such-entry") == -2


async def test_hit_counter_increments(state) -> None:
    entry_id = await seed(state, "what is python")
    for _ in range(3):
        result = await state.cache.lookup(make_request("what is python"), "fake")
        await state.cache.register_hit(result, "fake/echo")
    entry = await state.vectors.get(entry_id)
    assert entry.hits == 3


async def test_hits_accumulate_modelled_savings(state) -> None:
    await seed(state, "what is python")
    result = await state.cache.lookup(make_request("what is python"), "fake")
    saved = await state.cache.register_hit(result, "fake-echo")
    assert saved > 0
    stats = await state.cache.stats()
    assert stats["usd_saved"] == pytest.approx(saved, rel=1e-6)
    assert stats["tokens_saved"] == 30


async def test_near_misses_are_recorded_for_tuning(state) -> None:
    state.settings.threshold_factual = 0.95
    state.settings.near_miss_margin = 0.10
    await seed(state, "what is the python programming language")
    result = await state.cache.lookup(make_request("what is python programming language"), "fake")
    assert result.status == "miss"
    rows = await state.analytics.near_misses()
    assert len(rows) == 1
    assert 0.85 < rows[0]["similarity"] < 0.95


async def test_shadow_mode_reports_but_never_serves(state) -> None:
    await seed(state, "what is python")
    state.settings.shadow_mode = True
    result = await state.cache.lookup(make_request("what is python"), "fake")
    assert result.status == "shadow_hit"
    assert result.served_from_cache is False
    assert result.entry is not None


async def test_invalidate_by_namespace(state) -> None:
    await seed(state, "what is python")
    await seed(state, "what is redis")
    result = await state.cache.lookup(make_request("what is python"), "fake")
    removed = await state.cache.invalidate(namespace=result.namespace)
    assert removed > 0
    assert (await state.cache.lookup(make_request("what is python"), "fake")).status == "miss"


async def test_invalidate_by_model(state) -> None:
    await seed(state, "what is python")
    assert await state.cache.invalidate(model="fake/echo") > 0
    assert await state.vectors.count() == 0


async def test_invalidate_everything(state) -> None:
    await seed(state, "what is python")
    await seed(state, "what is redis")
    assert await state.cache.invalidate(drop_all=True) > 0
    assert await state.vectors.count() == 0


async def test_a_stale_exact_pointer_falls_through_instead_of_crashing(state) -> None:
    entry_id = await seed(state, "what is python")
    await state.vectors.delete(entry_id)  # entry gone, pointer left behind
    result = await state.cache.lookup(make_request("what is python"), "fake")
    assert result.status == "miss"


async def test_concurrent_lookups_are_safe(state) -> None:
    await seed(state, "what is python")
    results = await asyncio.gather(
        *(state.cache.lookup(make_request("what is python"), "fake") for _ in range(20))
    )
    assert all(r.status == "hit" for r in results)
