"""Cacheability rules. Each test here maps to a way a naive cache goes wrong."""

from __future__ import annotations

import pytest

from cachellm.cache.policy import decide, detect_pii
from cachellm.models import ChatCompletionRequest
from tests.conftest import make_settings


def decision(text: str, **kwargs):
    settings = kwargs.pop("settings", None) or make_settings()
    cache_control = kwargs.pop("cache_control", "")
    kwargs.setdefault("temperature", 0.0)
    request = ChatCompletionRequest(
        model="fake/echo", messages=[{"role": "user", "content": text}], **kwargs
    )
    return decide(request, settings, cache_control=cache_control)


def test_plain_question_is_cacheable_as_factual() -> None:
    result = decision("What is the capital of France?")
    assert result.cacheable
    assert result.category == "factual"


def test_time_sensitive_prompts_get_a_short_ttl() -> None:
    volatile = decision("What is the weather today in Jaipur?")
    factual = decision("What is the capital of France?")
    assert volatile.category == "volatile"
    assert volatile.ttl < factual.ttl


def test_creative_prompts_demand_a_higher_threshold() -> None:
    creative = decision("Write a poem about the monsoon")
    factual = decision("What is the capital of France?")
    assert creative.category == "creative"
    assert creative.threshold > factual.threshold


def test_classification_prompts_allow_a_lower_threshold() -> None:
    result = decision("Classify the sentiment of this review: it was great")
    assert result.category == "classification"
    assert result.threshold < make_settings().threshold_for("factual")


def test_short_max_tokens_reads_as_classification() -> None:
    assert decision("Is this spam: buy now", max_tokens=8).category == "classification"


def test_high_temperature_is_never_cached() -> None:
    result = decision("What is the capital of France?", temperature=0.9)
    assert not result.cacheable
    assert result.reason == "temperature_too_high"


def test_multiple_completions_are_never_cached() -> None:
    assert decision("What is Python?", n=3).reason == "multiple_completions_requested"


def test_tool_calls_are_not_cached_by_default() -> None:
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    assert decision("What is Python?", tools=tools).reason == "tool_calls"


def test_json_mode_is_not_cached_by_default() -> None:
    result = decision("Return JSON", response_format={"type": "json_object"})
    assert result.reason == "json_mode"


def test_json_mode_can_be_enabled() -> None:
    settings = make_settings(cache_json_mode=True)
    result = decision("Return JSON", response_format={"type": "json_object"}, settings=settings)
    assert result.cacheable


def test_client_can_bypass_with_a_header() -> None:
    assert decision("What is Python?", cache_control="no-store").reason == "client_requested_bypass"


def test_overlong_prompts_are_skipped() -> None:
    assert decision("x" * 9000).reason == "prompt_too_long"


def test_empty_prompt_is_skipped() -> None:
    assert decision("").reason == "no_user_message"


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("email me at adarsh@example.com", "email"),
        ("my order 123456789012 status", "long_digits"),
        ("key sk-abcdefghijklmnop", "api_key"),
        ("what is my account balance", "possessive"),
    ],
)
def test_pii_is_detected(text: str, label: str) -> None:
    assert detect_pii(text) == label


def test_pii_prompts_are_never_stored() -> None:
    result = decision("What is my order 123456789012 status?")
    assert not result.cacheable
    assert result.reason.startswith("pii:")


def test_pii_guard_can_be_switched_off() -> None:
    settings = make_settings(pii_guard=False)
    assert decision("email me at a@b.com", settings=settings).cacheable


def test_multi_turn_is_skipped_unless_enabled() -> None:
    request = ChatCompletionRequest(
        model="fake/echo",
        temperature=0.0,
        messages=[
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "A language."},
            {"role": "user", "content": "And Java?"},
        ],
    )
    assert decide(request, make_settings()).reason == "multi_turn_conversation"
    assert decide(request, make_settings(cache_multi_turn=True)).cacheable


def test_disabled_cache_bypasses_everything() -> None:
    assert (
        decision("What is Python?", settings=make_settings(enabled=False)).reason
        == "cache_disabled"
    )


def test_thresholds_are_calibrated_per_embedding_model() -> None:
    """A threshold copied between models is not a threshold, it is a guess."""
    mini = make_settings(
        embedding_backend="fastembed", embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    bge = make_settings(embedding_backend="fastembed", embedding_model="BAAI/bge-small-en-v1.5")
    assert mini.is_calibrated and bge.is_calibrated
    assert mini.threshold_for("factual") < bge.threshold_for("factual")


def test_unknown_models_fall_back_and_admit_it() -> None:
    unknown = make_settings(embedding_backend="fastembed", embedding_model="nobody/knows")
    assert unknown.is_calibrated is False
    assert unknown.threshold_for("factual") == 0.92


def test_explicit_thresholds_beat_calibration() -> None:
    forced = make_settings(threshold_factual=0.75)
    assert forced.threshold_for("factual") == 0.75


def test_category_ordering_holds_for_every_calibrated_model() -> None:
    from cachellm.settings import CALIBRATED_THRESHOLDS

    for model in CALIBRATED_THRESHOLDS:
        cfg = make_settings(embedding_backend="fastembed", embedding_model=model)
        assert cfg.threshold_for("classification") <= cfg.threshold_for("factual")
        assert cfg.threshold_for("factual") < cfg.threshold_for("creative")
        assert cfg.threshold_for("creative") < 1.0
