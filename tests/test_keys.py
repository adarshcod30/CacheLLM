"""Cache key derivation: the part that decides what may share an answer."""

from __future__ import annotations

import pytest

from cachellm.cache.keys import (
    embedding_text,
    exact_hash,
    namespace_for,
    normalise_text,
    strip_filler,
    temperature_bucket,
)
from cachellm.models import ChatCompletionRequest


def req(**kwargs) -> ChatCompletionRequest:
    kwargs.setdefault("model", "fake/echo")
    kwargs.setdefault("messages", [{"role": "user", "content": "What is Python?"}])
    return ChatCompletionRequest(**kwargs)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  What   is  Python?  ", "what is python"),
        ("WHAT IS PYTHON!!!", "what is python"),
        ("what is python", "what is python"),
        ("", ""),
    ],
)
def test_normalise_collapses_noise(raw: str, expected: str) -> None:
    assert normalise_text(raw) == expected


def test_exact_hash_ignores_case_and_trailing_punctuation() -> None:
    assert exact_hash("What is Python?") == exact_hash("what is python")
    assert exact_hash("What is Python?") != exact_hash("what is java")


def test_strip_filler_removes_politeness_with_punctuation() -> None:
    assert strip_filler("Hey, could you please explain what is Python?") == "what is python"


def test_strip_filler_never_empties_a_prompt() -> None:
    assert strip_filler("please thanks") != ""


def test_embedding_text_respects_the_flag() -> None:
    raw = "Please tell me what is Python"
    assert embedding_text(raw, strip_fillers=False) == "please tell me what is python"
    assert embedding_text(raw, strip_fillers=True) == "what is python"


@pytest.mark.parametrize(
    ("value", "bucket"), [(0.0, "0.0"), (0.04, "0.0"), (0.26, "0.3"), (1.0, "1.0")]
)
def test_temperature_bucketing(value: float, bucket: str) -> None:
    assert temperature_bucket(value) == bucket


def test_system_prompt_separates_namespaces() -> None:
    a = req(
        messages=[
            {"role": "system", "content": "You are a doctor."},
            {"role": "user", "content": "What is Python?"},
        ]
    )
    b = req(
        messages=[
            {"role": "system", "content": "You are a poet."},
            {"role": "user", "content": "What is Python?"},
        ]
    )
    assert namespace_for(a, "fake") != namespace_for(b, "fake")


def test_model_and_provider_separate_namespaces() -> None:
    a = req(model="fake/one")
    b = req(model="fake/two")
    assert namespace_for(a, "fake") != namespace_for(b, "fake")
    assert namespace_for(a, "fake") != namespace_for(a, "bedrock")


def test_close_temperatures_share_a_namespace() -> None:
    assert namespace_for(req(temperature=0.0), "fake") == namespace_for(
        req(temperature=0.04), "fake"
    )
    assert namespace_for(req(temperature=0.0), "fake") != namespace_for(
        req(temperature=0.9), "fake"
    )


def test_max_tokens_separates_namespaces() -> None:
    assert namespace_for(req(max_tokens=16), "fake") != namespace_for(req(max_tokens=512), "fake")
