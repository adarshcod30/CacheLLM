from __future__ import annotations

import json

import pytest

from cachellm.api import sse


def test_replay_splits_into_several_chunks() -> None:
    text = "Python is a high level programming language used widely for data work."
    pieces = sse.split_for_replay(text)
    assert len(pieces) > 1
    assert "".join(pieces) == text


def test_replay_of_empty_text_is_empty() -> None:
    assert sse.split_for_replay("") == []


@pytest.mark.parametrize(
    "text",
    [
        "a b c d e f g h i j k l m n o p q r s t u v w x y z",
        "line one\nline two\n\nline four",
        "  leading and trailing spaces  ",
        "single",
        "tabs\tand\tmore\ttabs in a longer sentence than the chunk size",
        "unicode: \u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e and emoji \U0001f680 kept intact",
    ],
)
def test_replay_concatenates_back_to_the_original(text: str) -> None:
    """SSE clients concatenate deltas, so joining must be lossless."""
    assert "".join(sse.split_for_replay(text)) == text


def test_chunk_is_valid_openai_shaped_sse() -> None:
    raw = sse.text_chunk("chatcmpl-1", "fake/echo", 123, "hello")
    assert raw.startswith("data: ") and raw.endswith("\n\n")
    payload = json.loads(raw[6:])
    assert payload["object"] == "chat.completion.chunk"
    assert payload["choices"][0]["delta"]["content"] == "hello"
    assert payload["choices"][0]["finish_reason"] is None


def test_final_chunk_carries_usage_and_finish_reason() -> None:
    raw = sse.final_chunk(
        "id", "m", 1, "stop", {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}
    )
    payload = json.loads(raw[6:])
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["usage"]["total_tokens"] == 8
