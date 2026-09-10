"""Server-sent event helpers for streaming chat completions."""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

DONE = "data: [DONE]\n\n"

# Non-space run plus whatever whitespace follows it, so no separator is lost.
_TOKEN_WITH_TRAILING_SPACE = re.compile(r"\S+\s*|\s+")


def new_stream_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def chunk(
    *,
    stream_id: str,
    model: str,
    delta: dict[str, Any] | None = None,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
    created: int | None = None,
) -> str:
    payload: dict[str, Any] = {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": created or int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta if delta is not None else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def role_chunk(stream_id: str, model: str, created: int) -> str:
    return chunk(
        stream_id=stream_id,
        model=model,
        delta={"role": "assistant", "content": ""},
        created=created,
    )


def text_chunk(stream_id: str, model: str, created: int, text: str) -> str:
    return chunk(stream_id=stream_id, model=model, delta={"content": text}, created=created)


def final_chunk(
    stream_id: str, model: str, created: int, finish_reason: str, usage: dict[str, int] | None
) -> str:
    return chunk(
        stream_id=stream_id,
        model=model,
        delta={},
        finish_reason=finish_reason,
        usage=usage,
        created=created,
    )


def error_event(message: str, err_type: str = "api_error", code: str | None = None) -> str:
    """A failure after the stream has started, in OpenAI's shape.

    The status line has already gone out as 200 by then, so this event is the
    only way left to say something broke. The official SDKs raise an error when
    they read it, instead of ending quietly with a half-written answer.
    """
    payload = {"error": {"message": message, "type": err_type, "param": None, "code": code}}
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def split_for_replay(text: str, max_chars: int = 24) -> list[str]:
    """Chop a cached answer into believable stream chunks.

    A cached hit has the whole answer already, but a client that asked for a
    stream still expects a stream. Replaying in small pieces keeps SDKs and UIs
    working unchanged; it arrives in a couple of milliseconds either way.

    The invariant that matters is ``"".join(pieces) == text``: an SSE client
    concatenates deltas directly, it does not re-insert separators. Splitting
    on spaces and dropping them silently eats a space at every chunk boundary,
    which is invisible in unit tests that join with a space and obvious the
    moment a real SDK reads the stream.
    """
    if not text:
        return []
    pieces: list[str] = []
    buffer = ""
    for token in _TOKEN_WITH_TRAILING_SPACE.findall(text):
        buffer += token
        if len(buffer) >= max_chars:
            pieces.append(buffer)
            buffer = ""
    if buffer:
        pieces.append(buffer)
    return pieces
