"""Upstream failures on a streaming request.

Found against the live Groq API. A stream asking for a model the host did not
have came back as HTTP 200 with an empty answer, because the proxy started
answering before the upstream did, and the caller had no way to tell what went
wrong. A failure before the first token must now carry the upstream's own
status, and a failure midway must end the stream with an error the SDKs raise.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from cachellm.errors import UpstreamError
from cachellm.models import ChatCompletionRequest
from cachellm.providers.base import Provider, ProviderResult, StreamEvent
from cachellm.providers.fake import FakeProvider

ASK = {
    "model": "fake-echo",
    "temperature": 0,
    "stream": True,
    "messages": [{"role": "user", "content": "What is the capital of Finland?"}],
}
MISSING = "Upstream returned 404: The model `nope` does not exist or you do not have access to it."


class FailingStream(Provider):
    """Sends `after` pieces of an answer, then fails the way a real host does."""

    name = "fake"

    def __init__(self, after: int) -> None:
        self.after = after

    async def complete(self, request: ChatCompletionRequest) -> ProviderResult:
        raise UpstreamError(MISSING, status_code=404)

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamEvent]:
        for i in range(self.after):
            yield StreamEvent(delta=f"part {i} ")
        raise UpstreamError(MISSING, status_code=404)


async def failing_app(asgi, after: int):
    http, app = await asgi()
    app.state.cachellm.providers.register("fake", FailingStream(after))
    return http, app


def data_lines(lines: list[str]) -> list[dict]:
    return [json.loads(ln[6:]) for ln in lines if ln.startswith("data: ") and ln != "data: [DONE]"]


async def test_failing_before_the_first_token_returns_the_upstream_status(asgi) -> None:
    http, _ = await failing_app(asgi, after=0)
    response = await http.post("/v1/chat/completions", json=ASK)
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert "does not exist" in response.json()["error"]["message"]


async def test_failing_midway_ends_with_an_error_event_not_a_fake_finish(asgi) -> None:
    http, _ = await failing_app(asgi, after=2)
    async with http.stream("POST", "/v1/chat/completions", json=ASK) as response:
        assert response.status_code == 200
        lines = [ln async for ln in response.aiter_lines()]
    payloads = data_lines(lines)
    errors = [p["error"] for p in payloads if "error" in p]
    assert len(errors) == 1
    assert "does not exist" in errors[0]["message"]
    finishes = [c["finish_reason"] for p in payloads for c in p.get("choices", [])]
    assert "error" not in finishes


async def test_the_official_sdk_raises_when_a_stream_fails_midway(asgi) -> None:
    from openai import APIError, AsyncOpenAI

    http, _ = await failing_app(asgi, after=2)
    sdk = AsyncOpenAI(api_key="unused", base_url="http://testserver/v1", http_client=http)
    stream = await sdk.chat.completions.create(
        model=ASK["model"], messages=ASK["messages"], temperature=0, stream=True
    )
    received: list[str] = []
    with pytest.raises(APIError, match="does not exist"):
        async for chunk in stream:
            received.extend(c.delta.content or "" for c in chunk.choices)
    assert "".join(received) == "part 0 part 1 "


async def test_a_failed_stream_leaves_nothing_in_the_cache(asgi) -> None:
    http, app = await failing_app(asgi, after=2)
    async with http.stream("POST", "/v1/chat/completions", json=ASK) as response:
        _ = [ln async for ln in response.aiter_lines()]

    app.state.cachellm.providers.register("fake", FakeProvider())
    again = await http.post("/v1/chat/completions", json={**ASK, "stream": False})
    assert again.status_code == 200
    assert again.headers["X-Cache"] == "MISS"
