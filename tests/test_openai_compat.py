"""The OpenAI-compatible adapter.

This one class serves fourteen of the seventeen catalogued hosts: OpenAI,
Anthropic, Gemini, xAI, Groq, DeepSeek, Mistral, OpenRouter, Together,
Fireworks, Cerebras, Perplexity, Moonshot and Ollama. A bug here breaks all of
them at once, so it is exercised against a fake upstream that speaks the real
wire format, including server-sent events.
"""

from __future__ import annotations

import json

import httpx
import pytest

from cachellm.errors import UpstreamError
from cachellm.models import ChatCompletionRequest
from cachellm.providers.openai_compat import OpenAICompatProvider
from tests.conftest import make_settings


def req(model: str = "llama3.2", **kw) -> ChatCompletionRequest:
    kw.setdefault("messages", [{"role": "user", "content": "What is Redis used for?"}])
    return ChatCompletionRequest(model=model, temperature=0, **kw)


def provider_with(handler, api_key: str = "sk-test") -> tuple[OpenAICompatProvider, list]:
    """An adapter whose HTTP client talks to `handler` instead of the network."""
    seen: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    # The transport goes in through the constructor, so the adapter builds its
    # client the real way, auth header included, and only the network is fake.
    p = OpenAICompatProvider(
        make_settings(),
        base_url="https://upstream.test/v1",
        api_key=api_key,
        transport=httpx.MockTransport(recording),
    )
    return p, seen


def completion(
    text: str = "Redis is an in-memory data store.",
    finish: str = "stop",
    prompt_tokens: int = 11,
    completion_tokens: int = 8,
) -> dict:
    return {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "model": "llama3.2",
        "choices": [
            {"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": text}}
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def sse(*events: dict, done: bool = True) -> bytes:
    lines = [f"data: {json.dumps(e)}\n\n" for e in events]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def chunk(content: str | None = None, finish: str | None = None) -> dict:
    delta = {"content": content} if content is not None else {}
    return {
        "id": "c",
        "object": "chat.completion.chunk",
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


# -------------------------------------------------------------- complete()
async def test_a_normal_completion_is_parsed() -> None:
    p, _ = provider_with(lambda r: httpx.Response(200, json=completion()))
    result = await p.complete(req())
    assert result.text == "Redis is an in-memory data store."
    assert (result.prompt_tokens, result.completion_tokens) == (11, 8)
    assert result.finish_reason == "stop"


async def test_the_api_key_is_sent_as_a_bearer_token() -> None:
    p, seen = provider_with(lambda r: httpx.Response(200, json=completion()), api_key="gsk-secret")
    await p.complete(req())
    assert seen[0].headers["authorization"] == "Bearer gsk-secret"


async def test_no_authorization_header_when_no_key_is_configured() -> None:
    """Ollama and local servers need no key; sending an empty bearer confuses some."""
    p, seen = provider_with(lambda r: httpx.Response(200, json=completion()), api_key="")
    await p.complete(req())
    assert "authorization" not in seen[0].headers


async def test_the_request_hits_chat_completions_with_the_real_body() -> None:
    p, seen = provider_with(lambda r: httpx.Response(200, json=completion()))
    await p.complete(req(max_tokens=64))
    assert seen[0].url.path == "/v1/chat/completions"
    body = json.loads(seen[0].content)
    assert body["model"] == "llama3.2"
    assert body["stream"] is False
    assert body["max_tokens"] == 64
    assert body["messages"][0]["content"] == "What is Redis used for?"


@pytest.mark.parametrize(
    "model",
    [
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "anthropic/claude-3.5-sonnet",
        "accounts/fireworks/models/llama-v3p3-70b-instruct",
    ],
)
async def test_vendor_ids_containing_slashes_reach_the_upstream_whole(model: str) -> None:
    p, seen = provider_with(lambda r: httpx.Response(200, json=completion()))
    await p.complete(req(model=model))
    assert json.loads(seen[0].content)["model"] == model


async def test_our_own_routing_prefix_is_removed_before_sending() -> None:
    p, seen = provider_with(lambda r: httpx.Response(200, json=completion()))
    await p.complete(req(model="openai/gpt-4o-mini"))
    assert json.loads(seen[0].content)["model"] == "gpt-4o-mini"


@pytest.mark.parametrize(
    ("finish", "expected"),
    [("stop", "stop"), ("length", "length"), ("tool_calls", "tool_calls"), (None, "stop")],
)
async def test_finish_reasons_are_normalised(finish, expected) -> None:
    p, _ = provider_with(lambda r: httpx.Response(200, json=completion(finish=finish)))
    assert (await p.complete(req())).finish_reason == expected


async def test_a_missing_usage_block_does_not_crash() -> None:
    body = completion()
    del body["usage"]
    p, _ = provider_with(lambda r: httpx.Response(200, json=body))
    result = await p.complete(req())
    assert result.text
    assert result.prompt_tokens == 0


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500, 503])
async def test_upstream_errors_become_upstream_errors_with_their_status(status) -> None:
    p, _ = provider_with(lambda r: httpx.Response(status, json={"error": {"message": "no"}}))
    with pytest.raises(UpstreamError) as caught:
        await p.complete(req())
    assert caught.value.status_code == status
    assert str(status) in caught.value.message


async def test_a_network_failure_becomes_an_upstream_error() -> None:
    def boom(request):
        raise httpx.ConnectError("connection refused")

    p, _ = provider_with(boom)
    with pytest.raises(UpstreamError) as caught:
        await p.complete(req())
    assert "failed" in caught.value.message.lower()


# ---------------------------------------------------------------- stream()
async def test_a_stream_yields_every_delta_then_the_finish_and_usage() -> None:
    body = sse(
        chunk("Redis "),
        chunk("is "),
        chunk("fast."),
        chunk(finish="stop"),
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "choices": [],
            "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
        },
    )
    p, _ = provider_with(
        lambda r: httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})
    )
    events = [e async for e in p.stream(req())]

    assert "".join(e.delta for e in events) == "Redis is fast."
    assert any(e.finish_reason == "stop" for e in events)
    usage = [e for e in events if e.prompt_tokens or e.completion_tokens]
    assert usage and (usage[-1].prompt_tokens, usage[-1].completion_tokens) == (9, 3)


async def test_streaming_asks_the_upstream_for_usage() -> None:
    """Without include_usage many hosts send no token counts on a stream."""
    p, seen = provider_with(
        lambda r: httpx.Response(
            200,
            content=sse(chunk("hi"), chunk(finish="stop")),
            headers={"content-type": "text/event-stream"},
        )
    )
    _ = [e async for e in p.stream(req())]
    body = json.loads(seen[0].content)
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


async def test_keepalives_comments_and_garbage_lines_are_ignored() -> None:
    raw = b": keep-alive\n\nevent: ping\n\ndata: not json at all\n\n" + sse(
        chunk("ok"), chunk(finish="stop")
    )
    p, _ = provider_with(
        lambda r: httpx.Response(200, content=raw, headers={"content-type": "text/event-stream"})
    )
    events = [e async for e in p.stream(req())]
    assert "".join(e.delta for e in events) == "ok"


async def test_a_stream_that_never_sends_done_still_finishes() -> None:
    p, _ = provider_with(
        lambda r: httpx.Response(
            200,
            content=sse(chunk("partial"), chunk(finish="stop"), done=False),
            headers={"content-type": "text/event-stream"},
        )
    )
    events = [e async for e in p.stream(req())]
    assert "".join(e.delta for e in events) == "partial"


async def test_a_stream_error_status_raises_with_the_body() -> None:
    p, _ = provider_with(lambda r: httpx.Response(401, content=b'{"error":{"message":"bad key"}}'))
    with pytest.raises(UpstreamError) as caught:
        _ = [e async for e in p.stream(req())]
    assert caught.value.status_code == 401
    assert "bad key" in caught.value.message


async def test_close_releases_the_client() -> None:
    p, _ = provider_with(lambda r: httpx.Response(200, json=completion()))
    await p.complete(req())
    await p.close()
    assert p._client is None
