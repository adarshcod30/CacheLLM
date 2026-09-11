from __future__ import annotations

import pytest

from cachellm.models import ChatCompletionRequest
from cachellm.providers.base import normalise_finish_reason
from cachellm.providers.bedrock import (
    build_converse_kwargs,
    looks_like_bedrock_model,
    to_converse_messages,
)
from cachellm.providers.fake import FakeProvider
from cachellm.providers.registry import ProviderRegistry
from tests.conftest import make_settings


def request_with(messages, **kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="bedrock/amazon.nova-micro-v1:0", messages=messages, **kwargs
    )


def test_system_prompt_is_hoisted_out_of_messages() -> None:
    body = build_converse_kwargs(
        request_with(
            [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Hi"}]
        ),
        "amazon.nova-micro-v1:0",
    )
    assert body["system"] == [{"text": "Be terse."}]
    assert [m["role"] for m in body["messages"]] == ["user"]


def test_consecutive_same_role_messages_are_merged() -> None:
    messages = to_converse_messages(
        request_with(
            [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}]
        ).messages
    )
    assert len(messages) == 1
    assert messages[0]["content"] == [{"text": "one"}, {"text": "two"}]


def test_conversation_must_open_with_a_user_turn() -> None:
    messages = to_converse_messages(
        request_with(
            [
                {"role": "assistant", "content": "leading"},
                {"role": "user", "content": "real question"},
            ]
        ).messages
    )
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == [{"text": "real question"}]


def test_inference_config_is_only_sent_when_asked_for() -> None:
    bare = build_converse_kwargs(request_with([{"role": "user", "content": "Hi"}]), "m")
    assert "inferenceConfig" not in bare
    tuned = build_converse_kwargs(
        request_with(
            [{"role": "user", "content": "Hi"}], temperature=0.2, max_tokens=64, stop=["END"]
        ),
        "m",
    )
    assert tuned["inferenceConfig"] == {
        "maxTokens": 64,
        "temperature": 0.2,
        "stopSequences": ["END"],
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("end_turn", "stop"),
        ("max_tokens", "length"),
        ("tool_use", "tool_calls"),
        ("content_filtered", "content_filter"),
        (None, "stop"),
    ],
)
def test_finish_reasons_are_normalised_to_openai_vocabulary(raw, expected) -> None:
    assert normalise_finish_reason(raw) == expected


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("us.amazon.nova-micro-v1:0", "bedrock"),
        ("anthropic.claude-3-haiku-20240307-v1:0", "bedrock"),
        ("bedrock/whatever", "bedrock"),
        ("openai/gpt-4o-mini", "openai"),
        ("gpt-4o-mini", "openai"),
        ("o3-mini", "openai"),
        ("fake/echo", "fake"),
    ],
)
def test_model_routing(model: str, provider: str) -> None:
    registry = ProviderRegistry(make_settings(default_provider="bedrock"))
    assert registry.provider_name_for(model) == provider


def test_unknown_models_fall_back_to_the_default_provider() -> None:
    registry = ProviderRegistry(make_settings(default_provider="fake"))
    assert registry.provider_name_for("some-local-model") == "fake"
    assert looks_like_bedrock_model("some-local-model") is False


async def test_fake_provider_is_deterministic() -> None:
    provider = FakeProvider()
    request = ChatCompletionRequest(
        model="fake/echo", messages=[{"role": "user", "content": "What is Python?"}]
    )
    first = await provider.complete(request)
    second = await provider.complete(request)
    assert first.text == second.text
    assert first.finish_reason == "stop"
    assert provider.calls == 2


async def test_fake_provider_streams_the_same_text_it_completes() -> None:
    provider = FakeProvider()
    request = ChatCompletionRequest(
        model="fake/echo", messages=[{"role": "user", "content": "Explain Redis"}]
    )
    whole = (await provider.complete(request)).text
    streamed = "".join([event.delta async for event in provider.stream(request)])
    assert streamed == whole


def _bedrock_region(monkeypatch, **env: str) -> str:
    pytest.importorskip("boto3")
    from cachellm.providers.bedrock import BedrockProvider

    for name in ("AWS_REGION", "AWS_DEFAULT_REGION", "AWS_PROFILE", "CACHELLM_AWS_REGION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", "/nonexistent/aws-config")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    settings = make_settings(aws_region=env.get("CACHELLM_AWS_REGION", ""))
    return BedrockProvider(settings)._get_client().meta.region_name


def test_bedrock_follows_the_standard_aws_region_variable(monkeypatch) -> None:
    """The region used to be passed explicitly, so AWS_REGION was ignored and
    everyone was sent to us-east-1."""
    assert _bedrock_region(monkeypatch, AWS_REGION="eu-west-1") == "eu-west-1"


def test_an_explicit_cachellm_region_still_wins(monkeypatch) -> None:
    assert (
        _bedrock_region(monkeypatch, AWS_REGION="eu-west-1", CACHELLM_AWS_REGION="ap-south-1")
        == "ap-south-1"
    )


def test_bedrock_falls_back_to_us_east_1_when_nothing_says(monkeypatch) -> None:
    assert _bedrock_region(monkeypatch) == "us-east-1"


def test_bedrock_also_reads_the_default_region_variable(monkeypatch) -> None:
    assert _bedrock_region(monkeypatch, AWS_DEFAULT_REGION="eu-central-1") == "eu-central-1"
