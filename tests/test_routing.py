"""Model-name routing: where a request goes, and what id the upstream receives.

Two things are checked here that used to be wrong. The default provider used to
be Bedrock, which a plain install cannot even reach now that AWS is an optional
extra. And any model id containing a slash had its first segment stripped,
which silently broke every OpenRouter and Together id.
"""

from __future__ import annotations

import pytest

from cachellm.providers.catalog import (
    HOSTS,
    looks_like_bedrock,
    looks_like_openai,
    strip_routing_prefix,
    suggested_models,
)
from cachellm.providers.registry import ProviderRegistry
from cachellm.settings import Settings
from tests.conftest import make_settings


def registry(default: str = "openai") -> ProviderRegistry:
    return ProviderRegistry(make_settings(default_provider=default))


# ------------------------------------------------------------------ the default
def test_the_default_provider_is_one_a_plain_install_can_reach() -> None:
    """AWS is an optional extra, so it cannot be the fallback.

    Read the declared field default rather than instantiating Settings: an
    instance still picks up CACHELLM_DEFAULT_PROVIDER from the environment, and
    CI sets that, so instantiating would test the runner's configuration rather
    than what the code ships.
    """
    assert Settings.model_fields["default_provider"].default == "openai"


# ------------------------------------------------- rule 1: explicit prefixes
@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("bedrock/us.amazon.nova-micro-v1:0", "bedrock"),
        ("openai/gpt-4o-mini", "openai"),
        ("fake/echo", "fake"),
        ("BEDROCK/amazon.nova-lite-v1:0", "bedrock"),
    ],
)
def test_an_explicit_prefix_wins(model: str, provider: str) -> None:
    assert registry().provider_name_for(model) == provider


# --------------------------------------- rule 2: recognisable vendor naming
@pytest.mark.parametrize(
    "model",
    [
        "amazon.nova-lite-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
        "meta.llama3-70b-instruct-v1:0",
        "mistral.mixtral-8x7b-instruct-v0:1",
        "cohere.embed-english-v3",
        "us.amazon.nova-pro-v1:0",
        "eu.anthropic.claude-3-5-sonnet-20240620-v1:0",
        "apac.amazon.nova-micro-v1:0",
    ],
)
def test_bedrock_ids_are_recognised_without_a_prefix(model: str) -> None:
    assert looks_like_bedrock(model)
    assert registry().provider_name_for(model) == "bedrock"


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4o-mini",
        "gpt-4.1",
        "o1-preview",
        "o3-mini",
        "o4-mini",
        "chatgpt-4o-latest",
        "text-embedding-3-small",
    ],
)
def test_openai_families_are_recognised(model: str) -> None:
    assert looks_like_openai(model)
    assert registry(default="bedrock").provider_name_for(model) == "openai"


def test_bedrock_wins_over_the_default_even_when_the_default_is_openai() -> None:
    assert registry(default="openai").provider_name_for("amazon.nova-lite-v1:0") == "bedrock"


# ------------------------------------------------------ rule 3: the fallback
@pytest.mark.parametrize(
    "model",
    [
        "llama-3.1-8b-instant",  # Groq
        "mixtral-8x7b-32768",  # Groq
        "gemma2-9b-it",  # Groq
        "llama3.2",  # Ollama
        "qwen2.5-coder:7b",  # Ollama
        "gemini-2.5-flash",  # Gemini
        "my-finetuned-model",  # self-hosted
    ],
)
def test_unattributable_names_go_to_the_openai_compatible_adapter(model: str) -> None:
    """These are served by several hosts, so only the configured endpoint knows."""
    assert not looks_like_bedrock(model)
    assert not looks_like_openai(model)
    assert registry().provider_name_for(model) == "openai"


# ------------------------------------------- the id forwarded to the upstream
@pytest.mark.parametrize(
    ("sent", "forwarded"),
    [
        ("openai/gpt-4o-mini", "gpt-4o-mini"),
        ("bedrock/us.amazon.nova-micro-v1:0", "us.amazon.nova-micro-v1:0"),
        ("fake/echo", "echo"),
        ("gpt-4o-mini", "gpt-4o-mini"),
    ],
)
def test_our_own_prefixes_are_stripped(sent: str, forwarded: str) -> None:
    assert strip_routing_prefix(sent) == forwarded


def test_groq_names_for_openai_models_reach_groq_whole() -> None:
    settings = make_settings(
        default_provider="openai", openai_base_url="https://api.groq.com/openai/v1"
    )
    registry = ProviderRegistry(settings)
    provider, name = registry.resolve("openai/gpt-oss-120b")
    assert name == "groq", "a configured Groq URL is known by its host name"
    assert registry.route("openai/gpt-oss-120b").upstream_model == "openai/gpt-oss-120b"
    assert provider.resolve_model("openai/gpt-oss-120b") == "openai/gpt-oss-120b"


@pytest.mark.parametrize(
    "model",
    [
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",  # Together
        "mistralai/Mixtral-8x7B-Instruct-v0.1",  # Together
        "anthropic/claude-3.5-sonnet",  # OpenRouter
        "google/gemini-2.0-flash-exp",  # OpenRouter
        "deepseek/deepseek-r1:free",  # OpenRouter
    ],
)
def test_vendor_ids_with_a_slash_are_forwarded_whole(model: str) -> None:
    """The slash is part of the real name on OpenRouter and Together.

    Stripping it makes the upstream reject the request as an unknown model,
    which is exactly what used to happen.
    """
    assert strip_routing_prefix(model) == model
    provider, name = registry().resolve(model)
    assert name == "openai"
    assert provider.resolve_model(model) == model


def test_a_dot_means_bedrock_and_a_slash_means_the_vendors_own_id() -> None:
    """`anthropic.claude-...` is Bedrock; `anthropic/claude-...` is OpenRouter."""
    r = registry()
    assert r.provider_name_for("anthropic.claude-3-haiku-20240307-v1:0") == "bedrock"
    assert r.provider_name_for("anthropic/claude-3.5-sonnet") == "openai"


# --------------------------------------------------------------- the catalogue
def test_every_catalogued_example_routes_to_the_host_that_serves_it() -> None:
    r = registry()
    for host in HOSTS:
        for example in host.examples:
            if example.startswith("whatever") or example == "my-finetuned-model":
                continue  # placeholders for a self-hosted endpoint
            assert r.provider_name_for(example) == host.provider, (
                f"{example!r} from {host.name} routed to "
                f"{r.provider_name_for(example)}, expected {host.provider}"
            )


def test_catalogue_entries_are_usable() -> None:
    for host in HOSTS:
        assert host.examples, f"{host.name} has no example model"
        assert host.provider in ("bedrock", "openai", "fake")
        if host.provider == "openai" and host.base_url:
            assert host.base_url.startswith(("http://", "https://"))


def test_models_endpoint_lists_the_catalogue() -> None:
    models = suggested_models()
    assert len(models) == len(set(models)), "duplicate model ids in the listing"
    assert any(m.startswith("bedrock/") for m in models)
    assert "gpt-5.6-luna" in models
