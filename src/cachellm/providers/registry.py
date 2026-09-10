"""Model to provider routing.

Two ways to reach a provider:

* explicit prefix, ``bedrock/us.amazon.nova-lite-v1:0`` or ``openai/gpt-4o-mini``,
  which is unambiguous and is what the docs recommend;
* bare model id, which is what a drop-in client actually sends, routed by
  recognising vendor prefixes and otherwise falling back to the configured
  default provider.

The provider name is part of the cache namespace, so a Bedrock answer can never
be served to an OpenAI request even when the model strings look alike.
"""

from __future__ import annotations

import structlog

from cachellm.providers.base import Provider
from cachellm.providers.bedrock import BedrockProvider, looks_like_bedrock_model
from cachellm.providers.fake import FakeProvider
from cachellm.providers.openai_compat import OpenAICompatProvider
from cachellm.settings import Settings

log = structlog.get_logger(__name__)

# Model families we can attribute to a vendor with confidence. A bare
# "gpt-4o-mini" from a drop-in client means OpenAI, whatever the configured
# default provider happens to be.
OPENAI_MODEL_PREFIXES = (
    "gpt-",
    "o1",
    "o3",
    "o4",
    "chatgpt-",
    "text-embedding-",
    "davinci",
    "babbage",
)

# Shown by GET /v1/models so a client can discover what this proxy will route.
SUGGESTED_MODELS = [
    "bedrock/us.amazon.nova-micro-v1:0",
    "bedrock/us.amazon.nova-lite-v1:0",
    "bedrock/us.amazon.nova-pro-v1:0",
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    "openai/gpt-4o-mini",
    "fake/echo",
]


class ProviderRegistry:
    def __init__(self, settings: Settings, providers: dict[str, Provider] | None = None) -> None:
        self._settings = settings
        self._providers: dict[str, Provider] = providers or {}

    def _get_or_create(self, name: str) -> Provider:
        if name not in self._providers:
            if name == "bedrock":
                self._providers[name] = BedrockProvider(self._settings)
            elif name == "openai":
                self._providers[name] = OpenAICompatProvider(self._settings)
            elif name == "fake":
                self._providers[name] = FakeProvider(latency_ms=self._settings.fake_latency_ms)
            else:  # pragma: no cover - guarded by resolve()
                raise ValueError(f"unknown provider {name!r}")
        return self._providers[name]

    def provider_name_for(self, model: str) -> str:
        model = model.strip()
        if "/" in model:
            prefix = model.split("/", 1)[0].lower()
            if prefix in ("bedrock", "openai", "fake"):
                return prefix
        if model.startswith("fake-"):
            return "fake"
        if looks_like_bedrock_model(model):
            return "bedrock"
        if model.lower().startswith(OPENAI_MODEL_PREFIXES):
            return "openai"
        return self._settings.default_provider

    def resolve(self, model: str) -> tuple[Provider, str]:
        name = self.provider_name_for(model)
        return self._get_or_create(name), name

    def register(self, name: str, provider: Provider) -> None:
        self._providers[name] = provider

    def models(self) -> list[str]:
        return list(SUGGESTED_MODELS)

    async def close(self) -> None:
        for provider in self._providers.values():
            await provider.close()
        self._providers.clear()
