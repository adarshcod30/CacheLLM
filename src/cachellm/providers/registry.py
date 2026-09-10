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
from cachellm.providers.bedrock import BedrockProvider
from cachellm.providers.catalog import (
    ROUTING_PREFIXES,
    looks_like_bedrock,
    looks_like_openai,
    suggested_models,
)
from cachellm.providers.fake import FakeProvider
from cachellm.providers.openai_compat import OpenAICompatProvider
from cachellm.settings import Settings

log = structlog.get_logger(__name__)


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
        """Three rules, in order.

        1. An explicit prefix this proxy owns wins outright.
        2. A recognisable vendor naming convention decides it: Bedrock ids are
           `vendor.model`, OpenAI ids belong to a small set of families.
        3. Otherwise the configured default, which is the OpenAI-compatible
           adapter. Names like `llama3.2` or `mixtral-8x7b-32768` are served by
           several hosts, so the endpoint the operator configured is the only
           sensible answer.
        """
        model = model.strip()
        head, sep, _ = model.partition("/")
        if sep and head.lower() in ROUTING_PREFIXES:
            return head.lower()
        if model.startswith("fake-"):
            return "fake"
        if looks_like_bedrock(model):
            return "bedrock"
        if looks_like_openai(model):
            return "openai"
        return self._settings.default_provider

    def resolve(self, model: str) -> tuple[Provider, str]:
        name = self.provider_name_for(model)
        return self._get_or_create(name), name

    def register(self, name: str, provider: Provider) -> None:
        self._providers[name] = provider

    def models(self) -> list[str]:
        return suggested_models()

    async def close(self) -> None:
        for provider in self._providers.values():
            await provider.close()
        self._providers.clear()
