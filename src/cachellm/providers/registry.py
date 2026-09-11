"""Model name to host routing, across every host this proxy can reach.

With keys for Gemini and Groq and Ollama running, one proxy serves all three at
once: `gemini-2.5-flash` goes to Gemini, `openai/gpt-oss-20b` to Groq and
`llama3.2:1b` to Ollama, with nothing to configure. Six rules decide, in order:

1. `bedrock/` and `fake/` prefixes pick the adapters this proxy owns.
2. Bedrock's `vendor.model` naming, which no other host uses.
3. A host's own live model list, fetched at startup. This is what settles the
   hard cases: `openai/gpt-oss-20b` is a Groq model, not an OpenAI one, and
   `groq/compound` is sent to Groq whole because Groq lists it that way.
4. An explicit host prefix, LiteLLM style: `groq/openai/gpt-oss-20b` sends
   `openai/gpt-oss-20b` to Groq.
5. A naming convention for an enabled host: `claude-`, `gemini-`, `grok-`.
6. Otherwise the default host, which is what a single-host setup always used.

The host is part of the cache namespace, so an answer from one host is never
served for another, even when the model names look alike.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import structlog

from cachellm.providers import detect
from cachellm.providers.base import Provider
from cachellm.providers.bedrock import BedrockProvider
from cachellm.providers.catalog import (
    HOST_ONLY_PREFIXES,
    HOSTS_BY_KEY,
    convention_host,
    looks_like_bedrock,
    suggested_models,
)
from cachellm.providers.fake import FakeProvider
from cachellm.providers.openai_compat import OpenAICompatProvider
from cachellm.settings import Settings

log = structlog.get_logger(__name__)

#: How long a host's model list is trusted before an unrecognised name
#: refreshes it in the background, so models added since startup are found.
LIST_MAX_AGE_SECONDS = 600.0


@dataclass(frozen=True)
class Decision:
    """Where one request goes, what the host receives, and why."""

    key: str  # the route: also the cache namespace and the metrics label
    adapter: str  # "openai", "bedrock" or "fake"
    upstream_model: str
    rule: str
    via: str = "default"  # prefix, bedrock, listed, host_prefix, convention, default
    local: bool = False  # runs on this machine, so its tokens cost nothing
    error: str = ""  # set when the name asks for a host that is not enabled


class ProviderRegistry:
    def __init__(
        self,
        settings: Settings,
        providers: dict[str, Provider] | None = None,
        plan: detect.RoutePlan | None = None,
    ) -> None:
        self._settings = settings
        self._providers: dict[str, Provider] = dict(providers or {})
        self.plan = plan or detect.plan(settings)
        self._listed: dict[str, set[str]] = {}
        self._listed_raw: dict[str, list[str]] = {}
        self._listed_at = 0.0
        self._refresh: asyncio.Task[Any] | None = None

    # ------------------------------------------------------------- routing
    def route(self, model: str) -> Decision:
        m = model.strip()
        head, sep, rest = m.partition("/")
        prefix = head.lower() if sep and rest else ""

        if prefix in ("bedrock", "fake"):
            return self._decide(prefix, rest, f"the {prefix}/ prefix", "prefix")
        if m.startswith("fake-"):
            return self._decide("fake", m, "a fake- test model", "prefix")
        if looks_like_bedrock(m):
            return self._decide("bedrock", m, "Bedrock vendor naming convention", "bedrock")

        enabled = [r.key for r in self.plan.openai_routes()]
        for key in enabled:
            if m in self._listed.get(key, ()):
                return self._decide(key, m, f"listed by {self.name_of(key)}", "listed")

        if prefix in enabled:
            return self._decide(prefix, rest, f"the {prefix}/ host prefix", "host_prefix")
        if prefix in HOST_ONLY_PREFIXES:
            host = HOSTS_BY_KEY[prefix]
            fix = f"set {host.env_key}" if host.env_key else "start it"
            return Decision(
                prefix,
                "openai",
                rest,
                "a host prefix for a host that is not enabled",
                via="host_prefix",
                error=(
                    f"The {prefix}/ prefix sends requests to {host.name}, which is not "
                    f"enabled on this proxy. To use it, {fix}, and include it in "
                    "CACHELLM_HOSTS if you have set that."
                ),
            )

        by_name = convention_host(m)
        if by_name in enabled:
            return self._decide(
                by_name, m, f"{self.name_of(by_name)} naming convention", "convention"
            )

        default = self.plan.default
        return self._decide(
            default,
            m,
            f"no host claimed it, so the default ({self.name_of(default)}) applies",
            "default",
        )

    def _decide(self, key: str, upstream_model: str, rule: str, via: str) -> Decision:
        route = self.plan.routes.get(key)
        adapter = route.adapter if route else key
        return Decision(key, adapter, upstream_model, rule, via, local=bool(route and route.local))

    def name_of(self, key: str) -> str:
        route = self.plan.routes.get(key)
        return route.name if route else key

    def provider_name_for(self, model: str) -> str:
        return self.route(model).key

    def provider_for(self, decision: Decision) -> Provider:
        return self._get_or_create(decision.key)

    def resolve(self, model: str) -> tuple[Provider, str]:
        decision = self.route(model)
        return self._get_or_create(decision.key), decision.key

    def _get_or_create(self, key: str) -> Provider:
        if key not in self._providers:
            route = self.plan.routes.get(key)
            if route is None:
                raise ValueError(f"no route named {key!r}")
            if route.adapter == "bedrock":
                self._providers[key] = BedrockProvider(self._settings)
            elif route.adapter == "fake":
                self._providers[key] = FakeProvider(latency_ms=self._settings.fake_latency_ms)
            else:
                self._providers[key] = OpenAICompatProvider(
                    self._settings, base_url=route.base_url, api_key=route.api_key or ""
                )
        return self._providers[key]

    def register(self, name: str, provider: Provider) -> None:
        self._providers[name] = provider

    # ---------------------------------------------------------- model lists
    async def discover(self) -> dict[str, int]:
        """Ask every enabled host which models it serves, all at once.

        A host that cannot answer, or has no key, simply has no list, and the
        naming rules and the default decide for it instead.
        """
        targets = [r for r in self.plan.openai_routes() if not r.keyless]

        async def one(key: str) -> tuple[str, list[str]]:
            provider = self._get_or_create(key)
            if not isinstance(provider, OpenAICompatProvider):
                return key, []
            try:
                timeout = self._settings.discover_timeout
                return key, await asyncio.wait_for(provider.list_models(), timeout)
            except Exception as exc:
                log.info("model_list_unavailable", host=key, error=str(exc)[:160])
                return key, []

        for key, ids in await asyncio.gather(*(one(r.key) for r in targets)):
            if ids:  # a failed refresh keeps the list we already had
                self._listed_raw[key] = ids
                self._listed[key] = {v for i in ids for v in (i, i.removeprefix("models/"))}
        self._listed_at = time.monotonic()
        counts = self.listed_counts()
        if counts:
            log.info("model_lists_loaded", hosts=counts)
        return counts

    def refresh_soon(self) -> None:
        """Refresh stale model lists in the background, at most once at a time."""
        if not self._settings.discover_models or not self._listed_at:
            return
        if time.monotonic() - self._listed_at < LIST_MAX_AGE_SECONDS:
            return
        if self._refresh is not None and not self._refresh.done():
            return
        self._listed_at = time.monotonic()
        self._refresh = asyncio.get_running_loop().create_task(self.discover())

    def listed_counts(self) -> dict[str, int]:
        return {key: len(ids) for key, ids in self._listed_raw.items()}

    def model_cards(self) -> list[tuple[str, str]]:
        """Every model on offer as (id, host), for GET /v1/models."""
        cards = [
            (model_id, key)
            for key in self.plan.enabled
            for model_id in self._listed_raw.get(key, [])
        ]
        if not cards:
            return [(m, "cachellm") for m in suggested_models()]
        return cards + [(m, "bedrock") for m in suggested_models() if m.startswith("bedrock/")]

    def models(self) -> list[str]:
        return [model_id for model_id, _ in self.model_cards()]

    async def close(self) -> None:
        if self._refresh is not None and not self._refresh.done():
            self._refresh.cancel()
        for provider in self._providers.values():
            await provider.close()
        self._providers.clear()
