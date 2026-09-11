"""One proxy, several hosts at once.

With keys for Gemini and Groq and Ollama running, `gemini-2.5-flash` should
reach Gemini, `openai/gpt-oss-20b` should reach Groq, and `llama3.2:1b` should
reach Ollama, through one proxy, with nothing configured. These tests pin the
plan that detection builds, every routing rule, model discovery and its
failures, and the whole path through the app against simulated hosts.
"""

from __future__ import annotations

import json

import httpx
import pytest

from cachellm.api.app import create_app
from cachellm.api.deps import build_state, shutdown_state
from cachellm.observability.metrics import reset_metrics
from cachellm.providers import detect
from cachellm.providers.catalog import HOSTS
from cachellm.providers.detect import Route, RoutePlan
from cachellm.providers.openai_compat import OpenAICompatProvider
from cachellm.providers.registry import ProviderRegistry
from cachellm.settings import Settings
from tests.conftest import make_settings, purge

BUILT_IN = {
    "bedrock": Route("bedrock", "AWS Bedrock", "bedrock"),
    "fake": Route("fake", "Test double", "fake"),
}


# ------------------------------------------------------------------ helpers
def only_these_keys(monkeypatch, **env: str) -> None:
    """A machine with exactly these keys, no Ollama and no AWS."""
    for name in (
        "CACHELLM_DEFAULT_PROVIDER",
        "CACHELLM_OPENAI_BASE_URL",
        "CACHELLM_OPENAI_API_KEY",
        "CACHELLM_HOSTS",
        "GOOGLE_API_KEY",
        "GOOGLE_GENAI_API_KEY",
        "OPENAI_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    for host in HOSTS:
        if host.env_key:
            monkeypatch.delenv(host.env_key, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(detect, "port_open", lambda *a, **k: False)
    monkeypatch.setattr(detect, "aws_credentials_available", lambda: False)


def completion(text: str, model: str) -> dict:
    return {
        "id": "x",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21},
    }


def stream_body(text: str) -> bytes:
    chunks = [
        {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 9}},
    ]
    return ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()


class SimulatedHost:
    """A stand-in for one real host: lists its models and answers as itself."""

    def __init__(self, key: str, models: list[str], status: int = 200) -> None:
        self.key = key
        self.models = models
        self.status = status
        self.asked_for: list[str] = []
        self.listed = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            self.listed += 1
            if self.status != 200:
                return httpx.Response(self.status, json={"error": {"message": "down"}})
            return httpx.Response(200, json={"data": [{"id": m} for m in self.models]})
        body = json.loads(request.content)
        self.asked_for.append(body["model"])
        if body.get("stream"):
            return httpx.Response(
                200,
                content=stream_body(f"answer from {self.key}"),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=completion(f"answer from {self.key}", body["model"]))

    def provider(self, settings: Settings) -> OpenAICompatProvider:
        return OpenAICompatProvider(
            settings,
            base_url=f"https://{self.key}.test/v1",
            api_key="k",
            transport=httpx.MockTransport(self.handler),
        )


def plan_of(*routes: Route, default: str | None = None) -> RoutePlan:
    keys = tuple(r.key for r in routes)
    return RoutePlan(
        routes={**{r.key: r for r in routes}, **BUILT_IN},
        default=default or keys[0],
        enabled=keys,
        source="detected",
    )


def route(key: str, local: bool = False, api_key: str | None = "k") -> Route:
    name = {"gemini": "Google Gemini", "groq": "Groq", "ollama": "Ollama"}.get(key, key)
    return Route(key, name, "openai", f"https://{key}.test/v1", api_key, "test", local=local)


GEMINI_MODELS = ["models/gemini-2.5-flash", "models/gemini-3.5-flash"]
GROQ_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "groq/compound", "qwen/qwen3.6-27b"]
OLLAMA_MODELS = ["llama3.2:1b"]


async def registry_with_hosts(
    *hosts: SimulatedHost, local: tuple[str, ...] = ()
) -> ProviderRegistry:
    settings = make_settings()
    registry = ProviderRegistry(
        settings,
        providers={h.key: h.provider(settings) for h in hosts},
        plan=plan_of(*(route(h.key, local=h.key in local) for h in hosts)),
    )
    await registry.discover()
    return registry


# -------------------------------------------------------------- the plan
def test_every_host_with_a_key_gets_a_route(monkeypatch) -> None:
    only_these_keys(monkeypatch, GEMINI_API_KEY="g-key", GROQ_API_KEY="gsk-key")
    plan = detect.plan(Settings(_env_file=None))
    assert plan.source == "detected"
    assert plan.enabled == ("gemini", "groq")
    assert plan.default == "gemini"
    assert plan.routes["groq"].api_key == "gsk-key"


def test_cachellm_hosts_chooses_the_hosts_and_the_default(monkeypatch) -> None:
    only_these_keys(monkeypatch, GEMINI_API_KEY="g-key", GROQ_API_KEY="gsk-key")
    assert detect.plan(Settings(_env_file=None, hosts="groq, gemini")).enabled == ("groq", "gemini")
    assert detect.plan(Settings(_env_file=None, hosts="groq")).enabled == ("groq",)


def test_a_running_ollama_joins_as_a_free_local_route(monkeypatch) -> None:
    only_these_keys(monkeypatch, GROQ_API_KEY="gsk-key")
    monkeypatch.setattr(detect, "port_open", lambda url, *a, **k: "11434" in url)
    plan = detect.plan(Settings(_env_file=None))
    assert plan.enabled == ("groq", "ollama")
    assert plan.routes["ollama"].local


def test_a_configured_gateway_still_gets_everything(monkeypatch) -> None:
    """Someone pointing the proxy at their own gateway wants every request to
    go there, not to whichever vendor key happens to be exported."""
    only_these_keys(monkeypatch, GEMINI_API_KEY="g-key")
    settings = Settings(
        _env_file=None, default_provider="openai", openai_base_url="https://gateway.internal/v1"
    )
    registry = ProviderRegistry(settings)
    assert registry.plan.enabled == ("custom",)
    assert registry.route("gemini-2.5-flash").key == "custom"


def test_a_configured_host_can_be_joined_by_named_ones(monkeypatch) -> None:
    only_these_keys(monkeypatch, GEMINI_API_KEY="g-key", GROQ_API_KEY="gsk-key")
    settings = Settings(
        _env_file=None,
        openai_base_url="https://api.groq.com/openai/v1",
        hosts="gemini",
    )
    plan = detect.plan(settings)
    assert plan.enabled == ("groq", "gemini")
    assert plan.routes["groq"].api_key == "gsk-key", "a known host's own key is used"


def test_a_named_host_without_its_key_is_left_out(monkeypatch) -> None:
    only_these_keys(monkeypatch, GROQ_API_KEY="gsk-key")
    assert detect.plan(Settings(_env_file=None, hosts="groq,anthropic")).enabled == ("groq",)


# ------------------------------------------------------------ the rules
async def test_a_host_s_own_list_settles_vendor_style_names() -> None:
    registry = await registry_with_hosts(
        SimulatedHost("gemini", GEMINI_MODELS), SimulatedHost("groq", GROQ_MODELS)
    )
    d = registry.route("openai/gpt-oss-20b")
    assert (d.key, d.upstream_model, d.via) == ("groq", "openai/gpt-oss-20b", "listed")
    assert registry.route("groq/compound").upstream_model == "groq/compound"
    assert registry.route("gemini-2.5-flash").key == "gemini", "models/ prefix is optional"
    assert registry.route("models/gemini-3.5-flash").key == "gemini"


async def test_a_host_prefix_picks_the_host_and_is_removed() -> None:
    registry = await registry_with_hosts(
        SimulatedHost("gemini", GEMINI_MODELS), SimulatedHost("groq", GROQ_MODELS)
    )
    d = registry.route("groq/llama-3.1-8b-instant")
    assert (d.key, d.upstream_model, d.via) == ("groq", "llama-3.1-8b-instant", "host_prefix")


async def test_naming_conventions_only_reach_hosts_you_have() -> None:
    registry = await registry_with_hosts(
        SimulatedHost("gemini", []), SimulatedHost("groq", GROQ_MODELS)
    )
    assert registry.route("gemini-9-ultra").key == "gemini"
    d = registry.route("claude-haiku-4-5")
    assert (d.key, d.via) == ("gemini", "default"), "no Anthropic key, so the default answers"


async def test_an_ollama_tag_goes_to_ollama_and_costs_nothing() -> None:
    registry = await registry_with_hosts(
        SimulatedHost("groq", GROQ_MODELS), SimulatedHost("ollama", []), local=("ollama",)
    )
    d = registry.route("qwen2.5:0.5b")
    assert (d.key, d.via, d.local) == ("ollama", "convention", True)


async def test_bedrock_and_the_test_double_are_always_reachable_by_name() -> None:
    registry = await registry_with_hosts(SimulatedHost("groq", GROQ_MODELS))
    assert registry.route("amazon.nova-lite-v1:0").key == "bedrock"
    assert registry.route("bedrock/us.amazon.nova-micro-v1:0").upstream_model == (
        "us.amazon.nova-micro-v1:0"
    )
    assert registry.route("fake/echo").key == "fake"


async def test_a_prefix_for_a_host_you_do_not_have_is_a_clear_error() -> None:
    registry = await registry_with_hosts(SimulatedHost("groq", GROQ_MODELS))
    d = registry.route("xai/grok-4")
    assert "XAI_API_KEY" in d.error


async def test_an_openai_prefix_falls_through_to_whoever_serves_it() -> None:
    """`openai/` is also a vendor namespace, so it must not be an error."""
    registry = await registry_with_hosts(SimulatedHost("groq", GROQ_MODELS))
    d = registry.route("openai/gpt-4o-mini")
    assert not d.error
    assert (d.key, d.upstream_model) == ("groq", "openai/gpt-4o-mini")


# ------------------------------------------------------------ discovery
async def test_discovery_skips_hosts_without_a_key() -> None:
    settings = make_settings()
    host = SimulatedHost("openai", ["gpt-4o-mini"])
    keyless = Route("openai", "OpenAI", "openai", "https://api.openai.com/v1", "", "test")
    registry = ProviderRegistry(
        settings, providers={"openai": host.provider(settings)}, plan=plan_of(keyless)
    )
    assert await registry.discover() == {}
    assert host.listed == 0, "a keyless call would only earn a 401"


async def test_a_host_that_cannot_list_its_models_is_not_fatal() -> None:
    registry = await registry_with_hosts(
        SimulatedHost("gemini", [], status=500), SimulatedHost("groq", GROQ_MODELS)
    )
    assert registry.listed_counts() == {"groq": 4}
    assert registry.route("gemini-2.5-flash").key == "gemini", "the naming rule still works"


async def test_a_failed_refresh_keeps_the_list_it_had() -> None:
    groq = SimulatedHost("groq", GROQ_MODELS)
    registry = await registry_with_hosts(groq)
    groq.status = 503
    await registry.discover()
    assert registry.route("openai/gpt-oss-20b").via == "listed"


# ------------------------------------------------------ through the app
async def app_for(registry: ProviderRegistry):
    reset_metrics()
    cfg = make_settings(discover_models=True)
    state = await build_state(cfg, providers=registry)
    app = create_app(cfg, state=state)
    app.state.cachellm = state
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")
    return http, state, cfg


def ask(model: str, text: str = "What is the capital of Finland?", **extra) -> dict:
    return {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": text}],
        **extra,
    }


@pytest.fixture
async def two_hosts():
    gemini, groq = SimulatedHost("gemini", GEMINI_MODELS), SimulatedHost("groq", GROQ_MODELS)
    settings = make_settings()
    registry = ProviderRegistry(
        settings,
        providers={"gemini": gemini.provider(settings), "groq": groq.provider(settings)},
        plan=plan_of(route("gemini"), route("groq")),
    )
    http, state, cfg = await app_for(registry)
    try:
        yield http, gemini, groq
    finally:
        await http.aclose()
        await shutdown_state(state)
        purge(cfg)


async def test_each_model_reaches_its_own_host_through_one_proxy(two_hosts) -> None:
    http, gemini, groq = two_hosts
    first = await http.post("/v1/chat/completions", json=ask("gemini-2.5-flash"))
    second = await http.post("/v1/chat/completions", json=ask("openai/gpt-oss-20b"))
    assert first.headers["X-Cache-Upstream"] == "gemini"
    assert second.headers["X-Cache-Upstream"] == "groq"
    assert gemini.asked_for == ["gemini-2.5-flash"]
    assert groq.asked_for == ["openai/gpt-oss-20b"]
    assert "gemini" in first.json()["choices"][0]["message"]["content"]


async def test_an_answer_from_one_host_is_never_served_for_another(two_hosts) -> None:
    http, _gemini, groq = two_hosts
    await http.post("/v1/chat/completions", json=ask("gemini-2.5-flash"))
    other = await http.post("/v1/chat/completions", json=ask("openai/gpt-oss-20b"))
    again = await http.post("/v1/chat/completions", json=ask("openai/gpt-oss-20b"))
    assert other.headers["X-Cache"] == "MISS"
    assert again.headers["X-Cache"] == "HIT"
    assert again.headers["X-Cache-Upstream"] == "groq"
    assert len(groq.asked_for) == 1


async def test_a_host_prefix_and_the_plain_name_share_one_cache_entry(two_hosts) -> None:
    http, _, groq = two_hosts
    await http.post("/v1/chat/completions", json=ask("groq/openai/gpt-oss-20b"))
    plain = await http.post("/v1/chat/completions", json=ask("openai/gpt-oss-20b"))
    assert plain.headers["X-Cache"] == "HIT", "same host, same upstream model, same answer"
    assert groq.asked_for == ["openai/gpt-oss-20b"]


async def test_streams_route_the_same_way(two_hosts) -> None:
    http, _, groq = two_hosts
    async with http.stream(
        "POST", "/v1/chat/completions", json=ask("qwen/qwen3.6-27b", stream=True)
    ) as response:
        assert response.headers["X-Cache-Upstream"] == "groq"
        text = "".join([line async for line in response.aiter_lines()])
    assert "answer from groq" in text
    assert groq.asked_for == ["qwen/qwen3.6-27b"]


async def test_the_model_list_shows_every_host_s_models(two_hosts) -> None:
    http, _, _ = two_hosts
    cards = {m["id"]: m["owned_by"] for m in (await http.get("/v1/models")).json()["data"]}
    assert cards["openai/gpt-oss-20b"] == "groq"
    assert cards["models/gemini-2.5-flash"] == "gemini"


async def test_the_route_endpoint_explains_the_decision(two_hosts) -> None:
    http, _, _ = two_hosts
    d = (await http.get("/admin/route/openai/gpt-oss-20b")).json()
    assert (d["provider"], d["forwarded_as"]) == ("groq", "openai/gpt-oss-20b")
    assert "listed by Groq" in d["reason"]


async def test_a_missing_host_is_explained_not_forwarded(two_hosts) -> None:
    http, gemini, groq = two_hosts
    response = await http.post("/v1/chat/completions", json=ask("mistral/mistral-large-latest"))
    assert response.status_code == 400
    assert "MISTRAL_API_KEY" in response.json()["error"]["message"]
    assert gemini.asked_for == groq.asked_for == []


async def test_local_models_cost_nothing_even_when_they_save_time() -> None:
    ollama = SimulatedHost("ollama", OLLAMA_MODELS)
    settings = make_settings()
    registry = ProviderRegistry(
        settings,
        providers={"ollama": ollama.provider(settings)},
        plan=plan_of(route("ollama", local=True, api_key=None)),
    )
    http, state, cfg = await app_for(registry)
    try:
        await http.post("/v1/chat/completions", json=ask("llama3.2:1b"))
        hit = await http.post("/v1/chat/completions", json=ask("llama3.2:1b"))
        assert hit.headers["X-Cache"] == "HIT"
        log = (await http.get("/admin/requests")).json()["requests"]
        assert all(float(r.get("spent_usd") or 0) == 0 for r in log)
        assert all(float(r.get("saved_usd") or 0) == 0 for r in log)
    finally:
        await http.aclose()
        await shutdown_state(state)
        purge(cfg)


async def test_invalidating_by_a_prefixed_name_clears_the_stored_entry(two_hosts) -> None:
    http, _, groq = two_hosts
    await http.post("/v1/chat/completions", json=ask("groq/openai/gpt-oss-20b"))
    dropped = await http.post("/admin/invalidate", json={"model": "groq/openai/gpt-oss-20b"})
    assert dropped.json()["removed_keys"] >= 1
    again = await http.post("/v1/chat/completions", json=ask("openai/gpt-oss-20b"))
    assert again.headers["X-Cache"] == "MISS"
    assert len(groq.asked_for) == 2
