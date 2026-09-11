"""The command line, which is how most people meet the product.

It sat at zero coverage while shipping a bug no test could see: `cachellm stats`
built its own application state, so against the default in-memory backend it
reported on an empty cache of its own. These tests drive the real commands.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator

import pytest
import uvicorn
from typer.testing import CliRunner

from cachellm import __version__
from cachellm.api.app import create_app
from cachellm.cli import app
from tests.conftest import make_settings

runner = CliRunner()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_proxy() -> Iterator[str]:
    """A real proxy on a real port, because `stats` must read it over HTTP."""
    port = free_port()
    cfg = make_settings(backend="memory", default_provider="fake")
    server = uvicorn.Server(
        uvicorn.Config(create_app(cfg), host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "proxy did not start"
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def send(url: str, content: str, temperature: float = 0.0) -> None:
    import httpx

    httpx.post(
        f"{url}/v1/chat/completions",
        timeout=30,
        json={
            "model": "fake/echo",
            "temperature": temperature,
            "messages": [{"role": "user", "content": content}],
        },
    ).raise_for_status()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("serve", "stats", "watch", "providers", "route", "invalidate", "config"):
        assert command in result.output


def test_providers_lists_every_catalogued_host() -> None:
    result = runner.invoke(app, ["providers"])
    assert result.exit_code == 0
    for name in ("OpenAI", "Groq", "Ollama", "AWS Bedrock", "OpenRouter", "xAI, Grok"):
        assert name in result.output


def test_route_explains_where_a_name_goes(live_proxy: str) -> None:
    result = runner.invoke(app, ["route", "fake/echo", "--url", live_proxy])
    assert result.exit_code == 0, result.output
    assert "Test double" in result.output
    assert "sent as   echo" in result.output


def test_route_explains_a_host_that_is_not_enabled(live_proxy: str) -> None:
    result = runner.invoke(app, ["route", "mistral/mistral-large-latest", "--url", live_proxy])
    assert result.exit_code == 1
    assert "MISTRAL_API_KEY" in result.output


def test_route_says_what_to_do_when_nothing_is_running() -> None:
    result = runner.invoke(
        app, ["route", "gpt-4o-mini", "--url", f"http://127.0.0.1:{free_port()}"]
    )
    assert result.exit_code == 1
    assert "No proxy answering" in result.output


def test_config_never_prints_secrets(monkeypatch) -> None:
    monkeypatch.setenv("CACHELLM_API_KEYS", "super-secret-client-key")
    monkeypatch.setenv("CACHELLM_OPENAI_API_KEY", "sk-super-secret-upstream")
    from cachellm.settings import reset_settings_cache

    reset_settings_cache()
    try:
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        assert "super-secret" not in result.output
        assert "1 key(s) configured" in result.output
    finally:
        reset_settings_cache()


def test_stats_says_what_to_do_when_nothing_is_running() -> None:
    result = runner.invoke(app, ["stats", "--url", f"http://127.0.0.1:{free_port()}"])
    assert result.exit_code == 1
    assert "No proxy answering" in result.output
    assert "cachellm serve" in result.output


def test_stats_reads_the_running_proxy_not_a_cache_of_its_own(live_proxy: str) -> None:
    """The regression that shipped in 0.2.0.

    With the in-memory backend the cache lives inside the serving process. A
    CLI that built its own state saw an empty cache and printed "no requests
    yet" no matter how much traffic had gone through.
    """
    send(live_proxy, "What is Redis used for?")
    send(live_proxy, "What is Redis used for?")
    send(live_proxy, "Write a poem about rain", temperature=0.9)

    result = runner.invoke(app, ["stats", "--url", live_proxy])
    assert result.exit_code == 0, result.output
    assert "no requests yet" not in result.output
    assert "1 of 3 requests" in result.output
    assert "HIT" in result.output
    assert "BYPASS" in result.output
    assert "temperature_too_high" in result.output
    assert "What is Redis used for?" in result.output


def test_stats_json_is_machine_readable(live_proxy: str) -> None:
    send(live_proxy, "What is Redis used for?")
    result = runner.invoke(app, ["stats", "--url", live_proxy, "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["stats"]["requests"] == 1
    assert data["recent"][0]["status"] == "MISS"


def test_watch_refuses_cleanly_when_nothing_is_running() -> None:
    result = runner.invoke(app, ["watch", "--url", f"http://127.0.0.1:{free_port()}"])
    assert result.exit_code == 1
    assert "No proxy answering" in result.output


def test_invalidate_requires_a_target() -> None:
    result = runner.invoke(app, ["invalidate"])
    assert result.exit_code == 2
    assert "--namespace" in result.output
