"""The terminal report and provider auto-detection.

Both exist so someone can install the package and get somewhere useful without
reading configuration docs, so both are checked the way a user meets them.
"""

from __future__ import annotations

import time

import pytest

from cachellm import report
from cachellm.providers import detect
from cachellm.providers.catalog import HOSTS, HOSTS_BY_KEY
from cachellm.settings import Settings
from tests.conftest import make_settings


# ------------------------------------------------------------------- reporting
def test_bar_is_proportional_and_clamped() -> None:
    assert report.bar(0.0, 10).count("█") == 0
    assert report.bar(1.0, 10).count("█") == 10
    assert report.bar(0.5, 10).count("█") == 5
    assert report.bar(-3, 10).count("█") == 0
    assert report.bar(9.9, 10).count("█") == 10


@pytest.mark.parametrize(
    ("amount", "shown"),
    [(12.5, "$12.50"), (0.25, "$0.2500"), (0.000123, "$0.000123"), (0.0, "$0.000000")],
)
def test_small_sums_keep_enough_decimals_to_say_something(amount, shown) -> None:
    assert report.money(amount) == shown


def test_summary_reports_the_headline_numbers() -> None:
    text = report.summary(
        {
            "cache_available": True,
            "backend": "memory",
            "requests": 2000,
            "hits": 1540,
            "hits_exact": 1265,
            "hits_semantic": 275,
            "misses": 460,
            "bypassed": 0,
            "hit_rate": 0.77,
            "usd_saved": 0.0135,
            "usd_spent": 0.0038,
            "tokens_saved": 116029,
            "entries": 451,
            "latency_ms": {"hit": {"p50": 2.6, "p95": 5.7}, "miss": {"p50": 797.0, "p95": 1022.8}},
        },
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    assert "77.0%" in text
    assert "1,540 of 2,000" in text
    assert "116,029 tokens" in text
    assert "179x faster" in text  # 1022.8 / 5.7
    assert "memory backend" in text
    assert "all-MiniLM-L6-v2" in text


def test_summary_says_so_when_there_is_nothing_to_report() -> None:
    assert "no requests yet" in report.summary(
        {"cache_available": True, "backend": "memory", "requests": 0}
    )


def test_summary_explains_a_degraded_cache_rather_than_printing_zeros() -> None:
    text = report.summary({"cache_available": False, "degraded_reason": "redis is down"})
    assert "redis is down" in text
    assert "still serving" in text


def test_a_bypassed_request_shows_its_reason_and_its_prompt() -> None:
    line = report.request_line(
        {
            "at": time.time(),
            "status": "BYPASS",
            "tier": "",
            "similarity": 0.0,
            "category": "factual",
            "model": "m",
            "latency_ms": 690.0,
            "saved_usd": 0.0,
            "spent_usd": 0.0002,
            "prompt": "What is my order 123456789012 status?",
            "reason": "pii:long_digits",
        }
    )
    assert "BYPASS" in line
    assert "pii:long_digits" in line
    assert "order" in line, "a blank prompt makes the log useless for the rows that matter"


def test_long_prompts_are_truncated_not_wrapped() -> None:
    line = report.request_line(
        {
            "at": time.time(),
            "status": "HIT",
            "tier": "exact",
            "similarity": 1.0,
            "latency_ms": 2.0,
            "saved_usd": 0.0,
            "prompt": "x" * 400,
        },
        width=40,
    )
    assert "\n" not in line
    assert "…" in line


def test_request_table_handles_an_empty_log() -> None:
    assert "no requests logged yet" in report.request_table([])


# ------------------------------------------------------------------ detection
def test_every_keyed_host_is_found_when_its_variable_is_set(monkeypatch) -> None:
    for host in HOSTS:
        if not host.env_key:
            continue
        for name in [h.env_key for h in HOSTS if h.env_key]:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv(host.env_key, "test-key-value")
        found = {d.key for d in detect.available(include_fake=False)}
        assert host.key in found, f"{host.name} not detected from {host.env_key}"


def test_gemini_also_reads_the_google_variable(monkeypatch) -> None:
    for h in HOSTS:
        if h.env_key:
            monkeypatch.delenv(h.env_key, raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-value")
    assert "gemini" in {d.key for d in detect.available(include_fake=False)}


def test_detection_prefers_an_explicit_key_over_a_local_server(monkeypatch) -> None:
    """A key someone set is a stronger signal than a server that happens to run."""
    order = detect.PREFERENCE
    assert order.index("openai") < order.index("ollama") < order.index("fake")


def test_the_test_double_is_always_the_last_resort() -> None:
    assert detect.PREFERENCE[-1] == "fake"
    assert detect.available()[-1].key == "fake"


def test_detection_stands_aside_when_the_operator_has_configured_things(monkeypatch) -> None:
    monkeypatch.setenv("CACHELLM_OPENAI_BASE_URL", "https://my-own-endpoint/v1")
    cfg = make_settings()
    before = (cfg.default_provider, cfg.openai_base_url)
    assert detect.apply(cfg) is None
    assert (cfg.default_provider, cfg.openai_base_url) == before


def test_detection_respects_settings_written_in_a_dotenv_file(monkeypatch, tmp_path) -> None:
    """Only exported variables used to count, so a vendor key in the shell won.

    Found while preparing the live Gemini run: a base URL written in .env was
    replaced by Gemini's the moment GEMINI_API_KEY was exported.
    """
    for name in (
        "CACHELLM_DEFAULT_PROVIDER",
        "CACHELLM_OPENAI_BASE_URL",
        "CACHELLM_OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    (tmp_path / ".env").write_text("CACHELLM_OPENAI_BASE_URL=http://my-vllm:8000/v1\n")
    monkeypatch.chdir(tmp_path)

    cfg = Settings()
    assert detect.apply(cfg) is None
    assert cfg.openai_base_url == "http://my-vllm:8000/v1"


def _only_groq_available(monkeypatch) -> None:
    for name in (
        "CACHELLM_DEFAULT_PROVIDER",
        "CACHELLM_OPENAI_BASE_URL",
        "CACHELLM_OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    for h in HOSTS:
        if h.env_key:
            monkeypatch.delenv(h.env_key, raising=False)
    for extra in ("GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY", "OPENAI_KEY"):
        monkeypatch.delenv(extra, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-value")
    monkeypatch.setattr(detect, "port_open", lambda *a, **k: False)
    monkeypatch.setattr(detect, "aws_credentials_available", lambda: False)


def test_an_empty_key_line_in_dotenv_does_not_switch_detection_off(monkeypatch, tmp_path) -> None:
    _only_groq_available(monkeypatch)
    (tmp_path / ".env").write_text("CACHELLM_OPENAI_API_KEY=\n")
    monkeypatch.chdir(tmp_path)
    cfg = Settings()
    assert "Groq" in (detect.apply(cfg) or "")


def test_copying_the_shipped_env_example_leaves_detection_on(monkeypatch, tmp_path) -> None:
    """The README says to start from .env.example, so it must not decide for you."""
    import shutil
    from pathlib import Path

    _only_groq_available(monkeypatch)
    shutil.copy(Path(__file__).resolve().parents[1] / ".env.example", tmp_path / ".env")
    monkeypatch.chdir(tmp_path)
    cfg = Settings()
    assert "Groq" in (detect.apply(cfg) or "")
    assert cfg.openai_base_url == HOSTS_BY_KEY["groq"].base_url


def test_detection_configures_settings_when_nothing_was_set(monkeypatch) -> None:
    for name in (
        "CACHELLM_DEFAULT_PROVIDER",
        "CACHELLM_OPENAI_BASE_URL",
        "CACHELLM_OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    for h in HOSTS:
        if h.env_key:
            monkeypatch.delenv(h.env_key, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-value")
    monkeypatch.setattr(detect, "port_open", lambda *a, **k: False)
    monkeypatch.setattr(detect, "aws_credentials_available", lambda: False)

    cfg = Settings(_env_file=None)
    described = detect.apply(cfg)
    assert described is not None
    assert "Groq" in described
    assert cfg.default_provider == "openai"
    assert cfg.openai_base_url == HOSTS_BY_KEY["groq"].base_url
    assert cfg.openai_api_key == "gsk-test-value"


def test_port_open_says_no_for_a_dead_port() -> None:
    assert detect.port_open("http://127.0.0.1:9/v1") is False


def test_every_catalogued_host_has_what_a_person_needs_to_use_it() -> None:
    for host in HOSTS:
        assert host.key and host.name and host.examples
        if host.provider == "openai" and host.key != "fake":
            assert host.base_url, f"{host.name} has no base URL to configure"
