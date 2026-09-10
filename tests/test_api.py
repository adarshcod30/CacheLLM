"""HTTP contract: the promise that switching base_url is the only change needed."""

from __future__ import annotations

import json

import pytest

from tests.conftest import requires_redis

pytestmark = [pytest.mark.integration, requires_redis]

ASK = {
    "model": "fake/echo",
    "temperature": 0,
    "messages": [{"role": "user", "content": "What is the Python programming language?"}],
}


def post(client, **overrides):
    body = {**ASK, **overrides}
    return client.post("/v1/chat/completions", json=body)


def test_first_call_misses_and_returns_openai_shape(client) -> None:
    response = post(client)
    assert response.status_code == 200
    assert response.headers["X-Cache"] == "MISS"
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert set(body["usage"]) >= {"prompt_tokens", "completion_tokens", "total_tokens"}


def test_second_identical_call_is_an_exact_hit(client) -> None:
    first = post(client).json()
    response = post(client)
    assert response.headers["X-Cache"] == "HIT"
    assert response.headers["X-Cache-Tier"] == "exact"
    assert response.headers["X-Cache-Similarity"] == "1.0000"
    assert float(response.headers["X-Cache-Saved-USD"]) > 0
    assert (
        response.json()["choices"][0]["message"]["content"]
        == first["choices"][0]["message"]["content"]
    )


def test_paraphrase_is_a_semantic_hit(client) -> None:
    client.app.state.cachellm.settings.threshold_factual = 0.90
    post(client)
    response = post(
        client, messages=[{"role": "user", "content": "what is python programming language"}]
    )
    assert response.headers["X-Cache"] == "HIT"
    assert response.headers["X-Cache-Tier"] == "semantic"
    assert 0.90 <= float(response.headers["X-Cache-Similarity"]) < 1.0


def test_headers_explain_every_decision(client) -> None:
    headers = post(client).headers
    for name in (
        "X-Cache",
        "X-Cache-Category",
        "X-Cache-Threshold",
        "X-Cache-Lookup-Ms",
        "X-Cache-Latency-Ms",
        "X-Cache-Namespace",
    ):
        assert name in headers


def test_debug_block_is_attached_to_the_body(client) -> None:
    body = post(client).json()
    assert body["cachellm"]["status"] == "miss"
    assert body["cachellm"]["category"] == "factual"


def test_uncacheable_request_is_reported_as_bypass(client) -> None:
    response = post(client, temperature=0.9)
    assert response.headers["X-Cache"] == "BYPASS"
    assert response.headers["X-Cache-Bypass-Reason"] == "temperature_too_high"


def test_client_can_force_a_bypass(client) -> None:
    post(client)
    response = client.post(
        "/v1/chat/completions", json=ASK, headers={"X-Cache-Control": "no-store"}
    )
    assert response.headers["X-Cache"] == "BYPASS"
    assert response.headers["X-Cache-Bypass-Reason"] == "client_requested_bypass"


def test_only_if_cached_fails_loudly_on_a_miss(client) -> None:
    response = client.post(
        "/v1/chat/completions", json=ASK, headers={"X-Cache-Control": "only-if-cached"}
    )
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "cache_miss"


def test_only_if_cached_succeeds_once_warm(client) -> None:
    post(client)
    response = client.post(
        "/v1/chat/completions", json=ASK, headers={"X-Cache-Control": "only-if-cached"}
    )
    assert response.status_code == 200
    assert response.headers["X-Cache"] == "HIT"


def test_streaming_miss_emits_valid_sse(client) -> None:
    with client.stream("POST", "/v1/chat/completions", json={**ASK, "stream": True}) as response:
        assert response.headers["X-Cache"] == "MISS"
        lines = [ln for ln in response.iter_lines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    payloads = [json.loads(ln[6:]) for ln in lines[:-1]]
    assert payloads[0]["choices"][0]["delta"]["role"] == "assistant"
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
    assert payloads[-1]["usage"]["total_tokens"] > 0


def test_a_streamed_miss_is_cached_and_replayed_as_a_stream(client) -> None:
    with client.stream("POST", "/v1/chat/completions", json={**ASK, "stream": True}) as first:
        streamed = "".join(
            json.loads(ln[6:])["choices"][0]["delta"].get("content", "")
            for ln in first.iter_lines()
            if ln.startswith("data: ") and ln != "data: [DONE]"
        )
    with client.stream("POST", "/v1/chat/completions", json={**ASK, "stream": True}) as second:
        assert second.headers["X-Cache"] == "HIT"
        replayed = "".join(
            json.loads(ln[6:])["choices"][0]["delta"].get("content", "")
            for ln in second.iter_lines()
            if ln.startswith("data: ") and ln != "data: [DONE]"
        )
    assert replayed == streamed


def test_models_endpoint_lists_routable_models(client) -> None:
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    assert any(m["id"].startswith("bedrock/") for m in body["data"])


def test_invalid_body_returns_an_openai_error_envelope(client) -> None:
    response = client.post("/v1/chat/completions", json={"model": "fake/echo"})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "messages"


def test_health_and_readiness(client) -> None:
    assert client.get("/healthz").json()["status"] == "ok"
    ready = client.get("/readyz").json()
    assert ready["status"] == "ready"
    assert ready["cache_available"] is True


def test_metrics_expose_cache_outcomes(client) -> None:
    post(client)
    post(client)
    text = client.get("/metrics").text
    assert 'cachellm_requests_total{category="factual"' in text
    assert 'cachellm_cost_usd_total{kind="saved"' in text
    assert "cachellm_request_duration_seconds" in text


def test_metrics_are_scrapable_by_prometheus(client) -> None:
    """The content type must match the body.

    `generate_latest` emits the Prometheus text format. Advertising the
    OpenMetrics type alongside it makes a real Prometheus reject the scrape
    with "data does not end with # EOF", and nothing short of a live scrape
    notices, so it is pinned here.
    """
    post(client)
    response = client.get("/metrics")
    assert response.status_code == 200
    content_type = response.headers["content-type"]
    body = response.text
    if "openmetrics" in content_type:
        assert body.rstrip().endswith("# EOF"), "OpenMetrics body must end with # EOF"
    else:
        assert content_type.startswith("text/plain")
        assert not body.rstrip().endswith("# EOF")
    # every metric line must be name{labels} value, which is what a scraper parses
    for line in body.splitlines():
        if line and not line.startswith("#"):
            assert len(line.rsplit(" ", 1)) == 2, f"unparseable metric line: {line!r}"


def test_admin_stats_reports_hit_rate(client) -> None:
    post(client)
    post(client)
    stats = client.get("/admin/stats").json()
    assert stats["requests"] == 2
    assert stats["hits"] == 1
    assert stats["hit_rate"] == 0.5
    assert stats["usd_saved"] > 0
    assert stats["latency_ms"]["hit"]["count"] == 1


def test_admin_config_shows_effective_policy(client) -> None:
    config = client.get("/admin/config").json()
    assert config["thresholds"]["creative"] > config["thresholds"]["classification"]
    assert config["rules"]["pii_guard"] is True


def test_admin_invalidate_clears_the_cache(client) -> None:
    post(client)
    assert client.post("/admin/invalidate", json={"all": True}).json()["removed_keys"] > 0
    assert post(client).headers["X-Cache"] == "MISS"


def test_admin_invalidate_requires_a_target(client) -> None:
    assert client.post("/admin/invalidate", json={}).status_code == 400


def test_admin_entries_lists_what_is_stored(client) -> None:
    post(client)
    body = client.get("/admin/entries").json()
    assert body["returned"] == 1
    assert body["entries"][0]["category"] == "factual"


def test_admin_near_misses_and_histogram(client) -> None:
    settings = client.app.state.cachellm.settings
    settings.threshold_factual = 0.95
    settings.near_miss_margin = 0.10
    post(client)
    post(client, messages=[{"role": "user", "content": "what is python programming language"}])
    rows = client.get("/admin/near-misses").json()
    assert rows["count"] == 1
    histogram = client.get("/admin/near-miss-histogram").json()["histogram"]
    assert any(row["would_hit_if_threshold_here"] > 0 for row in histogram)


def test_threshold_sweep_reports_the_precision_tradeoff(client) -> None:
    payload = {
        "pairs": [
            {
                "a": "what is the capital of france",
                "b": "what is the capital city of france",
                "duplicate": True,
            },
            {
                "a": "how do i reverse a list in python",
                "b": "how do i reverse a list in python",
                "duplicate": True,
            },
            {
                "a": "what is the capital of france",
                "b": "what is the capital of finland",
                "duplicate": False,
            },
            {
                "a": "how do i sort a list in python",
                "b": "how do i reverse a list in python",
                "duplicate": False,
            },
        ],
        "thresholds": [0.80, 0.90, 0.95, 0.99],
    }
    body = client.post("/admin/threshold-sweep", json=payload).json()
    assert body["pairs"] == 4
    assert body["duplicates"] == 2
    assert body["similarity"]["duplicates_mean"] > body["similarity"]["non_duplicates_mean"]
    hit_rates = [row["hit_rate"] for row in body["sweep"]]
    assert hit_rates == sorted(hit_rates, reverse=True)  # stricter threshold, fewer hits
    assert body["best_f1"]["f1"] > 0


def test_auth_rejects_missing_and_wrong_keys(client_factory) -> None:
    client = client_factory(api_keys="secret-one,secret-two")
    assert client.post("/v1/chat/completions", json=ASK).status_code == 401
    assert (
        client.post(
            "/v1/chat/completions", json=ASK, headers={"Authorization": "Bearer nope"}
        ).status_code
        == 401
    )
    ok = client.post(
        "/v1/chat/completions", json=ASK, headers={"Authorization": "Bearer secret-two"}
    )
    assert ok.status_code == 200


def test_shadow_mode_never_serves_from_cache(client_factory) -> None:
    client = client_factory(shadow_mode=True)
    client.post("/v1/chat/completions", json=ASK)
    response = client.post("/v1/chat/completions", json=ASK)
    assert response.headers["X-Cache"] == "SHADOW"
    assert response.headers["X-Cache-Tier"] == "exact"
    assert client.get("/admin/stats").json()["shadow_hits"] == 1


def test_disabled_cache_passes_everything_through(client_factory) -> None:
    client = client_factory(enabled=False)
    response = client.post("/v1/chat/completions", json=ASK)
    assert response.headers["X-Cache"] == "BYPASS"
    assert response.status_code == 200


async def test_the_official_openai_sdk_works_unchanged(asgi) -> None:
    """The drop-in claim, checked with the real client library."""
    from openai import AsyncOpenAI

    http, _app = await asgi()
    if True:
        sdk = AsyncOpenAI(api_key="unused", base_url="http://testserver/v1", http_client=http)

        first = await sdk.chat.completions.create(
            model="fake/echo",
            temperature=0,
            messages=[{"role": "user", "content": "What is Redis used for?"}],
        )
        assert first.choices[0].message.content
        assert first.object == "chat.completion"

        second = await sdk.chat.completions.with_raw_response.create(
            model="fake/echo",
            temperature=0,
            messages=[{"role": "user", "content": "What is Redis used for?"}],
        )
        assert second.headers["X-Cache"] == "HIT"
        assert second.parse().choices[0].message.content == first.choices[0].message.content

        chunks = []
        stream = await sdk.chat.completions.create(
            model="fake/echo",
            temperature=0,
            stream=True,
            messages=[{"role": "user", "content": "What is Redis used for?"}],
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        assert "".join(chunks) == first.choices[0].message.content
