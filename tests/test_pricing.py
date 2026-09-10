from __future__ import annotations

import json

from cachellm.pricing import estimate_cost, normalise_model_id, price_for


def test_region_prefixes_and_routing_prefixes_price_the_same() -> None:
    assert normalise_model_id("bedrock/us.amazon.nova-lite-v1:0") == "amazon.nova-lite-v1:0"
    assert normalise_model_id("eu.anthropic.claude-3-haiku-20240307-v1:0").startswith("anthropic.")
    assert price_for("bedrock/us.amazon.nova-micro-v1:0") == price_for("amazon.nova-micro-v1:0")


def test_cost_is_per_million_tokens() -> None:
    # nova-lite: $0.06 per 1M in, $0.24 per 1M out
    assert estimate_cost("amazon.nova-lite-v1:0", 1_000_000, 0) == 0.06
    assert estimate_cost("amazon.nova-lite-v1:0", 0, 1_000_000) == 0.24


def test_unknown_models_fall_back_rather_than_crash() -> None:
    assert estimate_cost("some-model-nobody-has-heard-of", 1000, 1000) > 0


def test_pricing_file_overrides_defaults(tmp_path, monkeypatch) -> None:
    override = tmp_path / "prices.json"
    override.write_text(
        json.dumps({"amazon.nova-lite-v1:0": {"input_per_m": 1.0, "output_per_m": 2.0}})
    )
    monkeypatch.setenv("CACHELLM_PRICING_FILE", str(override))
    import cachellm.pricing as pricing

    pricing._loaded_override = False
    assert pricing.estimate_cost("amazon.nova-lite-v1:0", 1_000_000, 0) == 1.0
    pricing._loaded_override = False
    pricing.PRICES["amazon.nova-lite-v1:0"] = pricing.ModelPrice(0.06, 0.24)
