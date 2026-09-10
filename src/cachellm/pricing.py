"""Token prices used to compute "money saved" for every cache hit.

Prices are USD per one million tokens and are *defaults*: providers change them,
so treat this table as a starting point and override it with
``CACHELLM_PRICING_FILE`` pointing at a JSON file of the same shape.

The saving reported by CacheLLM is a modelled number, not a bill. It is
"what these tokens would have cost at list price if we had called the provider",
which is exactly the number a team wants when deciding whether to deploy this.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

__all__ = ["PRICES", "ModelPrice", "estimate_cost", "price_for"]


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1M tokens."""

    input_per_m: float
    output_per_m: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_m + output_tokens * self.output_per_m) / 1_000_000


# Defaults verified against public list prices; override per deployment.
PRICES: dict[str, ModelPrice] = {
    # --- AWS Bedrock: Amazon Nova ---
    "amazon.nova-micro-v1:0": ModelPrice(0.035, 0.14),
    "amazon.nova-lite-v1:0": ModelPrice(0.06, 0.24),
    "amazon.nova-pro-v1:0": ModelPrice(0.80, 3.20),
    # --- AWS Bedrock: Anthropic ---
    "anthropic.claude-3-haiku-20240307-v1:0": ModelPrice(0.25, 1.25),
    "anthropic.claude-3-5-haiku-20241022-v1:0": ModelPrice(0.80, 4.00),
    "anthropic.claude-3-sonnet-20240229-v1:0": ModelPrice(3.00, 15.00),
    "anthropic.claude-3-5-sonnet-20240620-v1:0": ModelPrice(3.00, 15.00),
    # --- AWS Bedrock: Meta / Mistral ---
    "meta.llama3-8b-instruct-v1:0": ModelPrice(0.30, 0.60),
    "meta.llama3-70b-instruct-v1:0": ModelPrice(2.65, 3.50),
    "mistral.mistral-7b-instruct-v0:2": ModelPrice(0.15, 0.20),
    "mistral.mixtral-8x7b-instruct-v0:1": ModelPrice(0.45, 0.70),
    "mistral.mistral-large-2402-v1:0": ModelPrice(4.00, 12.00),
    # --- OpenAI ---
    "gpt-4o-mini": ModelPrice(0.15, 0.60),
    "gpt-4o": ModelPrice(2.50, 10.00),
    "gpt-4.1-mini": ModelPrice(0.40, 1.60),
    # --- Google Gemini, paid tier, output includes thinking (checked 2026-09-11) ---
    "gemini-2.5-flash": ModelPrice(0.30, 2.50),
    "gemini-2.5-flash-lite": ModelPrice(0.10, 0.40),
    "gemini-2.5-pro": ModelPrice(1.25, 10.00),
    "gemini-3.5-flash": ModelPrice(1.50, 9.00),
    "gemini-3.5-flash-lite": ModelPrice(0.30, 2.50),
    # --- Groq, keyed without the vendor segment (checked 2026-09-11) ---
    "gpt-oss-20b": ModelPrice(0.075, 0.30),
    "gpt-oss-120b": ModelPrice(0.15, 0.60),
    "gpt-oss-safeguard-20b": ModelPrice(0.075, 0.30),
    "qwen3.6-27b": ModelPrice(0.60, 3.00),
    "qwen3.8-27b": ModelPrice(0.80, 4.00),
    # --- embeddings (input only) ---
    "amazon.titan-embed-text-v2:0": ModelPrice(0.02, 0.0),
    "text-embedding-3-small": ModelPrice(0.02, 0.0),
    "text-embedding-3-large": ModelPrice(0.13, 0.0),
    # --- test double ---
    "fake-echo": ModelPrice(0.50, 1.50),
}

_FALLBACK = ModelPrice(0.15, 0.60)
_loaded_override = False


def _load_override() -> None:
    global _loaded_override
    if _loaded_override:
        return
    _loaded_override = True
    path = os.getenv("CACHELLM_PRICING_FILE", "").strip()
    if not path or not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    for model, entry in raw.items():
        PRICES[model] = ModelPrice(float(entry["input_per_m"]), float(entry["output_per_m"]))


def normalise_model_id(model: str) -> str:
    """Strip routing and inference-profile decorations from a model id.

    ``bedrock/us.amazon.nova-lite-v1:0`` and ``amazon.nova-lite-v1:0`` are the
    same billable model, and cross-region profiles just add a region prefix.
    """
    m = model.strip()
    if "/" in m:
        m = m.split("/", 1)[1]
    for region_prefix in ("us.", "eu.", "apac.", "ap."):
        if m.startswith(region_prefix):
            m = m[len(region_prefix) :]
            break
    return m


def price_for(model: str) -> ModelPrice:
    _load_override()
    key = normalise_model_id(model)
    if key in PRICES:
        return PRICES[key]
    # Prefix match so dated or preview revisions still price sensibly. The
    # longest match wins: `gemini-2.5-flash-lite-preview` is a Flash-Lite, and
    # taking the first match in table order priced it as the dearer Flash.
    stems = {known.split("-2024")[0].split("-v1:")[0]: known for known in PRICES}
    matches = [stem for stem in stems if key.startswith(stem)]
    if matches:
        return PRICES[stems[max(matches, key=len)]]
    return _FALLBACK


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    return price_for(model).cost(input_tokens, output_tokens)
