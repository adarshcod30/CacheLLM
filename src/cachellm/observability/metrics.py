"""Prometheus metrics.

The metric that matters most is ``cachellm_cost_usd_total{kind="saved"}``. It is
computed once, at serve time, from the token counts stored on the cache entry
multiplied by that model's price. Everything downstream (the Grafana panel, the
admin endpoint, the README headline) reads the same number, so they can never
disagree with each other.

Label cardinality is kept deliberately low: model and provider are bounded
sets, category has six values, result has five. Prompt text never becomes a
label.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST

# Buckets tuned for this workload: cache hits land in single-digit milliseconds,
# provider calls in hundreds of milliseconds to seconds.
LATENCY_BUCKETS = (0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
SIMILARITY_BUCKETS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 1.0)


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()

        self.requests = Counter(
            "cachellm_requests_total",
            "Chat completion requests handled, by cache outcome.",
            ["result", "category", "provider", "model"],
            registry=self.registry,
        )
        self.request_duration = Histogram(
            "cachellm_request_duration_seconds",
            "End-to-end request duration.",
            ["result"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.lookup_duration = Histogram(
            "cachellm_lookup_duration_seconds",
            "Time spent deciding hit or miss (policy, embedding, vector search).",
            ["tier"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.embed_duration = Histogram(
            "cachellm_embed_duration_seconds",
            "Time spent embedding the prompt.",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.similarity = Histogram(
            "cachellm_similarity",
            "Best similarity score seen at lookup time.",
            ["result"],
            buckets=SIMILARITY_BUCKETS,
            registry=self.registry,
        )
        self.tokens = Counter(
            "cachellm_tokens_total",
            "Tokens, split by whether they were spent upstream or avoided.",
            ["kind", "direction", "model"],
            registry=self.registry,
        )
        self.cost = Counter(
            "cachellm_cost_usd_total",
            "Modelled USD, split into spent upstream and saved by the cache.",
            ["kind", "model"],
            registry=self.registry,
        )
        self.entries = Gauge(
            "cachellm_cache_entries",
            "Documents currently in the vector index.",
            registry=self.registry,
        )
        self.provider_errors = Counter(
            "cachellm_provider_errors_total",
            "Upstream provider failures.",
            ["provider"],
            registry=self.registry,
        )
        self.coalesced = Counter(
            "cachellm_coalesced_total",
            "Requests that joined an in-flight identical call instead of duplicating it.",
            registry=self.registry,
        )
        self.near_misses = Counter(
            "cachellm_near_miss_total",
            "Lookups that landed just below the similarity threshold.",
            ["category"],
            registry=self.registry,
        )
        self.shadow_hits = Counter(
            "cachellm_shadow_hits_total",
            "Requests the cache would have served while running in shadow mode.",
            ["category"],
            registry=self.registry,
        )

    # ------------------------------------------------------------------ helpers
    def observe_tokens(
        self, model: str, prompt_tokens: int, completion_tokens: int, kind: str
    ) -> None:
        if prompt_tokens:
            self.tokens.labels(kind=kind, direction="input", model=model).inc(prompt_tokens)
        if completion_tokens:
            self.tokens.labels(kind=kind, direction="output", model=model).inc(completion_tokens)

    def observe_cost(self, model: str, usd: float, kind: str) -> None:
        if usd:
            self.cost.labels(kind=kind, model=model).inc(usd)

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST


_metrics: Metrics | None = None


def get_metrics() -> Metrics:
    global _metrics
    if _metrics is None:
        _metrics = Metrics()
    return _metrics


def reset_metrics() -> None:
    """Test helper: start from a clean registry."""
    global _metrics
    _metrics = None
