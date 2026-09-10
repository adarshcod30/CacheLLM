"""Typed configuration for CacheLLM.

Every value can be set through an environment variable prefixed with
``CACHELLM_`` (for example ``CACHELLM_THRESHOLD_DEFAULT=0.94``) or through a
``.env`` file sitting next to the process working directory. Defaults are
deliberately conservative: it is far worse to serve a confidently wrong cached
answer than to miss the cache and pay for a fresh call.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Embedding dimensions for the models we ship presets for. Used to catch the
# classic "changed the model but not the index" mistake at startup.
KNOWN_EMBEDDING_DIMS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "snowflake/snowflake-arctic-embed-s": 384,
    "jinaai/jina-embeddings-v2-small-en": 512,
    "thenlper/gte-base": 768,
    "hash-embedder": 384,
}

# Similarity thresholds DO NOT transfer between embedding models. Measured on
# the project's own labelled corpus (see `bench/compare_models.py` and
# docs/evaluation.md), these are the highest thresholds at which each model
# served zero hard negatives: pairs one word apart with opposite meaning, like
# "undo the last git commit" versus "undo the last git merge".
#
# The spread is the point. bge-small needs 0.96 where MiniLM needs 0.89, so a
# threshold copied from a blog post is meaningless without naming the model it
# was measured on.
CALIBRATED_THRESHOLDS: dict[str, float] = {
    "sentence-transformers/all-MiniLM-L6-v2": 0.89,
    "BAAI/bge-small-en-v1.5": 0.96,
    "BAAI/bge-base-en-v1.5": 0.94,
    "snowflake/snowflake-arctic-embed-s": 0.98,
    "jinaai/jina-embeddings-v2-small-en": 0.96,
    "thenlper/gte-base": 0.96,
}

Category = Literal[
    "factual",
    "classification",
    "creative",
    "volatile",
    "conversational",
    "default",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CACHELLM_",
        extra="ignore",
        protected_namespaces=(),
    )

    # ------------------------------------------------------------------ server
    host: str = "127.0.0.1"
    port: int = 8080
    log_level: str = "INFO"
    log_json: bool = True
    #: Prompt and completion text is NEVER written to logs unless this is on.
    log_prompts: bool = False

    # -------------------------------------------------------------------- auth
    #: Comma-separated client keys. Empty means auth is disabled (local dev only).
    api_keys: str = ""
    require_auth_for_admin: bool = True

    # ----------------------------------------------------------------- storage
    #: Where the cache lives. "memory" needs nothing installed and searches a
    #: numpy matrix in this process; "redis" shares one cache across workers and
    #: survives restarts; "auto" uses Redis when it is reachable and quietly
    #: falls back to memory when it is not, which is what makes
    #: `pip install cachellm-proxy && cachellm serve` work on a bare machine.
    #:
    #: Memory is not a lesser option below roughly 100k entries: scanning 20,000
    #: cached prompts takes 0.44 ms, where a Redis round trip alone costs 2-3 ms.
    backend: Literal["auto", "memory", "redis"] = "auto"
    #: Cap on entries held in memory. 50k of 384-dim vectors is about 73 MB.
    memory_max_entries: int = 50_000
    #: Optional file to persist the in-memory cache across restarts.
    memory_snapshot_path: str = ""

    # ------------------------------------------------------------------- redis
    redis_url: str = "redis://localhost:6379/0"
    index_name: str = "cachellm_idx"
    entry_prefix: str = "cachellm:e:"
    exact_prefix: str = "cachellm:x:"
    stats_prefix: str = "cachellm:s:"

    # -------------------------------------------------------------- embeddings
    embedding_backend: Literal["fastembed", "hash"] = "fastembed"
    #: MiniLM is the default because it gave four times the safe recall of
    #: bge-small on the evaluation corpus, at a third of the download size.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_cache_size: int = 2048

    # ---------------------------------------------------------------- caching
    enabled: bool = True
    #: Observe-only. Looks up, records what it *would* have served, then still
    #: calls the provider. The safe way to roll out onto real traffic.
    shadow_mode: bool = False
    cache_streaming: bool = True
    #: Store responses that stopped because they hit max_tokens. Off by default:
    #: a truncated answer served from cache forever is a silent quality bug. Turn
    #: it on if you cap max_tokens deliberately and want those answers cached.
    cache_truncated: bool = False
    top_k: int = 3
    #: A miss within this margin of the threshold is recorded as a near miss.
    near_miss_margin: float = 0.06
    near_miss_log_size: int = 500

    #: 0.0 means "use the calibrated value for the configured embedding model".
    #: Set any of these explicitly to override the calibration.
    threshold_default: float = 0.0
    threshold_factual: float = 0.0
    threshold_classification: float = 0.0
    threshold_creative: float = 0.0
    threshold_volatile: float = 0.0
    threshold_conversational: float = 0.0

    #: Offsets applied to the calibrated threshold, per category. Classification
    #: has a small, constrained answer space so it tolerates looser matching;
    #: creative generation tolerates almost none.
    category_threshold_offsets: dict[str, float] = {
        "classification": -0.02,
        "factual": 0.0,
        "conversational": +0.03,
        "volatile": +0.03,
        "creative": +0.06,
        "default": 0.0,
    }

    ttl_default: int = 86_400  # 1 day
    ttl_factual: int = 604_800  # 7 days
    ttl_classification: int = 604_800
    ttl_creative: int = 3_600
    ttl_volatile: int = 900  # 15 minutes
    ttl_conversational: int = 3_600

    # ------------------------------------------------------- cacheability rules
    max_cacheable_temperature: float = 0.3
    max_prompt_chars: int = 8_000
    cache_multi_turn: bool = False
    cache_json_mode: bool = False
    cache_tool_calls: bool = False
    #: Skip caching prompts that look like they carry personal data.
    pii_guard: bool = True
    #: Strip filler words before embedding. Raises hit rate, slightly raises risk.
    strip_filler_words: bool = False

    # --------------------------------------------------------------- providers
    default_provider: Literal["bedrock", "openai", "fake"] = "bedrock"
    aws_region: str = "us-east-1"
    aws_profile: str = ""
    #: Simulated upstream latency for the `fake` provider, so a reproducible
    #: benchmark can show a realistic hit-versus-miss gap without spending money.
    fake_latency_ms: float = 0.0
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    request_timeout: float = 120.0
    provider_max_retries: int = 2

    # ----------------------------------------------------------- observability
    metrics_enabled: bool = True
    tracing_enabled: bool = False
    otlp_endpoint: str = ""
    service_name: str = "cachellm"

    @field_validator(
        "threshold_default",
        "threshold_factual",
        "threshold_classification",
        "threshold_creative",
        "threshold_volatile",
        "threshold_conversational",
    )
    @classmethod
    def _valid_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("similarity thresholds must be between 0 and 1")
        return v

    # ------------------------------------------------------------------ helpers
    @property
    def client_keys(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def auth_enabled(self) -> bool:
        return bool(self.client_keys)

    @property
    def calibrated_threshold(self) -> float:
        """The safe threshold for the configured embedding model.

        0.92 is the fallback for an uncalibrated model. It is a guess, and the
        proxy says so at startup, because the measured safe points across six
        models span 0.89 to 0.98.
        """
        if self.embedding_backend == "hash":
            return 0.92
        return CALIBRATED_THRESHOLDS.get(self.embedding_model, 0.92)

    @property
    def is_calibrated(self) -> bool:
        return self.embedding_backend != "hash" and self.embedding_model in CALIBRATED_THRESHOLDS

    def threshold_for(self, category: str) -> float:
        explicit = float(getattr(self, f"threshold_{category}", 0.0) or 0.0)
        if explicit > 0.0:
            return explicit
        base = self.threshold_default if self.threshold_default > 0.0 else self.calibrated_threshold
        offset = self.category_threshold_offsets.get(category, 0.0)
        # Cap below 1.0: an exact-similarity requirement would make the semantic
        # tier dead weight, and the exact tier already covers literal repeats.
        return round(min(0.995, max(0.0, base + offset)), 4)

    def ttl_for(self, category: str) -> int:
        return int(getattr(self, f"ttl_{category}", self.ttl_default))

    def expected_dim(self) -> int:
        return KNOWN_EMBEDDING_DIMS.get(self.embedding_model, self.embedding_dim)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Test helper: drop the memoised Settings instance."""
    get_settings.cache_clear()
