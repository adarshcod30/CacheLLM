"""Durable counters and the near-miss log.

Prometheus holds the time series for dashboards, but it resets when the process
does and it cannot answer "what did the cache do last week". These Redis-backed
counters survive restarts and are shared across workers, which is what the
admin endpoints and the README numbers read from.

The near-miss log is the tuning instrument: every lookup that landed just below
the threshold is kept with its score, so you can see exactly what a slightly
looser threshold would have bought you, on your own traffic.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from cachellm.settings import Settings


def _trim_prompt(text: str, keep_full: bool) -> str:
    """Prompts are sensitive; keep enough to recognise, no more."""
    return text if keep_full else text[:96]


if TYPE_CHECKING:  # redis is an optional extra; this module imports without it
    import redis.asyncio as aioredis


@dataclass
class RequestRecord:
    """One line of the request log: what the cache did, and what it saved.

    Prompt text is truncated and only kept at all so a person reading
    `cachellm stats` can recognise their own traffic. Set
    CACHELLM_LOG_PROMPTS=false (the default) and it is trimmed to a fingerprint.
    """

    at: float
    status: str  # HIT | MISS | BYPASS | SHADOW
    tier: str  # exact | semantic | ""
    similarity: float
    category: str
    model: str
    latency_ms: float
    saved_usd: float
    spent_usd: float
    prompt: str
    reason: str = ""  # why it was bypassed, when it was


@dataclass
class NearMiss:
    prompt: str
    matched_prompt: str
    similarity: float
    threshold: float
    category: str
    namespace: str
    model: str
    at: float


class Analytics:
    def __init__(self, redis: aioredis.Redis, settings: Settings) -> None:
        self._redis = redis
        self._settings = settings
        self._counters = f"{settings.stats_prefix}counters"
        self._near = f"{settings.stats_prefix}near_misses"
        self._log = f"{settings.stats_prefix}requests"
        self._latency = f"{settings.stats_prefix}latency"

    # ------------------------------------------------------------------ counters
    async def incr(self, field: str, amount: int = 1) -> None:
        await self._redis.hincrby(self._counters, field, amount)

    async def incr_float(self, field: str, amount: float) -> None:
        if amount:
            await self._redis.hincrbyfloat(self._counters, field, amount)

    async def bulk(self, ints: dict[str, int], floats: dict[str, float] | None = None) -> None:
        async with self._redis.pipeline(transaction=False) as pipe:
            for field, count in ints.items():
                if count:
                    pipe.hincrby(self._counters, field, count)
            for field, value in (floats or {}).items():
                if value:
                    pipe.hincrbyfloat(self._counters, field, value)
            await pipe.execute()

    async def counters(self) -> dict[str, float]:
        raw = await self._redis.hgetall(self._counters)
        out: dict[str, float] = {}
        for key, value in raw.items():
            name = key.decode() if isinstance(key, bytes) else str(key)
            text = value.decode() if isinstance(value, bytes) else str(value)
            try:
                out[name] = float(text)
            except ValueError:
                continue
        return out

    async def reset(self) -> None:
        await self._redis.delete(self._counters, self._near, self._latency, self._log)

    # ---------------------------------------------------------------- latency
    async def record_latency(self, result: str, ms: float) -> None:
        """Keep a rolling sample per outcome so admin/stats can show percentiles."""
        await self._redis.lpush(f"{self._latency}:{result}", f"{ms:.3f}")
        await self._redis.ltrim(f"{self._latency}:{result}", 0, 4_999)

    async def latency_percentiles(self, result: str) -> dict[str, float]:
        raw = await self._redis.lrange(f"{self._latency}:{result}", 0, -1)
        values = sorted(float(v) for v in raw)
        if not values:
            return {}

        def pct(p: float) -> float:
            idx = min(len(values) - 1, max(0, round((p / 100.0) * (len(values) - 1))))
            return round(values[idx], 2)

        return {
            "count": len(values),
            "p50": pct(50),
            "p95": pct(95),
            "p99": pct(99),
            "min": round(values[0], 2),
            "max": round(values[-1], 2),
        }

    # -------------------------------------------------------------- request log
    async def record_request(self, record: RequestRecord) -> None:
        payload = asdict(record)
        payload["prompt"] = _trim_prompt(payload["prompt"], self._settings.log_prompts)
        await self._redis.lpush(self._log, json.dumps(payload))
        await self._redis.ltrim(self._log, 0, self._settings.request_log_size - 1)

    async def recent_requests(self, limit: int = 50) -> list[dict[str, Any]]:
        raw = await self._redis.lrange(self._log, 0, limit - 1)
        out: list[dict[str, Any]] = []
        for item in raw:
            with contextlib.suppress(json.JSONDecodeError):
                out.append(json.loads(item))
        return out

    # -------------------------------------------------------------- near misses
    async def record_near_miss(self, miss: NearMiss) -> None:
        payload = asdict(miss)
        if not self._settings.log_prompts:
            payload["prompt"] = payload["prompt"][:120]
            payload["matched_prompt"] = payload["matched_prompt"][:120]
        await self._redis.lpush(self._near, json.dumps(payload))
        await self._redis.ltrim(self._near, 0, self._settings.near_miss_log_size - 1)

    async def near_misses(self, limit: int = 50) -> list[dict[str, Any]]:
        raw = await self._redis.lrange(self._near, 0, limit - 1)
        out: list[dict[str, Any]] = []
        for item in raw:
            try:
                out.append(json.loads(item))
            except json.JSONDecodeError:
                continue
        return out

    async def near_miss_histogram(self, buckets: int = 20) -> list[dict[str, Any]]:
        """How many near misses sit in each similarity band.

        Read this as: "if I dropped the threshold to X, this many more requests
        would have been served from cache."
        """
        misses = await self.near_misses(limit=self._settings.near_miss_log_size)
        hist = [0] * buckets
        for miss in misses:
            sim = float(miss.get("similarity", 0.0))
            idx = min(buckets - 1, max(0, int(sim * buckets)))
            hist[idx] += 1
        width = 1.0 / buckets
        cumulative = 0
        rows: list[dict[str, Any]] = []
        for i in range(buckets - 1, -1, -1):
            cumulative += hist[i]
            rows.append(
                {
                    "similarity_at_least": round(i * width, 3),
                    "in_band": hist[i],
                    "would_hit_if_threshold_here": cumulative,
                }
            )
        return rows

    @staticmethod
    def now() -> float:
        return time.time()
