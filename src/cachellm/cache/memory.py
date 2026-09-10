"""In-memory backend: the cache with no server behind it.

Redis is the right choice once several worker processes share one cache, or
once the entry count reaches the hundreds of thousands. Below that it is
overhead, and requiring it is the single biggest obstacle to someone trying
this at all.

The measurement that justifies this module: finding the nearest of 20,000
cached prompts by multiplying one 20000x384 matrix takes **0.44 ms**. A Redis
round trip costs 2 to 3 ms before it does any work, and the embedding step that
precedes every lookup costs about 6 ms. So for a single process the brute-force
scan is not a compromise, it is faster.

An approximate index only starts to pay above roughly 100,000 entries, which is
also where the memory this holds stops being reasonable. That is the crossover,
and it is where Redis takes over.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from cachellm.cache.analytics import NearMiss
from cachellm.cache.entry import CacheEntry
from cachellm.settings import Settings

log = structlog.get_logger(__name__)

_INITIAL_CAPACITY = 256
_NO_MATCH = -2.0  # below any real cosine similarity, so masked rows never win


class MemoryVectorStore:
    """Vector search over a numpy matrix held in this process.

    Rows are added and removed in constant time. Deleting swaps the last row
    into the freed slot rather than rebuilding, so a busy cache never pays to
    recopy the whole matrix.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._dim = settings.expected_dim()
        self._prefix = settings.entry_prefix
        self._max_entries = settings.memory_max_entries

        self._entries: dict[str, CacheEntry] = {}
        self._expires_at: dict[str, float] = {}
        self._last_used: dict[str, float] = {}

        # Parallel arrays: position -> id, namespace, vector.
        self._ids: list[str] = []
        self._pos: dict[str, int] = {}
        self._namespaces = np.empty(_INITIAL_CAPACITY, dtype=object)
        self._matrix = np.zeros((_INITIAL_CAPACITY, self._dim), dtype=np.float32)
        self._size = 0
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ setup
    async def connect(self) -> None:
        path = self._settings.memory_snapshot_path
        if path:
            await asyncio.to_thread(self._load_snapshot, Path(path))
        log.info(
            "memory_store_ready",
            dims=self._dim,
            max_entries=self._max_entries,
            snapshot=path or None,
            entries=self._size,
        )

    async def close(self) -> None:
        path = self._settings.memory_snapshot_path
        if path:
            await asyncio.to_thread(self._save_snapshot, Path(path))

    def key(self, entry_id: str) -> str:
        return f"{self._prefix}{entry_id}"

    # ----------------------------------------------------------- matrix moves
    def _grow(self) -> None:
        capacity = max(_INITIAL_CAPACITY, self._matrix.shape[0] * 2)
        matrix = np.zeros((capacity, self._dim), dtype=np.float32)
        matrix[: self._size] = self._matrix[: self._size]
        namespaces = np.empty(capacity, dtype=object)
        namespaces[: self._size] = self._namespaces[: self._size]
        self._matrix, self._namespaces = matrix, namespaces

    def _add_row(self, entry_id: str, namespace: str, vector: np.ndarray) -> None:
        existing = self._pos.get(entry_id)
        if existing is not None:
            self._matrix[existing] = vector
            self._namespaces[existing] = namespace
            return
        if self._size == self._matrix.shape[0]:
            self._grow()
        self._matrix[self._size] = vector
        self._namespaces[self._size] = namespace
        self._ids.append(entry_id)
        self._pos[entry_id] = self._size
        self._size += 1

    def _drop_row(self, entry_id: str) -> None:
        """Remove one row by moving the last row into its slot."""
        position = self._pos.pop(entry_id, None)
        if position is None:
            return
        last = self._size - 1
        if position != last:
            self._matrix[position] = self._matrix[last]
            self._namespaces[position] = self._namespaces[last]
            moved = self._ids[last]
            self._ids[position] = moved
            self._pos[moved] = position
        self._ids.pop()
        self._size -= 1

    def _forget(self, entry_id: str) -> None:
        self._entries.pop(entry_id, None)
        self._expires_at.pop(entry_id, None)
        self._last_used.pop(entry_id, None)
        self._drop_row(entry_id)

    # -------------------------------------------------------------- lifecycle
    def _purge_expired(self) -> int:
        now = time.time()
        dead = [eid for eid, expiry in self._expires_at.items() if expiry and expiry <= now]
        for entry_id in dead:
            self._forget(entry_id)
        return len(dead)

    def _evict_if_full(self) -> None:
        """Drop the least recently used entries once over the cap."""
        overflow = self._size - self._max_entries
        if overflow <= 0:
            return
        by_age = sorted(self._ids, key=lambda eid: self._last_used.get(eid, 0.0))
        for entry_id in by_age[:overflow]:
            self._forget(entry_id)
        log.debug("memory_store_evicted", count=overflow, size=self._size)

    # ------------------------------------------------------------------ writes
    async def put(self, entry: CacheEntry) -> None:
        if entry.embedding is None:
            return
        vector = np.asarray(entry.embedding, dtype=np.float32)
        if vector.shape != (self._dim,):
            log.warning("memory_store_dim_mismatch", got=vector.shape, want=self._dim)
            return
        async with self._lock:
            self._entries[entry.entry_id] = entry
            self._expires_at[entry.entry_id] = time.time() + entry.ttl if entry.ttl > 0 else 0.0
            self._last_used[entry.entry_id] = time.time()
            self._add_row(entry.entry_id, entry.namespace, vector)
            self._evict_if_full()

    async def touch_hit(self, entry_id: str) -> None:
        async with self._lock:
            entry = self._entries.get(entry_id)
            if entry is not None:
                entry.hits += 1
                self._last_used[entry_id] = time.time()

    async def get(self, entry_id: str) -> CacheEntry | None:
        async with self._lock:
            expiry = self._expires_at.get(entry_id)
            if expiry and expiry <= time.time():
                self._forget(entry_id)
                return None
            entry = self._entries.get(entry_id)
            if entry is not None:
                self._last_used[entry_id] = time.time()
            return entry

    async def delete(self, entry_id: str) -> int:
        async with self._lock:
            existed = entry_id in self._entries
            self._forget(entry_id)
            return 1 if existed else 0

    async def ttl(self, entry_id: str) -> int:
        """Seconds until this entry expires, using Redis's return convention.

        -1 means stored with no expiry, -2 means nothing is there, so callers
        and tests do not have to care which backend answered.
        """
        async with self._lock:
            if entry_id not in self._entries:
                return -2
            expiry = self._expires_at.get(entry_id, 0.0)
            if not expiry:
                return -1
            remaining = expiry - time.time()
            if remaining <= 0:
                self._forget(entry_id)
                return -2
            return math.ceil(remaining)

    # ----------------------------------------------------------------- queries
    async def search(
        self, namespace: str, vector: np.ndarray, k: int
    ) -> list[tuple[CacheEntry, float]]:
        async with self._lock:
            self._purge_expired()
            if self._size == 0:
                return []

            query = np.asarray(vector, dtype=np.float32)
            # One matrix-vector product scores every cached prompt at once.
            scores = self._matrix[: self._size] @ query
            # Entries from other namespaces must never win, so push them below
            # any achievable cosine similarity rather than filtering after.
            mask = self._namespaces[: self._size] == namespace
            scores = np.where(mask, scores, _NO_MATCH)

            wanted = min(k, self._size)
            if wanted < self._size:
                candidates = np.argpartition(-scores, wanted - 1)[:wanted]
            else:
                candidates = np.arange(self._size)

            hits: list[tuple[CacheEntry, float]] = []
            for position in candidates:
                score = float(scores[position])
                if score <= _NO_MATCH:
                    continue
                entry = self._entries.get(self._ids[position])
                if entry is not None:
                    hits.append((entry, score))
            hits.sort(key=lambda pair: pair[1], reverse=True)
            return hits

    async def count(self) -> int:
        async with self._lock:
            self._purge_expired()
            return self._size

    async def scan_keys(self, match: str | None = None) -> list[str]:
        async with self._lock:
            self._purge_expired()
            return [self.key(entry_id) for entry_id in self._ids]

    async def invalidate(
        self, *, namespace: str | None = None, model: str | None = None, drop_all: bool = False
    ) -> int:
        async with self._lock:
            if drop_all:
                removed = self._size
                self._entries.clear()
                self._expires_at.clear()
                self._last_used.clear()
                self._ids.clear()
                self._pos.clear()
                self._size = 0
                return removed

            if namespace:
                doomed = [e for e in self._entries.values() if e.namespace == namespace]
            elif model:
                doomed = [e for e in self._entries.values() if e.model == model]
            else:
                return 0
            for entry in doomed:
                self._forget(entry.entry_id)
            return len(doomed)

    # --------------------------------------------------------------- snapshots
    def _save_snapshot(self, path: Path) -> None:
        """Persist the cache so a restart does not start cold. Best effort."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "dim": self._dim,
                "entries": [
                    {
                        **{k: v for k, v in asdict(entry).items() if k != "embedding"},
                        "expires_at": self._expires_at.get(entry.entry_id, 0.0),
                    }
                    for entry in self._entries.values()
                ],
            }
            order = [entry.entry_id for entry in self._entries.values()]
            vectors = (
                np.stack([self._matrix[self._pos[eid]] for eid in order])
                if order
                else np.zeros((0, self._dim), dtype=np.float32)
            )
            np.savez_compressed(path, meta=json.dumps(payload), vectors=vectors)
            log.info("memory_snapshot_saved", path=str(path), entries=len(order))
        except Exception as exc:  # a snapshot must never break shutdown
            log.warning("memory_snapshot_save_failed", error=str(exc)[:200])

    def _load_snapshot(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            blob = np.load(path, allow_pickle=False)
            payload = json.loads(str(blob["meta"]))
            if int(payload.get("dim", -1)) != self._dim:
                log.warning("memory_snapshot_dim_mismatch", ignoring=str(path))
                return
            vectors = blob["vectors"]
            now = time.time()
            restored = 0
            for row, raw in zip(vectors, payload["entries"], strict=True):
                expiry = float(raw.pop("expires_at", 0.0) or 0.0)
                if expiry and expiry <= now:
                    continue  # expired while we were away
                entry = CacheEntry(**raw)
                self._entries[entry.entry_id] = entry
                self._expires_at[entry.entry_id] = expiry
                self._last_used[entry.entry_id] = now
                self._add_row(entry.entry_id, entry.namespace, row)
                restored += 1
            log.info("memory_snapshot_loaded", path=str(path), entries=restored)
        except Exception as exc:  # a bad snapshot must not stop the proxy
            log.warning("memory_snapshot_load_failed", error=str(exc)[:200])


class MemoryExactStore:
    """Level 1 lookup, backed by a dict."""

    def __init__(self, settings: Settings) -> None:
        self._prefix = settings.exact_prefix
        self._pointers: dict[str, tuple[str, float]] = {}

    def key(self, namespace: str, exact_hash: str) -> str:
        return f"{self._prefix}{namespace}:{exact_hash}"

    async def get(self, namespace: str, exact_hash: str) -> str | None:
        record = self._pointers.get(self.key(namespace, exact_hash))
        if record is None:
            return None
        entry_id, expiry = record
        if expiry and expiry <= time.time():
            self._pointers.pop(self.key(namespace, exact_hash), None)
            return None
        return entry_id

    async def put(self, namespace: str, exact_hash: str, entry_id: str, ttl: int) -> None:
        expiry = time.time() + ttl if ttl > 0 else 0.0
        self._pointers[self.key(namespace, exact_hash)] = (entry_id, expiry)

    async def delete(self, namespace: str, exact_hash: str) -> None:
        self._pointers.pop(self.key(namespace, exact_hash), None)


class MemoryAnalytics:
    """Counters, latency samples and the near-miss log, all in this process."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._counters: dict[str, float] = {}
        self._latency: dict[str, deque[float]] = {}
        self._near: deque[dict[str, Any]] = deque(maxlen=settings.near_miss_log_size)

    async def incr(self, field: str, amount: int = 1) -> None:
        self._counters[field] = self._counters.get(field, 0.0) + amount

    async def incr_float(self, field: str, amount: float) -> None:
        if amount:
            self._counters[field] = self._counters.get(field, 0.0) + amount

    async def bulk(self, ints: dict[str, int], floats: dict[str, float] | None = None) -> None:
        for field, count in ints.items():
            if count:
                self._counters[field] = self._counters.get(field, 0.0) + count
        for field, value in (floats or {}).items():
            if value:
                self._counters[field] = self._counters.get(field, 0.0) + value

    async def counters(self) -> dict[str, float]:
        return dict(self._counters)

    async def reset(self) -> None:
        self._counters.clear()
        self._latency.clear()
        self._near.clear()

    async def record_latency(self, result: str, ms: float) -> None:
        samples = self._latency.setdefault(result, deque(maxlen=5_000))
        samples.append(ms)

    async def latency_percentiles(self, result: str) -> dict[str, float]:
        values = sorted(self._latency.get(result, ()))
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

    async def record_near_miss(self, miss: NearMiss) -> None:
        payload = asdict(miss)
        if not self._settings.log_prompts:
            payload["prompt"] = payload["prompt"][:120]
            payload["matched_prompt"] = payload["matched_prompt"][:120]
        self._near.appendleft(payload)

    async def near_misses(self, limit: int = 50) -> list[dict[str, Any]]:
        return list(self._near)[:limit]

    async def near_miss_histogram(self, buckets: int = 20) -> list[dict[str, Any]]:
        misses = await self.near_misses(limit=self._settings.near_miss_log_size)
        hist = [0] * buckets
        for miss in misses:
            sim = float(miss.get("similarity", 0.0))
            hist[min(buckets - 1, max(0, math.floor(sim * buckets)))] += 1
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
