"""Level 2: semantic nearest-neighbour lookup backed by Redis + RedisVL.

The index is HNSW over cosine distance with a ``namespace`` tag filter, so a
query only ever considers entries that are allowed to answer it. Redis returns
cosine *distance*; because every vector is unit length, similarity is simply
``1 - distance``.

Entries are written with the plain Redis client rather than through RedisVL so
that the hash write and its TTL land in one pipeline, and so an expiring key
takes itself out of the index with no sweeper process.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import redis.asyncio as aioredis
import structlog
from redis.exceptions import RedisError
from redisvl.index import AsyncSearchIndex
from redisvl.query import FilterQuery, VectorQuery
from redisvl.query.filter import Tag

from cachellm.cache.entry import CacheEntry
from cachellm.settings import Settings

log = structlog.get_logger(__name__)

RETURN_FIELDS = [
    "entry_id",
    "namespace",
    "category",
    "model",
    "provider",
    "prompt",
    "response_text",
    "prompt_tokens",
    "completion_tokens",
    "finish_reason",
    "created_at",
    "ttl",
    "hits",
    "exact_hash",
    "tool_calls",
]


class VectorStore:
    def __init__(self, redis: aioredis.Redis, settings: Settings) -> None:
        self._redis = redis
        self._settings = settings
        self._prefix = settings.entry_prefix
        self._index: AsyncSearchIndex | None = None

    # ------------------------------------------------------------------ schema
    def schema(self) -> dict[str, Any]:
        return {
            "index": {
                "name": self._settings.index_name,
                "prefix": self._prefix,
                "storage_type": "hash",
            },
            "fields": [
                {"name": "namespace", "type": "tag"},
                {"name": "category", "type": "tag"},
                {"name": "model", "type": "tag"},
                {"name": "provider", "type": "tag"},
                {"name": "created_at", "type": "numeric", "attrs": {"sortable": True}},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": self._settings.expected_dim(),
                        "distance_metric": "cosine",
                        "algorithm": "hnsw",
                        "datatype": "float32",
                    },
                },
            ],
        }

    async def connect(self) -> None:
        index = AsyncSearchIndex.from_dict(self.schema(), redis_client=self._redis)
        await index.create(overwrite=False)
        self._index = index
        log.info(
            "vector_index_ready",
            index=self._settings.index_name,
            dims=self._settings.expected_dim(),
        )

    @property
    def index(self) -> AsyncSearchIndex:
        if self._index is None:
            raise RuntimeError("VectorStore.connect() has not been awaited")
        return self._index

    def key(self, entry_id: str) -> str:
        return f"{self._prefix}{entry_id}"

    # ------------------------------------------------------------------- writes
    async def put(self, entry: CacheEntry) -> None:
        key = self.key(entry.entry_id)
        # redis-py types the mapping key as a wide union, and Mapping is
        # invariant in its key type, so a plain dict[str, ...] will not match.
        # The values really are Redis-compatible; this is a variance nit.
        mapping = cast("Mapping[Any, Any]", entry.to_redis_mapping())
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.hset(key, mapping=mapping)
            if entry.ttl > 0:
                pipe.expire(key, entry.ttl)
            await pipe.execute()

    async def touch_hit(self, entry_id: str) -> None:
        """Record that an entry served a request. Never resurrects an expired key."""
        await self._redis.hincrby(self.key(entry_id), "hits", 1)

    async def get(self, entry_id: str) -> CacheEntry | None:
        raw = await self._redis.hgetall(self.key(entry_id))
        if not raw:
            return None
        return CacheEntry.from_redis_mapping(raw)

    async def delete(self, entry_id: str) -> int:
        return int(await self._redis.delete(self.key(entry_id)))

    # ------------------------------------------------------------------ queries
    async def search(
        self, namespace: str, vector: np.ndarray, k: int
    ) -> list[tuple[CacheEntry, float]]:
        query = VectorQuery(
            vector=np.asarray(vector, dtype=np.float32).tolist(),
            vector_field_name="embedding",
            num_results=k,
            filter_expression=Tag("namespace") == namespace,
            return_fields=RETURN_FIELDS,
            return_score=True,
        )
        rows = await self.index.query(query)
        out: list[tuple[CacheEntry, float]] = []
        for row in rows:
            distance = float(row.get("vector_distance", 1.0))
            similarity = 1.0 - distance
            out.append((CacheEntry.from_redis_mapping(row), similarity))
        out.sort(key=lambda pair: pair[1], reverse=True)
        return out

    async def count(self) -> int:
        try:
            info = await self.index.info()
        except RedisError:  # pragma: no cover - index may not exist yet
            return 0
        return int(info.get("num_docs", 0) or 0)

    # ------------------------------------------------------------- invalidation
    async def scan_keys(self, match: str | None = None) -> list[str]:
        pattern = match or f"{self._prefix}*"
        keys: list[str] = []
        async for key in self._redis.scan_iter(match=pattern, count=500):
            keys.append(key.decode() if isinstance(key, bytes) else str(key))
        return keys

    async def invalidate(
        self, *, namespace: str | None = None, model: str | None = None, drop_all: bool = False
    ) -> int:
        """Delete entries by namespace, by model, or everything.

        Namespace and model are indexed tags, so this is a query, not a scan of
        the whole keyspace. That is what makes "the system prompt changed, drop
        its cache" a one-line operation.
        """
        if drop_all:
            keys = await self.scan_keys()
            keys += await self._scan_prefix(self._settings.exact_prefix)
            if not keys:
                return 0
            return int(await self._redis.delete(*keys))

        if namespace:
            expr = Tag("namespace") == namespace
        elif model:
            expr = Tag("model") == model
        else:
            return 0

        # A filter-only query, not a nearest-neighbour search. Namespace and
        # model are indexed tags, so "drop everything for this system prompt"
        # or "this model was upgraded" is an index lookup, never a keyspace
        # scan. RedisVL escapes tag values, which matters because model ids are
        # full of dots, slashes and colons.
        to_delete: list[str] = []
        query = FilterQuery(
            filter_expression=expr,
            return_fields=["entry_id", "namespace", "exact_hash"],
            num_results=500,
        )
        async for page in self.index.paginate(query, page_size=500):
            for row in page:
                entry_id = CacheEntry._s(row, "entry_id")
                ns = CacheEntry._s(row, "namespace")
                exact = CacheEntry._s(row, "exact_hash")
                if entry_id:
                    to_delete.append(self.key(entry_id))
                if ns and exact:
                    to_delete.append(f"{self._settings.exact_prefix}{ns}:{exact}")

        if not to_delete:
            return 0
        return int(await self._redis.delete(*to_delete))

    async def _scan_prefix(self, prefix: str) -> list[str]:
        keys: list[str] = []
        async for key in self._redis.scan_iter(match=f"{prefix}*", count=500):
            keys.append(key.decode() if isinstance(key, bytes) else str(key))
        return keys
