"""Level 1: exact-match lookup.

A literal repeat should never pay for an embedding. This tier is a plain
``GET`` that resolves a normalised-prompt hash to an entry id, so it answers in
about a millisecond and takes the pressure off the vector index.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cachellm.settings import Settings

if TYPE_CHECKING:  # redis is an optional extra; this module imports without it
    import redis.asyncio as aioredis


class ExactStore:
    def __init__(self, redis: aioredis.Redis, settings: Settings) -> None:
        self._redis = redis
        self._prefix = settings.exact_prefix

    def key(self, namespace: str, exact_hash: str) -> str:
        return f"{self._prefix}{namespace}:{exact_hash}"

    async def get(self, namespace: str, exact_hash: str) -> str | None:
        raw = await self._redis.get(self.key(namespace, exact_hash))
        if raw is None:
            return None
        return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)

    async def put(self, namespace: str, exact_hash: str, entry_id: str, ttl: int) -> None:
        await self._redis.set(self.key(namespace, exact_hash), entry_id, ex=ttl)

    async def delete(self, namespace: str, exact_hash: str) -> None:
        await self._redis.delete(self.key(namespace, exact_hash))
