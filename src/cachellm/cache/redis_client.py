"""Redis connection factory.

``decode_responses`` is deliberately False. Embeddings are stored as raw
float32 bytes and a decoding client would corrupt them on the way back out;
text fields are decoded explicitly where they are read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cachellm.settings import Settings

if TYPE_CHECKING:
    import redis.asyncio as aioredis


def build_redis(settings: Settings) -> aioredis.Redis:
    # Imported here rather than at module scope, so the package installs and
    # runs without the redis extra. Memory is the default backend.
    import redis.asyncio as aioredis

    client: Any = aioredis.from_url(
        settings.redis_url,
        decode_responses=False,
        health_check_interval=30,
        socket_connect_timeout=5,
        socket_keepalive=True,
    )
    return client
