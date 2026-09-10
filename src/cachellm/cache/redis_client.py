"""Redis connection factory.

``decode_responses`` is deliberately False. Embeddings are stored as raw
float32 bytes and a decoding client would corrupt them on the way back out;
text fields are decoded explicitly where they are read.
"""

from __future__ import annotations

import redis.asyncio as aioredis

from cachellm.settings import Settings


def build_redis(settings: Settings) -> aioredis.Redis:
    return aioredis.from_url(
        settings.redis_url,
        decode_responses=False,
        health_check_interval=30,
        socket_connect_timeout=5,
        socket_keepalive=True,
    )
