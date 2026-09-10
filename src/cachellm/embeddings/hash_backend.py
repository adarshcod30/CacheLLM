"""Deterministic hashing embedder.

No model download, no network, identical output on every machine, which makes
it the right backend for unit tests and CI. It uses the classic hashing trick
(signed token buckets), so it still produces meaningful lexical similarity:
shared words pull vectors together. It does NOT understand meaning, so it is
not a substitute for a real model in the evaluation numbers.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from cachellm.embeddings.base import Embedder

_TOKEN = re.compile(r"[a-z0-9']+")


class HashEmbedder(Embedder):
    def __init__(self, dim: int = 384) -> None:
        self.name = "hash-embedder"
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _TOKEN.findall(text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            bucket = value % self.dim
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vec[bucket] += sign
        return self.normalise(vec)

    async def embed(self, text: str) -> np.ndarray:
        return self._vector(text)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self._vector(t) for t in texts]
