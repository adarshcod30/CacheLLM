"""Local ONNX embeddings via fastembed.

Running the embedding model *inside* the proxy is the single most important
latency decision in the project. A hosted embedding API costs 80-200 ms from
India, which would become the floor for every cache hit and destroy the whole
point. bge-small on CPU is single-digit milliseconds and costs nothing.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any

import numpy as np

from cachellm.embeddings.base import Embedder


class FastEmbedEmbedder(Embedder):
    def __init__(self, model_name: str, dim: int, cache_size: int = 2048) -> None:
        self.name = model_name
        self.dim = dim
        self._cache_size = cache_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._model: Any = None
        self._lock = asyncio.Lock()

    async def _ensure_model(self) -> Any:
        """Load the ONNX model once, off the event loop.

        Loading takes a second or two and is CPU-bound, so it goes to a worker
        thread; the lock stops a burst of concurrent first requests from each
        loading their own copy.
        """
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    from fastembed import TextEmbedding

                    def load() -> Any:
                        return TextEmbedding(model_name=self.name)

                    self._model = await asyncio.to_thread(load)
        return self._model

    def _cache_get(self, text: str) -> np.ndarray | None:
        vec = self._cache.get(text)
        if vec is not None:
            self._cache.move_to_end(text)
        return vec

    def _cache_put(self, text: str, vector: np.ndarray) -> None:
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    async def embed(self, text: str) -> np.ndarray:
        cached = self._cache_get(text)
        if cached is not None:
            return cached
        vectors = await self.embed_batch([text])
        return vectors[0]

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed many texts, running the model once per distinct uncached text.

        Results are assembled from what was just computed, never read back out
        of the cache. The cache is bounded, so reading back used to fail two
        ways: a batch larger than the cache evicted its own first results, and
        a cache size of 0 made every call fail, which quietly turned caching off.
        """
        if not texts:
            return []
        model = await self._ensure_model()
        found: dict[str, np.ndarray] = {}
        for text in texts:
            if (hit := self._cache_get(text)) is not None:
                found[text] = hit
        pending = list(dict.fromkeys(t for t in texts if t not in found))
        if pending:
            raw = await asyncio.to_thread(lambda: list(model.embed(pending)))
            for text, vector in zip(pending, raw, strict=True):
                found[text] = self.normalise(np.asarray(vector, dtype=np.float32))
                self._cache_put(text, found[text])
        return [found[t] for t in texts]
