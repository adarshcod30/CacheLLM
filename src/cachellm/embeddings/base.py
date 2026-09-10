from __future__ import annotations

import abc

import numpy as np


class Embedder(abc.ABC):
    """Turns text into a unit-length float32 vector.

    Vectors are L2-normalised on the way out, which means the cosine distance
    Redis reports can be turned into a similarity with ``1 - distance`` and no
    extra maths at query time.
    """

    name: str
    dim: int

    @abc.abstractmethod
    async def embed(self, text: str) -> np.ndarray: ...

    @abc.abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]: ...

    async def warmup(self) -> None:
        await self.embed("warmup")

    @staticmethod
    def normalise(vector: np.ndarray) -> np.ndarray:
        vec = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return vec
        return (vec / norm).astype(np.float32)

    @staticmethod
    def to_bytes(vector: np.ndarray) -> bytes:
        return np.asarray(vector, dtype=np.float32).tobytes()
