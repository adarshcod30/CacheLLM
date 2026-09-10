"""The fastembed embedder, with the ONNX model swapped for a counting stand-in.

CI never downloads the real model, so this file is what covers the embedder's
own logic: the cache, batching, and normalisation.
"""

from __future__ import annotations

import numpy as np
import pytest

from cachellm.embeddings.fastembed_backend import FastEmbedEmbedder


class CountingModel:
    """Deterministic vectors, and a record of every text the model was asked for."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed(self, texts):
        for text in texts:
            self.seen.append(text)
            vector = np.zeros(8, dtype=np.float32)
            vector[len(text) % 8] = 3.0
            vector[(len(text) + 3) % 8] = 4.0
            yield vector


def embedder(cache_size: int = 2048) -> tuple[FastEmbedEmbedder, CountingModel]:
    model = CountingModel()
    emb = FastEmbedEmbedder("stand-in", dim=8, cache_size=cache_size)
    emb._model = model
    return emb, model


async def test_a_batch_larger_than_the_cache_returns_every_vector() -> None:
    """The admin threshold sweep passes user-sized lists straight in here."""
    emb, _ = embedder(cache_size=2)
    texts = [f"question number {i}" for i in range(5)]
    vectors = await emb.embed_batch(texts)
    assert len(vectors) == 5


@pytest.mark.parametrize("cache_size", [0, 1])
async def test_a_tiny_or_disabled_cache_still_embeds(cache_size: int) -> None:
    """CACHELLM_EMBEDDING_CACHE_SIZE=0 used to fail the startup warmup, which
    switched the whole cache off without an obvious reason."""
    emb, model = embedder(cache_size=cache_size)
    await emb.warmup()
    vector = await emb.embed("What is Redis?")
    assert vector.shape == (8,)
    assert model.seen[-1] == "What is Redis?"


async def test_repeats_within_a_batch_run_the_model_once() -> None:
    emb, model = embedder()
    vectors = await emb.embed_batch(["same", "other", "same"])
    assert model.seen == ["same", "other"]
    assert np.array_equal(vectors[0], vectors[2])


async def test_a_cached_text_is_not_embedded_again() -> None:
    emb, model = embedder()
    await emb.embed("What is Redis?")
    await emb.embed("What is Redis?")
    await emb.embed_batch(["What is Redis?", "new question"])
    assert model.seen == ["What is Redis?", "new question"]


async def test_vectors_come_back_unit_length_so_dot_product_is_cosine() -> None:
    emb, _ = embedder()
    vector = await emb.embed("any text at all")
    assert np.isclose(float(np.linalg.norm(vector)), 1.0)


async def test_an_empty_batch_is_free() -> None:
    emb, model = embedder()
    assert await emb.embed_batch([]) == []
    assert model.seen == []
