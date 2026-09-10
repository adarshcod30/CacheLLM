from __future__ import annotations

from cachellm.embeddings.base import Embedder
from cachellm.embeddings.hash_backend import HashEmbedder
from cachellm.settings import Settings

__all__ = ["Embedder", "HashEmbedder", "build_embedder"]


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "hash":
        return HashEmbedder(dim=settings.embedding_dim)
    from cachellm.embeddings.fastembed_backend import FastEmbedEmbedder

    return FastEmbedEmbedder(
        model_name=settings.embedding_model,
        dim=settings.expected_dim(),
        cache_size=settings.embedding_cache_size,
    )
