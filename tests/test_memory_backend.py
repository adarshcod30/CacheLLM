"""Behaviour specific to the in-memory store.

The shared cache tests already run against both backends, so what is left here
is what only the memory store can get wrong: the swap-remove that keeps its
matrix compact, the eviction cap, the snapshot round trip, and the fallback
that picks it when Redis is absent.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from cachellm.api.deps import build_state, shutdown_state
from cachellm.cache.entry import CacheEntry
from cachellm.cache.memory import MemoryAnalytics, MemoryVectorStore
from cachellm.providers.registry import ProviderRegistry
from tests.conftest import make_settings

DIM = 384


def unit(seed: int, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def entry(
    entry_id: str,
    namespace: str = "ns",
    vector: np.ndarray | None = None,
    ttl: int = 600,
    model: str = "fake/echo",
) -> CacheEntry:
    return CacheEntry(
        entry_id=entry_id,
        namespace=namespace,
        category="factual",
        model=model,
        provider="fake",
        prompt=f"prompt {entry_id}",
        response_text=f"answer {entry_id}",
        prompt_tokens=5,
        completion_tokens=7,
        ttl=ttl,
        exact_hash=entry_id,
        embedding=vector if vector is not None else unit(hash(entry_id) % 10_000),
    )


@pytest.fixture
def store() -> MemoryVectorStore:
    return MemoryVectorStore(make_settings(backend="memory"))


async def test_finds_the_nearest_vector(store: MemoryVectorStore) -> None:
    target = unit(1)
    await store.put(entry("a", vector=target))
    await store.put(entry("b", vector=unit(2)))
    await store.put(entry("c", vector=unit(3)))
    hits = await store.search("ns", target, k=3)
    assert hits[0][0].entry_id == "a"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    assert hits[0][1] > hits[-1][1]


async def test_namespaces_cannot_leak_into_each_other(store: MemoryVectorStore) -> None:
    shared = unit(7)
    await store.put(entry("mine", namespace="ns-a", vector=shared))
    await store.put(entry("theirs", namespace="ns-b", vector=shared))
    hits = await store.search("ns-a", shared, k=5)
    assert [e.entry_id for e, _ in hits] == ["mine"]


async def test_deleting_keeps_the_matrix_correct(store: MemoryVectorStore) -> None:
    """Deletes swap the last row into the freed slot; the index must follow."""
    vectors = {f"e{i}": unit(i) for i in range(12)}
    for eid, vec in vectors.items():
        await store.put(entry(eid, vector=vec))
    for eid in ("e0", "e5", "e11", "e3"):
        assert await store.delete(eid) == 1
    assert await store.count() == 8
    # every survivor must still be findable by its own vector
    for eid in ("e1", "e2", "e4", "e6", "e7", "e8", "e9", "e10"):
        hits = await store.search("ns", vectors[eid], k=1)
        assert hits[0][0].entry_id == eid, f"{eid} lost its row after deletes"
    # and the deleted ones must be gone
    for eid in ("e0", "e5", "e11", "e3"):
        assert await store.get(eid) is None


async def test_growing_past_the_initial_capacity(store: MemoryVectorStore) -> None:
    vectors = {f"e{i}": unit(i) for i in range(600)}  # initial capacity is 256
    for eid, vec in vectors.items():
        await store.put(entry(eid, vector=vec))
    assert await store.count() == 600
    for eid in ("e0", "e255", "e256", "e599"):
        hits = await store.search("ns", vectors[eid], k=1)
        assert hits[0][0].entry_id == eid


async def test_reinserting_the_same_id_updates_in_place(store: MemoryVectorStore) -> None:
    await store.put(entry("a", vector=unit(1)))
    await store.put(entry("a", vector=unit(2)))
    assert await store.count() == 1
    hits = await store.search("ns", unit(2), k=1)
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)


async def test_eviction_drops_the_least_recently_used() -> None:
    store = MemoryVectorStore(make_settings(backend="memory", memory_max_entries=5))
    for i in range(5):
        await store.put(entry(f"e{i}", vector=unit(i)))
    await store.get("e0")  # touch the oldest so it is no longer the coldest
    await store.put(entry("e5", vector=unit(5)))
    assert await store.count() == 5
    assert await store.get("e0") is not None, "recently used entry was evicted"
    assert await store.get("e1") is None, "least recently used entry survived"


async def test_expired_entries_disappear(store: MemoryVectorStore) -> None:
    vector = unit(1)
    await store.put(entry("gone", vector=vector, ttl=1))
    await store.put(entry("stays", vector=unit(2), ttl=600))
    assert await store.ttl("gone") <= 1
    await asyncio.sleep(1.05)
    assert await store.get("gone") is None
    assert await store.count() == 1
    assert [e.entry_id for e, _ in await store.search("ns", vector, k=5)] == ["stays"]


async def test_ttl_matches_redis_conventions(store: MemoryVectorStore) -> None:
    await store.put(entry("with-ttl", vector=unit(1), ttl=60))
    await store.put(entry("forever", vector=unit(2), ttl=0))
    assert 0 < await store.ttl("with-ttl") <= 60
    assert await store.ttl("forever") == -1
    assert await store.ttl("never-existed") == -2


async def test_a_wrong_sized_vector_is_refused_not_crashed(store: MemoryVectorStore) -> None:
    await store.put(entry("bad", vector=np.ones(7, dtype=np.float32)))
    assert await store.count() == 0


async def test_invalidate_by_namespace_and_model(store: MemoryVectorStore) -> None:
    await store.put(entry("a", namespace="ns-a", vector=unit(1)))
    await store.put(entry("b", namespace="ns-b", vector=unit(2)))
    await store.put(entry("c", namespace="ns-b", vector=unit(3), model="other/model"))
    assert await store.invalidate(namespace="ns-b") == 2
    assert await store.count() == 1
    await store.put(entry("d", namespace="ns-c", vector=unit(4), model="other/model"))
    assert await store.invalidate(model="other/model") == 1
    assert await store.invalidate(drop_all=True) == 1
    assert await store.count() == 0


async def test_snapshot_survives_a_restart(tmp_path) -> None:
    path = tmp_path / "cache.npz"
    cfg = make_settings(backend="memory", memory_snapshot_path=str(path))
    vectors = {f"e{i}": unit(i) for i in range(5)}

    first = MemoryVectorStore(cfg)
    await first.connect()
    for eid, vec in vectors.items():
        await first.put(entry(eid, vector=vec))
    await first.close()
    assert path.exists()

    second = MemoryVectorStore(cfg)
    await second.connect()
    assert await second.count() == 5
    for eid in vectors:
        hits = await second.search("ns", vectors[eid], k=1)
        assert hits[0][0].entry_id == eid
        assert hits[0][0].response_text == f"answer {eid}"


async def test_snapshot_drops_entries_that_expired_while_away(tmp_path) -> None:
    path = tmp_path / "cache.npz"
    cfg = make_settings(backend="memory", memory_snapshot_path=str(path))
    first = MemoryVectorStore(cfg)
    await first.connect()
    await first.put(entry("short", vector=unit(1), ttl=1))
    await first.put(entry("long", vector=unit(2), ttl=600))
    await first.close()

    await asyncio.sleep(1.05)
    second = MemoryVectorStore(cfg)
    await second.connect()
    assert await second.count() == 1
    assert await second.get("short") is None
    assert await second.get("long") is not None


async def test_a_corrupt_snapshot_does_not_stop_startup(tmp_path) -> None:
    path = tmp_path / "cache.npz"
    path.write_bytes(b"this is not a numpy archive")
    store = MemoryVectorStore(make_settings(backend="memory", memory_snapshot_path=str(path)))
    await store.connect()
    assert await store.count() == 0


async def test_analytics_percentiles_and_near_misses() -> None:
    analytics = MemoryAnalytics(make_settings(backend="memory"))
    await analytics.bulk({"requests": 3, "hits": 2}, {"usd_saved": 0.25})
    await analytics.incr_float("usd_saved", 0.25)
    counters = await analytics.counters()
    assert counters["requests"] == 3
    assert counters["usd_saved"] == pytest.approx(0.5)

    for ms in (1.0, 2.0, 3.0, 100.0):
        await analytics.record_latency("hit", ms)
    pct = await analytics.latency_percentiles("hit")
    assert pct["count"] == 4
    assert pct["min"] == 1.0
    assert pct["max"] == 100.0
    assert await analytics.latency_percentiles("miss") == {}

    await analytics.reset()
    assert await analytics.counters() == {}


# ------------------------------------------------------------------ selection
async def test_auto_falls_back_to_memory_when_redis_is_absent(fake_provider) -> None:
    cfg = make_settings(backend="auto", redis_url="redis://127.0.0.1:6399/0")
    state = await build_state(cfg, providers=ProviderRegistry(cfg, {"fake": fake_provider}))
    try:
        assert state.backend == "memory"
        assert state.cache_available is True
        assert state.degraded_reason == ""
    finally:
        await shutdown_state(state)


async def test_asking_for_redis_explicitly_does_not_silently_fall_back(fake_provider) -> None:
    """`backend=redis` means redis. Quietly using memory would hide an outage."""
    cfg = make_settings(backend="redis", redis_url="redis://127.0.0.1:6399/0")
    state = await build_state(cfg, providers=ProviderRegistry(cfg, {"fake": fake_provider}))
    try:
        assert state.backend == "none"
        assert state.cache_available is False
        assert "ConnectionError" in state.degraded_reason or state.degraded_reason
    finally:
        await shutdown_state(state)


async def test_memory_backend_needs_no_redis_at_all(fake_provider) -> None:
    cfg = make_settings(backend="memory", redis_url="redis://127.0.0.1:6399/0")
    state = await build_state(cfg, providers=ProviderRegistry(cfg, {"fake": fake_provider}))
    try:
        assert state.backend == "memory"
        assert state.redis is None
        assert state.cache_available is True
    finally:
        await shutdown_state(state)
