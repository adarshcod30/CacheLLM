"""When a Redis answers but cannot hold the cache.

A distribution Redis without the search module, or a URL pointing at database
1, used to produce one info-level line quoting a cryptic FT.INFO error before
the proxy quietly moved to memory. Nobody starts a Redis server by accident,
so the reason is now a warning, and `cachellm stats` shows it too.

The stand-in client lets this run anywhere. The same paths were checked by hand
against a real Redis 8.10 started without its modules.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from cachellm import report
from cachellm.api.app import create_app
from cachellm.api.deps import _where, build_state, shutdown_state
from cachellm.providers.fake import FakeProvider
from cachellm.providers.registry import ProviderRegistry
from tests.conftest import make_settings


class StandInRedis:
    def __init__(self, db: int = 0, search: bool = True, reachable: bool = True) -> None:
        self.connection_pool = SimpleNamespace(connection_kwargs={"db": db})
        self.search = search
        self.reachable = reachable
        self.closed = False

    async def ping(self) -> bool:
        if not self.reachable:
            raise ConnectionError("Error 61 connecting to localhost:6379. Connection refused.")
        return True

    async def execute_command(self, *args):
        if not self.search:
            from redis.exceptions import ResponseError

            raise ResponseError("unknown command 'FT._LIST', with args beginning with: ")
        return []

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def redis_like(monkeypatch):
    def install(**kwargs) -> StandInRedis:
        client = StandInRedis(**kwargs)
        monkeypatch.setattr("cachellm.cache.redis_client.build_redis", lambda settings: client)
        return client

    return install


async def started(backend: str):
    cfg = make_settings(backend=backend)
    return await build_state(cfg, providers=ProviderRegistry(cfg, {"fake": FakeProvider()}))


async def test_auto_explains_a_redis_without_search_and_uses_memory(redis_like) -> None:
    client = redis_like(search=False)
    state = await started("auto")
    try:
        assert state.backend == "memory"
        assert state.cache_available
        assert "no search module" in state.backend_note
        assert "Homebrew" in state.backend_note
        assert client.closed, "the unusable client should not be left open"
    finally:
        await shutdown_state(state)


async def test_auto_explains_a_url_that_is_not_database_zero(redis_like) -> None:
    redis_like(db=3)
    state = await started("auto")
    try:
        assert state.backend == "memory"
        assert "database 3" in state.backend_note
        assert "/0" in state.backend_note
    finally:
        await shutdown_state(state)


async def test_explicit_redis_fails_open_with_the_same_plain_reason(redis_like) -> None:
    redis_like(search=False)
    state = await started("redis")
    try:
        assert not state.cache_available
        assert "no search module" in state.degraded_reason
        assert state.backend_note == ""
    finally:
        await shutdown_state(state)


async def test_no_redis_at_all_stays_quiet(redis_like) -> None:
    """The default install has no Redis. That is normal, not worth a note."""
    redis_like(reachable=False)
    state = await started("auto")
    try:
        assert state.backend == "memory"
        assert state.backend_note == ""
    finally:
        await shutdown_state(state)


async def test_the_note_reaches_the_stats_endpoint_and_the_terminal(redis_like) -> None:
    redis_like(search=False)
    state = await started("auto")
    app = create_app(state.settings, state=state)
    app.state.cachellm = state
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as http:
            stats = (await http.get("/admin/stats")).json()
        assert "no search module" in stats["backend_note"]
        assert "no search module" in report.summary(stats)
    finally:
        await shutdown_state(state)


def test_the_logged_address_never_includes_the_password() -> None:
    assert _where("redis://:hunter2@cache.internal:6380/0") == "cache.internal:6380"
    assert _where("rediss://user:hunter2@cache.internal/0") == "cache.internal:6379"
    assert "hunter2" not in _where("redis://:hunter2@cache.internal:6380/0")
