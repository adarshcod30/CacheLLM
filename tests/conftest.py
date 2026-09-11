from __future__ import annotations

import contextlib
import os
import uuid
from collections.abc import AsyncIterator, Iterator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from cachellm.api.app import create_app
from cachellm.api.deps import build_state, shutdown_state
from cachellm.observability.metrics import reset_metrics
from cachellm.providers.fake import FakeProvider
from cachellm.providers.registry import ProviderRegistry
from cachellm.settings import Settings

# Redis Search only indexes database 0, so tests cannot hide in db 15. They are
# isolated instead by a unique index name and unique key prefixes per run, and
# every fixture deletes exactly what it created.
TEST_REDIS_URL = os.getenv("CACHELLM_TEST_REDIS_URL", "redis://localhost:6379/0")


def make_settings(**overrides) -> Settings:
    token = uuid.uuid4().hex[:8]
    base = {
        "backend": "memory",
        "_env_file": None,
        "embedding_backend": "hash",
        "embedding_dim": 384,
        "redis_url": TEST_REDIS_URL,
        "index_name": f"test_idx_{token}",
        "entry_prefix": f"test:{token}:e:",
        "exact_prefix": f"test:{token}:x:",
        "stats_prefix": f"test:{token}:s:",
        "default_provider": "fake",
        "api_keys": "",
        "log_json": True,
        "log_level": "WARNING",
        "metrics_enabled": True,
        # Tests never reach out to real hosts for their model lists.
        "discover_models": False,
    }
    base.update(overrides)
    return Settings(**base)


def purge(settings: Settings) -> None:
    """Remove everything a test created, using a synchronous client.

    Cleanup deliberately does not share the application's async Redis client:
    the app may own a different event loop, and reaching across loops is how
    test suites start failing for reasons that have nothing to do with the code
    under test.
    """
    try:
        import redis

        client = redis.from_url(settings.redis_url, socket_connect_timeout=2)
        keys: list[bytes] = []
        for prefix in (settings.entry_prefix, settings.exact_prefix, settings.stats_prefix):
            keys.extend(client.scan_iter(match=f"{prefix}*", count=500))
        if keys:
            client.delete(*keys)
        with contextlib.suppress(Exception):
            client.execute_command("FT.DROPINDEX", settings.index_name)
        client.close()
    except Exception:  # cleanup must never fail a test run
        pass


def redis_available(url: str) -> bool:
    try:
        import redis

        client = redis.from_url(url, socket_connect_timeout=1)
        client.ping()
        client.close()
        return True
    except Exception:
        return False


requires_redis = pytest.mark.skipif(
    not redis_available(TEST_REDIS_URL), reason=f"no Redis at {TEST_REDIS_URL}"
)


@pytest.fixture(params=["memory", "redis"])
def backend(request) -> str:
    """Run the storage-facing tests against both backends.

    The point of the in-memory store is that it is not a lesser mode: the same
    suite has to pass against it and against Redis, or "no server required" is
    a claim rather than a fact.
    """
    if request.param == "redis" and not redis_available(TEST_REDIS_URL):
        pytest.skip(f"no Redis at {TEST_REDIS_URL}")
    return str(request.param)


@pytest.fixture
def settings(backend: str) -> Iterator[Settings]:
    cfg = make_settings(backend=backend)
    yield cfg
    purge(cfg)


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()


# --------------------------------------------------------------------- async
@pytest_asyncio.fixture
async def state(backend: str, fake_provider: FakeProvider) -> AsyncIterator:
    """Application state built on the running test loop, for direct-call tests."""
    reset_metrics()
    cfg = make_settings(backend=backend)
    registry = ProviderRegistry(cfg, providers={"fake": fake_provider})
    app_state = await build_state(cfg, providers=registry)
    try:
        yield app_state
    finally:
        await shutdown_state(app_state)
        purge(cfg)


@pytest_asyncio.fixture
async def asgi():
    """Factory for an httpx client speaking to the app on this same loop."""
    import httpx

    created: list = []

    async def factory(**overrides):
        reset_metrics()
        overrides.setdefault("backend", "memory")
        cfg = make_settings(**overrides)
        app_state = await build_state(cfg, providers=ProviderRegistry(cfg))
        app = create_app(cfg, state=app_state)
        app.state.cachellm = app_state
        transport = httpx.ASGITransport(app=app)
        http = httpx.AsyncClient(transport=transport, base_url="http://testserver")
        created.append((http, app_state, cfg))
        return http, app

    try:
        yield factory
    finally:
        for http, app_state, cfg in created:
            await http.aclose()
            await shutdown_state(app_state)
            purge(cfg)


# ---------------------------------------------------------------------- sync
@pytest.fixture
def client_factory(backend: str) -> Iterator:
    """Factory for TestClients. The app builds its own state inside its own loop."""
    created: list[tuple[TestClient, Settings]] = []

    def factory(**overrides) -> TestClient:
        reset_metrics()
        overrides.setdefault("backend", backend)
        cfg = make_settings(**overrides)
        test_client = TestClient(create_app(cfg))
        test_client.__enter__()
        created.append((test_client, cfg))
        return test_client

    try:
        yield factory
    finally:
        for test_client, cfg in created:
            with contextlib.suppress(Exception):
                test_client.__exit__(None, None, None)
            purge(cfg)


@pytest.fixture
def client(client_factory) -> TestClient:
    return client_factory()
