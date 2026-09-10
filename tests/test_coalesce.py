"""Cache stampede protection."""

from __future__ import annotations

import asyncio

import pytest

from cachellm.cache.coalesce import SingleFlight


async def test_identical_concurrent_calls_run_the_factory_once() -> None:
    flight = SingleFlight()
    calls = 0

    async def slow() -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return "answer"

    results = await asyncio.gather(*(flight.do("same-key", slow) for _ in range(10)))
    assert calls == 1
    assert all(value == "answer" for value, _ in results)
    assert sum(1 for _, joined in results if joined) == 9
    assert flight.in_flight == 0


async def test_different_keys_do_not_share() -> None:
    flight = SingleFlight()
    calls = 0

    async def work() -> int:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return calls

    await asyncio.gather(flight.do("a", work), flight.do("b", work))
    assert calls == 2


async def test_failure_propagates_to_every_waiter_and_clears_state() -> None:
    flight = SingleFlight()

    async def boom() -> str:
        await asyncio.sleep(0.02)
        raise RuntimeError("upstream down")

    results = await asyncio.gather(
        *(flight.do("key", boom) for _ in range(5)), return_exceptions=True
    )
    assert len(results) == 5
    assert all(isinstance(r, RuntimeError) for r in results)
    assert flight.in_flight == 0

    async def ok() -> str:
        return "recovered"

    value, joined = await flight.do("key", ok)
    assert value == "recovered"
    assert joined is False


@pytest.mark.parametrize("concurrency", [2, 25])
async def test_scales_to_many_waiters(concurrency: int) -> None:
    flight = SingleFlight()
    calls = 0

    async def work() -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.02)
        return "x"

    await asyncio.gather(*(flight.do("k", work) for _ in range(concurrency)))
    assert calls == 1
