"""Single-flight: collapse identical in-flight misses into one upstream call.

Ten users asking the same brand-new question at the same moment is a cache
stampede: every one of them misses, and every one of them pays for a full
generation. The first caller here does the work and the rest await its result.
On a cold cache under real concurrency this is worth more than a few points of
similarity threshold.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")


class SingleFlight:
    def __init__(self) -> None:
        self._inflight: dict[str, asyncio.Future[Any]] = {}
        self._lock = asyncio.Lock()
        self.coalesced = 0

    async def do(self, key: str, factory: Callable[[], Awaitable[T]]) -> tuple[T, bool]:
        """Run ``factory`` for ``key``, or await the run already in progress.

        Returns ``(result, was_coalesced)``.
        """
        async with self._lock:
            existing = self._inflight.get(key)
            if existing is not None:
                self.coalesced += 1
                waiter = existing
                joined = True
            else:
                waiter = asyncio.get_running_loop().create_future()
                self._inflight[key] = waiter
                joined = False

        if joined:
            return await asyncio.shield(waiter), True

        try:
            result = await factory()
        except BaseException as exc:  # propagate to every waiter, then re-raise
            async with self._lock:
                self._inflight.pop(key, None)
            if not waiter.done():
                waiter.set_exception(exc)
            # Keep the future's exception from being reported as unretrieved.
            waiter.exception()
            raise
        else:
            async with self._lock:
                self._inflight.pop(key, None)
            if not waiter.done():
                waiter.set_result(result)
            return result, False

    @property
    def in_flight(self) -> int:
        return len(self._inflight)
