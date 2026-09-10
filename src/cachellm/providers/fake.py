"""Deterministic in-process provider.

Every test, the whole CI pipeline and the reproducible benchmark run against
this. It costs nothing, never rate-limits, and returns byte-identical answers
on every machine, which is what makes the published hit-rate numbers something
a reader can re-run rather than take on trust.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator

from cachellm.models import ChatCompletionRequest
from cachellm.providers.base import Provider, ProviderResult, StreamEvent


class FakeProvider(Provider):
    name = "fake"

    def __init__(self, latency_ms: float = 0.0, answers: dict[str, str] | None = None) -> None:
        self.latency_ms = latency_ms
        self.answers = answers or {}
        self.calls = 0

    def _answer(self, prompt: str) -> str:
        if prompt in self.answers:
            return self.answers[prompt]
        digest = hashlib.blake2b(prompt.strip().lower().encode(), digest_size=6).hexdigest()
        return f"[fake] answer to {prompt.strip()[:80]!r} (id {digest})"

    async def complete(self, request: ChatCompletionRequest) -> ProviderResult:
        self.calls += 1
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000.0)
        prompt = request.last_user_text()
        text = self._answer(prompt)
        return ProviderResult(
            text=text,
            model=self.resolve_model(request.model),
            prompt_tokens=self.approx_tokens(prompt),
            completion_tokens=self.approx_tokens(text),
            finish_reason="stop",
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamEvent]:
        result = await self.complete(request)
        words = result.text.split(" ")
        for i, word in enumerate(words):
            yield StreamEvent(delta=word if i == 0 else f" {word}")
        yield StreamEvent(
            finish_reason="stop",
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )
