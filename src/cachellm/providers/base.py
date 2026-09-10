"""Provider interface.

Everything the cache needs from an upstream model lives behind two methods.
Keeping this surface tiny is what makes the cache provider-agnostic: a cached
Bedrock answer never gets served to an OpenAI request (provider is part of the
namespace), but the caching logic itself is written once.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from cachellm.models import ChatCompletionRequest

# Upstream "finished cleanly" markers, normalised to the OpenAI vocabulary.
FINISH_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "stop": "stop",
    "max_tokens": "length",
    "length": "length",
    "content_filtered": "content_filter",
    "content_filter": "content_filter",
    "tool_use": "tool_calls",
    "tool_calls": "tool_calls",
    "guardrail_intervened": "content_filter",
}


def normalise_finish_reason(raw: str | None) -> str:
    if not raw:
        return "stop"
    return FINISH_REASON_MAP.get(str(raw).lower(), str(raw).lower())


@dataclass
class ProviderResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    tool_calls: list[dict[str, Any]] | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamEvent:
    """One step of a streamed completion."""

    delta: str = ""
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Provider(abc.ABC):
    name: str

    @abc.abstractmethod
    async def complete(self, request: ChatCompletionRequest) -> ProviderResult: ...

    @abc.abstractmethod
    def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamEvent]: ...

    def resolve_model(self, model: str) -> str:
        """Strip the routing prefix, if any, to get the upstream model id."""
        return model.split("/", 1)[1] if "/" in model else model

    async def close(self) -> None:  # pragma: no cover - default no-op
        return None

    @staticmethod
    def approx_tokens(text: str) -> int:
        """Rough token count for providers that do not report usage.

        Four characters per token is the widely used English approximation. It
        is only used when the upstream is silent, and it is flagged as an
        estimate wherever it reaches a number the user sees.
        """
        return max(1, len(text) // 4)
