"""Request and response schemas mirroring the OpenAI chat completions contract.

Unknown fields are preserved rather than rejected: OpenAI adds parameters
regularly and a proxy that 422s on an unrecognised key is not a drop-in.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool", "developer", "function"]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Role
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    def as_text(self) -> str:
        """Flatten multi-part content down to plain text for hashing/embedding."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        parts: list[str] = []
        for part in self.content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in (None, "text") and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)

    def has_non_text_parts(self) -> bool:
        if isinstance(self.content, list):
            return any(
                isinstance(p, dict) and p.get("type") not in (None, "text") for p in self.content
            )
        return False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stream: bool | None = None
    stream_options: dict[str, Any] | None = None
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    user: str | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None

    # ------------------------------------------------------------------ helpers
    @property
    def effective_temperature(self) -> float:
        return 1.0 if self.temperature is None else float(self.temperature)

    @property
    def effective_max_tokens(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens

    def system_text(self) -> str:
        return "\n".join(
            m.as_text() for m in self.messages if m.role in ("system", "developer")
        ).strip()

    def conversation_messages(self) -> list[ChatMessage]:
        return [m for m in self.messages if m.role not in ("system", "developer")]

    def last_user_text(self) -> str:
        for message in reversed(self.messages):
            if message.role == "user":
                return message.as_text().strip()
        return ""

    def user_turn_count(self) -> int:
        return sum(1 for m in self.messages if m.role == "user")

    def cache_text(self, include_history: bool) -> str:
        """The text that gets embedded and hashed."""
        if not include_history:
            return self.last_user_text()
        turns = [f"{m.role}: {m.as_text().strip()}" for m in self.conversation_messages()]
        return "\n".join(t for t in turns if t.strip())


class Usage(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChoiceMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class Choice(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int = 0
    message: ChoiceMessage
    finish_reason: str | None = "stop"
    logprobs: None = None


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)
    system_fingerprint: str | None = None

    @classmethod
    def from_text(
        cls,
        *,
        model: str,
        text: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        finish_reason: str = "stop",
    ) -> ChatCompletionResponse:
        return cls(
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(role="assistant", content=text),
                    finish_reason=finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    def text(self) -> str:
        if not self.choices:
            return ""
        return self.choices[0].message.content or ""


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "cachellm"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]
