"""The record stored in Redis for one cached answer."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class CacheEntry:
    entry_id: str
    namespace: str
    category: str
    model: str
    provider: str
    prompt: str
    response_text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    created_at: float = field(default_factory=time.time)
    ttl: int = 86_400
    hits: int = 0
    exact_hash: str = ""
    embedding: np.ndarray | None = None
    tool_calls: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------- serialisation
    def to_redis_mapping(self) -> dict[str, str | bytes | int | float]:
        """Flatten to a Redis hash. The vector goes in as raw float32 bytes."""
        mapping: dict[str, str | bytes | int | float] = {
            "entry_id": self.entry_id,
            "namespace": self.namespace,
            "category": self.category,
            "model": self.model,
            "provider": self.provider,
            "prompt": self.prompt,
            "response_text": self.response_text,
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "finish_reason": self.finish_reason,
            "created_at": float(self.created_at),
            "ttl": int(self.ttl),
            "hits": int(self.hits),
            "exact_hash": self.exact_hash,
            "tool_calls": json.dumps(self.tool_calls) if self.tool_calls else "",
        }
        if self.embedding is not None:
            mapping["embedding"] = np.asarray(self.embedding, dtype=np.float32).tobytes()
        return mapping

    @staticmethod
    def _s(raw: dict[Any, Any], key: str, default: str = "") -> str:
        value = raw.get(key, raw.get(key.encode(), default))
        if isinstance(value, bytes):
            return value.decode("utf-8", "replace")
        return str(value) if value is not None else default

    @classmethod
    def from_redis_mapping(cls, raw: dict[Any, Any]) -> CacheEntry:
        get = cls._s
        tool_calls_raw = get(raw, "tool_calls")
        return cls(
            entry_id=get(raw, "entry_id"),
            namespace=get(raw, "namespace"),
            category=get(raw, "category", "default"),
            model=get(raw, "model"),
            provider=get(raw, "provider"),
            prompt=get(raw, "prompt"),
            response_text=get(raw, "response_text"),
            prompt_tokens=int(float(get(raw, "prompt_tokens", "0") or 0)),
            completion_tokens=int(float(get(raw, "completion_tokens", "0") or 0)),
            finish_reason=get(raw, "finish_reason", "stop"),
            created_at=float(get(raw, "created_at", "0") or 0),
            ttl=int(float(get(raw, "ttl", "0") or 0)),
            hits=int(float(get(raw, "hits", "0") or 0)),
            exact_hash=get(raw, "exact_hash"),
            tool_calls=json.loads(tool_calls_raw) if tool_calls_raw else None,
        )

    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.created_at)

    def summary(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("embedding", None)
        data["age_seconds"] = round(self.age_seconds(), 1)
        return data
