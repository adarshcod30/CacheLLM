"""Any OpenAI-compatible upstream: OpenAI itself, Groq, Together, OpenRouter,
vLLM, LM Studio, llama.cpp server.

One adapter covers all of them because they all speak the same
``POST /chat/completions``. That is the whole reason the OpenAI shape became
the industry contract, and it is why CacheLLM speaks it on the front door too.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlparse

import httpx
import structlog

from cachellm.errors import UpstreamError
from cachellm.models import ChatCompletionRequest
from cachellm.providers.base import (
    Provider,
    ProviderResult,
    StreamEvent,
    normalise_finish_reason,
)
from cachellm.settings import Settings

log = structlog.get_logger(__name__)


class OpenAICompatProvider(Provider):
    name = "openai"

    def __init__(
        self,
        settings: Settings,
        base_url: str | None = None,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._base_url = (base_url or settings.openai_base_url).rstrip("/")
        self._api_key = api_key if api_key is not None else settings.openai_api_key
        # Injectable so tests exercise the real client construction, including
        # the auth header, against a fake upstream rather than the network.
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._is_openai_itself = urlparse(self._base_url).hostname == "api.openai.com"

    def resolve_model(self, model: str) -> str:
        """Drop our `openai/` routing prefix only when the upstream is OpenAI.

        OpenAI's own API wants `gpt-4o-mini`. Groq, OpenRouter and Together all
        name OpenAI's models `openai/gpt-oss-120b`, so to them the prefix is part
        of the model id. Stripping it there turned Groq's two main chat models
        into 404s, which only showed up against the live API.
        """
        if model.startswith("openai/") and not self._is_openai_itself:
            return model
        return super().resolve_model(model)

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=httpx.Timeout(self._settings.request_timeout, connect=10.0),
                transport=self._transport,
            )
        return self._client

    def _payload(self, request: ChatCompletionRequest, stream: bool) -> dict[str, Any]:
        body = request.model_dump(exclude_none=True, by_alias=True)
        body["model"] = self.resolve_model(request.model)
        body["stream"] = stream
        body.pop("stream_options", None)
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    async def complete(self, request: ChatCompletionRequest) -> ProviderResult:
        try:
            response = await self._http().post(
                "/chat/completions", json=self._payload(request, False)
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:400]
            log.warning("openai_http_error", status=exc.response.status_code, detail=detail)
            raise UpstreamError(
                f"Upstream returned {exc.response.status_code}: {detail}",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(f"Upstream request failed: {exc}") from exc

        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message", {}) or {}
        usage = data.get("usage", {}) or {}
        return ProviderResult(
            text=message.get("content") or "",
            model=data.get("model", self.resolve_model(request.model)),
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            finish_reason=normalise_finish_reason(choice.get("finish_reason")),
            tool_calls=message.get("tool_calls"),
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamEvent]:
        payload = self._payload(request, True)
        try:
            async with self._http().stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")[:400]
                    raise UpstreamError(
                        f"Upstream returned {response.status_code}: {body}",
                        status_code=response.status_code,
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    usage = chunk.get("usage") or {}
                    if usage:
                        yield StreamEvent(
                            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                        )
                    for choice in chunk.get("choices", []) or []:
                        delta = (choice.get("delta") or {}).get("content") or ""
                        finish = choice.get("finish_reason")
                        if delta:
                            yield StreamEvent(delta=delta)
                        if finish:
                            yield StreamEvent(finish_reason=normalise_finish_reason(finish))
        except httpx.HTTPError as exc:
            raise UpstreamError(f"Upstream stream failed: {exc}") from exc

    async def list_models(self) -> list[str]:
        """The model ids this host serves right now, from its GET /models."""
        response = await self._http().get("/models")
        response.raise_for_status()
        body = response.json()
        items = body.get("data", []) if isinstance(body, dict) else body
        return [str(item["id"]) for item in items if isinstance(item, dict) and item.get("id")]

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
