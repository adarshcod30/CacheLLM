"""AWS Bedrock provider using the Converse API.

Converse is the reason this project needs no Anthropic or Meta key: one request
shape reaches Nova, Claude, Llama and Mistral alike. The interesting work here
is the translation, because Bedrock's contract differs from OpenAI's in three
ways that bite in production:

* system prompts are a separate top-level field, not a message;
* messages must strictly alternate user/assistant and must start with user;
* boto3 is synchronous, so every call is pushed to a worker thread and the
  streaming iterator is bridged onto the event loop by hand.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import AsyncIterator
from typing import Any

import structlog

from cachellm.errors import UpstreamError
from cachellm.models import ChatCompletionRequest, ChatMessage
from cachellm.providers.base import (
    Provider,
    ProviderResult,
    StreamEvent,
    normalise_finish_reason,
)
from cachellm.settings import Settings

log = structlog.get_logger(__name__)

BEDROCK_VENDOR_PREFIXES = (
    "amazon.",
    "anthropic.",
    "meta.",
    "mistral.",
    "cohere.",
    "ai21.",
    "deepseek.",
    "qwen.",
    "openai.",
    "google.",
    "moonshot",
    "minimax.",
    "nvidia.",
    "zai.",
    "writer.",
)


def _explain(exc: Exception) -> str:
    """Turn boto's less obvious failures into something actionable.

    The credential one is worth special-casing: the AWS CLI's `aws login` flow
    stores credentials through a provider that needs the optional CRT extra,
    and boto's own message names the package without naming the install.
    """
    text = str(exc)
    if "botocore[crt]" in text or "Missing Dependency" in text:
        return (
            "Bedrock call failed: your AWS credentials come from `aws login`, which needs "
            "the optional CRT extra. Install it with `uv sync --extra aws` "
            "(or `pip install 'botocore[crt]'`), then restart the proxy. "
            "Static keys and SSO profiles do not need this."
        )
    if "ExpiredToken" in text or "security token included in the request is expired" in text:
        return "Bedrock call failed: AWS credentials have expired. Run `aws login` again."
    if "AccessDeniedException" in text:
        return (
            f"Bedrock call failed: access denied. Check the model is enabled in this region "
            f"and the identity has bedrock:InvokeModel. Original: {text[:200]}"
        )
    if "ThrottlingException" in text or "TooManyRequests" in text:
        return f"Bedrock call failed: throttled by AWS. Lower concurrency and retry. {text[:160]}"
    if "ValidationException" in text:
        return f"Bedrock rejected the request: {text[:280]}"
    return f"Bedrock call failed: {text[:300]}"


def looks_like_bedrock_model(model: str) -> bool:
    candidate = model.split("/", 1)[1] if model.startswith("bedrock/") else model
    for region in ("us.", "eu.", "apac.", "ap."):
        if candidate.startswith(region):
            candidate = candidate[len(region) :]
            break
    return candidate.startswith(BEDROCK_VENDOR_PREFIXES)


def to_converse_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Convert OpenAI messages into Converse messages, merging same-role runs."""
    converted: list[dict[str, Any]] = []
    for message in messages:
        if message.role in ("system", "developer"):
            continue
        role = "assistant" if message.role == "assistant" else "user"
        text = message.as_text()
        if not text.strip():
            continue
        if converted and converted[-1]["role"] == role:
            converted[-1]["content"].append({"text": text})
        else:
            converted.append({"role": role, "content": [{"text": text}]})
    # Converse rejects a conversation that opens with the assistant.
    while converted and converted[0]["role"] != "user":
        converted.pop(0)
    return converted


def build_converse_kwargs(request: ChatCompletionRequest, model_id: str) -> dict[str, Any]:
    inference: dict[str, Any] = {}
    if request.effective_max_tokens:
        inference["maxTokens"] = int(request.effective_max_tokens)
    if request.temperature is not None:
        inference["temperature"] = float(request.temperature)
    if request.top_p is not None:
        inference["topP"] = float(request.top_p)
    if request.stop:
        stops = [request.stop] if isinstance(request.stop, str) else list(request.stop)
        inference["stopSequences"] = stops[:4]

    kwargs: dict[str, Any] = {
        "modelId": model_id,
        "messages": to_converse_messages(request.messages),
    }
    system_text = request.system_text()
    if system_text:
        kwargs["system"] = [{"text": system_text}]
    if inference:
        kwargs["inferenceConfig"] = inference
    return kwargs


class BedrockProvider(Provider):
    name = "bedrock"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = None
        self._lock = threading.Lock()

    def _get_client(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    try:
                        import boto3
                        from botocore.config import Config
                    except ImportError as exc:
                        raise UpstreamError(
                            "Routing to Bedrock needs the optional AWS extra. Install it "
                            "with `pip install 'cachellm-proxy[aws]'`, or point "
                            "CACHELLM_DEFAULT_PROVIDER at an OpenAI-compatible endpoint "
                            "instead, which needs nothing extra."
                        ) from exc

                    # Passing a region overrides the profile's own region, so
                    # only pass one somebody chose. boto3 reads
                    # AWS_DEFAULT_REGION but not AWS_REGION, which the AWS CLI
                    # and the docs use, so that one is read here. Hardcoding
                    # us-east-1 once sent every user there.
                    profile = self._settings.aws_profile or None
                    region = self._settings.aws_region or os.environ.get("AWS_REGION") or None
                    session = boto3.Session(region_name=region, profile_name=profile)
                    if not session.region_name:
                        session = boto3.Session(region_name="us-east-1", profile_name=profile)
                    self._client = session.client(
                        "bedrock-runtime",
                        config=Config(
                            retries={
                                "max_attempts": self._settings.provider_max_retries,
                                "mode": "standard",
                            },
                            read_timeout=int(self._settings.request_timeout),
                            connect_timeout=10,
                        ),
                    )
        return self._client

    # ---------------------------------------------------------------- complete
    async def complete(self, request: ChatCompletionRequest) -> ProviderResult:
        model_id = self.resolve_model(request.model)
        kwargs = build_converse_kwargs(request, model_id)
        try:
            response = await asyncio.to_thread(lambda: self._get_client().converse(**kwargs))
        except UpstreamError:
            raise  # already carries an actionable message; do not re-wrap it
        except Exception as exc:  # boto raises many distinct client errors
            log.warning("bedrock_error", model=model_id, error=str(exc)[:300])
            raise UpstreamError(_explain(exc)) from exc

        message = response.get("output", {}).get("message", {})
        text = "".join(
            block.get("text", "") for block in message.get("content", []) if "text" in block
        )
        usage = response.get("usage", {})
        return ProviderResult(
            text=text,
            model=model_id,
            prompt_tokens=int(usage.get("inputTokens", 0) or 0),
            completion_tokens=int(usage.get("outputTokens", 0) or 0),
            finish_reason=normalise_finish_reason(response.get("stopReason")),
            raw={"metrics": response.get("metrics", {})},
        )

    # ------------------------------------------------------------------ stream
    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamEvent]:
        model_id = self.resolve_model(request.model)
        kwargs = build_converse_kwargs(request, model_id)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[StreamEvent | Exception | None] = asyncio.Queue()

        def pump() -> None:
            """Consume the synchronous boto EventStream on a worker thread."""
            try:
                response = self._get_client().converse_stream(**kwargs)
                for event in response.get("stream", []):
                    if "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"].get("delta", {}).get("text", "")
                        if delta:
                            loop.call_soon_threadsafe(queue.put_nowait, StreamEvent(delta=delta))
                    elif "messageStop" in event:
                        reason = normalise_finish_reason(event["messageStop"].get("stopReason"))
                        loop.call_soon_threadsafe(
                            queue.put_nowait, StreamEvent(finish_reason=reason)
                        )
                    elif "metadata" in event:
                        usage = event["metadata"].get("usage", {})
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            StreamEvent(
                                prompt_tokens=int(usage.get("inputTokens", 0) or 0),
                                completion_tokens=int(usage.get("outputTokens", 0) or 0),
                            ),
                        )
            except Exception as exc:  # surfaced to the consumer below
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        task = asyncio.create_task(asyncio.to_thread(pump))
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    log.warning("bedrock_stream_error", model=model_id, error=str(item)[:300])
                    raise UpstreamError(_explain(item)) from item
                yield item
        finally:
            task.cancel()
