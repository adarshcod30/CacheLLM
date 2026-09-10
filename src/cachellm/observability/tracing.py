"""Optional OpenTelemetry tracing.

OTel is the spine, not the destination. Instrument once and the same spans can
go to Langfuse (which reads them as LLM generations, with cost and prompts),
Grafana Tempo, Jaeger, or anything else that accepts OTLP. Nothing here is
required for the proxy to run: with tracing disabled every helper degrades to a
no-op context manager.

Span attributes follow the OpenTelemetry GenAI semantic conventions where they
exist (``gen_ai.*``), with CacheLLM's own facts under ``cachellm.*``.
"""

from __future__ import annotations

import base64
import contextlib
import os
from collections.abc import Iterator
from typing import Any

import structlog

from cachellm.settings import Settings

log = structlog.get_logger(__name__)

_tracer: Any = None
_enabled = False


def langfuse_otlp_headers(public_key: str, secret_key: str) -> str:
    """Langfuse authenticates OTLP with HTTP basic auth over the key pair."""
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    return f"Authorization=Basic {token}"


def setup_tracing(settings: Settings, app: Any = None) -> bool:
    """Configure the global tracer. Returns True when tracing is live."""
    global _tracer, _enabled
    if not settings.tracing_enabled:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        log.warning(
            "tracing_unavailable",
            hint="install the observability extra: uv sync --extra observability",
        )
        return False

    endpoint = settings.otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if not endpoint:
        log.warning("tracing_disabled_no_endpoint")
        return False

    provider = TracerProvider(resource=Resource.create({"service.name": settings.service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("cachellm")
    _enabled = True

    for module, instrument in (
        ("opentelemetry.instrumentation.redis", "RedisInstrumentor"),
        ("opentelemetry.instrumentation.httpx", "HTTPXClientInstrumentor"),
        ("opentelemetry.instrumentation.botocore", "BotocoreInstrumentor"),
    ):
        with contextlib.suppress(Exception):
            mod = __import__(module, fromlist=[instrument])
            getattr(mod, instrument)().instrument()

    if app is not None:
        with contextlib.suppress(Exception):
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz,readyz,metrics")

    log.info("tracing_enabled", endpoint=endpoint, service=settings.service_name)
    return True


@contextlib.contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Start a span, or do nothing at all when tracing is off."""
    if not _enabled or _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        yield current


def set_attributes(current: Any, **attributes: Any) -> None:
    if current is None:
        return
    for key, value in attributes.items():
        if value is not None:
            current.set_attribute(key, value)


def is_enabled() -> bool:
    return _enabled
