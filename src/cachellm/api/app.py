"""FastAPI application assembly."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from cachellm import __version__
from cachellm.api import routes_admin, routes_chat
from cachellm.api.deps import AppState, build_state, shutdown_state
from cachellm.errors import CacheLLMError
from cachellm.logging_setup import configure_logging
from cachellm.observability.tracing import setup_tracing
from cachellm.settings import Settings, get_settings

log = structlog.get_logger(__name__)

DESCRIPTION = """
A drop-in semantic cache for OpenAI-compatible LLM APIs.

Point your client's `base_url` at this service and repeated or reworded
questions are served from cache instead of the model. Every response carries
`X-Cache` headers explaining what happened.
""".strip()


def create_app(settings: Settings | None = None, state: AppState | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.cachellm = state or await build_state(settings)
        current: AppState = app.state.cachellm
        if not current.settings.auth_enabled:
            log.warning("auth_disabled", hint="set CACHELLM_API_KEYS before exposing this proxy")
        if current.settings.shadow_mode:
            log.warning("shadow_mode_on", hint="cache hits are logged, never served")
        if not current.settings.is_calibrated and current.settings.threshold_default <= 0:
            log.warning(
                "threshold_not_calibrated",
                embedding_model=current.settings.embedding_model,
                using=current.settings.calibrated_threshold,
                hint="measured safe thresholds span 0.89-0.98 across models; run "
                "`python -m bench.compare_models` on your own data before trusting this",
            )
        log.info(
            "cachellm_started",
            version=__version__,
            cache_available=current.cache_available,
            embedding_model=current.embedder.name,
            default_provider=current.settings.default_provider,
        )
        try:
            yield
        finally:
            if state is None:
                await shutdown_state(current)

    app = FastAPI(
        title="CacheLLM",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    setup_tracing(settings, app)

    app.include_router(routes_chat.router, prefix="/v1", tags=["openai"])
    app.include_router(routes_admin.router, prefix="/admin", tags=["admin"])

    # --------------------------------------------------------------- errors
    @app.exception_handler(CacheLLMError)
    async def _cachellm_error(_request: Request, exc: CacheLLMError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.envelope())

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(p) for p in first.get("loc", ())[1:]) or None
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": first.get("msg", "Invalid request."),
                    "type": "invalid_request_error",
                    "param": location,
                    "code": None,
                }
            },
        )

    # --------------------------------------------------------------- health
    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz", tags=["ops"])
    async def readyz(request: Request) -> JSONResponse:
        current: AppState = request.app.state.cachellm
        ready = current.cache_available
        body = {
            "status": "ready" if ready else "degraded",
            "cache_available": current.cache_available,
            "degraded_reason": current.degraded_reason or None,
            "embedding_model": current.embedder.name,
        }
        # Degraded still returns 200: the proxy is serving, just without cache.
        return JSONResponse(body, status_code=200)

    @app.get("/metrics", tags=["ops"])
    async def metrics(request: Request) -> Response:
        current: AppState = request.app.state.cachellm
        if not current.settings.metrics_enabled:
            return Response(status_code=404)
        if current.vectors is not None:
            with contextlib.suppress(Exception):
                current.metrics.entries.set(await current.vectors.count())
        body, content_type = current.metrics.render()
        return Response(content=body, media_type=content_type)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, Any]:
        return {
            "name": "CacheLLM",
            "version": __version__,
            "docs": "/docs",
            "openai_base_url": "/v1",
            "endpoints": ["/v1/chat/completions", "/v1/models", "/admin/stats", "/metrics"],
        }

    return app


app = create_app  # uvicorn factory entry point
