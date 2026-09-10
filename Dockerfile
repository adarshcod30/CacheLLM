# syntax=docker/dockerfile:1
# Multi-stage: dependencies resolve once into a layer that only changes when the
# lockfile changes, so day-to-day code edits rebuild in seconds.
FROM python:3.11-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, without the project itself, for a cacheable layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.11-slim-bookworm AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 cachellm

WORKDIR /app
COPY --from=builder --chown=cachellm:cachellm /app /app
COPY --chown=cachellm:cachellm bench/ ./bench/
COPY --chown=cachellm:cachellm eval/ ./eval/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/cachellm/.cache/huggingface \
    CACHELLM_HOST=0.0.0.0 \
    CACHELLM_PORT=8080

USER cachellm

# Bake the embedding model into the image so a cold container does not spend
# its first request downloading 90 MB from Hugging Face.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8080/healthz || exit 1

CMD ["uvicorn", "cachellm.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
