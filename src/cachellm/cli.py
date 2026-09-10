"""Command line interface."""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Annotated

import typer

from cachellm import __version__, report
from cachellm.settings import get_settings

app = typer.Typer(
    name="cachellm",
    help="A drop-in semantic cache for OpenAI-compatible LLM APIs.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "",
    port: Annotated[int, typer.Option(help="Bind port.")] = 0,
    reload: Annotated[bool, typer.Option(help="Auto-reload on code changes.")] = False,
) -> None:
    """Run the proxy."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "cachellm.api.app:create_app",
        factory=True,
        host=host or settings.host,
        port=port or settings.port,
        reload=reload,
        log_config=None,
    )


@app.command()
def config() -> None:
    """Show the effective configuration."""
    settings = get_settings()
    data = settings.model_dump()
    data["api_keys"] = f"<{len(settings.client_keys)} key(s) configured>"
    data["openai_api_key"] = "<set>" if settings.openai_api_key else "<unset>"
    typer.echo(json.dumps(data, indent=2, default=str))


@app.command()
def stats(
    limit: Annotated[int, typer.Option(help="How many recent requests to show.")] = 15,
    json_out: Annotated[bool, typer.Option("--json", help="Raw JSON instead.")] = False,
) -> None:
    """Show what the cache has been doing: hit rate, savings, latency, recent requests."""

    async def run() -> None:
        from cachellm.api.deps import build_state, shutdown_state

        settings = get_settings()
        state = await build_state(settings)
        try:
            if state.cache is None or state.analytics is None:
                typer.secho(f"cache unavailable: {state.degraded_reason}", fg="red")
                raise typer.Exit(1)
            data = await state.cache.stats()
            data["backend"] = state.backend
            data["cache_available"] = True
            data["caching_enabled"] = settings.enabled
            recent = await state.analytics.recent_requests(limit=limit)
            if json_out:
                typer.echo(json.dumps({"stats": data, "recent": recent}, indent=2))
                return
            typer.echo(report.summary(data, settings.embedding_model))
            typer.echo(report.request_table(recent, limit=limit))
        finally:
            await shutdown_state(state)

    asyncio.run(run())


@app.command()
def watch(
    interval: Annotated[float, typer.Option(help="Seconds between refreshes.")] = 2.0,
) -> None:
    """Follow requests as they happen, like `tail -f` for the cache."""

    async def run() -> None:
        from cachellm.api.deps import build_state, shutdown_state

        settings = get_settings()
        state = await build_state(settings)
        if state.cache is None or state.analytics is None:
            typer.secho(f"cache unavailable: {state.degraded_reason}", fg="red")
            raise typer.Exit(1)
        seen: set[tuple[float, str]] = set()
        typer.echo(f"watching {state.backend} backend, ctrl-c to stop\n")
        try:
            while True:
                for record in reversed(await state.analytics.recent_requests(limit=50)):
                    marker = (record.get("at", 0.0), record.get("prompt", ""))
                    if marker in seen:
                        continue
                    seen.add(marker)
                    typer.echo(report.request_line(record))
                if len(seen) > 5_000:
                    seen.clear()
                await asyncio.sleep(interval)
        except (KeyboardInterrupt, asyncio.CancelledError):
            data = await state.cache.stats()
            data["backend"] = state.backend
            data["cache_available"] = True
            typer.echo(report.summary(data, settings.embedding_model))
        finally:
            await shutdown_state(state)

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run())


@app.command()
def providers() -> None:
    """List every model host, and which ones this machine can already reach."""
    from cachellm.providers.catalog import HOSTS
    from cachellm.providers.detect import available

    payload = {
        "hosts": [
            {
                "name": h.name,
                "env_key": h.env_key,
                "pip_extra": h.extra,
                "example_models": list(h.examples),
            }
            for h in HOSTS
        ]
    }
    detected = [
        {"name": d.host.name, "reason": d.reason, "key": d.host.key}
        for d in available(include_fake=False)
    ]
    typer.echo(report.providers_table(payload, detected))
    settings = get_settings()
    if not detected:
        typer.echo(
            "  Nothing configured yet. To try it with no account at all:\n"
            "    CACHELLM_DEFAULT_PROVIDER=fake cachellm serve\n"
        )
    elif not settings.openai_base_url_is_explicit and detected[0]["key"] != "bedrock":
        typer.echo(f"  `cachellm serve` will use {detected[0]['name']}.\n")


@app.command()
def invalidate(
    namespace: Annotated[str, typer.Option(help="Namespace hash to drop.")] = "",
    model: Annotated[str, typer.Option(help="Drop every entry for this model.")] = "",
    all_entries: Annotated[bool, typer.Option("--all", help="Drop the whole cache.")] = False,
) -> None:
    """Remove cache entries by namespace, model, or all of them."""
    if not (namespace or model or all_entries):
        typer.secho("pass --namespace, --model or --all", fg="red")
        raise typer.Exit(2)

    async def run() -> None:
        from cachellm.api.deps import build_state, shutdown_state

        state = await build_state(get_settings())
        try:
            if state.cache is None:
                typer.secho(f"cache unavailable: {state.degraded_reason}", fg="red")
                raise typer.Exit(1)
            removed = await state.cache.invalidate(
                namespace=namespace or None, model=model or None, drop_all=all_entries
            )
            typer.echo(f"removed {removed} keys")
        finally:
            await shutdown_state(state)

    asyncio.run(run())


@app.command()
def tune(
    pairs_file: Annotated[Path, typer.Argument(help="JSONL of {a, b, duplicate} objects.")],
    output: Annotated[Path, typer.Option(help="Write the sweep to this JSON file.")] = Path(
        "threshold_sweep.json"
    ),
) -> None:
    """Sweep similarity thresholds against labelled prompt pairs."""
    from bench.tune_threshold import run_sweep

    asyncio.run(run_sweep(pairs_file, output))


if __name__ == "__main__":  # pragma: no cover
    app()
