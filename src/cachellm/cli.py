"""Command line interface."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer

from cachellm import __version__
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
def stats() -> None:
    """Print cache statistics straight from Redis."""

    async def run() -> None:
        from cachellm.api.deps import build_state, shutdown_state

        settings = get_settings()
        state = await build_state(settings)
        try:
            if state.cache is None:
                typer.secho(f"cache unavailable: {state.degraded_reason}", fg="red")
                raise typer.Exit(1)
            typer.echo(json.dumps(await state.cache.stats(), indent=2))
        finally:
            await shutdown_state(state)

    asyncio.run(run())


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
