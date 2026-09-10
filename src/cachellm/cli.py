"""Command line interface."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Annotated, Any

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


def _proxy_url(url: str) -> str:
    settings = get_settings()
    return (url or f"http://127.0.0.1:{settings.port}").rstrip("/")


def _fetch(url: str, path: str, params: dict[str, Any] | None = None) -> Any:
    """Read from a running proxy, or None when nothing is listening.

    Talking to the server rather than building our own state is not an
    optimisation: with the in-memory backend the cache lives inside the serving
    process, so a CLI that built its own would report on an empty cache of its
    own and always say "no requests yet".
    """
    import httpx

    settings = get_settings()
    headers = {}
    if keys := sorted(settings.client_keys):
        headers["Authorization"] = f"Bearer {keys[0]}"
    try:
        response = httpx.get(f"{url}{path}", params=params, headers=headers, timeout=5.0)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def _not_running(url: str) -> None:
    typer.secho(f"No proxy answering at {url}.", fg="yellow")
    typer.echo("  Start one with `cachellm serve`, or pass --url if it is elsewhere.")
    typer.echo(
        f"  {report.DIM}The cache lives inside the serving process unless you "
        f"use the redis backend, so there is nothing to read without it."
        f"{report.RESET}"
    )


@app.command()
def stats(
    url: Annotated[str, typer.Option(help="Proxy to read from.")] = "",
    limit: Annotated[int, typer.Option(help="How many recent requests to show.")] = 15,
    json_out: Annotated[bool, typer.Option("--json", help="Raw JSON instead.")] = False,
) -> None:
    """Show what the cache has been doing: hit rate, savings, latency, recent requests."""
    base = _proxy_url(url)
    data = _fetch(base, "/admin/stats")
    if data is None:
        _not_running(base)
        raise typer.Exit(1)
    log = _fetch(base, "/admin/requests", {"limit": limit}) or {}
    recent = log.get("requests", [])
    if json_out:
        typer.echo(json.dumps({"stats": data, "recent": recent}, indent=2))
        return
    typer.echo(report.summary(data, get_settings().embedding_model))
    typer.echo(report.request_table(recent, limit=limit))


@app.command()
def watch(
    url: Annotated[str, typer.Option(help="Proxy to follow.")] = "",
    interval: Annotated[float, typer.Option(help="Seconds between refreshes.")] = 2.0,
) -> None:
    """Follow requests as they happen, like `tail -f` for the cache."""
    base = _proxy_url(url)
    if _fetch(base, "/admin/stats") is None:
        _not_running(base)
        raise typer.Exit(1)

    seen: set[tuple[float, str]] = set()
    first = True
    typer.echo(f"following {base}, ctrl-c to stop\n")
    try:
        while True:
            payload = _fetch(base, "/admin/requests", {"limit": 50}) or {}
            for record in reversed(payload.get("requests", [])):
                marker = (record.get("at", 0.0), record.get("prompt", ""))
                if marker in seen:
                    continue
                seen.add(marker)
                if not first:
                    typer.echo(report.request_line(record))
            first = False
            if len(seen) > 5_000:
                seen.clear()
            time.sleep(interval)
    except KeyboardInterrupt:
        data = _fetch(base, "/admin/stats")
        if data:
            typer.echo("")
            typer.echo(report.summary(data, get_settings().embedding_model))


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
