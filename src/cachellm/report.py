"""Text reports for the terminal.

Everything a dashboard would show, printed where you already are. Prometheus
and Grafana remain available for people who want history and alerting, but
nobody should have to run two extra servers to answer "is the cache working".

Plain ANSI, no dependencies. Colour is dropped automatically when the output is
piped, so `cachellm stats | grep` behaves.
"""

from __future__ import annotations

import os
import sys
import textwrap
import time
from typing import Any

_COLOUR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str) -> str:
    return code if _COLOUR else ""


DIM, BOLD, RESET = _c("\033[2m"), _c("\033[1m"), _c("\033[0m")
GREEN, AMBER, MAGENTA, BLUE, RED = (
    _c("\033[32m"),
    _c("\033[33m"),
    _c("\033[35m"),
    _c("\033[36m"),
    _c("\033[31m"),
)
STATUS_COLOUR = {"HIT": GREEN, "MISS": AMBER, "BYPASS": MAGENTA, "SHADOW": MAGENTA, "ERROR": RED}


def bar(fraction: float, width: int = 22) -> str:
    """A proportion, drawn. Reads at a glance in a way a percentage does not."""
    fraction = max(0.0, min(1.0, fraction))
    filled = round(fraction * width)
    return f"{GREEN}{'█' * filled}{RESET}{DIM}{'░' * (width - filled)}{RESET}"


def money(amount: float) -> str:
    """Small sums need more decimals than large ones to say anything."""
    if amount >= 1:
        return f"${amount:,.2f}"
    if amount >= 0.01:
        return f"${amount:.4f}"
    return f"${amount:.6f}"


def ago(timestamp: float) -> str:
    seconds = max(0, int(time.time() - timestamp))
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86_400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86_400}d ago"


def label(text: str) -> str:
    return f"  {BOLD}{text:<11}{RESET}"


def header(stats: dict[str, Any], embedding_model: str = "") -> list[str]:
    backend = stats.get("backend", "?")
    bits = [f"{BOLD}CacheLLM{RESET}", f"{backend} backend"]
    if embedding_model:
        bits.append(embedding_model.split("/")[-1])
    if stats.get("shadow_mode"):
        bits.append(f"{MAGENTA}shadow mode{RESET}")
    if not stats.get("caching_enabled", True):
        bits.append(f"{AMBER}caching disabled{RESET}")
    lines = ["", "  " + f" {DIM}·{RESET} ".join(bits)]
    # A reachable Redis that was passed over is worth a line of its own:
    # whoever started it expected it to be used.
    if note := stats.get("backend_note"):
        lines += [f"  {AMBER}{row}{RESET}" for row in textwrap.wrap(str(note), 74)]
    return [*lines, ""]


def summary(stats: dict[str, Any], embedding_model: str = "") -> str:
    """The whole picture, in about fifteen lines."""
    if not stats.get("cache_available", False):
        return (
            f"\n  {AMBER}cache unavailable{RESET}: "
            f"{stats.get('degraded_reason', 'unknown')}\n"
            f"  {DIM}the proxy is still serving, it is just not caching{RESET}\n"
        )

    out = header(stats, embedding_model)
    requests = stats.get("requests", 0)
    hits, misses = stats.get("hits", 0), stats.get("misses", 0)
    rate = stats.get("hit_rate", 0.0)

    if not requests:
        out += ["  no requests yet", ""]
        return "\n".join(out)

    out.append(
        f"{label('HIT RATE')}{GREEN}{rate:6.1%}{RESET}  {bar(rate)}  "
        f"{hits:,} of {requests:,} requests"
    )
    out.append(
        f"  {' ' * 11}{DIM}{stats.get('hits_exact', 0):,} exact · "
        f"{stats.get('hits_semantic', 0):,} semantic · {misses:,} missed · "
        f"{stats.get('bypassed', 0):,} bypassed{RESET}"
    )

    saved, spent = stats.get("usd_saved", 0.0), stats.get("usd_spent", 0.0)
    total = saved + spent
    out.append("")
    reduction = f"  {GREEN}{saved / total:.0%} lower{RESET}" if total else ""
    out.append(
        f"{label('SAVED')}{GREEN}{money(saved)}{RESET}   "
        f"{DIM}spent {money(spent)}{RESET}{reduction}"
    )
    if stats.get("tokens_saved"):
        out.append(f"  {' ' * 11}{DIM}{stats['tokens_saved']:,} tokens never generated{RESET}")

    hit_lat = stats.get("latency_ms", {}).get("hit", {})
    miss_lat = stats.get("latency_ms", {}).get("miss", {})
    if hit_lat or miss_lat:
        out.append("")
        if hit_lat:
            out.append(
                f"{label('LATENCY')}{DIM}cached  {RESET}"
                f"{hit_lat.get('p50', 0):8.1f} ms p50 {DIM}·{RESET} "
                f"{hit_lat.get('p95', 0):8.1f} ms p95"
            )
        if miss_lat:
            speedup = ""
            if hit_lat.get("p95") and miss_lat.get("p95"):
                speedup = f"   {GREEN}{miss_lat['p95'] / hit_lat['p95']:.0f}x faster{RESET}"
            out.append(
                f"  {' ' * 11}{DIM}uncached{RESET}"
                f"{miss_lat.get('p50', 0):8.1f} ms p50 {DIM}·{RESET} "
                f"{miss_lat.get('p95', 0):8.1f} ms p95{speedup}"
            )

    out.append("")
    parts = [f"{stats.get('entries', 0):,} entries"]
    if stats.get("coalesced"):
        parts.append(f"{stats['coalesced']:,} coalesced")
    if stats.get("near_misses"):
        parts.append(f"{stats['near_misses']:,} near misses")
    if stats.get("errors"):
        parts.append(f"{RED}{stats['errors']:,} provider errors{RESET}")
    out.append(f"{label('CACHE')}{DIM}{f' {DIM}·{RESET}{DIM} '.join(parts)}{RESET}")

    for note in stats.get("diagnostics", []) or []:
        out.append("")
        out.append(f"  {AMBER}note{RESET}       {note}")

    out.append("")
    return "\n".join(out)


def request_line(record: dict[str, Any], width: int = 46) -> str:
    """One row of the request log."""
    status = str(record.get("status", "?"))
    colour = STATUS_COLOUR.get(status, "")
    tier = record.get("tier") or ""
    sim = record.get("similarity") or 0.0
    score = f"{sim:.3f}" if tier else f"{DIM}  ·  {RESET}"
    prompt = (record.get("prompt") or "").replace("\n", " ")
    if len(prompt) > width:
        prompt = prompt[: width - 1] + "…"
    reason = record.get("reason") or ""
    tail = f"  {DIM}({reason}){RESET}" if reason else ""
    saved = record.get("saved_usd") or 0.0
    money_col = f"{GREEN}+{money(saved)}{RESET}" if saved else f"{DIM}      ·     {RESET}"
    stamp = time.strftime("%H:%M:%S", time.localtime(record.get("at", 0)))
    return (
        f"  {DIM}{stamp}{RESET}  {colour}{status:<6}{RESET} {tier:<8} "
        f"{record.get('latency_ms', 0):8.1f}ms  {score}  {money_col}  "
        f"{prompt}{tail}"
    )


def request_table(records: list[dict[str, Any]], limit: int = 15) -> str:
    if not records:
        return f"  {DIM}no requests logged yet{RESET}\n"
    out = [
        f"  {BOLD}{'time':<8}  {'result':<6} {'tier':<8} {'latency':>10}  "
        f"{'score':<5}  {'saved':<11} prompt{RESET}"
    ]
    out += [request_line(r) for r in records[:limit]]
    out.append("")
    return "\n".join(out)


def providers_table(payload: dict[str, Any], detected: list[dict[str, Any]]) -> str:
    """Every host, and which of them this machine can reach right now."""
    out = ["", f"  {BOLD}This proxy routes to{RESET}", ""]
    if detected:
        for d in detected:
            chosen = f"  {DIM}(default){RESET}" if d.get("default") else ""
            out.append(f"  {GREEN}✓{RESET} {d['name']:<28}{DIM}{d['reason']}{RESET}{chosen}")
    else:
        out.append(f"  {DIM}nothing detected; the built-in test double is always available{RESET}")

    out += [
        "",
        f"  {BOLD}Everything supported{RESET}",
        f"  {DIM}export a host's key and it joins the routes when the proxy starts{RESET}",
        "",
    ]
    out.append(f"  {BOLD}{'host':<26} {'api key variable':<22} {'example model'}{RESET}")
    for host in payload.get("hosts", []):
        key = host.get("env_key") or (
            f"{DIM}pip extra: {host['pip_extra']}{RESET}"
            if host.get("pip_extra")
            else f"{DIM}none needed{RESET}"
        )
        example = (host.get("example_models") or [""])[0]
        out.append(f"  {host['name']:<26} {key:<22} {DIM}{example}{RESET}")
    out.append("")
    return "\n".join(out)
