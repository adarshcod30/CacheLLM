"""Formatting for the demo scripts. Kept out of bash so quoting stays sane."""

from __future__ import annotations

import json
import sys

GREEN, AMBER, MAGENTA, DIM, BOLD, RESET = (
    "\033[32m", "\033[33m", "\033[35m", "\033[2m", "\033[1m", "\033[0m",
)
COLOURS = {"HIT": GREEN, "MISS": AMBER, "BYPASS": MAGENTA, "SHADOW": MAGENTA}


def render_ask(headers_path: str, elapsed_s: str) -> None:
    headers: dict[str, str] = {}
    with open(headers_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if ":" in line:
                name, _, value = line.partition(":")
                headers[name.strip().lower()] = value.strip()

    status = headers.get("x-cache", "?")
    colour = COLOURS.get(status, RESET)
    ms = float(elapsed_s) * 1000

    bits = []
    if tier := headers.get("x-cache-tier"):
        bits.append(f"tier={tier}")
    if sim := headers.get("x-cache-similarity"):
        bits.append(f"similarity={sim}")
    if reason := headers.get("x-cache-bypass-reason"):
        bits.append(f"reason={reason}")
    if saved := headers.get("x-cache-saved-usd"):
        bits.append(f"saved=${float(saved):.6f}")

    print(f"{colour}{status:<6}{RESET} {ms:8.1f} ms   {DIM}{'  '.join(bits)}{RESET}")


def render_stats(raw: str) -> None:
    d = json.loads(raw)
    hits_exact, hits_semantic = d["hits_exact"], d["hits_semantic"]
    print(
        f"  {BOLD}hit rate{RESET}      {GREEN}{d['hit_rate']:.1%}{RESET}"
        f"   ({hits_exact} exact + {hits_semantic} semantic of {d['requests']} requests)"
    )
    print(
        f"  {BOLD}saved{RESET}         {GREEN}${d['usd_saved']:.6f}{RESET}"
        f"   spent ${d['usd_spent']:.6f}   {d['tokens_saved']:,} tokens never generated"
    )
    hit = d["latency_ms"]["hit"].get("p95")
    miss = d["latency_ms"]["miss"].get("p95")
    if hit and miss:
        print(
            f"  {BOLD}p95 latency{RESET}   {GREEN}{hit} ms{RESET} cached"
            f"   vs {miss} ms uncached   ({miss / hit:.0f}x faster)"
        )
    for note in d.get("diagnostics", []):
        print(f"  {AMBER}note{RESET}          {note}")


if __name__ == "__main__":
    if sys.argv[1] == "ask":
        render_ask(sys.argv[2], sys.argv[3])
    else:
        render_stats(sys.stdin.read())
