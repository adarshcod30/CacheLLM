"""Replay a workload through a running CacheLLM proxy and report the numbers.

This is the harness behind every figure in the README. It sends a realistic mix
of first-time, reworded and repeated questions at a chosen concurrency, records
what the cache did with each one, and prints latency percentiles split by
outcome alongside modelled cost.

Everything it needs is printed with the results: workload knobs, provider,
embedding model, threshold. A benchmark you cannot re-run and disagree with is
marketing, not evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from bench.workload import WorkloadConfig, WorkloadItem, build, summarise


@dataclass
class Record:
    index: int
    kind: str
    cache: str
    tier: str
    similarity: float
    latency_ms: float
    status: int
    saved_usd: float = 0.0
    error: str = ""


@dataclass
class ReplayConfig:
    base_url: str = "http://127.0.0.1:8080"
    model: str = "fake/echo"
    api_key: str = ""
    #: Optional system prompt sent with every request. Real applications almost
    #: always have one, and it also keeps verbose models from running into
    #: max_tokens, which would leave their answers uncacheable.
    system: str = ""
    concurrency: int = 8
    requests: int = 2000
    seed: int = 42
    max_tokens: int | None = None
    warmup: int = 0
    timeout: float = 120.0
    out_dir: Path = field(default=Path("results"))


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, round((p / 100) * (len(ordered) - 1))))
        return round(ordered[idx], 2)

    return {
        "count": len(ordered),
        "mean": round(statistics.fmean(ordered), 2),
        "p50": pct(50),
        "p95": pct(95),
        "p99": pct(99),
        "min": round(ordered[0], 2),
        "max": round(ordered[-1], 2),
    }


async def send_one(
    client: httpx.AsyncClient, config: ReplayConfig, index: int, item: WorkloadItem
) -> Record:
    messages: list[dict[str, str]] = []
    if config.system:
        messages.append({"role": "system", "content": config.system})
    messages.append({"role": "user", "content": item.prompt})
    payload: dict[str, Any] = {
        "model": config.model,
        "temperature": 0,
        "messages": messages,
    }
    if config.max_tokens:
        payload["max_tokens"] = config.max_tokens

    started = time.perf_counter()
    try:
        response = await client.post("/v1/chat/completions", json=payload)
        elapsed = (time.perf_counter() - started) * 1000
    except httpx.HTTPError as exc:
        return Record(
            index,
            item.kind,
            "ERROR",
            "",
            0.0,
            (time.perf_counter() - started) * 1000,
            0,
            error=str(exc)[:120],
        )

    headers = response.headers
    return Record(
        index=index,
        kind=item.kind,
        cache=headers.get("X-Cache", "?"),
        tier=headers.get("X-Cache-Tier", ""),
        similarity=float(headers.get("X-Cache-Similarity", 0.0) or 0.0),
        latency_ms=elapsed,
        status=response.status_code,
        saved_usd=float(headers.get("X-Cache-Saved-USD", 0.0) or 0.0),
        error="" if response.status_code == 200 else response.text[:120],
    )


async def replay(config: ReplayConfig, items: list[WorkloadItem]) -> list[Record]:
    headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
    limits = httpx.Limits(
        max_connections=config.concurrency * 2, max_keepalive_connections=config.concurrency * 2
    )
    semaphore = asyncio.Semaphore(config.concurrency)
    records: list[Record] = []

    async with httpx.AsyncClient(
        base_url=config.base_url, headers=headers, timeout=config.timeout, limits=limits
    ) as client:

        async def run(index: int, item: WorkloadItem) -> None:
            async with semaphore:
                records.append(await send_one(client, config, index, item))

        started = time.perf_counter()
        await asyncio.gather(*(run(i, item) for i, item in enumerate(items)))
        elapsed = time.perf_counter() - started

    records.sort(key=lambda r: r.index)
    records_meta.update(
        {
            "wall_seconds": round(elapsed, 2),
            "throughput_rps": round(len(items) / elapsed, 1) if elapsed else 0,
        }
    )
    return records


records_meta: dict[str, Any] = {}


def hit_rate_curve(records: list[Record], windows: int = 20) -> list[dict[str, float]]:
    """Hit rate over time, showing how fast a cold cache warms up."""
    if not records:
        return []
    size = max(1, len(records) // windows)
    curve: list[dict[str, float]] = []
    for start in range(0, len(records), size):
        chunk = records[start : start + size]
        hits = sum(1 for r in chunk if r.cache == "HIT")
        curve.append(
            {
                "through_request": start + len(chunk),
                "hit_rate": round(hits / len(chunk), 4),
            }
        )
    return curve


def analyse(
    records: list[Record],
    workload: dict[str, Any],
    config: ReplayConfig,
    stats: dict[str, Any] | None,
) -> dict[str, Any]:
    ok = [r for r in records if r.status == 200]
    hits = [r for r in ok if r.cache == "HIT"]
    misses = [r for r in ok if r.cache == "MISS"]
    bypass = [r for r in ok if r.cache == "BYPASS"]
    errors = [r for r in records if r.status != 200]

    exact = [r for r in hits if r.tier == "exact"]
    semantic = [r for r in hits if r.tier == "semantic"]
    hit_lat = percentiles([r.latency_ms for r in hits])
    miss_lat = percentiles([r.latency_ms for r in misses])

    by_kind: dict[str, dict[str, int]] = {}
    for record in ok:
        bucket = by_kind.setdefault(record.kind, {"HIT": 0, "MISS": 0, "BYPASS": 0})
        bucket[record.cache] = bucket.get(record.cache, 0) + 1

    speedup = (
        round(miss_lat["p95"] / hit_lat["p95"], 1)
        if hit_lat.get("p95") and miss_lat.get("p95")
        else None
    )
    return {
        "run": {
            "requests": len(records),
            "successful": len(ok),
            "errors": len(errors),
            "model": config.model,
            "concurrency": config.concurrency,
            "max_tokens": config.max_tokens,
            "system_prompt": config.system or None,
            **records_meta,
        },
        "workload": workload,
        "outcomes": {
            "hits": len(hits),
            "misses": len(misses),
            "bypassed": len(bypass),
            "hit_rate": round(len(hits) / len(ok), 4) if ok else 0,
            "hit_rate_of_cacheable": (
                round(len(hits) / (len(hits) + len(misses)), 4) if (hits or misses) else 0
            ),
            "exact_hits": len(exact),
            "semantic_hits": len(semantic),
            "semantic_share_of_hits": (round(len(semantic) / len(hits), 4) if hits else 0),
        },
        "latency_ms": {"hit": hit_lat, "miss": miss_lat, "p95_speedup_factor": speedup},
        "similarity_of_semantic_hits": percentiles([r.similarity for r in semantic]),
        "by_request_kind": by_kind,
        "hit_rate_curve": hit_rate_curve(ok),
        "server_stats": stats or {},
        "diagnostics": (stats or {}).get("diagnostics", []),
        "errors_sample": [asdict(r) for r in errors[:5]],
    }


def to_markdown(report: dict[str, Any]) -> str:
    run, out = report["run"], report["outcomes"]
    lat = report["latency_ms"]
    server = report.get("server_stats") or {}
    saved = server.get("usd_saved", 0.0)
    spent = server.get("usd_spent", 0.0)
    total = saved + spent
    lines = [
        "### Load test",
        "",
        f"{run['requests']} requests, concurrency {run['concurrency']}, "
        f"model `{run['model']}`, {run.get('wall_seconds', 0)}s wall "
        f"({run.get('throughput_rps', 0)} req/s).",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Hit rate (all requests) | {out['hit_rate']:.1%} |",
        f"| Hit rate (cacheable only) | {out['hit_rate_of_cacheable']:.1%} |",
        f"| Workload ceiling | {report['workload']['theoretical_max_hit_rate']:.1%} |",
        f"| Exact-tier hits | {out['exact_hits']} |",
        f"| Semantic-tier hits | {out['semantic_hits']} |",
        f"| Semantic share of hits | {out['semantic_share_of_hits']:.1%} |",
        f"| Errors | {run['errors']} |",
        "",
        "| Latency (ms) | p50 | p95 | p99 |",
        "| --- | ---: | ---: | ---: |",
        f"| Cache hit | {lat['hit'].get('p50', 0)} | {lat['hit'].get('p95', 0)} | "
        f"{lat['hit'].get('p99', 0)} |",
        f"| Cache miss | {lat['miss'].get('p50', 0)} | {lat['miss'].get('p95', 0)} | "
        f"{lat['miss'].get('p99', 0)} |",
    ]
    if lat.get("p95_speedup_factor"):
        lines.append(f"| p95 speedup | {lat['p95_speedup_factor']}x | | |")
    notes = report.get("diagnostics") or []
    if notes:
        lines += ["", "**Diagnostics**", ""] + [f"- {note}" for note in notes]
    if total:
        lines += [
            "",
            "| Modelled cost | USD |",
            "| --- | ---: |",
            f"| Spent upstream | {spent:.6f} |",
            f"| Saved by cache | {saved:.6f} |",
            f"| Reduction | {saved / total:.1%} |",
        ]
    return "\n".join(lines)


async def fetch_stats(config: ReplayConfig) -> dict[str, Any] | None:
    headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
    try:
        async with httpx.AsyncClient(
            base_url=config.base_url, headers=headers, timeout=10.0
        ) as client:
            response = await client.get("/admin/stats")
            return response.json() if response.status_code == 200 else None
    except httpx.HTTPError:
        return None


async def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a workload through CacheLLM.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="fake/echo")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--system", default="", help="System prompt sent with every request.")
    parser.add_argument("--unseen-ratio", type=float, default=0.18)
    parser.add_argument("--exact-repeat-ratio", type=float, default=0.40)
    parser.add_argument("--reset", action="store_true", help="Flush the cache before starting.")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    config = ReplayConfig(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        concurrency=args.concurrency,
        requests=args.requests,
        seed=args.seed,
        max_tokens=args.max_tokens,
        system=args.system,
        out_dir=args.out_dir,
    )
    workload_config = WorkloadConfig(
        requests=args.requests,
        seed=args.seed,
        unseen_ratio=args.unseen_ratio,
        exact_repeat_ratio=args.exact_repeat_ratio,
    )
    items = build(workload_config)
    workload = summarise(items, workload_config)

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    if args.reset:
        async with httpx.AsyncClient(
            base_url=args.base_url, headers=headers, timeout=30.0
        ) as client:
            await client.post("/admin/invalidate", json={"all": True})
            await client.post("/admin/reset-stats")
        print("cache flushed and counters reset")

    print(
        f"replaying {len(items)} requests at concurrency {args.concurrency} "
        f"against {args.base_url} ..."
    )
    records = await replay(config, items)
    report = analyse(records, workload, config, await fetch_stats(config))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "load_test.json").write_text(json.dumps(report, indent=2))
    markdown = to_markdown(report)
    (args.out_dir / "load_test.md").write_text(markdown)
    (args.out_dir / "load_test_records.json").write_text(
        json.dumps([asdict(r) for r in records], indent=2)
    )
    print()
    print(markdown)
    print()
    print(f"written to {args.out_dir}/load_test.json and {args.out_dir}/load_test.md")


if __name__ == "__main__":
    asyncio.run(main())
