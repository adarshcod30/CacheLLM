"""Compare embedding models on the one question that decides a cache's safety.

Hit rate is easy to buy: drop the threshold. The real measure of an embedding
model for caching is **how much recall it can give you before it starts serving
the wrong answer**. So the headline metric here is:

    max recall, subject to zero hits on hard negatives

Hard negatives are the "undo the last git commit" versus "undo the last git
merge" pairs: one word apart, opposite meaning. A model that cannot separate
those forces you into a threshold so strict that the semantic tier stops
earning its keep.

The second metric, ``separation``, is the gap between the mean duplicate score
and the highest non-duplicate score. Positive is good; negative means the score
distributions overlap and no single global threshold can split them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from eval.corpus import labelled_pairs

from cachellm.embeddings.fastembed_backend import FastEmbedEmbedder

CANDIDATES = [
    ("BAAI/bge-small-en-v1.5", 384),
    ("BAAI/bge-base-en-v1.5", 768),
    ("sentence-transformers/all-MiniLM-L6-v2", 384),
    ("snowflake/snowflake-arctic-embed-s", 384),
    ("jinaai/jina-embeddings-v2-small-en", 512),
    ("thenlper/gte-base", 768),
]

THRESHOLDS = [round(0.50 + 0.01 * i, 2) for i in range(51)]  # 0.50 .. 1.00


async def score_model(model: str, dim: int, pairs: list[dict[str, Any]]) -> dict[str, Any]:
    embedder = FastEmbedEmbedder(model_name=model, dim=dim)
    texts_a = [str(p["a"]) for p in pairs]
    texts_b = [str(p["b"]) for p in pairs]
    labels = np.array([bool(p["duplicate"]) for p in pairs])
    kinds = np.array([str(p.get("kind", "")) for p in pairs])
    hard = kinds == "hard_negative"

    started = time.perf_counter()
    vectors_a = await embedder.embed_batch(texts_a)
    vectors_b = await embedder.embed_batch(texts_b)
    embed_seconds = time.perf_counter() - started

    sims = np.array([float(np.dot(a, b)) for a, b in zip(vectors_a, vectors_b, strict=True)])

    # One embed call on a real question is the floor for every reworded hit.
    # Time the corpus's own questions through a copy with its cache off, so
    # every call runs the model. This used to time twenty synthetic strings
    # like "latency probe 123.4" and report the mean, which says little about
    # the questions people actually send. Still sensitive to CPU load: run it
    # on a quiet machine and read the column as a ranking first.
    probe = FastEmbedEmbedder(model_name=model, dim=dim, cache_size=0)
    questions = list(dict.fromkeys(texts_a + texts_b))
    for text in questions[:10]:
        await probe.embed(text)
    timings: list[float] = []
    for text in questions:
        started_one = time.perf_counter()
        await probe.embed(text)
        timings.append((time.perf_counter() - started_one) * 1000)
    embed_p50 = float(np.percentile(timings, 50))
    embed_p95 = float(np.percentile(timings, 95))

    best_safe: dict[str, Any] | None = None
    best_1pct: dict[str, Any] | None = None
    for threshold in THRESHOLDS:
        predicted = sims >= threshold
        hard_hits = int((predicted & hard).sum())
        false_hits = int((predicted & ~labels).sum())
        recall = float((predicted & labels).sum() / labels.sum())
        row = {
            "threshold": threshold,
            "recall": round(recall, 4),
            "false_hits": false_hits,
            "hard_negative_hits": hard_hits,
        }
        if hard_hits == 0 and (best_safe is None or recall > best_safe["recall"]):
            best_safe = row
        if false_hits <= max(1, int(0.01 * (~labels).sum())) and (
            best_1pct is None or recall > best_1pct["recall"]
        ):
            best_1pct = row

    dup_mean = float(sims[labels].mean())
    non_dup_max = float(sims[~labels].max())
    hard_max = float(sims[hard].max()) if hard.any() else 0.0
    return {
        "model": model,
        "dim": dim,
        "embed_ms_p50": round(embed_p50, 2),
        "embed_ms_p95": round(embed_p95, 2),
        "embed_questions_timed": len(timings),
        "embed_seconds_for_corpus": round(embed_seconds, 2),
        "duplicates_mean": round(dup_mean, 4),
        "duplicates_min": round(float(sims[labels].min()), 4),
        "non_duplicates_max": round(non_dup_max, 4),
        "hard_negative_max": round(hard_max, 4),
        "separation": round(dup_mean - hard_max, 4),
        "safe_operating_point": best_safe,
        "under_1pct_false_positives": best_1pct,
    }


def to_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Model | Dim | Embed ms p50 | Embed ms p95 | Mean dup score | Worst hard negative | "
        "Separation | Safe threshold | Recall there |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        safe = row["safe_operating_point"] or {}
        threshold = f"{safe['threshold']:.2f}" if safe else "n/a"
        recall = f"{safe['recall']:.1%}" if safe else "n/a"
        lines.append(
            f"| `{row['model']}` | {row['dim']} | {row['embed_ms_p50']} | {row['embed_ms_p95']} | "
            f"{row['duplicates_mean']:.3f} | {row['hard_negative_max']:.3f} | "
            f"{row['separation']:+.3f} | {threshold} | {recall} |"
        )
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Compare embedding models for caching.")
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--out", type=Path, default=Path("results/model_comparison.json"))
    parser.add_argument("--markdown", type=Path, default=Path("results/model_comparison.md"))
    args = parser.parse_args()

    pairs = labelled_pairs()
    candidates = [(m, 0) for m in args.models] if args.models else CANDIDATES

    rows: list[dict[str, Any]] = []
    for model, dim in candidates:
        print(f"scoring {model} ...", flush=True)
        try:
            rows.append(await score_model(model, dim, pairs))
        except Exception as exc:  # a model may be unavailable offline
            print(f"  skipped: {str(exc)[:160]}")

    rows.sort(key=lambda r: (r["safe_operating_point"] or {}).get("recall", 0), reverse=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"pairs": len(pairs), "models": rows}, indent=2))
    markdown = to_markdown(rows)
    args.markdown.write_text(markdown)
    print()
    print(markdown)


if __name__ == "__main__":
    asyncio.run(main())
