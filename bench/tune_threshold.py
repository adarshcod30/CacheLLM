"""Offline similarity-threshold sweep against labelled prompt pairs.

Answers the question every reviewer asks: *what does lowering the threshold
actually cost you?* For each threshold it reports how many pairs would be
served from cache, how many of those are genuine duplicates, and how many are
different questions that merely look alike.

Runs entirely locally against the embedding model. No server, no provider, no
spend, and identical output on any machine.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np

from cachellm.embeddings import build_embedder
from cachellm.settings import Settings

DEFAULT_THRESHOLDS = [round(0.70 + 0.02 * i, 2) for i in range(16)]  # 0.70 .. 1.00


async def sweep(
    pairs: list[dict[str, Any]],
    settings: Settings | None = None,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    settings = settings or Settings()
    embedder = build_embedder(settings)
    thresholds = thresholds or DEFAULT_THRESHOLDS

    texts_a = [str(p["a"]) for p in pairs]
    texts_b = [str(p["b"]) for p in pairs]
    labels = np.array([bool(p["duplicate"]) for p in pairs], dtype=bool)

    vectors_a = await embedder.embed_batch(texts_a)
    vectors_b = await embedder.embed_batch(texts_b)
    sims = np.array(
        [float(np.dot(a, b)) for a, b in zip(vectors_a, vectors_b, strict=True)], dtype=np.float32
    )

    kinds = np.array([str(p.get("kind", "")) for p in pairs])
    hard_mask = kinds == "hard_negative"
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    hard_count = int(hard_mask.sum())

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        predicted = sims >= threshold
        tp = int((predicted & labels).sum())
        fp = int((predicted & ~labels).sum())
        fn = int((~predicted & labels).sum())
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / positives if positives else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "threshold": threshold,
                "would_hit": tp + fp,
                "hit_rate": round((tp + fp) / len(pairs), 4),
                "true_hits": tp,
                "false_hits": fp,
                "missed_duplicates": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "false_positive_rate": round(fp / negatives, 4) if negatives else 0.0,
                "hard_negative_hits": int((predicted & hard_mask).sum()),
                "hard_negative_rate": (
                    round(int((predicted & hard_mask).sum()) / hard_count, 4) if hard_count else 0.0
                ),
            }
        )

    zero_fp = [r for r in rows if r["false_hits"] == 0]
    under_1pct = [r for r in rows if r["false_positive_rate"] <= 0.01]
    return {
        "embedding_model": embedder.name,
        "pairs": len(pairs),
        "duplicates": positives,
        "non_duplicates": negatives,
        "similarity": {
            "duplicates_mean": round(float(sims[labels].mean()), 4) if positives else None,
            "duplicates_min": round(float(sims[labels].min()), 4) if positives else None,
            "non_duplicates_mean": round(float(sims[~labels].mean()), 4) if negatives else None,
            "non_duplicates_max": round(float(sims[~labels].max()), 4) if negatives else None,
        },
        "riskiest_non_duplicates": [
            {
                "a": pairs[i]["a"],
                "b": pairs[i]["b"],
                "kind": str(kinds[i]),
                "similarity": round(float(sims[i]), 4),
            }
            for i in np.argsort(-np.where(labels, -1.0, sims))[:10]
            if not labels[i]
        ],
        "hardest_duplicates": [
            {"a": pairs[i]["a"], "b": pairs[i]["b"], "similarity": round(float(sims[i]), 4)}
            for i in np.argsort(np.where(labels, sims, 2.0))[:10]
            if labels[i]
        ],
        "sweep": rows,
        "best_f1": max(rows, key=lambda r: r["f1"]) if rows else None,
        "lowest_threshold_with_zero_false_hits": (
            min(zero_fp, key=lambda r: r["threshold"]) if zero_fp else None
        ),
        "lowest_threshold_under_1pct_false_positives": (
            min(under_1pct, key=lambda r: r["threshold"]) if under_1pct else None
        ),
    }


def to_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"Embedding model: `{result['embedding_model']}`  ",
        f"Pairs: {result['pairs']} "
        f"({result['duplicates']} duplicates, {result['non_duplicates']} non-duplicates)",
        "",
        "| Threshold | Recall | Precision | F1 | False hits | Hard-negative hits |",
        "| --------: | -----: | --------: | -: | ---------: | -----------------: |",
    ]
    for row in result["sweep"]:
        lines.append(
            f"| {row['threshold']:.2f} | {row['recall']:.1%} | {row['precision']:.3f} | "
            f"{row['f1']:.3f} | {row['false_hits']} | "
            f"{row['hard_negative_hits']} ({row['hard_negative_rate']:.1%}) |"
        )
    risky = result.get("riskiest_non_duplicates") or []
    if risky:
        lines += [
            "",
            "**Closest calls: different questions that look alike**",
            "",
            "| Similarity | Question A | Question B |",
            "| ---: | --- | --- |",
        ]
        for row in risky[:6]:
            lines.append(f"| {row['similarity']:.3f} | {row['a']} | {row['b']} |")
    hardest = result.get("hardest_duplicates") or []
    if hardest:
        lines += [
            "",
            "**Hardest duplicates: same question, lowest score**",
            "",
            "| Similarity | Question A | Question B |",
            "| ---: | --- | --- |",
        ]
        for row in hardest[:6]:
            lines.append(f"| {row['similarity']:.3f} | {row['a']} | {row['b']} |")
    return "\n".join(lines)


def _read_pairs(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


async def run_sweep(pairs_file: Path, output: Path) -> dict[str, Any]:
    rows = await asyncio.to_thread(_read_pairs, pairs_file)
    result = await sweep(rows)
    await asyncio.to_thread(Path(output).write_text, json.dumps(result, indent=2))
    print(to_markdown(result))
    return result


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Sweep similarity thresholds.")
    parser.add_argument("--pairs", type=Path, help="JSONL of {a, b, duplicate}.")
    parser.add_argument("--out", type=Path, default=Path("results/threshold_sweep.json"))
    parser.add_argument("--markdown", type=Path, default=Path("results/threshold_sweep.md"))
    parser.add_argument("--embedding-backend", default="", help="fastembed or hash.")
    args = parser.parse_args()

    if args.pairs:
        rows = [json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()]
    else:
        from eval.corpus import labelled_pairs

        rows = labelled_pairs()

    overrides: dict[str, Any] = {}
    if args.embedding_backend:
        overrides["embedding_backend"] = args.embedding_backend
    settings = Settings(**overrides) if overrides else Settings()

    result = await sweep(rows, settings=settings)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    markdown = to_markdown(result)
    args.markdown.write_text(markdown)
    print(markdown)
    print()
    print(f"Best F1               : {result['best_f1']['threshold']:.2f}")
    zero = result["lowest_threshold_with_zero_false_hits"]
    if zero:
        print(f"Lowest zero-false-hit : {zero['threshold']:.2f} (recall {zero['recall']:.1%})")
    print(f"Written to {args.out} and {args.markdown}")


if __name__ == "__main__":
    asyncio.run(main())
