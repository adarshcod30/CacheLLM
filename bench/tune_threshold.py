"""Offline similarity-threshold sweep against labelled prompt pairs.

The sweep itself lives in `cachellm.tuning`, so the installed `cachellm tune`
command and this script share one implementation. This wrapper adds the
defaults used for the published results: the built-in evaluation corpus, and
output files under results/.

Runs entirely locally against the embedding model. No server, no provider, no
spend, and identical output on any machine.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from cachellm.settings import Settings
from cachellm.tuning import DEFAULT_THRESHOLDS, read_pairs, run_sweep, sweep, to_markdown

__all__ = ["DEFAULT_THRESHOLDS", "read_pairs", "run_sweep", "sweep", "to_markdown"]


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Sweep similarity thresholds.")
    parser.add_argument("--pairs", type=Path, help="JSONL of {a, b, duplicate}.")
    parser.add_argument("--out", type=Path, default=Path("results/threshold_sweep.json"))
    parser.add_argument("--markdown", type=Path, default=Path("results/threshold_sweep.md"))
    parser.add_argument("--embedding-backend", default="", help="fastembed or hash.")
    args = parser.parse_args()

    if args.pairs:
        rows = read_pairs(args.pairs)
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
