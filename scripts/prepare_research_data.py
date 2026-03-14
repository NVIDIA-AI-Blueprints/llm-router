#!/usr/bin/env python3
"""Convert research-team pre-extracted features into aligned PrefillResult caches.

The research PT file has hidden states for N questions in an arbitrary order.
The training/test CSVs have labels for subsets of those questions in their own
insertion order. This script reindexes the PT tensors so row i of the output
matches the i-th unique question produced by _load_labels() on each CSV.

Usage:
    python scripts/prepare_research_data.py \
        --features working/assets-from-research/prefill_qwen3.5-35b-a3b_parquet_pool.pt \
        --train data/train_v1.csv \
        --test data/test_v1.csv \
        --output-dir data/parquet-pool/
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch


def normalize_question(q: str) -> str:
    return " ".join(q.split()).strip().lower()


def ordered_unique_questions(csv_path: Path) -> list[str]:
    """Return unique questions from a CSV in _load_labels insertion order."""
    seen: dict[str, str] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            nq = normalize_question(row["question"])
            if nq not in seen:
                seen[nq] = nq
    return list(seen.keys())


def build_pt_index(pt: dict) -> dict[str, int]:
    """Map normalize_question(q) -> row index in the PT file."""
    index: dict[str, int] = {}
    for i, q in enumerate(pt["questions"]):
        nq = normalize_question(q)
        if nq not in index:
            index[nq] = i
    return index


def reindex_and_save(
    pt: dict,
    pt_index: dict[str, int],
    questions: list[str],
    output_path: Path,
) -> None:
    """Slice PT tensors to match question ordering and save as PrefillResult."""
    indices = []
    missing = []
    for nq in questions:
        if nq in pt_index:
            indices.append(pt_index[nq])
        else:
            missing.append(nq[:80])

    if missing:
        print(f"  WARNING: {len(missing)} questions not found in PT file", file=sys.stderr)
        for m in missing[:5]:
            print(f"    {m}...", file=sys.stderr)
        sys.exit(1)

    idx = torch.tensor(indices, dtype=torch.long)
    cfg = pt["config"]

    data: dict = {"config": {"n_layers": cfg["n_layers"], "hidden_dim": cfg["hidden_dim"]}}
    data["questions"] = [pt["questions"][i] for i in indices]
    for key, val in pt.items():
        if not isinstance(val, torch.Tensor):
            continue
        if key.startswith("layer_"):
            data[key] = val[idx]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    sz_mb = output_path.stat().st_size / 1e6
    print(f"  Saved {output_path} ({len(indices)} questions, {sz_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert research pre-extracted features into aligned PrefillResult caches.",
    )
    parser.add_argument(
        "--features", required=True,
        help="Path to research PT file (e.g. prefill_qwen3.5-35b-a3b_parquet_pool.pt)",
    )
    parser.add_argument("--train", required=True, help="Training labels CSV")
    parser.add_argument("--test", default=None, help="Test labels CSV (optional)")
    parser.add_argument("--output-dir", required=True, help="Output directory for aligned caches")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    t0 = time.monotonic()
    print(f"Loading features: {args.features}")
    pt = torch.load(args.features, map_location="cpu", weights_only=False)

    n_questions = len(pt["questions"])
    n_layers = sum(1 for k in pt if k.startswith("layer_") and "_meanpool" not in k)
    n_meanpool = sum(1 for k in pt if "_meanpool" in k)
    print(f"  {n_questions} questions, {n_layers} last-token layers, {n_meanpool} meanpool layers")
    print(f"  hidden_dim={pt['config']['hidden_dim']}, n_layers={pt['config']['n_layers']}")

    pt_index = build_pt_index(pt)
    print(f"  Built index: {len(pt_index)} unique normalized questions")

    print(f"\nProcessing train split: {args.train}")
    train_qs = ordered_unique_questions(Path(args.train))
    print(f"  {len(train_qs)} unique questions")
    reindex_and_save(pt, pt_index, train_qs, output_dir / "train.pt")

    if args.test:
        print(f"\nProcessing test split: {args.test}")
        test_qs = ordered_unique_questions(Path(args.test))
        print(f"  {len(test_qs)} unique questions")
        reindex_and_save(pt, pt_index, test_qs, output_dir / "test.pt")

    elapsed = time.monotonic() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
