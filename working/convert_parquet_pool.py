#!/usr/bin/env python3
"""Convert parquet_pool JSONL files into train_v1.csv and test_v1.csv.

Reads all model subdirectories under working/assets-from-research/parquet_pool/,
combines train+val+cal splits into train, test split into test, and writes CSVs
matching the schema: question,model,isCorrect,output_tokens,embedding_id
"""

import csv
import json
import sys
from pathlib import Path

POOL_DIR = Path("working/assets-from-research/parquet_pool")
OUT_DIR = Path("data")

TRAIN_SPLITS = ["all_questions_train.jsonl", "all_questions_val.jsonl", "all_questions_cal.jsonl"]
TEST_SPLITS = ["all_questions_test.jsonl"]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def collect_rows(model_dirs: list[Path], split_files: list[str]) -> list[dict]:
    rows = []
    for model_dir in sorted(model_dirs):
        for split_file in split_files:
            p = model_dir / split_file
            if not p.exists():
                print(f"  WARNING: missing {p}", file=sys.stderr)
                continue
            rows.extend(read_jsonl(p))
    return rows


def build_embedding_map(rows: list[dict]) -> dict[str, int]:
    """Assign a sequential integer embedding_id to each unique query_id."""
    unique_ids = sorted(set(r["query_id"] for r in rows))
    return {qid: idx for idx, qid in enumerate(unique_ids)}


def write_csv(rows: list[dict], embedding_map: dict[str, int], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["question", "model", "isCorrect", "output_tokens", "embedding_id"])
        for r in rows:
            writer.writerow([
                r["query"],
                r["model_name"],
                r["performance"],
                r["output_tokens"],
                embedding_map[r["query_id"]],
            ])
    return len(rows)


def main():
    model_dirs = [d for d in POOL_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not model_dirs:
        print(f"ERROR: no model directories found in {POOL_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(model_dirs)} model directories:")
    for d in sorted(model_dirs):
        print(f"  {d.name}")

    print("\nCollecting train rows (train + val + cal splits)...")
    train_rows = collect_rows(model_dirs, TRAIN_SPLITS)

    print("Collecting test rows...")
    test_rows = collect_rows(model_dirs, TEST_SPLITS)

    all_rows = train_rows + test_rows
    print(f"\nTotal rows: {len(all_rows)} (train={len(train_rows)}, test={len(test_rows)})")

    embedding_map = build_embedding_map(all_rows)
    print(f"Unique questions (embedding_ids): {len(embedding_map)}")

    train_out = OUT_DIR / "train_v1.csv"
    test_out = OUT_DIR / "test_v1.csv"

    n_train = write_csv(train_rows, embedding_map, train_out)
    print(f"\nWrote {n_train} rows to {train_out}")

    n_test = write_csv(test_rows, embedding_map, test_out)
    print(f"Wrote {n_test} rows to {test_out}")

    print(f"\n{'split':5s}  {'model':30s}  {'rows':>6s}  {'tot_otokens':>12s}  {'mean_otokens':>12s}  {'med_otokens':>11s}")
    print("-" * 85)
    for label, rows in [("train", train_rows), ("test", test_rows)]:
        stats: dict[str, list[int]] = {}
        for r in rows:
            stats.setdefault(r["model_name"], []).append(int(r["output_tokens"]))
        for model in sorted(stats):
            tokens = stats[model]
            total = sum(tokens)
            mean = total / len(tokens)
            med = sorted(tokens)[len(tokens) // 2]
            print(f"  {label:5s}  {model:30s}  {len(tokens):>6d}  {total:>12,d}  {mean:>12.1f}  {med:>11,d}")


if __name__ == "__main__":
    main()
