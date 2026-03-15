"""Evaluate every non-empty subset of the 9-model pool.

Loads checkpoint, labels, and prefill cache once, runs trunk inference once,
then slices columns for each of the 511 subsets. Writes a CSV with per-subset
metrics matching the full evaluation report.

Usage:
    python working/evals/subset_eval.py
"""

from __future__ import annotations

import csv
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from model_router_toolkit.evaluate import _build_shared_features, _load_labels
from model_router_toolkit.prefill.extract import PrefillResult
from model_router_toolkit.prefill.trunk import predict_proba, reconstruct_trunk

CHECKPOINT = ROOT / "checkpoints" / "prefill_router.pt"
DATA = ROOT / "data" / "test_v1.csv"
PREFILL_CACHE = ROOT / "data" / "v1-9models-pool" / "test.pt"
OUTPUT = ROOT / "working" / "evals" / "subset_eval_results.csv"


def compute_metrics(
    model_names_subset: list[str],
    Y: np.ndarray,
    probs: np.ndarray,
) -> dict[str, object]:
    """Compute all eval metrics for a given (Y, probs) subset."""
    N, n_models = Y.shape
    choices = np.argmax(probs, axis=1)
    n_correct = Y.sum(axis=1)

    per_model_auc: dict[str, float] = {}
    for mi, mname in enumerate(model_names_subset):
        try:
            per_model_auc[mname] = float(roc_auc_score(Y[:, mi], probs[:, mi]))
        except ValueError:
            per_model_auc[mname] = float("nan")

    oracle_acc = float(Y.max(axis=1).mean())
    model_accs = [float(Y[:, mi].mean()) for mi in range(n_models)]
    best_mi = int(np.argmax(model_accs))
    best_acc = model_accs[best_mi]
    best_name = model_names_subset[best_mi]
    headroom = oracle_acc - best_acc

    router_acc = float(np.mean([Y[i, choices[i]] for i in range(N)]))
    lift = router_acc - best_acc
    pct_headroom = (lift / headroom * 100) if headroom > 0 else 0.0

    dist = np.bincount(choices, minlength=n_models)

    # Per-model accuracy when chosen
    acc_when_chosen: dict[str, float] = {}
    for mi, mname in enumerate(model_names_subset):
        mask = choices == mi
        acc_when_chosen[mname] = float(Y[mask, mi].mean()) if mask.sum() > 0 else float("nan")

    # Agreement zones
    n_all_correct = int((n_correct == n_models).sum())
    n_disagree = int(((n_correct > 0) & (n_correct < n_models)).sum())
    n_all_wrong = int((n_correct == 0).sum())

    disagree_mask = (n_correct > 0) & (n_correct < n_models)
    if disagree_mask.sum() > 0:
        dz_choices = choices[disagree_mask]
        dz_Y = Y[disagree_mask]
        disagree_acc = float(
            np.mean([dz_Y[i, dz_choices[i]] for i in range(int(disagree_mask.sum()))])
        )
    else:
        disagree_acc = float("nan")

    # Near-miss stats (disagree zone, wrong routing)
    near_miss_count = 0
    flippable_count = 0
    if disagree_mask.sum() > 0:
        dz_probs = probs[disagree_mask]
        for i in range(int(disagree_mask.sum())):
            chosen = dz_choices[i]
            if dz_Y[i, chosen] == 0:
                near_miss_count += 1
                correct_models = np.where(dz_Y[i] == 1)[0]
                if len(correct_models) > 0:
                    gap = dz_probs[i, chosen] - dz_probs[i, correct_models].max()
                    if gap < 0.05:
                        flippable_count += 1

    return {
        "oracle_accuracy": oracle_acc,
        "best_single_accuracy": best_acc,
        "best_single_model": best_name,
        "router_accuracy": router_acc,
        "lift_pp": lift * 100,
        "headroom_pct": pct_headroom,
        "mean_auc": float(np.nanmean(list(per_model_auc.values()))),
        "n_all_correct": n_all_correct,
        "n_disagree": n_disagree,
        "n_all_wrong": n_all_wrong,
        "disagree_zone_accuracy": disagree_acc,
        "near_miss_wrong": near_miss_count,
        "near_miss_flippable": flippable_count,
        "per_model_auc": per_model_auc,
        "routing_dist": {mname: dist[mi] / N * 100 for mi, mname in enumerate(model_names_subset)},
        "acc_when_chosen": acc_when_chosen,
    }


def main() -> None:
    t0 = time.time()

    # ── Load once ─────────────────────────────────────────────────────
    print("Loading checkpoint...")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    ckpt["_path"] = str(CHECKPOINT)
    all_model_names: list[str] = ckpt["model_names"]
    n_all = len(all_model_names)
    print(f"  Models ({n_all}): {all_model_names}")

    print("Loading labels...")
    questions_raw, label_map = _load_labels(DATA)
    N = len(questions_raw)
    Y_all = np.zeros((N, n_all), dtype=int)
    for qi, (_, d) in enumerate(label_map.items()):
        for mi, mname in enumerate(all_model_names):
            Y_all[qi, mi] = d.get(mname, 0)
    print(f"  {N} questions loaded")

    print("Loading prefill cache...")
    pr = PrefillResult.load(PREFILL_CACHE)
    prefill_results = {mname: pr for mname in all_model_names}

    print("Building features & running trunk inference (once for all 9 models)...")
    shared_feats = _build_shared_features(ckpt, prefill_results, all_model_names)
    trunk_nets = reconstruct_trunk(ckpt, device="cpu")
    probs_all = predict_proba(trunk_nets, shared_feats, device="cpu")
    print(f"  probs shape: {probs_all.shape}")

    t_load = time.time() - t0
    print(f"Setup complete in {t_load:.1f}s\n")

    # ── CSV header ────────────────────────────────────────────────────
    base_cols = [
        "subset_id", "subset_size", "models",
        "n_questions", "oracle_accuracy", "best_single_accuracy",
        "best_single_model", "router_accuracy", "lift_pp", "headroom_pct",
        "mean_auc", "n_all_correct", "n_disagree", "n_all_wrong",
        "disagree_zone_accuracy", "near_miss_wrong", "near_miss_flippable",
    ]
    per_model_cols: list[str] = []
    for mname in all_model_names:
        per_model_cols.extend([
            f"{mname}__auc",
            f"{mname}__routed_pct",
            f"{mname}__acc_when_chosen",
        ])
    all_cols = base_cols + per_model_cols

    # ── Enumerate all non-empty subsets ───────────────────────────────
    total_subsets = 2**n_all - 1
    print(f"Evaluating {total_subsets} subsets...")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()

        subset_id = 0
        for size in range(1, n_all + 1):
            for combo in combinations(range(n_all), size):
                subset_id += 1
                idx = list(combo)
                model_subset = [all_model_names[i] for i in idx]
                Y_sub = Y_all[:, idx]
                probs_sub = probs_all[:, idx]

                metrics = compute_metrics(model_subset, Y_sub, probs_sub)

                row: dict[str, object] = {
                    "subset_id": subset_id,
                    "subset_size": size,
                    "models": "|".join(model_subset),
                    "n_questions": N,
                    "oracle_accuracy": f"{metrics['oracle_accuracy']:.6f}",
                    "best_single_accuracy": f"{metrics['best_single_accuracy']:.6f}",
                    "best_single_model": metrics["best_single_model"],
                    "router_accuracy": f"{metrics['router_accuracy']:.6f}",
                    "lift_pp": f"{metrics['lift_pp']:.4f}",
                    "headroom_pct": f"{metrics['headroom_pct']:.2f}",
                    "mean_auc": f"{metrics['mean_auc']:.6f}",
                    "n_all_correct": metrics["n_all_correct"],
                    "n_disagree": metrics["n_disagree"],
                    "n_all_wrong": metrics["n_all_wrong"],
                    "disagree_zone_accuracy": f"{metrics['disagree_zone_accuracy']:.6f}"
                        if not np.isnan(metrics["disagree_zone_accuracy"]) else "",
                    "near_miss_wrong": metrics["near_miss_wrong"],
                    "near_miss_flippable": metrics["near_miss_flippable"],
                }

                for mname in all_model_names:
                    auc_val = metrics["per_model_auc"].get(mname)
                    routed_val = metrics["routing_dist"].get(mname)
                    awc_val = metrics["acc_when_chosen"].get(mname)
                    row[f"{mname}__auc"] = f"{auc_val:.6f}" if auc_val is not None and not np.isnan(auc_val) else ""
                    row[f"{mname}__routed_pct"] = f"{routed_val:.2f}" if routed_val is not None else ""
                    row[f"{mname}__acc_when_chosen"] = f"{awc_val:.6f}" if awc_val is not None and not np.isnan(awc_val) else ""

                writer.writerow(row)

                if subset_id % 50 == 0 or subset_id == total_subsets:
                    elapsed = time.time() - t0
                    print(f"  [{subset_id:3d}/{total_subsets}] size={size}  ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\nDone. {subset_id} subsets evaluated in {elapsed:.1f}s")
    print(f"Results: {OUTPUT}")


if __name__ == "__main__":
    main()
