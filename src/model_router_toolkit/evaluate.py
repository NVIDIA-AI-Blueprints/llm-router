"""Unified evaluation for routing checkpoints.

Runs batch prefill extraction and produces rich diagnostics:
per-model AUC, oracle/router accuracy, agreement zones, near-miss
analysis, and pairwise win rates.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from model_router_toolkit.config import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared data loading
# ---------------------------------------------------------------------------

def _load_labels(
    data_path: str | Path,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Load CSV -> (unique raw questions, {norm_q: {model: isCorrect, _raw: q}})."""
    from model_router_toolkit.prefill.extract import normalize_question

    label_map: dict[str, dict[str, Any]] = {}
    with open(data_path) as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {"question", "model", "isCorrect"}
        if not required.issubset(fields):
            missing = required - fields
            raise ValueError(f"CSV missing required columns: {missing}")

        for row in reader:
            q_norm = normalize_question(row["question"])
            model = row["model"].strip()
            is_correct = int(row["isCorrect"])
            if q_norm not in label_map:
                label_map[q_norm] = {"_raw": row["question"]}
            label_map[q_norm][model] = is_correct

    questions_raw = [d["_raw"] for d in label_map.values()]
    return questions_raw, label_map


# ---------------------------------------------------------------------------
# Prefill evaluation (batch extraction + trunk)
# ---------------------------------------------------------------------------

def _build_shared_features(
    ckpt: dict[str, Any],
    prefill_results: dict,
    model_names: list[str],
) -> np.ndarray:
    """Apply checkpoint transforms and stack into the shared feature matrix."""
    from model_router_toolkit.prefill.transforms import apply_pipeline

    feat: dict[str, np.ndarray] = {}
    for mname in model_names:
        t = ckpt["transforms"][mname]
        pr = prefill_results.get(mname)
        if pr is None:
            raise RuntimeError(
                f"No prefill result for target '{mname}' "
                f"(encoder='{t.get('encoder', '?')}')"
            )
        if t["mode"] == "mean":
            import torch
            raw = pr.hidden_mean[t["layer"]]
            raw = raw.float().numpy() if isinstance(raw, torch.Tensor) else raw
        else:
            import torch
            raw = pr.hidden_last[t["layer"]]
            raw = raw.float().numpy() if isinstance(raw, torch.Tensor) else raw
        feat[mname] = apply_pipeline(raw, t["scaler"], t["pca"])

    return np.hstack([feat[m] for m in model_names])


def _print_eval_report(
    model_names: list[str],
    Y: np.ndarray,
    probs: np.ndarray,
    ckpt: dict[str, Any],
) -> dict[str, Any]:
    """Compute and print the full evaluation report."""
    N, n_models = Y.shape
    choices = np.argmax(probs, axis=1)
    n_correct = Y.sum(axis=1)

    # Per-model AUC
    per_model_auc: dict[str, float] = {}
    for mi, mname in enumerate(model_names):
        try:
            per_model_auc[mname] = float(roc_auc_score(Y[:, mi], probs[:, mi]))
        except ValueError:
            per_model_auc[mname] = float("nan")

    # Accuracy metrics
    oracle_acc = float(Y.max(axis=1).mean())
    best_mi = int(np.argmax([Y[:, mi].mean() for mi in range(n_models)]))
    best_acc = float(Y[:, best_mi].mean())
    best_name = model_names[best_mi]
    headroom = oracle_acc - best_acc

    router_acc = float(np.mean([Y[i, choices[i]] for i in range(N)]))
    lift = router_acc - best_acc
    pct_headroom = (lift / headroom * 100) if headroom > 0 else 0.0

    # Routing distribution
    dist = np.bincount(choices, minlength=n_models)

    # ── Print ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  EVALUATION REPORT")
    print("=" * 60)

    ckpt_path = ckpt.get("_path", "")
    if ckpt_path:
        print(f"  Checkpoint: {ckpt_path}")
    print(f"  Questions:  {N}")

    if "trunk_config" in ckpt:
        tc = ckpt["trunk_config"]
        trunk_size = len(ckpt.get("shared_trunk", []))
        print(
            f"  Trunk:      d_in={tc['d_in']}, "
            f"hidden={tc['hidden']}, ensemble={trunk_size}",
        )

    # Transforms summary
    if "transforms" in ckpt:
        print()
        print("  Transforms:")
        for mname in model_names:
            t = ckpt["transforms"][mname]
            enc = t.get("encoder", "")
            enc_short = enc.split("/")[-1] if enc else ""
            print(
                f"    {mname:20s}: L{t['layer']} {t['mode']} "
                f"PCA{t['pca_dim']} ({enc_short})",
            )

    # Per-model metrics
    print()
    print(f"  {'Model':20s}  {'Accuracy':>8s}  {'AUC':>7s}")
    print(f"  {'-' * 20}  {'-' * 8}  {'-' * 7}")
    for mi, mname in enumerate(model_names):
        print(
            f"  {mname:20s}  {Y[:, mi].mean():8.4f}  "
            f"{per_model_auc[mname]:7.4f}",
        )

    # Summary
    print()
    print(f"  Oracle:       {oracle_acc:.4f}")
    print(f"  Best single:  {best_acc:.4f} ({best_name})")
    print(f"  Headroom:     {(oracle_acc - best_acc) * 100:.1f}pp")
    print()
    print(f"  Router (argmax):")
    print(
        f"    Accuracy:     {router_acc:.4f} ({lift * 100:+.2f}pp, "
        f"{pct_headroom:.1f}% headroom captured)",
    )

    # Distribution
    print("    Distribution:")
    for mi, mname in enumerate(model_names):
        n_routed = dist[mi]
        pct = n_routed / N * 100
        local_acc = Y[choices == mi, mi].mean() if n_routed > 0 else 0
        print(
            f"      {mname:20s}: {n_routed:4d} ({pct:5.1f}%)  "
            f"acc_when_chosen={local_acc:.4f}",
        )

    # Agreement zones
    zones = [
        ("All correct", n_correct == n_models),
        ("Disagree", (n_correct > 0) & (n_correct < n_models)),
        ("All wrong", n_correct == 0),
    ]
    print()
    print("  By agreement zone:")
    for zone_name, mask in zones:
        zn = int(mask.sum())
        if zn == 0:
            continue
        zone_choices = choices[mask]
        zone_Y = Y[mask]
        zone_acc = float(
            np.mean([zone_Y[i, zone_choices[i]] for i in range(zn)]),
        )
        zone_dist = np.bincount(zone_choices, minlength=n_models)
        dist_str = "  ".join(
            f"{m}={zone_dist[mi]}" for mi, m in enumerate(model_names)
        )
        print(
            f"    {zone_name:15s} ({zn:4d}, {zn / N * 100:5.1f}%): "
            f"acc={zone_acc:.4f}  [{dist_str}]",
        )

    # Deep analysis
    _print_deep_analysis(model_names, Y, probs, choices, n_correct)

    print()

    return {
        "n_questions": N,
        "model_names": model_names,
        "per_model_auc": per_model_auc,
        "oracle_accuracy": oracle_acc,
        "best_single_accuracy": best_acc,
        "best_single_model": best_name,
        "router_accuracy": router_acc,
        "lift_pp": lift * 100,
        "headroom_pct": pct_headroom,
    }


def _print_deep_analysis(
    model_names: list[str],
    Y: np.ndarray,
    probs: np.ndarray,
    choices: np.ndarray,
    n_correct: np.ndarray,
) -> None:
    """Deep routing diagnostics: near-miss and pairwise analysis."""
    N, n_models = Y.shape
    disagree = (n_correct > 0) & (n_correct < n_models)

    if disagree.sum() == 0:
        return

    print()
    print("=" * 60)
    print("  DEEP ROUTING ANALYSIS")
    print("=" * 60)

    # Near-miss analysis
    dz_Y = Y[disagree]
    dz_probs = probs[disagree]
    dz_choices = choices[disagree]
    dz_n = int(disagree.sum())

    gaps = []
    for i in range(dz_n):
        chosen = dz_choices[i]
        if dz_Y[i, chosen] == 1:
            continue
        correct_models = np.where(dz_Y[i] == 1)[0]
        if len(correct_models) == 0:
            continue
        best_correct_conf = dz_probs[i, correct_models].max()
        chosen_conf = dz_probs[i, chosen]
        gaps.append(chosen_conf - best_correct_conf)

    if gaps:
        gaps_arr = np.array(gaps)
        n_wrong = len(gaps_arr)
        n_flippable = int((gaps_arr < 0.05).sum())
        n_tiny = int((gaps_arr < 0.02).sum())
        print()
        print(f"  Near-miss (disagree zone, wrong routing):")
        print(
            f"    Wrong decisions: {n_wrong}/{dz_n} "
            f"({n_wrong / dz_n * 100:.1f}%)",
        )
        print(
            f"    Confidence gap (chosen_wrong - best_correct): "
            f"mean={gaps_arr.mean():.4f}  median={np.median(gaps_arr):.4f}",
        )
        print(
            f"    Flippable (gap < 0.05): {n_flippable} "
            f"({n_flippable / n_wrong * 100:.1f}%)",
        )
        print(
            f"    Tiny gap   (gap < 0.02): {n_tiny} "
            f"({n_tiny / n_wrong * 100:.1f}%)",
        )

    # Pairwise win rates (only for small model pools)
    if n_models <= 6:
        print()
        print("  Pairwise confidence win rates (disagree zone):")
        print(
            f"  When A correct & B wrong, P(conf_A > conf_B):",
        )
        header = f"    {'':20s}"
        for mname in model_names:
            header += f"  {mname[:8]:>8s}"
        print(header + "  (B wrong)")
        for ai, aname in enumerate(model_names):
            line = f"    {aname:20s}"
            for bi, bname in enumerate(model_names):
                if ai == bi:
                    line += f"  {'---':>8s}"
                    continue
                mask = disagree & (Y[:, ai] == 1) & (Y[:, bi] == 0)
                if mask.sum() == 0:
                    line += f"  {'N/A':>8s}"
                    continue
                win_rate = float(
                    (probs[mask, ai] > probs[mask, bi]).mean(),
                )
                line += f"  {win_rate:8.3f}"
            print(line + "  (A correct)")


def _run_prefill_evaluate(
    checkpoint_path: str | Path,
    data_path: str | Path,
    *,
    device: str = "cpu",
    batch_size: int = 4,
    prefill_dir: str | Path | None = None,
    prefill_cache: str | Path | None = None,
    hf_cache_dir: str | None = None,
    features_from: str | Path | None = None,
    models: list[str] | None = None,
) -> dict[str, Any]:
    """Rich prefill evaluation: batch extraction + trunk + full metrics."""
    import torch

    from model_router_toolkit.prefill.extract import PrefillResult
    from model_router_toolkit.prefill.trunk import predict_proba, reconstruct_trunk

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt["_path"] = str(checkpoint_path)
    all_model_names = ckpt["model_names"]
    n_all = len(all_model_names)

    # Load labels
    questions_raw, label_map = _load_labels(data_path)
    N = len(questions_raw)
    Y_all = np.zeros((N, n_all), dtype=int)
    for qi, (_, d) in enumerate(label_map.items()):
        for mi, mname in enumerate(all_model_names):
            Y_all[qi, mi] = d.get(mname, 0)

    print(f"  Loaded {N} questions for {n_all} targets")

    if features_from:
        print(f"  Loading pre-transformed features: {features_from}")
        feat_data = torch.load(features_from, map_location="cpu", weights_only=False)
        shared_feats = feat_data["features"]
        if isinstance(shared_feats, torch.Tensor):
            shared_feats = shared_feats.numpy()
    elif prefill_cache:
        print(f"  Loading prefill cache: {prefill_cache}")
        pr = PrefillResult.load(prefill_cache)
        prefill_results = {mname: pr for mname in all_model_names}
        shared_feats = _build_shared_features(ckpt, prefill_results, all_model_names)
    else:
        from model_router_toolkit.prefill.extract import extract_from_checkpoint

        print("  Extracting prefill features...")
        prefill_results = extract_from_checkpoint(
            ckpt, questions_raw,
            device=device, batch_size=batch_size,
            cache_dir=prefill_dir, hf_cache_dir=hf_cache_dir,
        )
        shared_feats = _build_shared_features(ckpt, prefill_results, all_model_names)

    print("  Running trunk inference...")
    trunk_nets = reconstruct_trunk(ckpt, device=device)
    probs_all = predict_proba(trunk_nets, shared_feats, device=device)

    # Subset filtering
    if models:
        unknown = set(models) - set(all_model_names)
        if unknown:
            raise ValueError(f"Models not in checkpoint: {unknown}")
        subset_idx = [all_model_names.index(m) for m in models]
        model_names = [all_model_names[i] for i in subset_idx]
        Y = Y_all[:, subset_idx]
        probs = probs_all[:, subset_idx]
        print(f"  Filtering to {len(models)} models: {models}")
    else:
        model_names = all_model_names
        Y = Y_all
        probs = probs_all

    return _print_eval_report(model_names, Y, probs, ckpt)


# ---------------------------------------------------------------------------
# Fallback: BaseRouter evaluation (generic)
# ---------------------------------------------------------------------------

def _run_baserouter_evaluate(
    config_path: str | Path,
    checkpoint_path: str | Path,
    data_path: str | Path,
    *,
    models: list[str] | None = None,
) -> None:
    """Per-question evaluation through the BaseRouter interface."""
    config = load_config(config_path)
    config.routing.checkpoint = str(checkpoint_path)

    from model_router_toolkit.config import build_router_from_config

    router = build_router_from_config(config)

    questions: list[str] = []
    seen: set[str] = set()
    by_question: dict[str, dict[str, tuple[bool, int]]] = defaultdict(dict)

    with open(data_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("question", "").strip()
            model = row.get("model", "").strip()
            is_correct = str(row.get("isCorrect", "0")).lower() in (
                "1", "true", "yes",
            )
            out_tokens = 0
            try:
                out_tokens = int(row.get("output_tokens", 0))
            except ValueError:
                pass
            if q and model:
                by_question[q][model] = (is_correct, out_tokens)
                if q not in seen:
                    seen.add(q)
                    questions.append(q)

    if not questions:
        print("No test data found.")
        return

    model_names = config.model_names
    router_correct = 0
    routing_counts: dict[str, int] = defaultdict(int)

    for q in questions:
        result = router.route(q, tolerance=config.routing.tolerance, models=models)
        selected = result.selected_model
        routing_counts[selected] += 1
        qdata = by_question[q]
        if qdata.get(selected, (False, 0))[0]:
            router_correct += 1

    n = len(questions)
    print()
    print("=" * 60)
    print("  EVALUATION REPORT (BaseRouter)")
    print("=" * 60)
    print(f"  Test samples: {n}")
    print(f"  Router accuracy: {router_correct / n:.2%}")
    print()
    print("  Routing distribution:")
    for m in model_names:
        pct = routing_counts[m] / n * 100 if n else 0
        print(f"    {m}: {pct:.1f}%")
    print()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_evaluate(
    config_path: str | Path,
    checkpoint_path: str | Path,
    data_path: str | Path,
    **kwargs,
) -> None:
    """Evaluate a routing checkpoint. Dispatches by method."""
    config = load_config(config_path)
    method = config.routing.method.lower()

    if method == "prefill":
        _run_prefill_evaluate(checkpoint_path, data_path, **kwargs)
    else:
        models = kwargs.get("models")
        _run_baserouter_evaluate(
            config_path, checkpoint_path, data_path, models=models,
        )
