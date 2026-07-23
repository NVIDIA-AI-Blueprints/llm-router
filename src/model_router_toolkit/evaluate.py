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
# Cost-coverage curve helpers (ported from prefill-complexity-router)
# ---------------------------------------------------------------------------


def _pareto_frontier(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Non-dominated (cost, accuracy) points sorted by cost (ascending)."""
    pareto: list[tuple[float, float]] = []
    best_acc = -1.0
    for cost, acc in sorted(points, key=lambda x: x[0]):
        if acc > best_acc:
            pareto.append((cost, acc))
            best_acc = acc
    return pareto


def _route_at_tolerance(
    probs: np.ndarray,
    costs_arr: np.ndarray,
    tol: float,
) -> np.ndarray:
    """Per-query routing: cheapest model with p >= p_max - tol."""
    p_max = probs.max(axis=1, keepdims=True)
    eligible = probs >= (p_max - tol)
    cost_matrix = np.where(eligible, costs_arr[np.newaxis, :], np.inf)
    return np.argmin(cost_matrix, axis=1)


def _build_routing_curve(
    Y: np.ndarray,
    probs: np.ndarray,
    model_names: list[str],
    costs: dict[str, float],
) -> tuple[list[tuple[float, float]], list[dict[str, float]]]:
    """Cost-accuracy curve by sweeping tolerance over observed probability gaps.

    Returns (curve, distributions) where curve is deduplicated (avg_cost, accuracy)
    points sorted by cost, and distributions[i] is {model_name: pct_routed} for
    each curve point.
    """
    N = len(Y)
    costs_arr = np.array([costs[mn] for mn in model_names])

    p_max = probs.max(axis=1, keepdims=True)
    gaps = (p_max - probs).ravel()
    tol_values = np.unique(gaps)

    curve: list[tuple[float, float]] = []
    dists: list[dict[str, float]] = []
    seen: set[tuple[float, float]] = set()

    for tol in tol_values:
        choices = _route_at_tolerance(probs, costs_arr, float(tol))
        avg_cost = float(np.mean(costs_arr[choices]))
        acc = float(np.mean([Y[i, choices[i]] for i in range(N)]))
        key = (round(avg_cost, 10), round(acc, 10))
        if key not in seen:
            seen.add(key)
            counts = np.bincount(choices, minlength=len(model_names))
            dist = {mn: round(float(counts[mi]) / N, 6) for mi, mn in enumerate(model_names)}
            dist["_tol"] = float(tol)
            curve.append((avg_cost, acc))
            dists.append(dist)

    order = sorted(range(len(curve)), key=lambda i: curve[i][0])
    return [curve[i] for i in order], [dists[i] for i in order]


def _padded_auc(
    curve: list[tuple[float, float]],
    c_min: float,
    c_max: float,
    floor_acc: float,
) -> float:
    """Area under *curve* padded to [c_min, c_max], normalised by range.

    Floor rule: for c < curve_min, accuracy = floor_acc (cheapest model).
    Plateau rule: for c > curve_max, accuracy = curve's rightmost value.
    """
    if c_max <= c_min:
        return 0.0
    xs = [p[0] for p in curve]
    ys = [p[1] for p in curve]

    padded_x: list[float] = []
    padded_y: list[float] = []

    if not xs or c_min < xs[0]:
        padded_x.append(c_min)
        padded_y.append(floor_acc)

    for x, y in zip(xs, ys):
        if c_min <= x <= c_max:
            padded_x.append(x)
            padded_y.append(y)

    rightmost = ys[-1] if ys else floor_acc
    if not xs or c_max > xs[-1]:
        padded_x.append(c_max)
        padded_y.append(rightmost)

    if len(padded_x) < 2:
        return (padded_y[0] - floor_acc) if padded_y else 0.0
    above_floor = [y - floor_acc for y in padded_y]
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(_trapz(above_floor, padded_x)) / (c_max - c_min)


def _build_cost_map(
    ckpt: dict[str, Any],
    model_names: list[str],
    config: Any | None = None,
) -> dict[str, float] | None:
    """Resolve per-model cost from checkpoint pool_config, falling back to config.

    Returns {model_name: cost_per_m_input_tokens} or None when costs are
    unavailable or all zero.
    """
    costs: dict[str, float] = {}

    pool_config = ckpt.get("pool_config", [])
    if isinstance(pool_config, list):
        for entry in pool_config:
            if isinstance(entry, dict):
                name = entry.get("name", "")
                cost = entry.get("cost_per_m_input_tokens", 0.0)
                if name and cost:
                    costs[name] = cost

    if config is not None:
        for m in getattr(config, "models", []):
            if m.name not in costs and m.cost_per_m_input_tokens > 0:
                costs[m.name] = m.cost_per_m_input_tokens

    if not all(mn in costs for mn in model_names):
        return None
    if all(costs[mn] == 0 for mn in model_names):
        return None
    return {mn: costs[mn] for mn in model_names}


def _load_pricing_csv(
    pricing_path: str | Path,
) -> dict[str, float]:
    """Load a pricing CSV with columns (model, cost_per_m_input_tokens)."""
    costs: dict[str, float] = {}
    with open(pricing_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("model", "").strip()
            cost = float(row.get("cost_per_m_input_tokens", 0))
            if name and cost > 0:
                costs[name] = cost
    return costs


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
    from model_router_toolkit.prefill.transforms import build_trunk_features

    feature_layout = ckpt.get("trunk_config", {}).get(
        "feature_layout",
        "per_target",
    )
    return build_trunk_features(
        prefill_results,
        ckpt["transforms"],
        model_names,
        feature_layout,
    )


def _print_eval_report(
    model_names: list[str],
    Y: np.ndarray,
    probs: np.ndarray,
    ckpt: dict[str, Any],
    config: Any | None = None,
    pricing: str | Path | None = None,
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
            f"  Trunk:      d_in={tc['d_in']}, hidden={tc['hidden']}, ensemble={trunk_size}",
        )

    # Transforms summary
    if "transforms" in ckpt:
        print()
        print("  Transforms:")
        for mname in model_names:
            t = ckpt["transforms"][mname]
            enc = t.get("encoder", "")
            enc_short = enc.split("/")[-1] if enc else ""
            feature_spec = t.get("feature_spec")
            if feature_spec:
                layers = feature_spec.get("layers", [])
                layer_summary = (
                    f"{layers[0]}..{layers[-1]}" if layers else "none"
                )
                print(
                    f"    {mname:20s}: {feature_spec['aggregation']} "
                    f"L{layer_summary} {feature_spec['pooling']} "
                    f"PCA{t['pca_dim']} ({enc_short})",
                )
            else:
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
            f"  {mname:20s}  {Y[:, mi].mean():8.4f}  {per_model_auc[mname]:7.4f}",
        )

    # Summary
    print()
    print(f"  Oracle:       {oracle_acc:.4f}")
    print(f"  Best single:  {best_acc:.4f} ({best_name})")
    print(f"  Headroom:     {(oracle_acc - best_acc) * 100:.1f}pp")
    print()
    print("  Router (argmax):")
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
            f"      {mname:20s}: {n_routed:4d} ({pct:5.1f}%)  acc_when_chosen={local_acc:.4f}",
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
        dist_str = "  ".join(f"{m}={zone_dist[mi]}" for mi, m in enumerate(model_names))
        print(
            f"    {zone_name:15s} ({zn:4d}, {zn / N * 100:5.1f}%): "
            f"acc={zone_acc:.4f}  [{dist_str}]",
        )

    # Deep analysis
    _print_deep_analysis(model_names, Y, probs, choices, n_correct)

    # ── Cost-coverage curve (P-AUCCC) ─────────────────────────────────
    p_auccc = pareto_auccc = pdp_auccc = mdp_auccc = None
    routing_curve = None

    if pricing:
        pricing_costs = _load_pricing_csv(pricing)
        costs_map = {mn: pricing_costs[mn] for mn in model_names if mn in pricing_costs} or None
        if costs_map and not all(mn in costs_map for mn in model_names):
            missing = [mn for mn in model_names if mn not in costs_map]
            logger.warning("Pricing CSV missing models: %s", missing)
            costs_map = None
    else:
        costs_map = _build_cost_map(ckpt, model_names, config)
    if costs_map:
        c_min = min(costs_map.values())
        c_max = max(costs_map.values())
        cheapest_mi = int(np.argmin([costs_map[mn] for mn in model_names]))
        floor_acc = float(Y[:, cheapest_mi].mean())

        model_points = [
            (costs_map[mn], float(Y[:, mi].mean())) for mi, mn in enumerate(model_names)
        ]
        routing_curve, _ = _build_routing_curve(Y, probs, model_names, costs_map)
        pareto_curve = _pareto_frontier(model_points)
        combined_pareto_curve = _pareto_frontier(model_points + routing_curve)

        p_auccc = _padded_auc(routing_curve, c_min, c_max, floor_acc)
        pareto_auccc = _padded_auc(pareto_curve, c_min, c_max, floor_acc)
        combined_pareto_auccc = _padded_auc(
            combined_pareto_curve,
            c_min,
            c_max,
            floor_acc,
        )
        pdp_auccc = combined_pareto_auccc - p_auccc
        mdp_auccc = p_auccc - pareto_auccc

        print()
        print("  Cost Coverage (P-AUCCC):")
        print(f"    Model-only Pareto AUCCC : {pareto_auccc:.4f}")
        print(f"    Router P-AUCCC          : {p_auccc:.4f}")
        print(f"    MDP-AUCCC (lift)        : {mdp_auccc:+.4f}")
        print(f"    PDP-AUCCC (gap)         : {pdp_auccc:+.4f}")

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
        "p_auccc": p_auccc,
        "pareto_auccc": pareto_auccc,
        "mdp_auccc": mdp_auccc,
        "pdp_auccc": pdp_auccc,
        "routing_curve": routing_curve,
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
        print("  Near-miss (disagree zone, wrong routing):")
        print(
            f"    Wrong decisions: {n_wrong}/{dz_n} ({n_wrong / dz_n * 100:.1f}%)",
        )
        print(
            f"    Confidence gap (chosen_wrong - best_correct): "
            f"mean={gaps_arr.mean():.4f}  median={np.median(gaps_arr):.4f}",
        )
        print(
            f"    Flippable (gap < 0.05): {n_flippable} ({n_flippable / n_wrong * 100:.1f}%)",
        )
        print(
            f"    Tiny gap   (gap < 0.02): {n_tiny} ({n_tiny / n_wrong * 100:.1f}%)",
        )

    # Pairwise win rates (only for small model pools)
    if n_models <= 6:
        print()
        print("  Pairwise confidence win rates (disagree zone):")
        print(
            "  When A correct & B wrong, P(conf_A > conf_B):",
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
    config: Any | None = None,
    device: str = "cpu",
    batch_size: int = 4,
    models: list[str] | None = None,
    pricing: str | Path | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Rich prefill evaluation: batch extraction + trunk + full metrics."""
    import torch

    from model_router_toolkit.prefill.extract import extract_from_checkpoint
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

    print("  Extracting prefill features...")
    prefill_results = extract_from_checkpoint(
        ckpt,
        questions_raw,
        device=device,
        batch_size=batch_size,
        cache_dir="cache/",
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

    report = _print_eval_report(
        model_names,
        Y,
        probs,
        ckpt,
        config=config,
        pricing=pricing,
    )

    if output:
        import json as _json

        serializable = {
            k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in report.items()
        }
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(_json.dumps(serializable, indent=2, default=str))
        print(f"\n  Report written to {output}")

    return report


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
                "1",
                "true",
                "yes",
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
) -> dict | None:
    """Evaluate a routing checkpoint. Dispatches by method."""
    config = load_config(config_path)
    method = config.routing.method.lower()

    if method == "prefill":
        return _run_prefill_evaluate(checkpoint_path, data_path, config=config, **kwargs)
    else:
        models = kwargs.get("models")
        _run_baserouter_evaluate(
            config_path,
            checkpoint_path,
            data_path,
            models=models,
        )
        return None
