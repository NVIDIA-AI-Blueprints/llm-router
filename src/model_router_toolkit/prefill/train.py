"""Prefill training pipeline: config + labeled CSV -> checkpoint.

Runs the full pipeline: extract prefill features, sweep layer/mode/PCA,
train a SharedTrunkNet ensemble, and save a self-contained checkpoint.

Usage (via CLI):
    model-router train --config configs/v1-9models-qwen08b.yaml --data train.csv
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from model_router_toolkit.config import PoolConfig
from model_router_toolkit.prefill.extract import (
    normalize_question,
    run_extraction,
)
from model_router_toolkit.prefill.sweep import SweepResult, sweep_model
from model_router_toolkit.prefill.transforms import fit_pca_pipeline, raw_hidden
from model_router_toolkit.prefill.trunk import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_TRUNK_HIDDEN,
    DEFAULT_WEIGHT_DECAY,
    SharedTrunkNet,
    predict_proba,
    train_ensemble,
)

logger = logging.getLogger(__name__)

DEFAULT_N_SEEDS = 10
DEFAULT_N_KEEP = 5
TRUNK_EPOCHS = 150
TRUNK_PATIENCE = 15


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------


def _load_labels(csv_path: str | Path) -> dict[str, dict[str, Any]]:
    """Read labeled CSV -> ``{normalized_question: {model: isCorrect, ...}}``."""
    rows: dict[str, dict[str, Any]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {"question", "model", "isCorrect"}
        if not required.issubset(fields):
            missing = required - fields
            raise ValueError(f"CSV missing required columns: {missing}")

        for row in reader:
            q = normalize_question(row["question"])
            model = row["model"]
            if q not in rows:
                rows[q] = {"_raw_question": row["question"]}
            rows[q][model] = int(row["isCorrect"])
            if "output_tokens" in row and row["output_tokens"]:
                try:
                    rows[q][f"{model}_osl"] = int(row["output_tokens"])
                except ValueError:
                    pass
    return rows


def _compute_output_token_stats(
    label_data: dict[str, dict[str, Any]],
    model_name: str,
) -> dict[str, float]:
    """Compute median/mean/p25/p75 output token stats for one model."""
    osl_key = f"{model_name}_osl"
    tokens = [d[osl_key] for d in label_data.values() if osl_key in d]
    if not tokens:
        return {}
    arr = np.array(tokens, dtype=float)
    return {
        "median_output_tokens": float(np.median(arr)),
        "mean_output_tokens": float(np.mean(arr)),
        "p25_output_tokens": float(np.percentile(arr, 25)),
        "p75_output_tokens": float(np.percentile(arr, 75)),
    }


# ---------------------------------------------------------------------------
# Checkpoint building
# ---------------------------------------------------------------------------


def _build_checkpoint(
    config: PoolConfig,
    model_names: list[str],
    transforms: dict[str, dict[str, Any]],
    trunk_state_dicts: list[dict[str, torch.Tensor]],
    trunk_config: dict[str, Any],
    cost_table: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Build a self-contained checkpoint dict."""
    active = set(model_names)
    pool_config = {
        "encoders": [{"hf_path": config.routing.encoder}],
        "targets": [
            {
                "name": m.name,
                "encoder": config.routing.encoder,
                "cost_per_m_input_tokens": m.cost_per_m_input_tokens,
                "cost_per_m_output_tokens": m.cost_per_m_output_tokens,
            }
            for m in config.models
            if m.name in active
        ],
    }

    ckpt_transforms: dict[str, Any] = {}
    for mname in model_names:
        t = transforms[mname]
        ckpt_transforms[mname] = {
            "scaler": t["scaler"],
            "pca": t["pca"],
            "layer": t["layer"],
            "mode": t["mode"],
            "pca_dim": t["pca_dim"],
            "encoder": t["encoder"],
            "chat_template_kwargs": t.get("chat_template_kwargs", {}),
        }

    return {
        "version": 2,
        "pool_config": pool_config,
        "model_names": model_names,
        "transforms": ckpt_transforms,
        "shared_trunk": trunk_state_dicts,
        "trunk_config": trunk_config,
        "cost_table": cost_table,
    }


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def train_prefill(
    config: PoolConfig,
    data_path: str | Path,
    output_dir: str | Path,
    *,
    mode: str = "auto",
    device: str | None = None,
    batch_size: int = 4,
    n_seeds: int = DEFAULT_N_SEEDS,
    n_keep: int = DEFAULT_N_KEEP,
    models: list[str] | None = None,
    pca_dims: list[int] | None = None,
    epochs: int = TRUNK_EPOCHS,
    patience: int = TRUNK_PATIENCE,
) -> Path:
    """Full prefill training pipeline: extract -> sweep -> train -> save.

    Returns the path to the saved checkpoint.
    """
    from model_router_toolkit.prefill.extract import detect_device

    dev = device or detect_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder = config.routing.encoder
    if not encoder:
        raise ValueError(
            "Config routing.encoder must be set for prefill training (e.g., 'Qwen/Qwen3.5-0.8B')"
        )
    encoder_tpl: dict[str, Any] = {}
    all_model_names = config.model_names
    if models:
        unknown = set(models) - set(all_model_names)
        if unknown:
            raise ValueError(f"Models not in config: {unknown}")
        model_names = [m for m in all_model_names if m in set(models)]
    else:
        model_names = all_model_names

    print("", flush=True)
    print("=" * 60)
    print("  MODEL ROUTER — PREFILL TRAINING")
    print("=" * 60)
    print(f"  Encoder:  {encoder}")
    print(f"  Targets:  {model_names}")
    print(f"  Device:   {dev}")
    print(f"  Ensemble: {n_seeds} seeds, keep {n_keep}")
    print("", flush=True)

    # ── 1. Load labels ────────────────────────────────────────────────
    print("  [1/6] Loading labels...")
    label_data = _load_labels(data_path)
    questions_raw = [d["_raw_question"] for d in label_data.values()]
    N = len(questions_raw)

    n_models = len(model_names)
    Y = np.zeros((N, n_models), dtype=int)
    for qi, (_, data) in enumerate(label_data.items()):
        for mi, mname in enumerate(model_names):
            Y[qi, mi] = data.get(mname, 0)

    train_mask = np.ones(N, dtype=bool)
    print(f"         {N} unique questions, {n_models} targets")
    for mi, mname in enumerate(model_names):
        print(f"         {mname}: acc={Y[:, mi].mean():.3f}")

    # ── 2. Output token stats ─────────────────────────────────────────
    cost_table: dict[str, dict[str, float]] = {}
    for m in config.models:
        if m.name not in set(model_names):
            continue
        stats = _compute_output_token_stats(label_data, m.name)
        if stats:
            cost_table[m.name] = stats

    # ── 3–5. Extract, sweep, transform ─────────────────────────────
    print()
    print("  [2/6] Extracting prefill features...")
    prefill = run_extraction(
        encoder,
        questions_raw,
        chat_template_kwargs=encoder_tpl,
        device=dev,
        batch_size=batch_size,
        cache_dir="cache/",
    )

    print(flush=True)
    print("  [3/6] Sweeping layer/mode/PCA per target...", flush=True)
    sweep_results: dict[str, SweepResult] = {}
    for mi, mname in enumerate(model_names):
        sweep_results[mname] = sweep_model(
            prefill,
            Y[:, mi],
            train_mask,
            pca_dims=pca_dims,
            target_name=mname,
        )

    print()
    print("  [4/6] Fitting transforms...")
    transforms: dict[str, dict[str, Any]] = {}
    feat_per_model: dict[str, np.ndarray] = {}

    for mname in model_names:
        sr = sweep_results[mname]
        raw = raw_hidden(prefill, sr.layer, sr.mode)
        scaler, pca, feats = fit_pca_pipeline(raw, train_mask, sr.pca_dim)
        transforms[mname] = {
            "scaler": scaler,
            "pca": pca,
            "layer": sr.layer,
            "mode": sr.mode,
            "pca_dim": sr.pca_dim,
            "encoder": encoder,
            "chat_template_kwargs": encoder_tpl,
        }
        feat_per_model[mname] = feats
        print(
            f"         {mname}: L{sr.layer} {sr.mode} PCA{sr.pca_dim} (AUC={sr.cv_auc:.4f})",
        )

    shared_feats = np.hstack([feat_per_model[m] for m in model_names])

    # ── 6. Train shared trunk ─────────────────────────────────────────
    print()
    print("  [5/6] Training shared trunk ensemble...")
    d_shared = shared_feats.shape[1]
    print(
        f"         Shared input dim: {d_shared} "
        f"({' + '.join(str(feat_per_model[m].shape[1]) for m in model_names)})",
    )

    trunk_hidden = DEFAULT_TRUNK_HIDDEN
    trunk_nets = train_ensemble(
        lambda: SharedTrunkNet(d_shared, n_models, hidden=trunk_hidden),
        shared_feats[train_mask],
        Y[train_mask].astype(np.float32),
        n_seeds=n_seeds,
        n_keep=n_keep,
        device=dev,
        lr=DEFAULT_LR,
        epochs=epochs,
        batch_size=DEFAULT_BATCH_SIZE,
        patience=patience,
        weight_decay=DEFAULT_WEIGHT_DECAY,
    )
    print(f"         Trained {n_seeds}-seed ensemble, kept {len(trunk_nets)}")

    trunk_probs = predict_proba(trunk_nets, shared_feats, device=dev)
    for mi, mname in enumerate(model_names):
        try:
            auc = roc_auc_score(Y[:, mi], trunk_probs[:, mi])
        except ValueError:
            auc = float("nan")
        print(f"         {mname}: AUC={auc:.4f}")

    # ── 7. Save checkpoint ────────────────────────────────────────────
    print()
    print("  [6/6] Saving checkpoint...")
    trunk_state_dicts = [net.cpu().state_dict() for net in trunk_nets]
    trunk_config = {
        "d_in": d_shared,
        "n_outputs": n_models,
        "hidden": list(trunk_hidden),
    }

    ckpt = _build_checkpoint(
        config,
        model_names,
        transforms,
        trunk_state_dicts,
        trunk_config,
        cost_table,
    )

    ckpt_path = output_dir / "prefill_router.pt"
    torch.save(ckpt, ckpt_path)
    sz_mb = ckpt_path.stat().st_size / 1e6
    print(f"         Checkpoint: {ckpt_path} ({sz_mb:.1f} MB)")

    print()
    print("=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print()

    return ckpt_path
