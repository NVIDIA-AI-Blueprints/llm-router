"""Layer / mode / PCA grid-search: find best prefill configuration per model.

Uses ternary search over layers (unimodal AUC assumption) and grid
search over mode (last/mean) x PCA dimensions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from model_router_toolkit.prefill.extract import PrefillResult
from model_router_toolkit.prefill.transforms import fit_pca_pipeline, raw_hidden

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
DEFAULT_N_FOLDS = 5
DEFAULT_PCA_DIMS = [50, 100, 150, 200, 300]
DEFAULT_MODES = ["last", "mean"]


@dataclass
class SweepResult:
    """Best configuration found for one target model."""

    layer: int
    mode: str
    pca_dim: int
    cv_auc: float


def cv_auc(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = DEFAULT_N_FOLDS,
) -> float:
    """K-fold CV AUC using logistic regression."""
    if len(set(y)) < 2:
        return 0.5

    n_folds = min(n_folds, int(min(np.sum(y == 0), np.sum(y == 1))))
    if n_folds < 2:
        return 0.5

    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE,
    )
    aucs: list[float] = []
    for tr, va in skf.split(X, y):
        lr = LogisticRegression(
            max_iter=500, C=1.0, solver="lbfgs", random_state=RANDOM_STATE,
        )
        lr.fit(X[tr], y[tr])
        probs = lr.predict_proba(X[va])[:, 1]
        try:
            aucs.append(roc_auc_score(y[va], probs))
        except ValueError:
            pass

    return float(np.mean(aucs)) if aucs else 0.5


def _eval_layer(
    prefill: PrefillResult,
    layer: int,
    mode: str,
    pca_dim: int,
    train_mask: np.ndarray,
    y_train: np.ndarray,
    n_folds: int,
) -> float:
    """Evaluate a single (layer, mode, pca_dim) combo."""
    raw = raw_hidden(prefill, layer, mode)
    _, _, feats = fit_pca_pipeline(raw, train_mask, pca_dim)
    return cv_auc(feats[train_mask], y_train, n_folds=n_folds)


def _ternary_search_layer(
    prefill: PrefillResult,
    layers: list[int],
    mode: str,
    pca_dim: int,
    train_mask: np.ndarray,
    y_train: np.ndarray,
    n_folds: int,
) -> tuple[int, float]:
    """Ternary search over sorted layers for peak AUC (unimodal assumption)."""
    cache: dict[int, float] = {}

    def _eval(idx: int) -> float:
        li = layers[idx]
        if li not in cache:
            cache[li] = _eval_layer(
                prefill, li, mode, pca_dim, train_mask, y_train, n_folds,
            )
        return cache[li]

    lo, hi = 0, len(layers) - 1
    while hi - lo > 2:
        m1 = lo + (hi - lo) // 3
        m2 = hi - (hi - lo) // 3
        if _eval(m1) < _eval(m2):
            lo = m1 + 1
        else:
            hi = m2 - 1

    best_idx = lo
    best_auc = _eval(lo)
    for idx in range(lo + 1, hi + 1):
        a = _eval(idx)
        if a > best_auc:
            best_auc = a
            best_idx = idx

    return layers[best_idx], best_auc


def sweep_model(
    prefill: PrefillResult,
    labels: np.ndarray,
    train_mask: np.ndarray,
    *,
    layers: list[int] | None = None,
    modes: list[str] | None = None,
    pca_dims: list[int] | None = None,
    n_folds: int = DEFAULT_N_FOLDS,
    target_name: str = "",
) -> SweepResult:
    """Grid-search over mode x PCA dim, ternary-search over layers."""
    layers = layers or prefill.available_layers
    modes = modes or DEFAULT_MODES
    pca_dims = pca_dims or DEFAULT_PCA_DIMS
    y_train = labels[train_mask]

    n_combos = len(modes) * len(pca_dims)
    tag = target_name or "sweep"
    logger.info(
        "  [%s] %d layers x %d modes x %d PCA = %d combos",
        tag, len(layers), len(modes), len(pca_dims), n_combos,
    )

    best_auc = 0.0
    best_cfg: dict | None = None
    t0 = time.monotonic()

    for mi, mode in enumerate(modes):
        for pi, pd in enumerate(pca_dims):
            step = mi * len(pca_dims) + pi + 1
            layer, auc = _ternary_search_layer(
                prefill, layers, mode, pd, train_mask, y_train, n_folds,
            )
            logger.info(
                "    [%s] %d/%d  L%02d %4s PCA%3d  AUC=%.4f",
                tag, step, n_combos, layer, mode, pd, auc,
            )
            if auc > best_auc:
                best_auc = auc
                best_cfg = {"layer": layer, "mode": mode, "pca_dim": pd}

    elapsed = time.monotonic() - t0
    if best_cfg is None:
        raise RuntimeError("Sweep produced no results — check labels/data")

    logger.info(
        "  [%s] done in %.1fs -> best L%d %s PCA%d AUC=%.4f",
        tag, elapsed, best_cfg["layer"], best_cfg["mode"],
        best_cfg["pca_dim"], best_auc,
    )

    return SweepResult(
        layer=best_cfg["layer"],
        mode=best_cfg["mode"],
        pca_dim=best_cfg["pca_dim"],
        cv_auc=best_auc,
    )
