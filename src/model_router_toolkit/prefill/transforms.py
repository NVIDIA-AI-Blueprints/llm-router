"""Feature transforms: raw hidden states -> PCA-reduced features.

Each target model has its own fitted StandardScaler + PCA pipeline
stored in the checkpoint.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from model_router_toolkit.prefill.extract import PrefillResult

RANDOM_STATE = 42


def raw_hidden(result: PrefillResult, layer: int, mode: str) -> np.ndarray:
    """Pull the right hidden state array from a PrefillResult as numpy."""
    tensor = result.hidden_mean[layer] if mode == "mean" else result.hidden_last[layer]
    if isinstance(tensor, torch.Tensor):
        return tensor.numpy()
    return tensor


def fit_pca_pipeline(
    raw: np.ndarray,
    train_mask: np.ndarray,
    pca_dim: int,
) -> tuple[StandardScaler, PCA, np.ndarray]:
    """Fit scaler + PCA on training rows, transform the full array."""
    n_comp = min(pca_dim, raw.shape[1], int(train_mask.sum()))
    scaler = StandardScaler().fit(raw[train_mask])
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE).fit(
        scaler.transform(raw[train_mask]),
    )
    features = pca.transform(scaler.transform(raw))
    return scaler, pca, features


def apply_pipeline(raw: np.ndarray, scaler: StandardScaler, pca: PCA) -> np.ndarray:
    """Transform with fitted scaler + PCA."""
    return pca.transform(scaler.transform(raw))


def build_features(
    result: PrefillResult,
    layer: int,
    mode: str,
    scaler: StandardScaler,
    pca: PCA,
) -> np.ndarray:
    """End-to-end: PrefillResult -> reduced feature array."""
    raw = raw_hidden(result, layer, mode)
    return apply_pipeline(raw, scaler, pca)
