"""Feature transforms: raw hidden states -> PCA-reduced features.

Each target model has its own fitted StandardScaler + PCA pipeline
stored in the checkpoint.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from model_router_toolkit.prefill.extract import PrefillResult


def raw_hidden(result: PrefillResult, layer: int, mode: str) -> np.ndarray:
    """Pull the right hidden state array from a PrefillResult."""
    if mode == "mean":
        return result.hidden_mean[layer]
    return result.hidden_last[layer]


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
    """End-to-end: PrefillResult -> reduced feature array (1, pca_dim)."""
    raw = raw_hidden(result, layer, mode)
    return apply_pipeline(raw, scaler, pca)
