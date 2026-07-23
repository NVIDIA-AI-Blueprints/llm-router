"""Feature transforms: raw hidden states -> PCA-reduced features.

Supports legacy single-layer features and explicit multi-layer feature
specifications stored in newer checkpoints.
"""

from __future__ import annotations

from typing import Any

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
        return tensor.float().numpy()
    return tensor


def resolve_feature_layers(result: PrefillResult, feature_spec: dict[str, Any]) -> list[int]:
    """Resolve and validate the ordered layer list for a feature specification."""
    requested = feature_spec.get("layers", "all")
    if requested == "all":
        layers = list(range(result.n_layers))
    else:
        layers = [int(layer) for layer in requested]

    if not layers:
        raise ValueError("Feature specification resolved to an empty layer list")
    if len(set(layers)) != len(layers):
        raise ValueError(f"Feature specification contains duplicate layers: {layers}")
    if feature_spec.get("hidden_state_indexing", "direct") != "direct":
        raise ValueError("Only direct hidden-state indexing is supported")
    return layers


def raw_features(result: PrefillResult, feature_spec: dict[str, Any]) -> np.ndarray:
    """Construct raw features from an explicit checkpoint/config recipe."""
    aggregation = feature_spec.get("aggregation", "single_layer")
    pooling = feature_spec.get("pooling", "last")
    hidden = result.hidden_mean if pooling == "mean" else result.hidden_last
    layers = resolve_feature_layers(result, feature_spec)

    if aggregation == "single_layer":
        if len(layers) != 1:
            raise ValueError(
                "single_layer aggregation requires exactly one resolved layer; "
                f"got {layers}"
            )
    elif aggregation != "all_layers_concat":
        raise ValueError(f"Unknown feature aggregation: {aggregation!r}")

    missing = [layer for layer in layers if layer not in hidden]
    if missing:
        raise ValueError(
            f"Prefill result is missing {pooling} layers {missing}; "
            f"available={sorted(hidden)}"
        )

    arrays = [raw_hidden(result, layer, pooling) for layer in layers]
    n_rows = {array.shape[0] for array in arrays}
    hidden_dims = {array.shape[1] for array in arrays}
    if len(n_rows) != 1 or len(hidden_dims) != 1:
        raise ValueError(
            "Prefill layers have inconsistent shapes: "
            f"{[array.shape for array in arrays]}"
        )
    return arrays[0] if aggregation == "single_layer" else np.concatenate(arrays, axis=1)


def fit_pca_pipeline(
    raw: np.ndarray,
    train_mask: np.ndarray,
    pca_dim: int,
    *,
    inplace: bool = False,
    randomized: bool = False,
) -> tuple[StandardScaler, PCA, np.ndarray]:
    """Fit scaler + PCA on training rows, transform the full array.

    ``inplace`` is intended for newly allocated feature matrices whose original
    values are no longer needed. It avoids an additional full-size scaled copy.
    """
    n_comp = min(pca_dim, raw.shape[1], int(train_mask.sum()))
    all_rows_train = bool(train_mask.all())
    train_rows = raw if all_rows_train else raw[train_mask]

    scaler = StandardScaler(copy=not inplace).fit(train_rows)
    scaled = scaler.transform(raw, copy=not inplace)
    pca_kwargs: dict[str, Any] = {}
    if randomized:
        pca_kwargs = {"svd_solver": "randomized", "iterated_power": 5}
    pca = PCA(
        n_components=n_comp,
        random_state=RANDOM_STATE,
        **pca_kwargs,
    ).fit(
        scaled if all_rows_train else scaled[train_mask],
    )
    features = pca.transform(scaled)
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
    *,
    feature_spec: dict[str, Any] | None = None,
) -> np.ndarray:
    """End-to-end: PrefillResult -> reduced feature array."""
    raw = (
        raw_features(result, feature_spec)
        if feature_spec is not None
        else raw_hidden(result, layer, mode)
    )
    return apply_pipeline(raw, scaler, pca)


def build_features_from_transform(
    result: PrefillResult,
    transform: dict[str, Any],
) -> np.ndarray:
    """Apply a legacy or feature-spec transform from a checkpoint."""
    return build_features(
        result,
        int(transform.get("layer", 0)),
        str(transform.get("mode", "last")),
        transform["scaler"],
        transform["pca"],
        feature_spec=transform.get("feature_spec"),
    )


def assemble_trunk_features(
    feature_blocks: dict[str, np.ndarray],
    model_names: list[str],
    feature_layout: str,
) -> np.ndarray:
    """Assemble transformed feature blocks into the trunk input matrix."""
    if not model_names:
        raise ValueError("At least one target model is required")
    if feature_layout == "shared_once":
        first_model = model_names[0]
        if first_model not in feature_blocks:
            raise ValueError(f"Missing shared feature block for {first_model!r}")
        return feature_blocks[first_model]
    if feature_layout == "per_target":
        missing = [model for model in model_names if model not in feature_blocks]
        if missing:
            raise ValueError(f"Missing per-target feature blocks for {missing}")
        return np.hstack([feature_blocks[model] for model in model_names])
    raise ValueError(f"Unknown trunk feature layout: {feature_layout!r}")


def build_trunk_features(
    prefill_results: dict[str, PrefillResult],
    transforms: dict[str, dict[str, Any]],
    model_names: list[str],
    feature_layout: str,
) -> np.ndarray:
    """Build the trunk input identically for batch evaluation and scoring."""
    targets = model_names[:1] if feature_layout == "shared_once" else model_names
    blocks: dict[str, np.ndarray] = {}
    for model_name in targets:
        if model_name not in prefill_results:
            raise ValueError(f"Missing prefill result for {model_name!r}")
        if model_name not in transforms:
            raise ValueError(f"Missing transform for {model_name!r}")
        blocks[model_name] = build_features_from_transform(
            prefill_results[model_name],
            transforms[model_name],
        )
    return assemble_trunk_features(blocks, model_names, feature_layout)
