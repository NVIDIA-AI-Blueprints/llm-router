"""Tests for prefill transform pipeline (pure math, no GPU needed)."""

import numpy as np
import pytest
import torch

from model_router_toolkit.prefill.extract import PrefillResult
from model_router_toolkit.prefill.transforms import (
    apply_pipeline,
    assemble_trunk_features,
    build_features,
    build_features_from_transform,
    fit_pca_pipeline,
    raw_features,
    raw_hidden,
    resolve_feature_layers,
)


def _synthetic_prefill_result(n_samples=100, hidden_dim=768, n_layers=24):
    """Create a synthetic PrefillResult with random hidden states."""
    half = n_layers // 2
    layers = list(range(half, n_layers))
    hidden_last = {li: torch.randn(n_samples, hidden_dim) for li in layers}
    hidden_mean = {li: torch.randn(n_samples, hidden_dim) for li in layers}
    return PrefillResult(
        hidden_last=hidden_last,
        hidden_mean=hidden_mean,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
    )


class TestTransforms:
    def test_raw_hidden_last(self):
        result = _synthetic_prefill_result(n_samples=50, hidden_dim=128, n_layers=8)
        layer = result.available_layers[0]
        arr = raw_hidden(result, layer, "last")
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (50, 128)

    def test_raw_hidden_mean(self):
        result = _synthetic_prefill_result(n_samples=50, hidden_dim=128, n_layers=8)
        layer = result.available_layers[0]
        arr = raw_hidden(result, layer, "mean")
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (50, 128)

    def test_fit_pca_pipeline_shapes(self):
        raw = np.random.randn(100, 768).astype(np.float32)
        train_mask = np.zeros(100, dtype=bool)
        train_mask[:80] = True
        scaler, pca, features = fit_pca_pipeline(raw, train_mask, pca_dim=50)
        assert features.shape == (100, 50)
        assert pca.n_components == 50

    def test_apply_pipeline_matches_fit(self):
        raw = np.random.randn(100, 768).astype(np.float32)
        train_mask = np.ones(100, dtype=bool)
        scaler, pca, features_fit = fit_pca_pipeline(raw, train_mask, pca_dim=50)
        features_apply = apply_pipeline(raw, scaler, pca)
        np.testing.assert_allclose(features_fit, features_apply, atol=1e-5)

    def test_build_features_end_to_end(self):
        result = _synthetic_prefill_result(n_samples=60, hidden_dim=256, n_layers=8)
        layer = result.available_layers[0]
        raw = raw_hidden(result, layer, "last")
        train_mask = np.ones(60, dtype=bool)
        scaler, pca, _ = fit_pca_pipeline(raw, train_mask, pca_dim=32)
        features = build_features(result, layer, "last", scaler, pca)
        assert features.shape == (60, 32)

    def test_fit_pca_uses_only_train_mask(self):
        rng = np.random.default_rng(42)
        raw = rng.standard_normal((100, 200)).astype(np.float32)
        train_mask = np.zeros(100, dtype=bool)
        train_mask[:60] = True
        scaler, pca, _ = fit_pca_pipeline(raw, train_mask, pca_dim=20)
        assert scaler.mean_.shape == (200,)
        assert scaler.n_samples_seen_ == 60

    def test_all_layer_meanpool_concatenation_order(self):
        result = PrefillResult(
            hidden_last={},
            hidden_mean={
                0: torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
                1: torch.tensor([[10.0, 11.0], [12.0, 13.0]]),
                2: torch.tensor([[20.0, 21.0], [22.0, 23.0]]),
            },
            n_layers=3,
            hidden_dim=2,
        )
        feature_spec = {
            "aggregation": "all_layers_concat",
            "layers": "all",
            "pooling": "mean",
            "hidden_state_indexing": "direct",
        }
        assert resolve_feature_layers(result, feature_spec) == [0, 1, 2]
        features = raw_features(result, feature_spec)
        expected = np.array(
            [
                [0.0, 1.0, 10.0, 11.0, 20.0, 21.0],
                [2.0, 3.0, 12.0, 13.0, 22.0, 23.0],
            ]
        )
        np.testing.assert_array_equal(features, expected)

    def test_all_layer_features_fail_on_missing_layer(self):
        result = PrefillResult(
            hidden_last={},
            hidden_mean={0: torch.randn(4, 3)},
            n_layers=2,
            hidden_dim=3,
        )
        feature_spec = {
            "aggregation": "all_layers_concat",
            "layers": [0, 1],
            "pooling": "mean",
            "hidden_state_indexing": "direct",
        }
        with pytest.raises(ValueError, match="missing mean layers"):
            raw_features(result, feature_spec)

    def test_feature_spec_transform_matches_fit_pipeline(self):
        rng = np.random.default_rng(42)
        result = PrefillResult(
            hidden_last={},
            hidden_mean={
                layer: torch.from_numpy(
                    rng.standard_normal((40, 6)).astype(np.float32)
                )
                for layer in range(4)
            },
            n_layers=4,
            hidden_dim=6,
        )
        feature_spec = {
            "aggregation": "all_layers_concat",
            "layers": [0, 1, 2, 3],
            "pooling": "mean",
            "hidden_state_indexing": "direct",
        }
        raw = raw_features(result, feature_spec)
        train_mask = np.ones(40, dtype=bool)
        scaler, pca, expected = fit_pca_pipeline(raw, train_mask, pca_dim=8)
        transform = {
            "layer": 0,
            "mode": "mean",
            "scaler": scaler,
            "pca": pca,
            "feature_spec": feature_spec,
        }
        actual = build_features_from_transform(result, transform)
        np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_inplace_pipeline_preserves_transform_parity(self):
        rng = np.random.default_rng(7)
        raw = rng.standard_normal((50, 30)).astype(np.float32)
        original = raw.copy()
        train_mask = np.ones(50, dtype=bool)

        scaler, pca, expected = fit_pca_pipeline(
            raw,
            train_mask,
            pca_dim=8,
            inplace=True,
        )

        assert not np.array_equal(raw, original)
        assert pca.svd_solver == "auto"
        assert pca.iterated_power == "auto"
        actual = pca.transform(scaler.transform(original.copy()))
        np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_shared_once_layout_does_not_repeat_feature_block(self):
        block = np.arange(20, dtype=np.float32).reshape(4, 5)
        assembled = assemble_trunk_features(
            {"model-a": block, "model-b": block},
            ["model-a", "model-b"],
            "shared_once",
        )
        assert assembled.shape == (4, 5)
        assert assembled is block

    def test_legacy_layout_still_concatenates_per_target(self):
        block_a = np.ones((4, 5), dtype=np.float32)
        block_b = np.zeros((4, 5), dtype=np.float32)
        assembled = assemble_trunk_features(
            {"model-a": block_a, "model-b": block_b},
            ["model-a", "model-b"],
            "per_target",
        )
        assert assembled.shape == (4, 10)
        np.testing.assert_array_equal(assembled[:, :5], block_a)
        np.testing.assert_array_equal(assembled[:, 5:], block_b)
