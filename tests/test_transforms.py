"""Tests for prefill transform pipeline (pure math, no GPU needed)."""

import numpy as np
import torch

from model_router_toolkit.prefill.extract import PrefillResult
from model_router_toolkit.prefill.transforms import (
    apply_pipeline,
    build_features,
    fit_pca_pipeline,
    raw_hidden,
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
