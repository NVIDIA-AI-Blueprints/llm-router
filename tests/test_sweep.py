"""Tests for hyperparameter sweep module."""

import numpy as np
import pytest
import torch

from model_router_toolkit.prefill.extract import PrefillResult
from model_router_toolkit.prefill.sweep import SweepResult, cv_auc, sweep_model


class TestSweep:
    def test_cv_auc_perfect_data(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 10))
        y = (X[:, 0] > 0).astype(float)
        auc = cv_auc(X, y, n_folds=5)
        assert auc > 0.9

    def test_cv_auc_random_data(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 10))
        y = rng.integers(0, 2, size=200).astype(float)
        auc = cv_auc(X, y, n_folds=5)
        assert 0.3 < auc < 0.7

    def test_sweep_model_returns_best(self):
        n_samples = 80
        hidden_dim = 64
        n_layers = 8
        half = n_layers // 2
        layers = list(range(half, n_layers))

        rng = np.random.default_rng(42)
        hidden_last = {li: torch.from_numpy(rng.standard_normal((n_samples, hidden_dim)).astype(np.float32)) for li in layers}
        hidden_mean = {li: torch.from_numpy(rng.standard_normal((n_samples, hidden_dim)).astype(np.float32)) for li in layers}

        result = PrefillResult(
            hidden_last=hidden_last,
            hidden_mean=hidden_mean,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
        )
        labels = rng.integers(0, 2, size=n_samples).astype(np.float64)
        train_mask = np.zeros(n_samples, dtype=bool)
        train_mask[:60] = True

        sr = sweep_model(
            result, labels, train_mask,
            layers=layers, modes=["last"], pca_dims=[16, 32],
        )
        assert isinstance(sr, SweepResult)
        assert sr.layer in layers
        assert sr.mode == "last"
        assert sr.pca_dim in [16, 32]
        assert 0.0 <= sr.cv_auc <= 1.0
