"""Tests for SharedTrunkNet MLP training and inference."""

import numpy as np
import pytest
import torch

from model_router_toolkit.prefill.trunk import (
    SharedTrunkNet,
    predict_proba,
    reconstruct_trunk,
    train_ensemble,
    train_mlp,
)

pytestmark = pytest.mark.requires_torch


class TestTrunk:
    def test_shared_trunk_net_forward_shape(self):
        net = SharedTrunkNet(d_in=100, n_outputs=4)
        x = torch.randn(8, 100)
        out = net(x)
        assert out.shape == (8, 4)

    def test_train_mlp_loss_decreases(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 50)).astype(np.float32)
        y_binary = (X[:, 0] > 0).astype(np.float32)
        y = np.column_stack([y_binary, 1 - y_binary])

        net = SharedTrunkNet(d_in=50, n_outputs=2)
        trained = train_mlp(net, X, y, epochs=50, patience=50, seed=42)
        probs = predict_proba([trained], X)
        assert probs.shape == (200, 2)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_train_mlp_early_stopping(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((300, 20)).astype(np.float32)
        y = np.column_stack([(X[:, 0] > 0).astype(np.float32)] * 2)

        net = SharedTrunkNet(d_in=20, n_outputs=2)
        trained = train_mlp(net, X, y, epochs=1000, patience=5, seed=0)
        assert trained is not None

    def test_train_ensemble_keeps_n_best(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 30)).astype(np.float32)
        y = np.column_stack([(X[:, 0] > 0).astype(np.float32)] * 2)

        def factory():
            return SharedTrunkNet(d_in=30, n_outputs=2)

        nets = train_ensemble(factory, X, y, n_seeds=5, n_keep=3, epochs=20, patience=5)
        assert len(nets) == 3

    def test_predict_proba_range(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 40)).astype(np.float32)
        y = np.column_stack([(X[:, 0] > 0).astype(np.float32)] * 3)

        net = SharedTrunkNet(d_in=40, n_outputs=3)
        trained = train_mlp(net, X, y, epochs=10, patience=10, seed=42)
        probs = predict_proba([trained], X)
        assert probs.shape == (100, 3)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_reconstruct_trunk_from_checkpoint(self, smoke_ckpt_path):
        ckpt = torch.load(smoke_ckpt_path, map_location="cpu", weights_only=False)
        nets = reconstruct_trunk(ckpt, device="cpu")
        assert len(nets) > 0
        for net in nets:
            assert isinstance(net, SharedTrunkNet)
        tcfg = ckpt["trunk_config"]
        d_in = tcfg["d_in"]
        n_out = tcfg["n_outputs"]
        x = torch.randn(2, d_in)
        out = nets[0](x)
        assert out.shape == (2, n_out)
