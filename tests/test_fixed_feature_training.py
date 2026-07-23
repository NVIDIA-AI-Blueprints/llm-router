"""End-to-end tests for fixed all-layer prefill training."""

import csv

import numpy as np
import pytest
import torch

from model_router_toolkit.config import PoolConfig
from model_router_toolkit.prefill.extract import PrefillResult
from model_router_toolkit.prefill.train import train_prefill


def _write_complete_labels(path, n_rows: int) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "model", "isCorrect", "output_tokens"],
        )
        writer.writeheader()
        for row in range(n_rows):
            for model_index, model_name in enumerate(["model-a", "model-b"]):
                writer.writerow(
                    {
                        "question": f"question {row}",
                        "model": model_name,
                        "isCorrect": (row + model_index) % 2,
                        "output_tokens": 10 + row,
                    }
                )


def test_fixed_all_layer_training_builds_versioned_checkpoint(
    tmp_path,
    monkeypatch,
):
    n_rows = 40
    n_layers = 4
    hidden_dim = 6
    rng = np.random.default_rng(42)
    prefill = PrefillResult(
        hidden_last={},
        hidden_mean={
            layer: torch.from_numpy(
                rng.standard_normal((n_rows, hidden_dim)).astype(np.float32)
            )
            for layer in range(n_layers)
        },
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        metadata={
            "resolved_layers": list(range(n_layers)),
            "pooling_modes": ["mean"],
            "hidden_state_indexing": "direct",
        },
    )

    def fake_run_extraction(*args, **kwargs):
        assert kwargs["extract_layers"] == "all"
        assert kwargs["pooling_modes"] == ["mean"]
        assert kwargs["hidden_state_indexing"] == "direct"
        return prefill

    monkeypatch.setattr(
        "model_router_toolkit.prefill.train.run_extraction",
        fake_run_extraction,
    )

    labels_path = tmp_path / "train.csv"
    _write_complete_labels(labels_path, n_rows)
    config = PoolConfig.model_validate(
        {
            "routing": {
                "method": "prefill",
                "encoder": "test/encoder",
                "features": {
                    "aggregation": "all_layers_concat",
                    "layers": "all",
                    "pooling": "mean",
                    "pca_dim": 5,
                    "hidden_state_indexing": "direct",
                },
            },
            "models": [
                {"name": "model-a"},
                {"name": "model-b"},
            ],
        }
    )

    checkpoint_path = train_prefill(
        config,
        labels_path,
        tmp_path / "output",
        device="cpu",
        n_seeds=2,
        n_keep=1,
        epochs=2,
        patience=1,
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    assert checkpoint["version"] == 3
    assert checkpoint["trunk_config"]["feature_layout"] == "shared_once"
    assert checkpoint["trunk_config"]["feature_width"] == 5
    assert checkpoint["trunk_config"]["d_in"] == 5

    transform_a = checkpoint["transforms"]["model-a"]
    transform_b = checkpoint["transforms"]["model-b"]
    assert transform_a["feature_spec"] == {
        "aggregation": "all_layers_concat",
        "layers": [0, 1, 2, 3],
        "pooling": "mean",
        "pca_dim": 5,
        "hidden_state_indexing": "direct",
    }
    assert transform_a["pca"].components_.shape == (5, n_layers * hidden_dim)
    assert transform_a["scaler"] is transform_b["scaler"]
    assert transform_a["pca"] is transform_b["pca"]


def test_fixed_feature_training_rejects_unattainable_pca_width(
    tmp_path,
    monkeypatch,
):
    n_rows = 10
    prefill = PrefillResult(
        hidden_last={},
        hidden_mean={0: torch.randn(n_rows, 4)},
        n_layers=1,
        hidden_dim=4,
    )
    monkeypatch.setattr(
        "model_router_toolkit.prefill.train.run_extraction",
        lambda *args, **kwargs: prefill,
    )

    labels_path = tmp_path / "train.csv"
    _write_complete_labels(labels_path, n_rows)
    config = PoolConfig.model_validate(
        {
            "routing": {
                "encoder": "test/encoder",
                "features": {
                    "aggregation": "all_layers_concat",
                    "layers": "all",
                    "pooling": "mean",
                    "pca_dim": 20,
                },
            },
            "models": [{"name": "model-a"}, {"name": "model-b"}],
        }
    )

    with pytest.raises(ValueError, match="PCA dimension 20 exceeds"):
        train_prefill(
            config,
            labels_path,
            tmp_path / "output",
            device="cpu",
            n_seeds=1,
            n_keep=1,
            epochs=1,
            patience=1,
        )
