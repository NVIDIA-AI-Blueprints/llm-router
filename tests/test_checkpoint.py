"""Tests for checkpoint loading and type detection."""

import pytest

from model_router_toolkit.checkpoint import detect_checkpoint_type, load_checkpoint


class TestCheckpoint:
    def test_detect_pkl_type(self):
        assert detect_checkpoint_type("model.pkl") == "kmeans"

    def test_detect_pt_type(self):
        assert detect_checkpoint_type("model.pt") == "prefill"

    def test_detect_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown checkpoint format"):
            detect_checkpoint_type("model.safetensors")

    def test_load_pkl_checkpoint(self, pkl_path):
        db = load_checkpoint(pkl_path)
        assert isinstance(db, dict)
        assert "kmeans_model" in db
        assert "models" in db
        assert "cluster_acc" in db
        assert isinstance(db["models"], list)
        assert len(db["models"]) > 0

    def test_load_pt_checkpoint(self, prefill_ckpt_path):
        ckpt = load_checkpoint(prefill_ckpt_path)
        assert isinstance(ckpt, dict)
        assert "model_names" in ckpt
        assert "transforms" in ckpt
        assert "shared_trunk" in ckpt
        assert isinstance(ckpt["model_names"], list)
        assert len(ckpt["model_names"]) > 0
