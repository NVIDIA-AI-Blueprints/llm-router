"""Checkpoint save/load for both routing methods.

KMeans checkpoints are sklearn pickle files (.pkl).
Prefill checkpoints are PyTorch state dicts (.pt).
"""

from __future__ import annotations

from pathlib import Path


def detect_checkpoint_type(path: str | Path) -> str:
    """Detect checkpoint type from file extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        return "kmeans"
    elif suffix == ".pt":
        return "prefill"
    else:
        raise ValueError(f"Unknown checkpoint format: {suffix}. Expected .pkl or .pt")


def load_checkpoint(path: str | Path):
    """Load a checkpoint and return the appropriate data structure."""
    path = Path(path)
    ctype = detect_checkpoint_type(path)

    if ctype == "kmeans":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
    elif ctype == "prefill":
        import torch
        return torch.load(path, map_location="cpu", weights_only=False)
