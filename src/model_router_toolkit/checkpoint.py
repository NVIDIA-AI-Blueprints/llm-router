"""Checkpoint save/load for both routing methods.

KMeans checkpoints are sklearn pickle files (.pkl).
Prefill checkpoints are PyTorch state dicts (.pt).

Security note: Both pickle and torch.load with weights_only=False can
execute arbitrary code. Only load checkpoints from trusted sources.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


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
    """Load a checkpoint and return the appropriate data structure.

    Warning: Only load checkpoints from trusted sources. Both pickle and
    torch.load can execute arbitrary code from malicious files.
    """
    path = Path(path)
    ctype = detect_checkpoint_type(path)

    if ctype == "kmeans":
        import pickle

        logger.debug("Loading pickle checkpoint: %s (only load from trusted sources)", path)
        with open(path, "rb") as f:
            return pickle.load(f)  # noqa: S301
    elif ctype == "prefill":
        import torch

        logger.debug("Loading torch checkpoint: %s (only load from trusted sources)", path)
        return torch.load(path, map_location="cpu", weights_only=False)
