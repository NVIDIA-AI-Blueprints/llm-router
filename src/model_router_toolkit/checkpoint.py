"""Checkpoint save/load for prefill routing.

Prefill checkpoints are PyTorch state dicts (.pt).

Security note: torch.load with weights_only=False can execute arbitrary
code. Only load checkpoints from trusted sources.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def detect_checkpoint_type(path: str | Path) -> str:
    """Detect checkpoint type from file extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pt":
        return "prefill"
    else:
        raise ValueError(f"Unknown checkpoint format: {suffix}. Expected .pt")


def load_checkpoint(path: str | Path):
    """Load a checkpoint and return the appropriate data structure.

    Warning: Only load checkpoints from trusted sources. torch.load can
    execute arbitrary code from malicious files.
    """
    import torch

    path = Path(path)
    detect_checkpoint_type(path)

    logger.debug("Loading torch checkpoint: %s (only load from trusted sources)", path)
    return torch.load(path, map_location="cpu", weights_only=False)
