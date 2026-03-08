"""KMeans training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def train_kmeans(config: Any, data_path: str | Path, output_dir: str | Path) -> Path:
    raise NotImplementedError(
        "KMeans training not yet implemented. Pipeline:\n"
        "1. Embed all questions via embed client\n"
        "2. Fit KMeans (n_clusters=100)\n"
        "3. Compute per-cluster per-model accuracy\n"
        "4. Fit Platt calibrators\n"
        "5. Save pkl to output_dir"
    )
