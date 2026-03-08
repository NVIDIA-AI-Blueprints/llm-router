"""Prefill training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def train_prefill(config: Any, data_path: str | Path, output_dir: str | Path) -> Path:
    raise NotImplementedError(
        "Prefill training not yet implemented. Pipeline:\n"
        "1. Extract prefill hidden states via encoder models\n"
        "2. Sweep layer/mode/PCA per target model\n"
        "3. Train SharedTrunkNet MLP ensemble\n"
        "4. Save .pt checkpoint + serve.yaml"
    )
