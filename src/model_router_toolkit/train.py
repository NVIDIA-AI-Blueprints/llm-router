"""Unified training dispatcher for KMeans and prefill routing methods."""

from __future__ import annotations

import csv
from pathlib import Path

from model_router_toolkit.config import load_config


def run_train(
    config_path: str | Path,
    data_path: str | Path,
    output_dir: str | Path = "checkpoints/",
    **kwargs,
) -> None:
    config = load_config(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(data_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        required = {"question", "model", "isCorrect", "output_tokens"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must have columns: {required}")
        if not rows:
            raise ValueError("No training data found")

    method = config.routing.method.lower()
    if method == "kmeans":
        from model_router_toolkit.kmeans.train import train_kmeans

        checkpoint_path = train_kmeans(config, data_path, output_dir)
    elif method == "prefill":
        from model_router_toolkit.prefill.train import train_prefill

        checkpoint_path = train_prefill(config, data_path, output_dir)
    else:
        raise ValueError(f"Unknown routing method: {method!r}. Use 'kmeans' or 'prefill'.")

    print(f"Trained {method} router")
    print(f"Checkpoint: {checkpoint_path}")
