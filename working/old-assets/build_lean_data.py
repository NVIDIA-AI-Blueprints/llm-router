#!/usr/bin/env python3
"""Build lean pre-transformed feature files from a trained checkpoint + full cache.

Applies the checkpoint's fitted scaler+PCA per model to the raw hidden states,
concatenates into the shared feature matrix, and saves alongside the transforms.
The output files can be used with `model-router train --features-from` to skip
extraction, sweep, and transform fitting — only MLP trunk training runs.

Usage:
    python scripts/build_lean_data.py \
        --checkpoint checkpoints/prefill_router.pt \
        --train-cache data/v1-9models-pool/train.pt \
        --test-cache data/v1-9models-pool/test.pt \
        --output-dir data/v1-9models-lean/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch


def build_features(cache_path: Path, ckpt: dict, output_path: Path) -> None:
    from model_router_toolkit.prefill.extract import PrefillResult
    from model_router_toolkit.prefill.transforms import apply_pipeline, raw_hidden

    print(f"  Loading cache: {cache_path}")
    prefill = PrefillResult.load(cache_path)

    model_names = ckpt["model_names"]
    transforms = ckpt["transforms"]

    feat_per_model = {}
    for mname in model_names:
        t = transforms[mname]
        raw = raw_hidden(prefill, t["layer"], t["mode"])
        feat_per_model[mname] = apply_pipeline(raw, t["scaler"], t["pca"])

    shared_feats = np.hstack([feat_per_model[m] for m in model_names])

    data = {
        "features": shared_feats,
        "model_names": model_names,
        "transforms": transforms,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    sz_mb = output_path.stat().st_size / 1e6
    n = shared_feats.shape[0]
    d = shared_feats.shape[1]
    print(f"  Saved {output_path} ({n} questions, {d} dims, {sz_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Build lean pre-transformed feature files for fast training.",
    )
    parser.add_argument("--checkpoint", required=True, help="Trained checkpoint (.pt)")
    parser.add_argument("--train-cache", required=True, help="Full train PrefillResult cache")
    parser.add_argument("--test-cache", default=None, help="Full test PrefillResult cache (optional)")
    parser.add_argument("--output-dir", required=True, help="Output directory for lean files")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    t0 = time.monotonic()
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_names = ckpt["model_names"]
    print(f"  {len(model_names)} models: {model_names}")

    print("\nBuilding train features:")
    build_features(Path(args.train_cache), ckpt, output_dir / "train_features.pt")

    if args.test_cache:
        print("\nBuilding test features:")
        build_features(Path(args.test_cache), ckpt, output_dir / "test_features.pt")

    elapsed = time.monotonic() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
