"""Prefill complexity scorer: loads checkpoint, runs extraction + MLP scoring.

Bridges PrefillExtractor, transforms, and SharedTrunkNet into a single
score() call that returns P(correct) and cost estimates per target model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model_router_toolkit.prefill.extract import PrefillExtractor, PrefillResult, detect_device
from model_router_toolkit.prefill.transforms import build_features
from model_router_toolkit.prefill.trunk import SharedTrunkNet, predict_proba, reconstruct_trunk
from model_router_toolkit.router import CostEstimate


@dataclass
class RawScores:
    model_names: list[str]
    confidences: list[float]
    costs: list[CostEstimate]


class PrefillScorer:
    """Loads a trained prefill checkpoint and scores questions."""

    def __init__(self, checkpoint_path: str | Path, *, config: Any = None):
        self._path = Path(checkpoint_path)
        self._config = config
        self._ckpt: dict | None = None
        self._trunk_nets: list[SharedTrunkNet] = []
        self._extractor: PrefillExtractor | None = None
        self.model_names: list[str] = []
        self._device = detect_device()

    def _ensure_loaded(self) -> None:
        if self._ckpt is not None:
            return

        self._ckpt = torch.load(self._path, map_location="cpu", weights_only=False)

        self.model_names = self._ckpt["model_names"]
        self._trunk_nets = reconstruct_trunk(self._ckpt, device=self._device)

        first_transform = next(iter(self._ckpt["transforms"].values()))
        encoder_path = first_transform["encoder"]

        self._extractor = PrefillExtractor(encoder_path, device=self._device)

    def _needed_layers(self) -> list[int]:
        """Collect all unique layers referenced by the transforms."""
        layers = set()
        for t in self._ckpt["transforms"].values():
            layers.add(t["layer"])
        return sorted(layers)

    def score(self, question: str) -> RawScores:
        self._ensure_loaded()

        needed_layers = self._needed_layers()

        # Cache extraction results by (encoder, template_kwargs) combo
        extraction_cache: dict[str, PrefillResult] = {}
        per_model_feats: dict[str, np.ndarray] = {}

        for mname in self.model_names:
            t = self._ckpt["transforms"][mname]
            encoder = t["encoder"]
            tpl_kwargs = t.get("chat_template_kwargs", {})
            cache_key = f"{encoder}:{sorted(tpl_kwargs.items())}"

            if cache_key not in extraction_cache:
                extraction_cache[cache_key] = self._extractor.extract(
                    question,
                    chat_template_kwargs=tpl_kwargs,
                    extract_layers=needed_layers,
                )

            result = extraction_cache[cache_key]
            feat = build_features(result, t["layer"], t["mode"], t["scaler"], t["pca"])
            per_model_feats[mname] = feat

        shared_feats = np.hstack([per_model_feats[m] for m in self.model_names])
        probs = predict_proba(self._trunk_nets, shared_feats, device=self._device)
        confidences = probs[0].tolist()

        costs = []
        cost_table = self._ckpt.get("cost_table", {})
        for mname in self.model_names:
            ct = cost_table.get(mname, {})

            pool_targets = self._ckpt.get("pool_config", {})
            if isinstance(pool_targets, dict):
                pool_targets = pool_targets.get("targets", [])
            rate_in = 0.0
            rate_out = 0.0
            for pt in pool_targets:
                if isinstance(pt, dict) and pt.get("name") == mname:
                    rate_in = pt.get("cost_per_m_input_tokens", 0.0)
                    rate_out = pt.get("cost_per_m_output_tokens", 0.0)
                    break

            median_out = int(ct.get("median_output_tokens", 500))
            est_in_tokens = len(question.split()) * 2
            est_out_cost = median_out * rate_out / 1_000_000
            est_in_cost = est_in_tokens * rate_in / 1_000_000

            costs.append(
                CostEstimate(
                    median_output_tokens=median_out,
                    cost_per_m_input_tokens=rate_in,
                    cost_per_m_output_tokens=rate_out,
                    estimated_input_tokens=est_in_tokens,
                    estimated_output_cost=est_out_cost,
                    estimated_input_cost=est_in_cost,
                    estimated_total_cost=est_in_cost + est_out_cost,
                )
            )

        return RawScores(
            model_names=self.model_names,
            confidences=confidences,
            costs=costs,
        )

    def unload(self) -> None:
        if self._extractor is not None:
            self._extractor.unload()
            self._extractor = None
        self._trunk_nets = []
        self._ckpt = None


def load_scorer(checkpoint_path: str | Path, *, config: Any = None) -> PrefillScorer:
    return PrefillScorer(checkpoint_path, config=config)
