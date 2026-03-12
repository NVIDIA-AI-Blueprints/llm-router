"""Prefill-based routing: single forward pass -> hidden states -> MLP scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_router_toolkit.router import BaseRouter, CostEstimate, RoutingResult


class PrefillRouter(BaseRouter):
    """Routes queries by running prefill through encoder models and scoring with an MLP.

    This is a wrapper that adapts the complexity_router.ComplexityScorer to
    the BaseRouter interface. The actual extraction and scoring logic lives in
    the prefill/ subpackage modules (extract, trunk, transforms, sweep).
    """

    def __init__(self, *, config: Any = None):
        self._config = config
        self._scorer = None
        self._model_names: list[str] = []

    def load(self, checkpoint_path: str | Path) -> None:
        from model_router_toolkit.prefill.scorer import load_scorer

        self._scorer = load_scorer(checkpoint_path, config=self._config)
        self._model_names = self._scorer.model_names

    def route(self, question: str, *, tolerance: float = 0.20) -> RoutingResult:
        if self._scorer is None:
            raise RuntimeError("Router not loaded. Call load() first.")

        raw = self._scorer.score(question)

        p_max = max(raw.confidences)
        threshold = p_max - tolerance

        cost_sorted = sorted(
            zip(raw.model_names, raw.confidences, raw.costs),
            key=lambda x: x[2].cost_per_m_input_tokens,
        )

        selected = cost_sorted[-1][0]
        for name, conf, _ in cost_sorted:
            if conf >= threshold:
                selected = name
                break

        return RoutingResult(
            model_names=raw.model_names,
            confidences=raw.confidences,
            costs=raw.costs,
            selected_model=selected,
            metadata={
                "p_max": p_max,
                "threshold": threshold,
                "tolerance": tolerance,
            },
        )

    def unload(self) -> None:
        if self._scorer is not None:
            self._scorer.unload()
            self._scorer = None
