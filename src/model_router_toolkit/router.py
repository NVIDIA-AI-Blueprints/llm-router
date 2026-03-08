"""Base routing abstraction and result types.

All routing methods (KMeans, prefill) implement BaseRouter.
Downstream code (strategy, server, CLI) only depends on this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CostEstimate:
    median_output_tokens: int
    cost_per_m_input_tokens: float
    cost_per_m_output_tokens: float
    estimated_input_tokens: int = 0
    estimated_output_cost: float = 0.0
    estimated_input_cost: float = 0.0
    estimated_total_cost: float = 0.0


@dataclass
class RoutingResult:
    model_names: list[str]
    confidences: list[float]
    costs: list[CostEstimate]
    selected_model: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def selected_confidence(self) -> float:
        idx = self.model_names.index(self.selected_model)
        return self.confidences[idx]

    @property
    def selected_cost(self) -> CostEstimate:
        idx = self.model_names.index(self.selected_model)
        return self.costs[idx]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_names": self.model_names,
            "confidences": self.confidences,
            "selected_model": self.selected_model,
            "metadata": self.metadata,
        }


class BaseRouter(ABC):
    """Abstract base for all routing methods."""

    @abstractmethod
    def route(self, question: str, *, tolerance: float = 0.20) -> RoutingResult:
        """Score all models and select the best cost-efficient one above threshold."""
        ...

    @abstractmethod
    def load(self, checkpoint_path: str | Path) -> None:
        """Load a trained checkpoint (pkl or pt)."""
        ...

    def unload(self) -> None:
        """Release resources (GPU memory, model weights, etc.)."""
        pass
