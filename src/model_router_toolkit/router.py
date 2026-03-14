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


def extract_user_text(messages: list[dict[str, str]] | None) -> str:
    """Extract the last user message text from an OpenAI-format message list.

    Handles both plain string content and multipart content arrays.
    Returns empty string if no user message is found.
    """
    if not messages:
        return ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = [p.get("text", "") for p in content if p.get("type") == "text"]
                return " ".join(texts)
    return ""


class BaseRouter(ABC):
    """Abstract base for all routing methods."""

    @abstractmethod
    def route(
        self, question: str, *, tolerance: float = 0.20,
        models: list[str] | None = None,
    ) -> RoutingResult:
        """Score all models and select the best cost-efficient one above threshold.

        If *models* is provided, only those models are considered for the
        routing decision.  Confidence scores are still computed for the full
        pool so callers can inspect them.
        """
        ...

    @abstractmethod
    def load(self, checkpoint_path: str | Path) -> None:
        """Load a trained checkpoint (pkl or pt)."""
        ...

    def unload(self) -> None:
        """Release resources (GPU memory, model weights, etc.)."""
        pass

    def has_model(self, model_name: str) -> bool:
        """Check whether *model_name* is a known model in the pool.

        Subclasses that store a PoolConfig should override this.
        The default returns False so callers fall through to route().
        """
        return False

    def resolve(self, model_name: str) -> RoutingResult | None:
        """Return a RoutingResult that pins *model_name* without ML inference.

        Use this for "router-per-subagent" flows: call resolve() once at
        subagent start to lock a model, then send subsequent requests
        directly to that model.

        Returns None if the model is not in the pool (caller should
        fall through to route()).
        """
        return None
