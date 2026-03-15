"""Shared utilities for HTTP server application factories."""

from __future__ import annotations

import logging
import time
from typing import Any

from model_router_toolkit.config import PoolConfig
from model_router_toolkit.router import BaseRouter

logger = logging.getLogger(__name__)


def warmup_router(router: BaseRouter, config: PoolConfig) -> None:
    """Pre-load encoder and run a dummy route so the first real request is fast."""
    print(f"Warming up {config.routing.method} router...")
    t0 = time.time()
    try:
        result = router.route("warmup test query", tolerance=0.5)
        elapsed = time.time() - t0
        print(f"  Warmup complete in {elapsed:.1f}s")
        print(f"  Models: {result.model_names}")
        print(
            f"  Test route -> {result.selected_model} "
            f"(confidences: {', '.join(f'{c:.3f}' for c in result.confidences)})"
        )
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  Warmup failed after {elapsed:.1f}s: {e}")
        logger.warning("Router warmup failed: %s", e)


def health_dict(config: PoolConfig, *, mode: str = "full") -> dict[str, Any]:
    """Build the /health response payload."""
    return {
        "status": "ok",
        "mode": mode,
        "method": config.routing.method,
        "models": config.model_names,
    }


def models_list(config: PoolConfig) -> list[dict[str, Any]]:
    """Build the /api/models response payload."""
    return [
        {
            "name": m.name,
            "display_name": m.display_name or m.name,
            "cost_per_m_input_tokens": m.cost_per_m_input_tokens,
            "cost_per_m_output_tokens": m.cost_per_m_output_tokens,
        }
        for m in config.models
    ]
