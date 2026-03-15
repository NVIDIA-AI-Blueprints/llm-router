"""Model Router Toolkit -- intelligent LLM routing via prefill complexity analysis."""

__version__ = "0.1.0"

from model_router_toolkit.config import ModelSpec, PoolConfig, load_config
from model_router_toolkit.router import BaseRouter, CostEstimate, RoutingResult


def __getattr__(name: str):
    if name == "ModelRoutingStrategy":
        try:
            from model_router_toolkit.adapters.litellm.strategy import ModelRoutingStrategy
        except ImportError:
            raise ImportError(
                "ModelRoutingStrategy requires litellm. "
                "Install with: pip install 'model-router-toolkit[litellm]'"
            ) from None
        return ModelRoutingStrategy
    if name == "PrefillRouter":
        from model_router_toolkit.prefill.router import PrefillRouter

        return PrefillRouter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseRouter",
    "RoutingResult",
    "CostEstimate",
    "ModelRoutingStrategy",
    "PoolConfig",
    "ModelSpec",
    "load_config",
    "PrefillRouter",
]
