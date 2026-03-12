"""Model Router Toolkit -- intelligent LLM routing via KMeans or prefill strategies."""

__version__ = "0.1.0"

from model_router_toolkit.config import PoolConfig, ModelSpec, load_config
from model_router_toolkit.router import BaseRouter, RoutingResult, CostEstimate


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
    if name == "KMeansRouter":
        from model_router_toolkit.kmeans.router import KMeansRouter
        return KMeansRouter
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
    "KMeansRouter",
    "PrefillRouter",
]
