"""Model Router Toolkit -- intelligent LLM routing via KMeans or prefill strategies."""

__version__ = "0.1.0"

from model_router_toolkit.config import PoolConfig, ModelSpec, load_config
from model_router_toolkit.router import BaseRouter, RoutingResult, CostEstimate
from model_router_toolkit.strategy import ModelRoutingStrategy


def __getattr__(name: str):
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
