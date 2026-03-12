"""LiteLLM adapter — strategy, proxy, and full serve mode.

Requires: pip install model-router-toolkit[litellm]
"""

from model_router_toolkit.adapters.litellm.strategy import ModelRoutingStrategy

__all__ = ["ModelRoutingStrategy"]
