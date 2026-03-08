"""LiteLLM Proxy integration for Model Router Toolkit.

Wraps the litellm proxy server and injects ModelRoutingStrategy as
the custom routing strategy, replacing litellm's built-in strategies
(simple-shuffle, least-busy, etc.) with intelligent model routing.

Usage:
    model-router proxy --litellm-config litellm.yaml --router-config pool.yaml
"""

from __future__ import annotations

from model_router_toolkit.proxy.startup import start_proxy

__all__ = ["start_proxy"]
