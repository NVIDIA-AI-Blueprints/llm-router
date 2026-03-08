"""LiteLLM custom routing strategy wrapping any BaseRouter.

This is the primary integration point. Usage:

    from litellm import Router
    from model_router_toolkit import ModelRoutingStrategy

    router = Router(model_list=my_models)
    strategy = ModelRoutingStrategy.from_config("pool_config.yaml")
    router.set_custom_routing_strategy(strategy)
"""

from __future__ import annotations

import threading
from typing import Any

from model_router_toolkit.router import BaseRouter, RoutingResult


class ModelRoutingStrategy:
    """Wraps a BaseRouter and implements the LiteLLM custom routing interface.

    Implements async_get_available_deployment() and get_available_deployment()
    as required by litellm.router.CustomRoutingStrategyBase.
    """

    def __init__(
        self,
        router: BaseRouter,
        *,
        tolerance: float = 0.20,
    ):
        self._router = router
        self._tolerance = tolerance
        self._litellm_router: Any = None
        self._local = threading.local()

    @classmethod
    def from_config(cls, config_path: str, **kwargs: Any) -> ModelRoutingStrategy:
        """Build a strategy from a pool_config.yaml file."""
        from model_router_toolkit.config import load_config, build_router_from_config

        config = load_config(config_path)
        router = build_router_from_config(config)
        return cls(router, tolerance=config.routing.tolerance, **kwargs)

    @property
    def tolerance(self) -> float:
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value: float) -> None:
        self._tolerance = max(0.0, min(1.0, value))

    @property
    def last_result(self) -> RoutingResult | None:
        return getattr(self._local, "last_result", None)

    @property
    def router(self) -> BaseRouter:
        return self._router

    def _extract_user_text(self, messages: list[dict[str, str]] | None) -> str:
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

    def _find_deployment(self, model_name: str) -> dict | None:
        if self._litellm_router is None:
            return None
        for dep in self._litellm_router.model_list:
            if isinstance(dep, dict) and dep.get("model_name") == model_name:
                return dep
        return None

    def _route_and_select(
        self,
        model: str,
        messages: list[dict[str, str]] | None = None,
        input: str | list | None = None,
        **kwargs: Any,
    ) -> dict:
        text = self._extract_user_text(messages)
        if not text and isinstance(input, str):
            text = input

        if not text:
            self._local.last_result = None
            if self._litellm_router:
                return self._litellm_router.model_list[0]
            return {}

        result = self._router.route(text, tolerance=self._tolerance)
        self._local.last_result = result

        dep = self._find_deployment(result.selected_model)
        if dep:
            return dep

        if self._litellm_router:
            return self._litellm_router.model_list[0]
        return {}

    async def async_get_available_deployment(
        self,
        model: str,
        messages: list[dict[str, str]] | None = None,
        input: str | list | None = None,
        specific_deployment: bool | None = False,
        request_kwargs: dict | None = None,
    ) -> dict:
        return self._route_and_select(model, messages, input)

    def get_available_deployment(
        self,
        model: str,
        messages: list[dict[str, str]] | None = None,
        input: str | list | None = None,
        specific_deployment: bool | None = False,
        request_kwargs: dict | None = None,
    ) -> dict:
        return self._route_and_select(model, messages, input)

    def set_litellm_router(self, litellm_router: Any) -> None:
        """Called internally when plugged into a litellm.Router."""
        self._litellm_router = litellm_router
