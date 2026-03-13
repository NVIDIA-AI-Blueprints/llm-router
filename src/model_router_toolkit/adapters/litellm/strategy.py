"""LiteLLM custom routing strategy wrapping any BaseRouter.

This is the primary integration point. Usage:

    from litellm import Router
    from model_router_toolkit import ModelRoutingStrategy

    router = Router(model_list=my_models)
    strategy = ModelRoutingStrategy.from_config("pool_config.yaml")
    router.set_custom_routing_strategy(strategy)
"""

from __future__ import annotations

import asyncio
import contextvars
from typing import Any

from model_router_toolkit.router import BaseRouter, RoutingResult, extract_user_text

_request_tolerance: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "request_tolerance", default=None,
)


class ModelRoutingStrategy:
    """Wraps a BaseRouter and implements the LiteLLM custom routing interface.

    Implements async_get_available_deployment() and get_available_deployment()
    as required by litellm.router.CustomRoutingStrategyBase.

    Tolerance can be overridden per-request via set_request_tolerance() which
    uses contextvars for async-safe, per-request scoping.
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
        self._last_result: RoutingResult | None = None

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

    def set_request_tolerance(self, value: float) -> None:
        """Set tolerance for the current async request context only."""
        _request_tolerance.set(max(0.0, min(1.0, value)))

    @property
    def effective_tolerance(self) -> float:
        """Tolerance for the current request: per-request override or default."""
        val = _request_tolerance.get()
        return val if val is not None else self._tolerance

    @property
    def last_result(self) -> RoutingResult | None:
        return self._last_result

    @property
    def router(self) -> BaseRouter:
        return self._router

    def _extract_user_text(self, messages: list[dict[str, str]] | None) -> str:
        return extract_user_text(messages)

    def _find_deployment(self, model_name: str) -> dict | None:
        if self._litellm_router is None:
            return None
        for dep in self._litellm_router.model_list:
            if isinstance(dep, dict) and dep.get("model_name") == model_name:
                return dep
        return None

    def _try_pin(self, request_kwargs: dict | None) -> dict | None:
        """Check request_kwargs for an explicit pin_model directive.

        Returns the pinned deployment dict, or None to fall through to
        normal routing.  The pin_model value must match a model in the
        pool; unknown names are silently ignored.
        """
        if not request_kwargs:
            return None
        metadata = request_kwargs.get("metadata") or {}
        pin = metadata.get("pin_model")
        if not pin or not self._router.has_model(pin):
            return None
        pinned = self._router.resolve(pin)
        if pinned is None:
            return None
        self._last_result = pinned
        return self._find_deployment(pin)

    def _route_and_select(
        self,
        model: str,
        messages: list[dict[str, str]] | None = None,
        input: str | list | None = None,
        request_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> dict:
        # Explicit pin via metadata — for router-per-subagent flows.
        dep = self._try_pin(request_kwargs)
        if dep:
            return dep

        text = self._extract_user_text(messages)
        if not text and isinstance(input, str):
            text = input

        if not text:
            self._last_result = None
            if self._litellm_router:
                return self._litellm_router.model_list[0]
            return {}

        result = self._router.route(text, tolerance=self.effective_tolerance)
        self._last_result = result

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
        return await asyncio.to_thread(
            self._route_and_select, model, messages, input, request_kwargs,
        )

    def get_available_deployment(
        self,
        model: str,
        messages: list[dict[str, str]] | None = None,
        input: str | list | None = None,
        specific_deployment: bool | None = False,
        request_kwargs: dict | None = None,
    ) -> dict:
        return self._route_and_select(model, messages, input, request_kwargs)

    def set_litellm_router(self, litellm_router: Any) -> None:
        """Called internally when plugged into a litellm.Router."""
        self._litellm_router = litellm_router
