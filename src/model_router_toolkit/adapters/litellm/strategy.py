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
import logging
from typing import Any

from model_router_toolkit.router import BaseRouter, RoutingResult, extract_user_text

logger = logging.getLogger(__name__)

_request_tolerance: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "request_tolerance",
    default=None,
)

# Holds a mutable per-request slot for the routing result. A mutable dict
# (rather than the result itself) so writes made inside copied contexts —
# asyncio.to_thread and task groups copy the context — stay visible to the
# request handler that installed the slot.
_request_result: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "request_routing_result",
    default=None,
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
        models: list[str] | None = None,
    ):
        self._router = router
        self._tolerance = tolerance
        self._models = models
        self._litellm_router: Any = None
        self._last_result: RoutingResult | None = None

    @classmethod
    def from_config(cls, config_path: str, **kwargs: Any) -> ModelRoutingStrategy:
        """Build a strategy from a pool_config.yaml file."""
        from model_router_toolkit.config import build_router_from_config, load_config

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

    def begin_request(self) -> None:
        """Install a request-scoped slot for the routing result.

        Call this in the request handler before dispatching to LiteLLM so
        that last_result reads back this request's routing decision instead
        of whichever concurrent request routed most recently.
        """
        _request_result.set({"result": None})

    def _record_result(self, result: RoutingResult | None) -> None:
        slot = _request_result.get()
        if slot is not None:
            slot["result"] = result
        self._last_result = result

    @property
    def models(self) -> list[str] | None:
        return self._models

    @models.setter
    def models(self, value: list[str] | None) -> None:
        self._models = value

    @property
    def effective_tolerance(self) -> float:
        """Tolerance for the current request: per-request override or default."""
        val = _request_tolerance.get()
        return val if val is not None else self._tolerance

    @property
    def last_result(self) -> RoutingResult | None:
        slot = _request_result.get()
        if slot is not None:
            return slot["result"]
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
        self._record_result(pinned)
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
            self._record_result(None)
            if self._litellm_router:
                return self._litellm_router.model_list[0]
            return {}

        req_models = ((request_kwargs or {}).get("metadata") or {}).get("models")
        allowed = req_models or self._models
        result = self._router.route(
            text,
            tolerance=self.effective_tolerance,
            models=allowed,
        )
        self._record_result(result)

        dep = self._find_deployment(result.selected_model)
        if dep:
            return dep

        logger.warning(
            "Routed model %r not found in litellm model_list. "
            "LiteLLM will fall back to default routing. "
            "Ensure all pool models are in model_list.",
            result.selected_model,
        )

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
            self._route_and_select,
            model,
            messages,
            input,
            request_kwargs,
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
