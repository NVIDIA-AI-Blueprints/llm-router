"""LiteLLM Proxy callback that routes via an external router sidecar.

Pairs with the router-only sidecar (``model-router serve-router``). The hook
runs inside LiteLLM Proxy's request path, calls the sidecar over HTTP, and
rewrites ``data["model"]`` to the routed deployment before LiteLLM dispatches.

Unlike ``proxy.py``, the encoder does not run in-process — only the LiteLLM
Proxy and an HTTP client live here. The router can be scaled, restarted, or
GPU-pinned independently.

Loading in a LiteLLM Proxy ``config.yaml``::

    litellm_settings:
      callbacks: model_router_toolkit.adapters.litellm.external_hook.external_router_hook

Configuration is environment-driven so the hook can be wired without code:

  ROUTER_SIDECAR_URL                  required, e.g. http://router:8079
  ROUTER_SIDECAR_TIMEOUT_S            default 2.0
  ROUTER_SIDECAR_TOLERANCE            default 0.20
  ROUTER_SIDECAR_DEFAULT_MODEL        fallback model when sidecar fails (no
                                      default — unset means re-raise on error)
  ROUTER_SIDECAR_FAILURES_BEFORE_OPEN circuit-breaker threshold, default 5
  ROUTER_SIDECAR_OPEN_DURATION_S      cooldown after open, default 30
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

_ROUTING_CALL_TYPES = frozenset(
    {"completion", "acompletion", "text_completion", "atext_completion"}
)


class _CircuitBreaker:
    """Tiny in-memory breaker: trip after N failures, cool down for D seconds."""

    def __init__(self, *, failures_before_open: int, open_duration_s: float):
        self._threshold = max(1, failures_before_open)
        self._cooldown = max(0.0, open_duration_s)
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if (time.monotonic() - self._opened_at) >= self._cooldown:
            self._opened_at = None
            self._consecutive_failures = 0
            return False
        return True

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._threshold and self._opened_at is None:
            self._opened_at = time.monotonic()
            logger.warning(
                "router sidecar circuit opened after %d consecutive failures; "
                "skipping for %.0fs",
                self._consecutive_failures,
                self._cooldown,
            )


def _import_custom_logger():
    try:
        from litellm.integrations.custom_logger import CustomLogger
    except ImportError as e:
        raise ImportError(
            "ExternalRouterHook requires litellm. Install it in the proxy "
            "environment: pip install 'litellm>=1.50,<2.0'"
        ) from e
    return CustomLogger


CustomLogger = _import_custom_logger()


class ExternalRouterHook(CustomLogger):  # type: ignore[misc,valid-type]
    """LiteLLM ``CustomLogger`` that delegates routing to an external sidecar.

    The hook mutates ``data["model"]`` in ``async_pre_call_hook`` before
    LiteLLM resolves the deployment, so the routed model name must exist in
    the proxy's ``model_list``.
    """

    def __init__(
        self,
        *,
        sidecar_url: str,
        timeout_s: float = 2.0,
        tolerance: float = 0.20,
        default_model: str | None = None,
        failures_before_open: int = 5,
        open_duration_s: float = 30.0,
    ):
        if not sidecar_url:
            raise ValueError("sidecar_url is required")
        self._url = sidecar_url.rstrip("/") + "/v1/route"
        self._timeout_s = float(timeout_s)
        self._tolerance = max(0.0, min(1.0, float(tolerance)))
        self._default_model = default_model
        self._breaker = _CircuitBreaker(
            failures_before_open=failures_before_open,
            open_duration_s=open_duration_s,
        )

    @classmethod
    def from_env(cls) -> ExternalRouterHook:
        """Build a hook from ``ROUTER_SIDECAR_*`` environment variables."""
        url = os.environ.get("ROUTER_SIDECAR_URL", "").strip()
        if not url:
            raise RuntimeError(
                "ROUTER_SIDECAR_URL is not set. Point it at the running "
                "router sidecar, e.g. http://router:8079"
            )
        return cls(
            sidecar_url=url,
            timeout_s=float(os.environ.get("ROUTER_SIDECAR_TIMEOUT_S", "2.0")),
            tolerance=float(os.environ.get("ROUTER_SIDECAR_TOLERANCE", "0.20")),
            default_model=os.environ.get("ROUTER_SIDECAR_DEFAULT_MODEL") or None,
            failures_before_open=int(
                os.environ.get("ROUTER_SIDECAR_FAILURES_BEFORE_OPEN", "5")
            ),
            open_duration_s=float(os.environ.get("ROUTER_SIDECAR_OPEN_DURATION_S", "30")),
        )

    async def _ask_sidecar(self, messages: list[dict[str, Any]], tolerance: float) -> str:
        import httpx

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(
                self._url, json={"messages": messages, "tolerance": tolerance}
            )
            r.raise_for_status()
            payload = r.json()
        selected = payload.get("selected_model")
        if not isinstance(selected, str) or not selected:
            raise ValueError(f"sidecar returned no selected_model: {payload!r}")
        return selected

    def _fallback_or_raise(self, data: dict, exc: BaseException) -> dict:
        if self._default_model:
            logger.warning(
                "router sidecar unavailable (%s: %s); falling back to %s",
                type(exc).__name__,
                exc,
                self._default_model,
            )
            data["model"] = self._default_model
            return data
        raise exc

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> dict:
        if call_type not in _ROUTING_CALL_TYPES:
            return data

        metadata = data.get("metadata") or {}
        if metadata.get("pin_model"):
            data["model"] = metadata["pin_model"]
            return data

        messages = data.get("messages")
        if not messages:
            return data

        tolerance = float(metadata.get("tolerance", self._tolerance))
        tolerance = max(0.0, min(1.0, tolerance))

        if self._breaker.is_open:
            return self._fallback_or_raise(
                data, RuntimeError("router sidecar circuit open")
            )

        try:
            data["model"] = await self._ask_sidecar(messages, tolerance)
            self._breaker.record_success()
            return data
        except Exception as exc:  # network, timeout, bad payload, non-2xx
            self._breaker.record_failure()
            return self._fallback_or_raise(data, exc)


def _build_default() -> ExternalRouterHook | None:
    """Eagerly construct the module-level hook if env is configured.

    Returns ``None`` when ``ROUTER_SIDECAR_URL`` is unset so importing the
    module (e.g. during tests) doesn't fail. LiteLLM's callback loader
    requires a non-None instance, so deployments must set the env var.
    """
    if not os.environ.get("ROUTER_SIDECAR_URL"):
        return None
    return ExternalRouterHook.from_env()


external_router_hook = _build_default()
