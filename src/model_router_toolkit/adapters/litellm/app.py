"""Full-mode FastAPI application: routing + LLM inference via LiteLLM.

Includes playground UI, chat completions, SSE chat, and auto-review endpoints.
Requires: pip install model-router-toolkit[litellm]
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from model_router_toolkit.adapters.http._shared import health_dict, models_list, warmup_router
from model_router_toolkit.config import PoolConfig, load_config

logger = logging.getLogger(__name__)


_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]+"),
    re.compile(r'"api[-_]?key"\s*:\s*"[^"]+"', re.IGNORECASE),
)


def _redact_secrets(msg: str) -> str:
    """Strip obvious credential patterns from upstream error messages.

    litellm sometimes echoes the upstream request (headers, JSON body) into
    exception strings, which can include short-lived API keys or bearer
    tokens. Best-effort scrubbing keeps those out of client-facing responses.
    """
    for pat in _SECRET_PATTERNS:
        msg = pat.sub("***REDACTED***", msg)
    return msg


def _install_openai_compat_error_handlers(app: FastAPI) -> None:
    """Translate litellm exceptions into OpenAI-compatible error JSON responses.

    Without this handler, FastAPI returns a generic 500 Internal Server Error
    with a plain-text body for every litellm exception (BadRequestError,
    AuthenticationError, RateLimitError, ...). This prevents OpenAI-compatible
    clients from distinguishing client errors (4xx) from server errors (5xx)
    and from extracting the upstream error message, leading to wasted retries
    for what are really 400-class user errors.
    """
    from fastapi.exceptions import HTTPException, RequestValidationError
    from litellm import exceptions as le

    # (exception class, HTTP status, OpenAI error type string).
    # Order matters: more specific subclasses appear first.
    # The OpenAI `error.type` strings mirror the values returned by the
    # OpenAI API itself (snake_case) so that clients which key off
    # `error.type` behave consistently against the router.
    exception_map: list[tuple[type[Exception], int, str]] = [
        (le.AuthenticationError, 401, "authentication_error"),
        (le.NotFoundError, 404, "not_found_error"),
        (le.ContentPolicyViolationError, 400, "content_policy_violation"),
        (le.ContextWindowExceededError, 400, "context_length_exceeded"),
        (le.UnprocessableEntityError, 422, "invalid_request_error"),
        (le.BadRequestError, 400, "invalid_request_error"),
        (le.BudgetExceededError, 429, "billing_hard_limit_reached"),
        (le.RateLimitError, 429, "rate_limit_exceeded"),
        (le.Timeout, 504, "timeout"),
        (le.APIConnectionError, 502, "api_connection_error"),
        (le.ServiceUnavailableError, 503, "service_unavailable"),
        (le.InternalServerError, 500, "server_error"),
    ]

    async def litellm_error_handler(_request: Request, exc: Exception):
        # FastAPI's own exceptions (HTTPException raised by route handlers,
        # request validation failures) carry their own status code and shape;
        # let the default handlers process them.
        if isinstance(exc, (HTTPException, RequestValidationError)):
            raise exc

        for cls, status, error_type in exception_map:
            if isinstance(exc, cls):
                # 5xx upstream failures are worth a full traceback (server-side
                # bug or outage); 4xx are typically client errors so a one-line
                # warning is enough to debug without flooding the log.
                if status >= 500:
                    logger.exception(
                        "Upstream %s (HTTP %d)", type(exc).__name__, status
                    )
                else:
                    logger.warning(
                        "Upstream %s (HTTP %d): %s",
                        type(exc).__name__,
                        status,
                        exc,
                    )
                return JSONResponse(
                    status_code=status,
                    content={
                        "error": {
                            "message": _redact_secrets(str(exc)),
                            "type": error_type,
                            "param": getattr(exc, "param", None),
                            "code": getattr(exc, "code", None),
                        }
                    },
                )

        # Catch-all for exceptions that don't match the map above. This includes
        # litellm classes that inherit from Exception directly (e.g.
        # BudgetExceededError when it was raised without the proper subclass,
        # guardrail violations from third-party callbacks). Surface them as 500
        # and keep a full traceback in the server log.
        logger.exception(
            "Unhandled exception in OpenAI-compatible endpoint: %s",
            type(exc).__name__,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": _redact_secrets(str(exc)),
                    "type": "server_error",
                    "param": None,
                    "code": None,
                }
            },
        )

    # Register on openai.APIError (covers every litellm exception that
    # inherits from openai's hierarchy — BadRequestError, RateLimitError, ...)
    # AND on Exception so the catch-all branch above can map the few classes
    # that inherit Exception directly. The Exception handler re-raises
    # FastAPI's own exceptions so it does not interfere with normal 4xx flow.
    from openai import APIError as OpenAIAPIError

    app.add_exception_handler(OpenAIAPIError, litellm_error_handler)
    app.add_exception_handler(Exception, litellm_error_handler)


def _resolve_api_key(litellm_model: str, api_base: str) -> str:
    """Pick the right API key based on model prefix or api_base."""
    if litellm_model.startswith("openrouter/"):
        return os.environ.get("OPENROUTER_API_KEY", "")
    if litellm_model.startswith("nvidia_nim/"):
        return os.environ.get("NVIDIA_API_KEY", "")

    if "nvidia" in api_base or "integrate.api.nvidia" in api_base:
        return os.environ.get("NVIDIA_API_KEY", "")
    if "openrouter" in api_base:
        return os.environ.get("OPENROUTER_API_KEY", "")

    return (
        os.environ.get("NVIDIA_API_KEY", "")
        or os.environ.get("OPENROUTER_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
    )


def _build_model_list(config: PoolConfig) -> list[dict]:
    model_list = []

    for m in config.models:
        api_base = m.api_base or ""
        api_key = _resolve_api_key(m.litellm_model, api_base)

        litellm_model = m.litellm_model
        has_provider = any(
            litellm_model.startswith(p)
            for p in (
                "openrouter/",
                "nvidia_nim/",
                "openai/",
                "anthropic/",
                "ollama/",
            )
        )
        if not has_provider:
            if "integrate.api.nvidia" in api_base:
                litellm_model = f"nvidia_nim/{litellm_model}"
            elif "openrouter" in api_base:
                litellm_model = f"openrouter/{litellm_model}"
            else:
                litellm_model = f"openai/{litellm_model}"

        params: dict = {
            "model": litellm_model,
            "api_key": api_key,
        }
        if m.api_base:
            params["api_base"] = m.api_base

        model_list.append(
            {
                "model_name": m.name,
                "litellm_params": params,
            }
        )
    return model_list


def create_app(
    config_path: str,
    *,
    warmup: bool = True,
    models: list[str] | None = None,
) -> FastAPI:
    """Create full-mode FastAPI app with routing + LLM inference."""
    from litellm import Router

    from model_router_toolkit.adapters.litellm.strategy import ModelRoutingStrategy

    config = load_config(config_path)
    model_list = _build_model_list(config)
    litellm_router = Router(model_list=model_list)
    strategy = ModelRoutingStrategy.from_config(config_path)
    strategy.models = models
    strategy.set_litellm_router(litellm_router)
    litellm_router.set_custom_routing_strategy(strategy)
    base_router = strategy.router

    if warmup:
        warmup_router(base_router, config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        base_router.unload()

    app = FastAPI(title="Model Router Toolkit", lifespan=lifespan)

    _install_openai_compat_error_handlers(app)

    cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in cors_origins],
        allow_credentials=cors_origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.router = base_router
    app.state.config = config
    app.state.litellm_router = litellm_router
    app.state.strategy = strategy
    app.state.allowed_models = models

    @app.get("/health")
    async def health():
        return health_dict(config, mode="full")

    @app.get("/api/models")
    async def get_models():
        return models_list(config)

    review_available = bool(
        os.environ.get("OPENROUTER_API_KEY") or os.environ.get("NVIDIA_API_KEY")
    )
    judge_model = (
        max(config.models, key=lambda m: m.cost_per_m_output_tokens).display_name
        if config.models
        else None
    )

    @app.get("/api/config")
    async def get_config():
        return {
            "routing_method": config.routing.method,
            "review_available": review_available,
            "judge_model": judge_model,
            "model_count": len(config.models),
            "tolerance": config.routing.tolerance,
        }

    from model_router_toolkit.adapters.litellm.chat import router as chat_router
    from model_router_toolkit.adapters.litellm.completions import router as completions_router
    from model_router_toolkit.adapters.litellm.review import router as review_router

    app.include_router(chat_router, prefix="/api", tags=["chat"])
    app.include_router(completions_router, prefix="/v1", tags=["completions"])
    app.include_router(review_router, prefix="/api", tags=["review"])

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
