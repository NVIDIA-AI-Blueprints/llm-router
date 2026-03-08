"""FastAPI application factory for model-router-toolkit.

Supports two modes:
- Full mode (default): routing + LLM inference via LiteLLM, playground UI
- Router-only mode: routing decisions only, no inference, no API keys needed
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from model_router_toolkit.config import PoolConfig, build_router_from_config, load_config
from model_router_toolkit.server._shared import health_dict, models_list, warmup_router

logger = logging.getLogger(__name__)


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

    return os.environ.get("NVIDIA_API_KEY", "") or os.environ.get("OPENROUTER_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")


def _build_model_list(config: PoolConfig) -> list[dict]:
    model_list = []
    embed_base = getattr(config.routing, "embed_api_base", "")

    for m in config.models:
        api_base = m.api_base or embed_base
        api_key = _resolve_api_key(m.litellm_model, api_base)

        litellm_model = m.litellm_model
        has_provider = any(litellm_model.startswith(p) for p in (
            "openrouter/", "nvidia_nim/", "openai/", "anthropic/", "ollama/",
        ))
        if not has_provider:
            if "nvidia" in api_base:
                litellm_model = f"nvidia_nim/{litellm_model}"
            elif "openrouter" in api_base:
                litellm_model = f"openrouter/{litellm_model}"
            else:
                litellm_model = f"openai/{litellm_model}"

        params: dict = {
            "model": litellm_model,
            "api_key": api_key,
        }

        model_list.append({
            "model_name": m.name,
            "litellm_params": params,
        })
    return model_list


def create_app(
    config_path: str,
    *,
    router_only: bool = False,
    warmup: bool = True,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config_path: Path to pool config YAML.
        router_only: If True, serve only ``POST /v1/route`` for routing
            decisions — no LiteLLM, no inference, no API keys needed.
        warmup: If True, run a warmup route on startup.
    """
    config = load_config(config_path)

    if router_only:
        base_router = build_router_from_config(config)
        strategy = None
    else:
        from litellm import Router

        from model_router_toolkit.strategy import ModelRoutingStrategy

        model_list = _build_model_list(config)
        litellm_router = Router(model_list=model_list)
        strategy = ModelRoutingStrategy.from_config(config_path)
        strategy.set_litellm_router(litellm_router)
        litellm_router.set_custom_routing_strategy(strategy)
        base_router = strategy.router

    if warmup:
        warmup_router(base_router, config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        base_router.unload()

    mode = "router-only" if router_only else "full"
    title = "Model Router Toolkit" + (" — Router Only" if router_only else "")
    app = FastAPI(title=title, lifespan=lifespan)

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

    @app.get("/health")
    async def health():
        return health_dict(config, mode=mode)

    @app.get("/api/models")
    async def get_models():
        return models_list(config)

    if router_only:
        from model_router_toolkit.server.route import router as route_router

        app.include_router(route_router, prefix="/v1", tags=["route"])
    else:
        app.state.litellm_router = litellm_router
        app.state.strategy = strategy

        review_available = bool(os.environ.get("OPENROUTER_API_KEY"))
        judge_model = max(config.models, key=lambda m: m.cost_per_m_output_tokens).display_name if config.models else None

        @app.get("/api/config")
        async def get_config():
            return {
                "routing_method": config.routing.method,
                "review_available": review_available,
                "judge_model": judge_model,
                "model_count": len(config.models),
                "tolerance": config.routing.tolerance,
            }

        from model_router_toolkit.server.chat import router as chat_router
        from model_router_toolkit.server.completions import router as completions_router
        from model_router_toolkit.server.review import router as review_router

        app.include_router(chat_router, prefix="/api", tags=["chat"])
        app.include_router(completions_router, prefix="/v1", tags=["completions"])
        app.include_router(review_router, prefix="/api", tags=["review"])

        static_dir = Path(__file__).parent / "static"
        if static_dir.exists():
            app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
