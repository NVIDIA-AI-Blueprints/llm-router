"""FastAPI application factory for model-router-toolkit."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from litellm import Router

from model_router_toolkit.config import PoolConfig, load_config
from model_router_toolkit.strategy import ModelRoutingStrategy

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


def _warmup(strategy: ModelRoutingStrategy, config: PoolConfig) -> None:
    """Pre-load all models and run a dummy route so first real request is fast."""
    method = config.routing.method
    print(f"Warming up {method} router...")
    t0 = time.time()

    try:
        result = strategy.router.route("warmup test query", tolerance=0.5)
        elapsed = time.time() - t0
        print(f"  Warmup complete in {elapsed:.1f}s")
        print(f"  Models: {result.model_names}")
        print(f"  Test route -> {result.selected_model} "
              f"(confidences: {', '.join(f'{c:.3f}' for c in result.confidences)})")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  Warmup failed after {elapsed:.1f}s: {e}")
        logger.warning("Router warmup failed: %s", e)


def create_app(config_path: str) -> FastAPI:
    """Create and configure the FastAPI application."""
    config = load_config(config_path)
    model_list = _build_model_list(config)

    litellm_router = Router(model_list=model_list)
    strategy = ModelRoutingStrategy.from_config(config_path)
    strategy.set_litellm_router(litellm_router)
    litellm_router.set_custom_routing_strategy(strategy)

    _warmup(strategy, config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        strategy.router.unload()

    app = FastAPI(title="Model Router Toolkit", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.litellm_router = litellm_router
    app.state.strategy = strategy
    app.state.config = config

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "method": config.routing.method,
            "models": config.model_names,
        }

    @app.get("/api/models")
    async def get_models():
        return [
            {
                "name": m.name,
                "display_name": m.display_name or m.name,
                "cost_per_m_input_tokens": m.cost_per_m_input_tokens,
                "cost_per_m_output_tokens": m.cost_per_m_output_tokens,
            }
            for m in config.models
        ]

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
