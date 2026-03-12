"""Router-only FastAPI application: routing decisions without LLM inference.

No litellm dependency. Serves POST /v1/route and /health.
Requires: pip install model-router-toolkit[server]
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from model_router_toolkit.config import build_router_from_config, load_config
from model_router_toolkit.adapters.http._shared import health_dict, models_list, warmup_router

logger = logging.getLogger(__name__)


def create_app(
    config_path: str,
    *,
    warmup: bool = True,
) -> FastAPI:
    """Create router-only FastAPI app (no inference, no API keys needed)."""
    config = load_config(config_path)
    base_router = build_router_from_config(config)

    if warmup:
        warmup_router(base_router, config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        base_router.unload()

    app = FastAPI(title="Model Router Toolkit — Router Only", lifespan=lifespan)

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
        return health_dict(config, mode="router-only")

    @app.get("/api/models")
    async def get_models():
        return models_list(config)

    from model_router_toolkit.adapters.http.route import router as route_router

    app.include_router(route_router, prefix="/v1", tags=["route"])

    return app
