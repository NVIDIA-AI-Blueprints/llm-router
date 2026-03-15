"""Configuration system for model-router-toolkit.

Loads pool_config.yaml and validates it with pydantic.
The config determines routing method, model pool, and provider settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    name: str
    display_name: str = ""
    litellm_model: str = ""
    cost_per_m_input_tokens: float = 0.0
    cost_per_m_output_tokens: float = 0.0
    system_prompt: str = ""
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    api_base: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.display_name:
            self.display_name = self.name


class RoutingConfig(BaseModel):
    method: str = "prefill"
    checkpoint: str = ""
    tolerance: float = 0.20

    encoder: str = ""
    encoder_server: str = ""
    training_mode: str = "auto"
    encoder_backend: str = "transformers"


class PoolConfig(BaseModel):
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    models: list[ModelSpec] = Field(default_factory=list)

    @property
    def model_names(self) -> list[str]:
        return [m.name for m in self.models]

    def get_model(self, name: str) -> ModelSpec | None:
        for m in self.models:
            if m.name == name:
                return m
        return None


def load_config(path: str | Path) -> PoolConfig:
    """Load and validate a pool_config.yaml file."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    return PoolConfig.model_validate(raw)


def build_router_from_config(config: PoolConfig):
    """Construct the appropriate BaseRouter from config."""
    from model_router_toolkit.router import BaseRouter

    method = config.routing.method.lower()

    if method == "prefill":
        from model_router_toolkit.prefill.router import PrefillRouter

        router = PrefillRouter(config=config)
        if config.routing.checkpoint:
            router.load(config.routing.checkpoint)
        return router

    else:
        raise ValueError(f"Unknown routing method: {method!r}. Supported: 'prefill'.")
