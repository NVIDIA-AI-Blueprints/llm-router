"""Configuration system for model-router-toolkit.

Loads pool_config.yaml and validates it with pydantic.
The config determines routing method, model pool, and provider settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


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


class PrefillFeatureConfig(BaseModel):
    """Fixed prefill feature recipe used instead of the feature sweep."""

    aggregation: Literal["single_layer", "all_layers_concat"] = "single_layer"
    layers: Literal["all"] | list[int] = "all"
    pooling: Literal["last", "mean"] = "mean"
    pca_dim: int = Field(default=200, gt=0)
    hidden_state_indexing: Literal["direct"] = "direct"

    @field_validator("layers")
    @classmethod
    def validate_layers(cls, value: Literal["all"] | list[int]):
        if value == "all":
            return value
        if not value:
            raise ValueError("layers must be 'all' or a non-empty list")
        if any(layer < 0 for layer in value):
            raise ValueError("layers must contain only non-negative integers")
        if len(set(value)) != len(value):
            raise ValueError("layers must not contain duplicates")
        return value

    @model_validator(mode="after")
    def validate_aggregation_layers(self):
        if self.aggregation == "single_layer":
            if self.layers == "all" or len(self.layers) != 1:
                raise ValueError(
                    "single_layer aggregation requires one explicit layer"
                )
        return self


class RoutingConfig(BaseModel):
    method: str = "prefill"
    checkpoint: str = ""
    tolerance: float = 0.20

    encoder: str = ""
    encoder_server: str = ""
    training_mode: str = "auto"
    encoder_backend: str = "transformers"
    features: PrefillFeatureConfig | None = None


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

    method = config.routing.method.lower()

    if method == "prefill":
        from model_router_toolkit.prefill.router import PrefillRouter

        router = PrefillRouter(config=config)
        if config.routing.checkpoint:
            router.load(config.routing.checkpoint)
        return router

    else:
        raise ValueError(f"Unknown routing method: {method!r}. Supported: 'prefill'.")
