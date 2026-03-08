"""Config bridge between pool_config.yaml and litellm proxy config.yaml.

- Generates a litellm-compatible config.yaml from a pool_config.yaml
- Validates that model names align between both configs
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from model_router_toolkit.config import PoolConfig, load_config

logger = logging.getLogger(__name__)


def generate_litellm_config(
    pool_config: str | Path | PoolConfig,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a litellm proxy config.yaml from a pool_config.yaml.

    Returns the config dict and optionally writes it to *output*.
    """
    if isinstance(pool_config, (str, Path)):
        config = load_config(pool_config)
    else:
        config = pool_config

    model_list = []
    for m in config.models:
        litellm_model = m.litellm_model
        api_base = m.api_base or getattr(config.routing, "embed_api_base", "")
        env_var = _api_key_env_var(litellm_model, api_base)

        params: dict[str, Any] = {"model": litellm_model}
        if env_var:
            params["api_key"] = f"os.environ/{env_var}"

        entry: dict[str, Any] = {
            "model_name": m.name,
            "litellm_params": params,
        }
        model_list.append(entry)

    litellm_config: dict[str, Any] = {
        "model_list": model_list,
        "router_settings": {
            "routing_strategy": "simple-shuffle",
        },
    }

    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            yaml.dump(litellm_config, f, default_flow_style=False, sort_keys=False)

    return litellm_config


def validate_model_alignment(
    litellm_config_path: str | Path,
    pool_config_path: str | Path,
) -> list[str]:
    """Check that model names in pool_config exist in the litellm config.

    Returns a list of warning messages (empty if everything aligns).
    """
    pool_config = load_config(pool_config_path)
    pool_names = set(pool_config.model_names)

    with open(litellm_config_path) as f:
        litellm_raw = yaml.safe_load(f)

    litellm_names = set()
    for entry in litellm_raw.get("model_list", []):
        name = entry.get("model_name", "")
        if name:
            litellm_names.add(name)

    warnings: list[str] = []

    missing_in_litellm = pool_names - litellm_names
    if missing_in_litellm:
        warnings.append(
            f"Models in pool_config but not in litellm config: {sorted(missing_in_litellm)}. "
            "The router may select models that the proxy cannot dispatch."
        )

    extra_in_litellm = litellm_names - pool_names
    if extra_in_litellm:
        warnings.append(
            f"Models in litellm config but not in pool_config: {sorted(extra_in_litellm)}. "
            "These models will never be selected by the router."
        )

    return warnings


def _api_key_env_var(litellm_model: str, api_base: str) -> str:
    """Return the env var name for the API key (not the value)."""
    if litellm_model.startswith("openrouter/"):
        return "OPENROUTER_API_KEY"
    if litellm_model.startswith("nvidia_nim/"):
        return "NVIDIA_API_KEY"
    if litellm_model.startswith("openai/"):
        return "OPENAI_API_KEY"
    if litellm_model.startswith("anthropic/"):
        return "ANTHROPIC_API_KEY"

    if "nvidia" in api_base or "integrate.api.nvidia" in api_base:
        return "NVIDIA_API_KEY"
    if "openrouter" in api_base:
        return "OPENROUTER_API_KEY"

    logger.warning(
        "Could not determine API key env var for model %r (api_base=%r); "
        "falling back to OPENAI_API_KEY",
        litellm_model, api_base,
    )
    return "OPENAI_API_KEY"
