"""Integration tests for the LiteLLM Proxy deployment feature.

Tests config bridge (generation + validation), strategy injection onto
a litellm Router, and the CLI subcommands.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from model_router_toolkit.config import ModelSpec, PoolConfig, RoutingConfig
from model_router_toolkit.adapters.litellm.config_bridge import (
    generate_litellm_config,
    validate_model_alignment,
)
from model_router_toolkit.router import BaseRouter, CostEstimate, RoutingResult
from model_router_toolkit.adapters.litellm.strategy import ModelRoutingStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StubRouter(BaseRouter):
    def __init__(self, model_names: list[str], selected: str | None = None):
        self._model_names = model_names
        self._selected = selected or model_names[0]

    def load(self, checkpoint_path):
        pass

    def route(self, question: str, *, tolerance: float = 0.20, models: list[str] | None = None) -> RoutingResult:
        n = len(self._model_names)
        return RoutingResult(
            model_names=self._model_names,
            confidences=[0.9 - i * 0.1 for i in range(n)],
            costs=[
                CostEstimate(
                    median_output_tokens=100,
                    cost_per_m_input_tokens=0.1 * (i + 1),
                    cost_per_m_output_tokens=0.1 * (i + 1),
                )
                for i in range(n)
            ],
            selected_model=self._selected,
            metadata={},
        )

    def unload(self):
        pass


def _pool_config() -> PoolConfig:
    return PoolConfig(
        routing=RoutingConfig(method="prefill", tolerance=0.20),
        models=[
            ModelSpec(
                name="model-a",
                litellm_model="nvidia_nim/nvidia/test-a",
                cost_per_m_input_tokens=0.10,
                cost_per_m_output_tokens=0.10,
            ),
            ModelSpec(
                name="model-b",
                litellm_model="openrouter/openai/test-b",
                cost_per_m_input_tokens=1.00,
                cost_per_m_output_tokens=5.00,
            ),
        ],
    )


def _write_pool_yaml(tmp_path: Path) -> Path:
    cfg = _pool_config()
    path = tmp_path / "pool.yaml"
    data = {
        "routing": {
            "method": cfg.routing.method,
            "tolerance": cfg.routing.tolerance,
        },
        "models": [
            {
                "name": m.name,
                "litellm_model": m.litellm_model,
                "cost_per_m_input_tokens": m.cost_per_m_input_tokens,
                "cost_per_m_output_tokens": m.cost_per_m_output_tokens,
            }
            for m in cfg.models
        ],
    }
    path.write_text(yaml.dump(data, sort_keys=False))
    return path


def _write_litellm_yaml(tmp_path: Path, model_names: list[str]) -> Path:
    path = tmp_path / "litellm.yaml"
    data = {
        "model_list": [
            {
                "model_name": name,
                "litellm_params": {"model": f"openai/{name}", "api_key": "test"},
            }
            for name in model_names
        ],
    }
    path.write_text(yaml.dump(data, sort_keys=False))
    return path


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------


class TestGenerateLiteLLMConfig:
    def test_generates_model_list(self):
        config = generate_litellm_config(_pool_config())
        assert "model_list" in config
        assert len(config["model_list"]) == 2

    def test_model_names_match_pool(self):
        config = generate_litellm_config(_pool_config())
        names = [e["model_name"] for e in config["model_list"]]
        assert names == ["model-a", "model-b"]

    def test_litellm_model_preserved(self):
        config = generate_litellm_config(_pool_config())
        models = {
            e["model_name"]: e["litellm_params"]["model"]
            for e in config["model_list"]
        }
        assert models["model-a"] == "nvidia_nim/nvidia/test-a"
        assert models["model-b"] == "openrouter/openai/test-b"

    def test_api_key_env_var_resolved(self):
        config = generate_litellm_config(_pool_config())
        keys = {
            e["model_name"]: e["litellm_params"]["api_key"]
            for e in config["model_list"]
        }
        assert keys["model-a"] == "os.environ/NVIDIA_API_KEY"
        assert keys["model-b"] == "os.environ/OPENROUTER_API_KEY"

    def test_router_settings_included(self):
        config = generate_litellm_config(_pool_config())
        assert "router_settings" in config
        assert config["router_settings"]["routing_strategy"] == "simple-shuffle"

    def test_writes_to_file(self, tmp_path):
        out = tmp_path / "litellm.yaml"
        generate_litellm_config(_pool_config(), output=out)
        assert out.exists()
        loaded = yaml.safe_load(out.read_text())
        assert len(loaded["model_list"]) == 2

    def test_from_yaml_path(self, tmp_path):
        pool_path = _write_pool_yaml(tmp_path)
        config = generate_litellm_config(pool_path)
        assert len(config["model_list"]) == 2


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestValidateModelAlignment:
    def test_matching_models_no_warnings(self, tmp_path):
        pool_path = _write_pool_yaml(tmp_path)
        litellm_path = _write_litellm_yaml(tmp_path, ["model-a", "model-b"])
        warnings = validate_model_alignment(litellm_path, pool_path)
        assert warnings == []

    def test_missing_in_litellm(self, tmp_path):
        pool_path = _write_pool_yaml(tmp_path)
        litellm_path = _write_litellm_yaml(tmp_path, ["model-a"])
        warnings = validate_model_alignment(litellm_path, pool_path)
        assert len(warnings) == 1
        assert "model-b" in warnings[0]
        assert "not in litellm" in warnings[0]

    def test_extra_in_litellm(self, tmp_path):
        pool_path = _write_pool_yaml(tmp_path)
        litellm_path = _write_litellm_yaml(
            tmp_path, ["model-a", "model-b", "model-c"],
        )
        warnings = validate_model_alignment(litellm_path, pool_path)
        assert len(warnings) == 1
        assert "model-c" in warnings[0]
        assert "never be selected" in warnings[0]

    def test_both_missing_and_extra(self, tmp_path):
        pool_path = _write_pool_yaml(tmp_path)
        litellm_path = _write_litellm_yaml(tmp_path, ["model-a", "model-c"])
        warnings = validate_model_alignment(litellm_path, pool_path)
        assert len(warnings) == 2


# ---------------------------------------------------------------------------
# Strategy injection
# ---------------------------------------------------------------------------


class TestStrategyInjection:
    def test_inject_strategy_patches_router(self):
        """_inject_strategy patches the proxy's global llm_router."""
        from litellm import Router as LiteLLMRouter

        model_list = [
            {"model_name": "m-a", "litellm_params": {"model": "openai/a", "api_key": "k"}},
            {"model_name": "m-b", "litellm_params": {"model": "openai/b", "api_key": "k"}},
        ]
        litellm_router = LiteLLMRouter(model_list=model_list)
        stub = StubRouter(["m-a", "m-b"], selected="m-a")
        strategy = ModelRoutingStrategy(stub, tolerance=0.20)
        strategy.set_litellm_router(litellm_router)
        litellm_router.set_custom_routing_strategy(strategy)

        dep = strategy.get_available_deployment(
            model="test",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert dep["model_name"] == "m-a"

    def test_inject_raises_when_no_router(self):
        """_inject_strategy raises when llm_router is None."""
        import sys
        import types

        from model_router_toolkit.adapters.litellm.proxy import _inject_strategy

        fake_proxy_mod = types.ModuleType("litellm.proxy.proxy_server")
        fake_proxy_mod.llm_router = None

        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": fake_proxy_mod}):
            with pytest.raises(RuntimeError, match="did not initialize"):
                _inject_strategy("fake.yaml")

    def test_inject_with_mock_proxy_module(self, tmp_path):
        """Full injection flow using a mocked proxy module global."""
        import sys
        import types

        from litellm import Router as LiteLLMRouter

        from model_router_toolkit.adapters.litellm.proxy import _inject_strategy

        model_list = [
            {"model_name": "m-a", "litellm_params": {"model": "openai/a", "api_key": "k"}},
        ]
        litellm_router = LiteLLMRouter(model_list=model_list)

        fake_proxy_mod = types.ModuleType("litellm.proxy.proxy_server")
        fake_proxy_mod.llm_router = litellm_router

        pool_path = _write_pool_yaml(tmp_path)
        pool_path_str = str(pool_path)

        with (
            patch.dict(sys.modules, {"litellm.proxy.proxy_server": fake_proxy_mod}),
            patch(
                "model_router_toolkit.adapters.litellm.strategy.ModelRoutingStrategy.from_config",
            ) as mock_from_config,
        ):
            stub = StubRouter(["m-a"])
            mock_strategy = ModelRoutingStrategy(stub, tolerance=0.20)
            mock_from_config.return_value = mock_strategy

            _inject_strategy(pool_path_str)

        mock_from_config.assert_called_once_with(pool_path_str)
        assert mock_strategy._litellm_router is litellm_router


# ---------------------------------------------------------------------------
# CLI subcommands (argument parsing only — no server startup)
# ---------------------------------------------------------------------------


class TestCLIProxyConfig:
    def test_proxy_config_stdout(self, tmp_path, capsys):
        pool_path = _write_pool_yaml(tmp_path)

        from model_router_toolkit.__main__ import _cmd_proxy_config

        args = MagicMock()
        args.config = str(pool_path)
        args.output = None
        _cmd_proxy_config(args)

        captured = capsys.readouterr()
        assert "model_list" in captured.out
        assert "model-a" in captured.out

    def test_proxy_config_file(self, tmp_path):
        pool_path = _write_pool_yaml(tmp_path)
        out_path = tmp_path / "out.yaml"

        from model_router_toolkit.__main__ import _cmd_proxy_config

        args = MagicMock()
        args.config = str(pool_path)
        args.output = str(out_path)
        _cmd_proxy_config(args)

        assert out_path.exists()
        loaded = yaml.safe_load(out_path.read_text())
        assert len(loaded["model_list"]) == 2
