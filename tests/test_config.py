import pytest
import yaml
from pathlib import Path

from model_router_toolkit.config import PoolConfig, ModelSpec, RoutingConfig, load_config


class TestPoolConfig:
    def test_from_dict(self, sample_pool_config_dict):
        config = PoolConfig.model_validate(sample_pool_config_dict)
        assert config.routing.method == "prefill"
        assert config.routing.tolerance == 0.20
        assert len(config.models) == 2
        assert config.model_names == ["nem-think", "gpt-5.2"]

    def test_get_model(self, sample_pool_config_dict):
        config = PoolConfig.model_validate(sample_pool_config_dict)
        m = config.get_model("nem-think")
        assert m is not None
        assert m.display_name == "Nemotron 3 Nano Think"
        assert config.get_model("nonexistent") is None

    def test_defaults(self):
        config = PoolConfig.model_validate({"routing": {}, "models": []})
        assert config.routing.method == "prefill"
        assert config.routing.tolerance == 0.20

    def test_model_spec_auto_display_name(self):
        m = ModelSpec(name="test-model")
        assert m.display_name == "test-model"

    def test_load_prefill_yaml_file(self, project_root):
        config_path = project_root / "configs" / "prefill-qwen08b.yaml"
        if not config_path.exists():
            pytest.skip("prefill-qwen08b.yaml not found")
        config = load_config(config_path)
        assert config.routing.method == "prefill"
        assert len(config.models) > 0

    def test_prefill_config(self):
        config = PoolConfig.model_validate({
            "routing": {
                "method": "prefill",
                "encoder": "Qwen/Qwen3.5-35B-A3B",
                "encoder_server": "http://localhost:8421",
            },
            "models": [],
        })
        assert config.routing.method == "prefill"
        assert config.routing.encoder == "Qwen/Qwen3.5-35B-A3B"
