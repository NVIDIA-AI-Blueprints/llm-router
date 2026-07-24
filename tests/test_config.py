import pytest
from pydantic import ValidationError

from model_router_toolkit.config import (
    ModelSpec,
    PoolConfig,
    PrefillFeatureConfig,
    load_config,
)


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
        config_path = project_root / "configs" / "v1-9models-qwen08b.yaml"
        if not config_path.exists():
            pytest.skip("v1-9models-qwen08b.yaml not found")
        config = load_config(config_path)
        assert config.routing.method == "prefill"
        assert len(config.models) > 0

    def test_prefill_config(self):
        config = PoolConfig.model_validate(
            {
                "routing": {
                    "method": "prefill",
                    "encoder": "Qwen/Qwen3.5-35B-A3B",
                    "encoder_server": "http://localhost:8421",
                },
                "models": [],
            }
        )
        assert config.routing.method == "prefill"
        assert config.routing.encoder == "Qwen/Qwen3.5-35B-A3B"

    def test_all_layer_meanpool_feature_config(self):
        config = PoolConfig.model_validate(
            {
                "routing": {
                    "encoder": "Qwen/Qwen3.6-35B-A3B",
                    "features": {
                        "aggregation": "all_layers_concat",
                        "layers": "all",
                        "pooling": "mean",
                        "pca_dim": 200,
                        "hidden_state_indexing": "direct",
                    },
                },
                "models": [],
            }
        )
        features = config.routing.features
        assert features is not None
        assert features.aggregation == "all_layers_concat"
        assert features.layers == "all"
        assert features.pooling == "mean"
        assert features.pca_dim == 200

    @pytest.mark.parametrize("layers", [[], [0, 0], [-1, 0]])
    def test_feature_config_rejects_invalid_layers(self, layers):
        with pytest.raises(ValidationError):
            PrefillFeatureConfig(
                aggregation="all_layers_concat",
                layers=layers,
            )
