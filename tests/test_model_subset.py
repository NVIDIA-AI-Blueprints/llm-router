"""Tests for model subset filtering in routing and evaluation."""

import numpy as np
import pytest

from model_router_toolkit.router import CostEstimate, RoutingResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_kmeans_router(pkl_path):
    from model_router_toolkit.kmeans.router import KMeansRouter

    router = KMeansRouter()
    router.load(pkl_path)

    class MockEmbed:
        def embed(self, text):
            np.random.seed(hash(text) % 2**31)
            return np.random.randn(2048).astype(np.float32)

    router.set_embed_client(MockEmbed())
    router.set_cost_table({
        m: {"cost": i * 0.1, "median_output_tokens": 500}
        for i, m in enumerate(router.model_names)
    })
    return router


# ---------------------------------------------------------------------------
# BaseRouter / PrefillRouter subset tests
# ---------------------------------------------------------------------------

class TestPrefillRouterSubset:
    def test_unknown_model_raises_without_scoring(self, prefill_ckpt_path, prefill_config_path):
        """Validation happens before encoder extraction, so no network needed."""
        from model_router_toolkit.config import load_config
        from model_router_toolkit.prefill.router import PrefillRouter

        config = load_config(prefill_config_path)
        router = PrefillRouter(config=config)
        router.load(prefill_ckpt_path)

        with pytest.raises(ValueError, match="not in pool"):
            router.route("test", models=["nonexistent-model"])

    def test_unknown_mixed_with_valid_raises(self, prefill_ckpt_path, prefill_config_path):
        from model_router_toolkit.config import load_config
        from model_router_toolkit.prefill.router import PrefillRouter

        config = load_config(prefill_config_path)
        router = PrefillRouter(config=config)
        router.load(prefill_ckpt_path)

        with pytest.raises(ValueError, match="not in pool"):
            router.route("test", models=[config.model_names[0], "fake-model"])

    @pytest.mark.slow
    def test_route_no_filter_returns_all(self, prefill_ckpt_path, prefill_config_path):
        from model_router_toolkit.config import load_config
        from model_router_toolkit.prefill.router import PrefillRouter

        config = load_config(prefill_config_path)
        router = PrefillRouter(config=config)
        router.load(prefill_ckpt_path)

        result = router.route("What is 2+2?")
        assert len(result.model_names) == len(config.model_names)
        assert result.selected_model in config.model_names

    @pytest.mark.slow
    def test_route_with_subset(self, prefill_ckpt_path, prefill_config_path):
        from model_router_toolkit.config import load_config
        from model_router_toolkit.prefill.router import PrefillRouter

        config = load_config(prefill_config_path)
        router = PrefillRouter(config=config)
        router.load(prefill_ckpt_path)
        pool = config.model_names

        subset = pool[:2]
        result = router.route("What is 2+2?", models=subset)
        assert result.selected_model in subset
        assert len(result.model_names) == len(pool)
        assert result.metadata["allowed_models"] == sorted(subset)

    @pytest.mark.slow
    def test_route_single_model(self, prefill_ckpt_path, prefill_config_path):
        from model_router_toolkit.config import load_config
        from model_router_toolkit.prefill.router import PrefillRouter

        config = load_config(prefill_config_path)
        router = PrefillRouter(config=config)
        router.load(prefill_ckpt_path)

        single = [config.model_names[0]]
        result = router.route("Explain quantum entanglement", models=single)
        assert result.selected_model == single[0]


# ---------------------------------------------------------------------------
# KMeansRouter subset tests
# ---------------------------------------------------------------------------

class TestKMeansRouterSubset:
    def test_route_no_filter(self, mock_kmeans_router):
        result = mock_kmeans_router.route("What is gravity?")
        assert result.selected_model in mock_kmeans_router.model_names

    def test_route_with_subset(self, mock_kmeans_router):
        pool = mock_kmeans_router.model_names
        subset = pool[:2]
        result = mock_kmeans_router.route("What is gravity?", models=subset)
        assert result.selected_model in subset
        assert len(result.model_names) == len(pool)

    def test_route_single_model(self, mock_kmeans_router):
        single = [mock_kmeans_router.model_names[-1]]
        result = mock_kmeans_router.route("test", models=single)
        assert result.selected_model == single[0]

    def test_route_unknown_model_raises(self, mock_kmeans_router):
        with pytest.raises(ValueError, match="not in pool"):
            mock_kmeans_router.route("test", models=["fake-model-xyz"])


# ---------------------------------------------------------------------------
# Evaluation subset tests
# ---------------------------------------------------------------------------

class TestEvalSubsetFiltering:
    """Test the column-slicing logic used in _run_prefill_evaluate."""

    @pytest.fixture
    def mock_eval_data(self):
        model_names = ["model-a", "model-b", "model-c"]
        N = 100
        np.random.seed(42)
        Y = np.random.randint(0, 2, size=(N, 3))
        probs = np.random.rand(N, 3)
        return model_names, Y, probs

    def test_subset_slicing(self, mock_eval_data):
        model_names, Y, probs = mock_eval_data
        subset = ["model-a", "model-c"]
        subset_idx = [model_names.index(m) for m in subset]
        Y_sub = Y[:, subset_idx]
        probs_sub = probs[:, subset_idx]

        assert Y_sub.shape == (100, 2)
        assert probs_sub.shape == (100, 2)
        np.testing.assert_array_equal(Y_sub[:, 0], Y[:, 0])
        np.testing.assert_array_equal(Y_sub[:, 1], Y[:, 2])

    def test_subset_preserves_order(self, mock_eval_data):
        model_names, Y, probs = mock_eval_data
        subset = ["model-c", "model-a"]
        subset_idx = [model_names.index(m) for m in subset]
        sub_names = [model_names[i] for i in subset_idx]
        assert sub_names == ["model-c", "model-a"]

    def test_unknown_model_detected(self, mock_eval_data):
        model_names, _, _ = mock_eval_data
        subset = ["model-a", "model-z"]
        unknown = set(subset) - set(model_names)
        assert unknown == {"model-z"}

    def test_full_pool_no_filter(self, mock_eval_data):
        model_names, Y, probs = mock_eval_data
        subset_idx = [model_names.index(m) for m in model_names]
        np.testing.assert_array_equal(Y[:, subset_idx], Y)

    def test_argmax_changes_with_subset(self, mock_eval_data):
        """Routing decisions change when models are removed from consideration."""
        model_names, Y, probs = mock_eval_data
        full_choices = np.argmax(probs, axis=1)

        subset = ["model-a", "model-c"]
        subset_idx = [model_names.index(m) for m in subset]
        probs_sub = probs[:, subset_idx]
        sub_choices = np.argmax(probs_sub, axis=1)

        assert full_choices.shape == (100,)
        assert sub_choices.shape == (100,)
        assert not np.array_equal(full_choices, sub_choices) or len(model_names) == 2
