import numpy as np
import pytest

from model_router_toolkit.kmeans.router import KMeansRouter


class TestKMeansRouter:
    def test_load_pkl(self, pkl_path):
        router = KMeansRouter()
        router.load(pkl_path)
        assert router.n_clusters == 100
        assert len(router.model_names) > 0

    def test_predict_probs(self, pkl_path):
        router = KMeansRouter()
        router.load(pkl_path)
        fake_embedding = np.random.randn(2048).astype(np.float32)
        cluster, probs = router.predict_probs(fake_embedding)
        assert 0 <= cluster < 100
        assert len(probs) == len(router.model_names)
        for p in probs.values():
            assert 0.0 <= p <= 1.0

    def test_route_requires_load(self):
        router = KMeansRouter()
        with pytest.raises(RuntimeError, match="not loaded"):
            router.route("test question")

    def test_route_with_mock_embed(self, pkl_path):
        router = KMeansRouter()
        router.load(pkl_path)

        class MockEmbed:
            def embed(self, text):
                return np.random.randn(2048).astype(np.float32)

        router.set_embed_client(MockEmbed())
        router.set_cost_table({
            m: {"cost": i * 0.1, "median_output_tokens": 500}
            for i, m in enumerate(router.model_names)
        })

        result = router.route("What is the capital of France?", tolerance=0.10)
        assert result.selected_model in router.model_names
        assert len(result.confidences) == len(router.model_names)
        assert result.metadata["cluster"] is not None

    def test_unload(self, pkl_path):
        router = KMeansRouter()
        router.load(pkl_path)
        assert router.n_clusters == 100
        router.unload()
        assert router._kmeans_model is None
