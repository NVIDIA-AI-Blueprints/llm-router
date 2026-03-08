"""Integration tests for embedding clients.

API tests require NVIDIA_API_KEY.
"""

import numpy as np
import pytest

from model_router_toolkit.kmeans.embed import (
    APIEmbedClient,
    LocalEmbedClient,
    get_default_embed_client,
)


class TestGetDefaultClient:
    """Factory function tests (no API keys needed)."""

    def test_get_default_embed_client_api_mode(self):
        class FakeConfig:
            class routing:
                embed_mode = "api"
                embed_model = "test-model"
                embed_api_base = "https://example.com/v1"

        client = get_default_embed_client(FakeConfig())
        assert isinstance(client, APIEmbedClient)

    def test_get_default_embed_client_local_mode(self):
        class FakeConfig:
            class routing:
                embed_mode = "local"
                embed_model = "test-model"

        client = get_default_embed_client(FakeConfig())
        assert isinstance(client, LocalEmbedClient)


@pytest.mark.slow
class TestAPIEmbedClientReal:
    """Tests calling the real NVIDIA embedding API."""

    @pytest.mark.requires_nvidia_api_key
    def test_api_embed_client_real(self):
        client = APIEmbedClient()
        embedding = client.embed("Hello world")
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert embedding.ndim == 1
        assert len(embedding) > 100

    @pytest.mark.requires_nvidia_api_key
    def test_api_embed_deterministic(self):
        client = APIEmbedClient()
        e1 = client.embed("Deterministic test")
        e2 = client.embed("Deterministic test")
        np.testing.assert_allclose(e1, e2, atol=1e-6)

    def test_api_embed_client_no_key_raises(self):
        import requests

        client = APIEmbedClient(api_key="invalid-key-for-test")
        with pytest.raises(requests.HTTPError):
            client.embed("Hello")
