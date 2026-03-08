"""Embedding clients for the KMeans router.

Supports API-based embedding (build.nvidia.com, OpenRouter) and local
sentence-transformers. The API client uses raw requests -- no SDK dependency.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

import numpy as np
import requests


class EmbedClient(Protocol):
    def embed(self, text: str) -> np.ndarray: ...


class APIEmbedClient:
    """Embed via OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(
        self,
        model: str = "nvidia/llama-nemotron-embed-1b-v2",
        api_base: str = "https://integrate.api.nvidia.com/v1",
        api_key: str | None = None,
    ):
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")

    def embed(self, text: str) -> np.ndarray:
        resp = requests.post(
            f"{self.api_base}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": text,
                "input_type": "query",
                "encoding_format": "float",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return np.array(data["data"][0]["embedding"], dtype=np.float32)


class LocalEmbedClient:
    """Embed via sentence-transformers in-process."""

    def __init__(
        self,
        model_name: str = "nvidia/llama-nemotron-embed-1b-v2",
        trust_remote_code: bool = True,
    ):
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self._model_name, trust_remote_code=self._trust_remote_code,
            )

    def embed(self, text: str) -> np.ndarray:
        self._ensure_model()
        return self._model.encode(text, prompt_name="query").astype(np.float32)


def get_default_embed_client(config: Any = None) -> EmbedClient:
    """Build an embed client from config or environment defaults."""
    if config is None:
        return APIEmbedClient()

    routing = config.routing if hasattr(config, "routing") else config
    mode = getattr(routing, "embed_mode", "api")

    if mode == "local":
        model = getattr(routing, "embed_model", "nvidia/llama-nemotron-embed-1b-v2")
        return LocalEmbedClient(model_name=model)

    api_base = getattr(routing, "embed_api_base", "https://integrate.api.nvidia.com/v1")
    model = getattr(routing, "embed_model", "nvidia/llama-nemotron-embed-1b-v2")

    api_key = os.environ.get("NVIDIA_API_KEY", "") or os.environ.get("OPENROUTER_API_KEY", "")
    return APIEmbedClient(model=model, api_base=api_base, api_key=api_key)
