"""KMeans-based routing: embed query -> cluster -> Platt calibration -> select."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from model_router_toolkit.router import BaseRouter, CostEstimate, RoutingResult


class KMeansRouter(BaseRouter):
    """Routes queries by embedding them and looking up cluster-level model accuracy."""

    def __init__(self, *, config: Any = None):
        self._config = config
        self._kmeans_model = None
        self._platt_models: dict = {}
        self._cluster_acc: dict = {}
        self._models: list[str] = []
        self._split_models: list[str] = []
        self._model_to_split: dict[str, str] = {}
        self._n_clusters: int = 0
        self._embed_client: Any = None
        self._cost_table: dict[str, dict] = {}

    def load(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        with open(path, "rb") as f:
            db = pickle.load(f)

        self._kmeans_model = db["kmeans_model"]
        self._platt_models = db.get("platt_models", {})
        self._cluster_acc = db["cluster_acc"]
        self._models = db["models"]
        self._split_models = db.get("split_models", [])
        self._model_to_split = {}
        if isinstance(self._split_models, list) and len(self._split_models) == len(self._models):
            self._model_to_split = dict(zip(self._models, self._split_models))
        elif isinstance(self._split_models, dict):
            self._model_to_split = self._split_models
        self._n_clusters = db.get("n_clusters", 100)

    @property
    def n_clusters(self) -> int:
        return self._n_clusters

    @property
    def model_names(self) -> list[str]:
        return list(self._models)

    @property
    def models_by_cost(self) -> list[str]:
        if self._cost_table:
            return sorted(self._models, key=lambda m: self._cost_table.get(m, {}).get("cost", 0))
        return list(self._models)

    def set_embed_client(self, client: Any) -> None:
        self._embed_client = client

    def set_cost_table(self, table: dict[str, dict]) -> None:
        self._cost_table = table

    def predict_probs(self, embedding: np.ndarray) -> tuple[int, dict[str, float]]:
        """Predict P(correct) for each model given an embedding vector."""
        cluster = int(self._kmeans_model.predict(embedding.reshape(1, -1))[0])
        probs = {}
        for model in self._models:
            split_key = self._model_to_split.get(model, model)
            raw_acc = self._cluster_acc.get(cluster, {}).get(split_key, 0.5)

            if model in self._platt_models:
                platt = self._platt_models[model]
                probs[model] = float(platt.predict_proba([[raw_acc]])[0, 1])
            else:
                probs[model] = raw_acc

        return cluster, probs

    def route(self, question: str, *, tolerance: float = 0.20) -> RoutingResult:
        if self._kmeans_model is None:
            raise RuntimeError("Router not loaded. Call load() first.")

        if self._embed_client is None:
            from model_router_toolkit.kmeans.embed import get_default_embed_client
            self._embed_client = get_default_embed_client(self._config)

        embedding = self._embed_client.embed(question)
        cluster, probs = self.predict_probs(embedding)

        p_max = max(probs.values())
        threshold = p_max - tolerance

        candidates = sorted(
            self._models,
            key=lambda m: self._cost_table.get(m, {}).get("cost", 0),
        )

        selected = candidates[-1]
        for model in candidates:
            if probs.get(model, 0) >= threshold:
                selected = model
                break

        model_names = list(probs.keys())
        confidences = [probs[m] for m in model_names]
        costs = []
        for m in model_names:
            ct = self._cost_table.get(m, {})
            costs.append(CostEstimate(
                median_output_tokens=ct.get("median_output_tokens", 500),
                cost_per_m_input_tokens=ct.get("cost_per_m_input_tokens", 0),
                cost_per_m_output_tokens=ct.get("cost_per_m_output_tokens", 0),
            ))

        return RoutingResult(
            model_names=model_names,
            confidences=confidences,
            costs=costs,
            selected_model=selected,
            metadata={
                "cluster": cluster,
                "probs": probs,
                "p_max": p_max,
                "threshold": threshold,
                "tolerance": tolerance,
            },
        )

    def unload(self) -> None:
        self._kmeans_model = None
        self._platt_models = {}
        self._cluster_acc = {}
