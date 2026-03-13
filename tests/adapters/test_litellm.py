import numpy as np
import pytest

from model_router_toolkit.router import BaseRouter, RoutingResult, CostEstimate
from model_router_toolkit.adapters.litellm.strategy import ModelRoutingStrategy


class FakeRouter(BaseRouter):
    """Deterministic router for testing the strategy wrapper."""

    def __init__(self, selected: str = "model-a"):
        self._selected = selected
        self._pool = ["model-a", "model-b"]

    def load(self, checkpoint_path):
        pass

    def route(self, question, *, tolerance=0.10):
        return RoutingResult(
            model_names=["model-a", "model-b"],
            confidences=[0.9, 0.7],
            costs=[
                CostEstimate(median_output_tokens=100, cost_per_m_input_tokens=0.1,
                             cost_per_m_output_tokens=0.1),
                CostEstimate(median_output_tokens=200, cost_per_m_input_tokens=1.0,
                             cost_per_m_output_tokens=1.0),
            ],
            selected_model=self._selected,
            metadata={"test": True},
        )

    def has_model(self, model_name):
        return model_name in self._pool

    def resolve(self, model_name):
        if model_name not in self._pool:
            return None
        return RoutingResult(
            model_names=self._pool,
            confidences=[1.0 if m == model_name else 0.0 for m in self._pool],
            costs=[
                CostEstimate(median_output_tokens=100, cost_per_m_input_tokens=0.1,
                             cost_per_m_output_tokens=0.1),
                CostEstimate(median_output_tokens=200, cost_per_m_input_tokens=1.0,
                             cost_per_m_output_tokens=1.0),
            ],
            selected_model=model_name,
            metadata={"pinned": True},
        )


class TestModelRoutingStrategy:
    def test_sync_routing(self):
        strategy = ModelRoutingStrategy(FakeRouter("model-a"), tolerance=0.20)
        strategy._litellm_router = type("R", (), {
            "model_list": [
                {"model_name": "model-a", "litellm_params": {"model": "openai/a"}},
                {"model_name": "model-b", "litellm_params": {"model": "openai/b"}},
            ]
        })()

        dep = strategy.get_available_deployment(
            model="test",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert dep["model_name"] == "model-a"
        assert strategy.last_result is not None
        assert strategy.last_result.selected_model == "model-a"

    def test_tolerance_bounds(self):
        strategy = ModelRoutingStrategy(FakeRouter(), tolerance=0.10)
        strategy.tolerance = -0.5
        assert strategy.tolerance == 0.0
        strategy.tolerance = 1.5
        assert strategy.tolerance == 1.0

    def test_empty_messages(self):
        strategy = ModelRoutingStrategy(FakeRouter(), tolerance=0.20)
        strategy._litellm_router = type("R", (), {"model_list": [{"model_name": "x"}]})()
        dep = strategy.get_available_deployment(model="test", messages=[])
        assert strategy.last_result is None

    def test_extract_user_text(self):
        strategy = ModelRoutingStrategy(FakeRouter())
        text = strategy._extract_user_text([
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What is 2+2?"},
        ])
        assert text == "What is 2+2?"

    def test_extract_multipart_content(self):
        strategy = ModelRoutingStrategy(FakeRouter())
        text = strategy._extract_user_text([
            {"role": "user", "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ]},
        ])
        assert text == "Hello World"

    @pytest.mark.asyncio
    async def test_async_routing(self):
        strategy = ModelRoutingStrategy(FakeRouter("model-b"), tolerance=0.20)
        strategy._litellm_router = type("R", (), {
            "model_list": [
                {"model_name": "model-a", "litellm_params": {"model": "openai/a"}},
                {"model_name": "model-b", "litellm_params": {"model": "openai/b"}},
            ]
        })()

        dep = await strategy.async_get_available_deployment(
            model="test",
            messages=[{"role": "user", "content": "Test"}],
        )
        assert dep["model_name"] == "model-b"

    def test_pin_model_metadata_bypasses_routing(self):
        """When request_kwargs has pin_model, skip ML and return pinned model."""
        strategy = ModelRoutingStrategy(FakeRouter("model-a"), tolerance=0.20)
        strategy._litellm_router = type("R", (), {
            "model_list": [
                {"model_name": "model-a", "litellm_params": {"model": "openai/a"}},
                {"model_name": "model-b", "litellm_params": {"model": "openai/b"}},
            ]
        })()

        dep = strategy.get_available_deployment(
            model="model-a",
            messages=[{"role": "user", "content": "Hello"}],
            request_kwargs={"metadata": {"pin_model": "model-b"}},
        )
        assert dep["model_name"] == "model-b"
        assert strategy.last_result is not None
        assert strategy.last_result.selected_model == "model-b"
        assert strategy.last_result.metadata.get("pinned") is True

    def test_pin_model_unknown_falls_through(self):
        """When pin_model is not in the pool, route normally via ML."""
        strategy = ModelRoutingStrategy(FakeRouter("model-a"), tolerance=0.20)
        strategy._litellm_router = type("R", (), {
            "model_list": [
                {"model_name": "model-a", "litellm_params": {"model": "openai/a"}},
                {"model_name": "model-b", "litellm_params": {"model": "openai/b"}},
            ]
        })()

        dep = strategy.get_available_deployment(
            model="model-a",
            messages=[{"role": "user", "content": "Hello"}],
            request_kwargs={"metadata": {"pin_model": "unknown-model"}},
        )
        assert dep["model_name"] == "model-a"
        assert strategy.last_result.metadata.get("pinned") is None

    def test_no_pin_model_routes_normally(self):
        """Without pin_model metadata, always route via ML even if model is a pool name."""
        strategy = ModelRoutingStrategy(FakeRouter("model-a"), tolerance=0.20)
        strategy._litellm_router = type("R", (), {
            "model_list": [
                {"model_name": "model-a", "litellm_params": {"model": "openai/a"}},
                {"model_name": "model-b", "litellm_params": {"model": "openai/b"}},
            ]
        })()

        dep = strategy.get_available_deployment(
            model="model-b",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert dep["model_name"] == "model-a"
        assert strategy.last_result.metadata.get("test") is True

    @pytest.mark.asyncio
    async def test_pin_model_async(self):
        """Async path also respects pin_model metadata."""
        strategy = ModelRoutingStrategy(FakeRouter("model-a"), tolerance=0.20)
        strategy._litellm_router = type("R", (), {
            "model_list": [
                {"model_name": "model-a", "litellm_params": {"model": "openai/a"}},
                {"model_name": "model-b", "litellm_params": {"model": "openai/b"}},
            ]
        })()

        dep = await strategy.async_get_available_deployment(
            model="model-a",
            messages=[{"role": "user", "content": "Test"}],
            request_kwargs={"metadata": {"pin_model": "model-b"}},
        )
        assert dep["model_name"] == "model-b"
        assert strategy.last_result.metadata.get("pinned") is True
