"""Integration test for OpenClaw sidecar: real router behind HTTP, mock plugin calls."""

import pytest

pytest.importorskip("fastapi")

from unittest.mock import patch

from httpx import AsyncClient

from model_router_toolkit.adapters.http.app import create_app


@pytest.fixture
def fake_config_path(tmp_path, sample_pool_config_dict):
    import yaml
    p = tmp_path / "pool.yaml"
    p.write_text(yaml.dump(sample_pool_config_dict))
    return str(p)


class FakeRouter:
    def route(self, question, *, tolerance=0.20, models=None):
        from model_router_toolkit.router import RoutingResult, CostEstimate
        return RoutingResult(
            model_names=["strong", "cheap"],
            confidences=[0.95, 0.70],
            costs=[
                CostEstimate(median_output_tokens=100, cost_per_m_input_tokens=3.0, cost_per_m_output_tokens=15.0),
                CostEstimate(median_output_tokens=100, cost_per_m_input_tokens=0.04, cost_per_m_output_tokens=0.04),
            ],
            selected_model="cheap",
            metadata={"tolerance": tolerance},
        )

    def load(self, path):
        pass

    def unload(self):
        pass


@pytest.mark.asyncio
class TestOpenClawSidecar:
    """Tests mimicking how the OpenClaw TS plugin calls the sidecar."""

    async def test_route_with_question(self, fake_config_path):
        with patch("model_router_toolkit.adapters.http.app.build_router_from_config", return_value=FakeRouter()):
            app = create_app(fake_config_path, warmup=False)
            async with AsyncClient(app=app, base_url="http://test") as client:
                resp = await client.post("/v1/route", json={
                    "question": "What is the capital of France?",
                    "tolerance": 0.20,
                })
                assert resp.status_code == 200
                data = resp.json()
                assert data["selected_model"] == "cheap"
                assert "strong" in data["confidences"]
                assert "cheap" in data["confidences"]

    async def test_route_with_messages(self, fake_config_path):
        with patch("model_router_toolkit.adapters.http.app.build_router_from_config", return_value=FakeRouter()):
            app = create_app(fake_config_path, warmup=False)
            async with AsyncClient(app=app, base_url="http://test") as client:
                resp = await client.post("/v1/route", json={
                    "messages": [{"role": "user", "content": "Hello"}],
                    "tolerance": 0.30,
                })
                assert resp.status_code == 200
                assert resp.json()["selected_model"] == "cheap"

    async def test_health(self, fake_config_path):
        with patch("model_router_toolkit.adapters.http.app.build_router_from_config", return_value=FakeRouter()):
            app = create_app(fake_config_path, warmup=False)
            async with AsyncClient(app=app, base_url="http://test") as client:
                resp = await client.get("/health")
                assert resp.status_code == 200
                assert resp.json()["mode"] == "router-only"

    async def test_pool_mapping_pattern(self, fake_config_path):
        """Verify the response format matches what the OpenClaw TS plugin expects."""
        with patch("model_router_toolkit.adapters.http.app.build_router_from_config", return_value=FakeRouter()):
            app = create_app(fake_config_path, warmup=False)
            async with AsyncClient(app=app, base_url="http://test") as client:
                resp = await client.post("/v1/route", json={"question": "test"})
                data = resp.json()
                assert "selected_model" in data
                assert "confidences" in data
                assert "costs" in data
                assert "metadata" in data
                assert isinstance(data["confidences"], dict)
                assert isinstance(data["costs"], list)
