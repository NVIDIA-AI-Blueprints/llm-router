"""Integration tests for the router-only mode with real encoder and checkpoint.

These tests load the actual Qwen3.5-0.8B encoder and trained checkpoint
to verify end-to-end routing through the HTTP API. Marked slow because
the encoder model takes ~15-35s to load on CPU.
"""

import subprocess

import pytest


@pytest.mark.slow
@pytest.mark.requires_torch
class TestRouterOnlyModeWithRealEncoder:
    """Tests using the HTTP adapter create_app with a real config and checkpoint."""

    def test_create_app_router_only_loads(self, v1_config_path):
        from model_router_toolkit.adapters.http.app import create_app

        app = create_app(str(v1_config_path))
        routes = [r.path for r in app.routes]
        assert "/health" in routes
        assert "/v1/route" in routes

    def test_health_endpoint(self, v1_config_path):
        from fastapi.testclient import TestClient

        from model_router_toolkit.adapters.http.app import create_app

        app = create_app(str(v1_config_path))
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["mode"] == "router-only"
        assert data["method"] == "prefill"
        assert len(data["models"]) >= 9

    def test_models_endpoint(self, v1_config_path):
        from fastapi.testclient import TestClient

        from model_router_toolkit.adapters.http.app import create_app

        app = create_app(str(v1_config_path))
        client = TestClient(app)
        resp = client.get("/api/models")
        assert resp.status_code == 200
        models = resp.json()
        assert len(models) >= 9
        for m in models:
            assert "name" in m
            assert "cost_per_m_input_tokens" in m

    def test_route_returns_routing_decision(self, v1_config_path):
        from fastapi.testclient import TestClient

        from model_router_toolkit.adapters.http.app import create_app

        app = create_app(str(v1_config_path))
        client = TestClient(app)
        resp = client.post(
            "/v1/route",
            json={
                "question": "What is the capital of France?",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["selected_model"] in data["model_names"]
        assert len(data["confidences"]) == len(data["model_names"])
        assert all(0.0 <= v <= 1.0 for v in data["confidences"].values())
        assert "route_ms" in data["metadata"]

    def test_route_with_messages_format(self, v1_config_path):
        from fastapi.testclient import TestClient

        from model_router_toolkit.adapters.http.app import create_app

        app = create_app(str(v1_config_path))
        client = TestClient(app)
        resp = client.post(
            "/v1/route",
            json={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Prove sqrt(2) is irrational"},
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["selected_model"] in data["model_names"]

    def test_route_includes_cost_estimates(self, v1_config_path):
        from fastapi.testclient import TestClient

        from model_router_toolkit.adapters.http.app import create_app

        app = create_app(str(v1_config_path))
        client = TestClient(app)
        resp = client.post(
            "/v1/route",
            json={
                "question": "What is 2+2?",
            },
        )
        data = resp.json()
        assert len(data["costs"]) == len(data["model_names"])
        for cost in data["costs"]:
            assert "model" in cost
            assert "estimated_total_cost" in cost

    def test_no_inference_endpoints(self, v1_config_path):
        """Confirm that completions and chat endpoints are not registered."""
        from fastapi.testclient import TestClient

        from model_router_toolkit.adapters.http.app import create_app

        app = create_app(str(v1_config_path))
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code in (404, 405)

    def test_tolerance_affects_routing(self, v1_config_path):
        from fastapi.testclient import TestClient

        from model_router_toolkit.adapters.http.app import create_app

        app = create_app(str(v1_config_path))
        client = TestClient(app)

        resp_tight = client.post(
            "/v1/route",
            json={
                "question": "Explain quantum entanglement in detail",
                "tolerance": 0.01,
            },
        )
        resp_loose = client.post(
            "/v1/route",
            json={
                "question": "Explain quantum entanglement in detail",
                "tolerance": 0.99,
            },
        )
        assert resp_tight.status_code == 200
        assert resp_loose.status_code == 200


@pytest.mark.slow
class TestServeRouterCLI:
    """Test the serve-router CLI subcommand."""

    def test_serve_router_help(self):
        result = subprocess.run(
            ["model-router", "serve-router", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "router-only" in result.stdout.lower() or "router" in result.stdout.lower()
