"""Unit tests for the router-only HTTP adapter and webhook auth middleware.

Uses a FakeRouter to test the app factory, endpoints, request parsing,
and response formatting without loading any encoder or checkpoint.
"""

import hashlib
import hmac as hmac_mod
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient

from model_router_toolkit.config import PoolConfig
from model_router_toolkit.router import BaseRouter, CostEstimate, RoutingResult
from model_router_toolkit.adapters.http.app import create_app
from model_router_toolkit.adapters.http.route import (
    RouteRequest,
    _extract_question,
    _result_to_response,
)
from model_router_toolkit.adapters.http.auth import WebhookAuthMiddleware


class FakeRouter(BaseRouter):
    """Deterministic router for unit testing."""

    def __init__(self, model_names=None):
        self._model_names = model_names or ["model-a", "model-b"]
        self._load_called = False
        self._unload_called = False

    def load(self, checkpoint_path):
        self._load_called = True

    def unload(self):
        self._unload_called = True

    def route(self, question, *, tolerance=0.20):
        return RoutingResult(
            model_names=self._model_names,
            confidences=[0.92, 0.71],
            costs=[
                CostEstimate(
                    median_output_tokens=100,
                    cost_per_m_input_tokens=0.04,
                    cost_per_m_output_tokens=0.16,
                    estimated_total_cost=0.0001,
                ),
                CostEstimate(
                    median_output_tokens=200,
                    cost_per_m_input_tokens=1.75,
                    cost_per_m_output_tokens=14.00,
                    estimated_total_cost=0.003,
                ),
            ],
            selected_model=self._model_names[0],
            metadata={"p_max": 0.92, "threshold": 0.72, "tolerance": tolerance},
        )


@pytest.fixture
def fake_config_path(tmp_path):
    config = {
        "routing": {
            "method": "prefill",
            "checkpoint": "",
            "tolerance": 0.20,
            "encoder": "Qwen/Qwen3.5-0.8B",
        },
        "models": [
            {
                "name": "model-a",
                "display_name": "Model A",
                "litellm_model": "openrouter/test/model-a",
                "cost_per_m_input_tokens": 0.04,
                "cost_per_m_output_tokens": 0.16,
            },
            {
                "name": "model-b",
                "display_name": "Model B",
                "litellm_model": "openrouter/test/model-b",
                "cost_per_m_input_tokens": 1.75,
                "cost_per_m_output_tokens": 14.00,
            },
        ],
    }
    p = tmp_path / "test-config.yaml"
    p.write_text(yaml.dump(config))
    return str(p)


@pytest.fixture
def client(fake_config_path):
    with patch(
        "model_router_toolkit.adapters.http.app.build_router_from_config",
        return_value=FakeRouter(),
    ):
        app = create_app(fake_config_path, warmup=False)
        yield TestClient(app)


@pytest.fixture
def authed_client(fake_config_path):
    """Client with webhook auth middleware enabled."""
    with patch(
        "model_router_toolkit.adapters.http.app.build_router_from_config",
        return_value=FakeRouter(),
    ):
        app = create_app(fake_config_path, warmup=False)
        app.add_middleware(WebhookAuthMiddleware, secret="test-secret-123")
        yield TestClient(app)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestExtractQuestion:
    def test_from_question_field(self):
        req = RouteRequest(question="What is 2+2?")
        assert _extract_question(req) == "What is 2+2?"

    def test_from_messages(self):
        req = RouteRequest(messages=[
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What is gravity?"},
        ])
        assert _extract_question(req) == "What is gravity?"

    def test_last_user_message(self):
        req = RouteRequest(messages=[
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Follow-up question"},
        ])
        assert _extract_question(req) == "Follow-up question"

    def test_question_field_takes_priority(self):
        req = RouteRequest(
            question="Direct question",
            messages=[{"role": "user", "content": "Message question"}],
        )
        assert _extract_question(req) == "Direct question"

    def test_empty_request(self):
        req = RouteRequest()
        assert _extract_question(req) == ""

    def test_no_user_message(self):
        req = RouteRequest(messages=[
            {"role": "system", "content": "System prompt"},
        ])
        assert _extract_question(req) == ""


class TestResultToResponse:
    def test_basic_conversion(self):
        result = RoutingResult(
            model_names=["a", "b"],
            confidences=[0.9, 0.7],
            costs=[
                CostEstimate(
                    median_output_tokens=100,
                    cost_per_m_input_tokens=0.1,
                    cost_per_m_output_tokens=0.2,
                    estimated_total_cost=0.001,
                ),
                CostEstimate(
                    median_output_tokens=200,
                    cost_per_m_input_tokens=1.0,
                    cost_per_m_output_tokens=2.0,
                    estimated_total_cost=0.01,
                ),
            ],
            selected_model="a",
            metadata={"test": True},
        )
        resp = _result_to_response(result)
        assert resp.selected_model == "a"
        assert resp.confidences == {"a": 0.9, "b": 0.7}
        assert len(resp.costs) == 2
        assert resp.costs[0]["model"] == "a"
        assert resp.metadata == {"test": True}


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["mode"] == "router-only"
        assert data["method"] == "prefill"
        assert len(data["models"]) == 2

    def test_health_lists_model_names(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "model-a" in data["models"]
        assert "model-b" in data["models"]


class TestModelsEndpoint:
    def test_models_returns_list(self, client):
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["name"] == "model-a"
        assert data[0]["display_name"] == "Model A"
        assert data[0]["cost_per_m_input_tokens"] == 0.04

    def test_models_include_costs(self, client):
        resp = client.get("/api/models")
        data = resp.json()
        for model in data:
            assert "cost_per_m_input_tokens" in model
            assert "cost_per_m_output_tokens" in model


class TestRouteEndpoint:
    def test_route_with_question(self, client):
        resp = client.post("/v1/route", json={"question": "What is 2+2?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["selected_model"] == "model-a"
        assert "model-a" in data["confidences"]
        assert "model-b" in data["confidences"]
        assert len(data["costs"]) == 2
        assert "route_ms" in data["metadata"]

    def test_route_with_messages(self, client):
        resp = client.post("/v1/route", json={
            "messages": [{"role": "user", "content": "Hello world"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["selected_model"] == "model-a"

    def test_route_with_tolerance(self, client):
        resp = client.post("/v1/route", json={
            "question": "Hard question",
            "tolerance": 0.05,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["metadata"]["tolerance"] == 0.05

    def test_route_empty_question_returns_fallback(self, client):
        resp = client.post("/v1/route", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["metadata"].get("error") == "no question text provided"

    def test_route_response_has_cost_details(self, client):
        resp = client.post("/v1/route", json={"question": "Test"})
        data = resp.json()
        cost = data["costs"][0]
        assert "estimated_total_cost" in cost
        assert "cost_per_m_input_tokens" in cost
        assert "median_output_tokens" in cost

    def test_route_confidences_are_dict(self, client):
        resp = client.post("/v1/route", json={"question": "Test"})
        data = resp.json()
        assert isinstance(data["confidences"], dict)
        assert data["confidences"]["model-a"] == pytest.approx(0.92)
        assert data["confidences"]["model-b"] == pytest.approx(0.71)


class TestNoInferenceEndpoints:
    """Verify that inference endpoints from the full server are NOT present."""

    def test_no_completions_endpoint(self, client):
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert resp.status_code in (404, 405)

    def test_no_chat_endpoint(self, client):
        resp = client.post("/api/chat", json={"message": "Hello"})
        assert resp.status_code in (404, 405)

    def test_no_review_endpoint(self, client):
        resp = client.post("/api/review", json={"question": "Test"})
        assert resp.status_code in (404, 405)


# ---------------------------------------------------------------------------
# Webhook auth middleware tests
# ---------------------------------------------------------------------------

class TestWebhookAuth:
    """Tests for adapters/http/auth.py WebhookAuthMiddleware."""

    SECRET = "test-secret-123"

    def _sign(self, body: bytes) -> str:
        return hmac_mod.new(self.SECRET.encode(), body, hashlib.sha256).hexdigest()

    def test_hmac_valid_signature(self, authed_client):
        import json

        body = json.dumps({"question": "What is 2+2?"}).encode()
        sig = self._sign(body)
        resp = authed_client.post(
            "/v1/route",
            content=body,
            headers={"Content-Type": "application/json", "X-Webhook-Signature": sig},
        )
        assert resp.status_code == 200

    def test_hmac_invalid_signature(self, authed_client):
        import json

        body = json.dumps({"question": "What is 2+2?"}).encode()
        resp = authed_client.post(
            "/v1/route",
            content=body,
            headers={"Content-Type": "application/json", "X-Webhook-Signature": "bad-sig"},
        )
        assert resp.status_code == 401
        assert "Invalid webhook signature" in resp.json()["error"]

    def test_hmac_missing_header_returns_401(self, authed_client):
        resp = authed_client.post(
            "/v1/route",
            json={"question": "What is 2+2?"},
        )
        assert resp.status_code == 401
        assert "Missing authentication" in resp.json()["error"]

    def test_bearer_valid_token(self, authed_client):
        resp = authed_client.post(
            "/v1/route",
            json={"question": "What is 2+2?"},
            headers={"Authorization": f"Bearer {self.SECRET}"},
        )
        assert resp.status_code == 200

    def test_bearer_wrong_token(self, authed_client):
        resp = authed_client.post(
            "/v1/route",
            json={"question": "What is 2+2?"},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401
        assert "Invalid bearer token" in resp.json()["error"]

    def test_bearer_missing_auth_returns_401(self, authed_client):
        resp = authed_client.post(
            "/v1/route",
            json={"question": "What is 2+2?"},
        )
        assert resp.status_code == 401

    def test_no_secret_passes_all_requests(self, fake_config_path):
        """When no secret is configured, all requests pass through."""
        with patch(
            "model_router_toolkit.adapters.http.app.build_router_from_config",
            return_value=FakeRouter(),
        ):
            app = create_app(fake_config_path, warmup=False)
            app.add_middleware(WebhookAuthMiddleware, secret="")
            client = TestClient(app)
            resp = client.post("/v1/route", json={"question": "Test"})
            assert resp.status_code == 200

    def test_health_bypasses_auth(self, authed_client):
        resp = authed_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
