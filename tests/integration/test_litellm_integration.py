"""LiteLLM integration tests.

Tests the wiring between model-router-toolkit and litellm:
- Model list construction from PoolConfig
- API key resolution by provider prefix and api_base
- ModelRoutingStrategy plugged into litellm.Router
- FastAPI endpoints using the litellm.Router internally
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from model_router_toolkit.config import ModelSpec, PoolConfig, RoutingConfig
from model_router_toolkit.router import BaseRouter, CostEstimate, RoutingResult
from model_router_toolkit.server.app import _build_model_list, _resolve_api_key
from model_router_toolkit.strategy import ModelRoutingStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StubRouter(BaseRouter):
    """Deterministic router for integration testing."""

    def __init__(self, model_names: list[str], selected: str | None = None):
        self._model_names = model_names
        self._selected = selected or model_names[0]

    def load(self, checkpoint_path):
        pass

    def route(self, question: str, *, tolerance: float = 0.20) -> RoutingResult:
        n = len(self._model_names)
        return RoutingResult(
            model_names=self._model_names,
            confidences=[0.9 - i * 0.1 for i in range(n)],
            costs=[
                CostEstimate(
                    median_output_tokens=100 * (i + 1),
                    cost_per_m_input_tokens=0.1 * (i + 1),
                    cost_per_m_output_tokens=0.1 * (i + 1),
                )
                for i in range(n)
            ],
            selected_model=self._selected,
            metadata={"source": "stub"},
        )

    def unload(self):
        pass


def _make_config(models=None) -> PoolConfig:
    if models is None:
        models = [
            ModelSpec(
                name="cheap-model",
                display_name="Cheap Model",
                litellm_model="nvidia_nim/nvidia/test-small",
                cost_per_m_input_tokens=0.10,
                cost_per_m_output_tokens=0.10,
            ),
            ModelSpec(
                name="expensive-model",
                display_name="Expensive Model",
                litellm_model="openrouter/openai/gpt-4o",
                cost_per_m_input_tokens=2.50,
                cost_per_m_output_tokens=10.00,
            ),
        ]
    return PoolConfig(
        routing=RoutingConfig(method="kmeans", tolerance=0.20),
        models=models,
    )


def _fake_litellm_response(content: str = "test response", model: str = "test"):
    """Create a mock object mimicking a litellm ModelResponse."""
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = content
    mock_resp.choices[0].delta.content = content
    mock_resp.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
    mock_resp.model = model
    mock_resp.model_dump.return_value = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    return mock_resp


def _make_test_app(
    model_names: list[str] | None = None,
    selected: str | None = None,
) -> FastAPI:
    """Build a minimal FastAPI app with mocked routing components."""
    from litellm import Router as LiteLLMRouter

    from model_router_toolkit.server.chat import router as chat_router
    from model_router_toolkit.server.completions import router as completions_router

    config = _make_config()
    if model_names is None:
        model_names = [m.name for m in config.models]

    model_list = [
        {
            "model_name": m.name,
            "litellm_params": {"model": f"openai/{m.name}", "api_key": "test-key"},
        }
        for m in config.models
    ]

    litellm_router = LiteLLMRouter(model_list=model_list)
    stub = StubRouter(model_names, selected=selected or model_names[0])
    strategy = ModelRoutingStrategy(stub, tolerance=0.20)
    strategy.set_litellm_router(litellm_router)
    litellm_router.set_custom_routing_strategy(strategy)

    app = FastAPI()
    app.state.litellm_router = litellm_router
    app.state.strategy = strategy
    app.state.config = config

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "method": config.routing.method,
            "models": config.model_names,
        }

    @app.get("/api/models")
    async def get_models():
        return [
            {
                "name": m.name,
                "display_name": m.display_name or m.name,
                "cost_per_m_input_tokens": m.cost_per_m_input_tokens,
                "cost_per_m_output_tokens": m.cost_per_m_output_tokens,
            }
            for m in config.models
        ]

    app.include_router(chat_router, prefix="/api", tags=["chat"])
    app.include_router(completions_router, prefix="/v1", tags=["completions"])
    return app


# ---------------------------------------------------------------------------
# _resolve_api_key
# ---------------------------------------------------------------------------


class TestResolveApiKey:
    def test_nvidia_nim_prefix(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvda-key-123")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key-456")
        assert _resolve_api_key("nvidia_nim/some-model", "") == "nvda-key-123"

    def test_openrouter_prefix(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvda-key-123")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key-456")
        assert _resolve_api_key("openrouter/some-model", "") == "or-key-456"

    def test_nvidia_api_base_fallback(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvda-key-123")
        result = _resolve_api_key("plain-model", "https://integrate.api.nvidia.com/v1")
        assert result == "nvda-key-123"

    def test_openrouter_api_base_fallback(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key-456")
        result = _resolve_api_key("plain-model", "https://openrouter.ai/api/v1")
        assert result == "or-key-456"

    def test_fallback_returns_empty_when_no_keys(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert _resolve_api_key("plain-model", "https://custom.example.com/v1") == ""

    def test_prefix_takes_precedence_over_api_base(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvda-key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        result = _resolve_api_key("openrouter/model", "https://integrate.api.nvidia.com/v1")
        assert result == "or-key"


# ---------------------------------------------------------------------------
# _build_model_list
# ---------------------------------------------------------------------------


class TestBuildModelList:
    def test_preserves_nvidia_nim_prefix(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        config = _make_config()
        model_list = _build_model_list(config)
        nvidia_entry = next(e for e in model_list if e["model_name"] == "cheap-model")
        assert nvidia_entry["litellm_params"]["model"] == "nvidia_nim/nvidia/test-small"

    def test_preserves_openrouter_prefix(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        config = _make_config()
        model_list = _build_model_list(config)
        or_entry = next(e for e in model_list if e["model_name"] == "expensive-model")
        assert or_entry["litellm_params"]["model"] == "openrouter/openai/gpt-4o"

    def test_adds_nvidia_nim_prefix_for_bare_model(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        config = _make_config(
            models=[
                ModelSpec(
                    name="bare",
                    litellm_model="my-model",
                    api_base="https://integrate.api.nvidia.com/v1",
                ),
            ]
        )
        model_list = _build_model_list(config)
        assert model_list[0]["litellm_params"]["model"] == "nvidia_nim/my-model"

    def test_adds_openrouter_prefix_for_bare_model(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        config = _make_config(
            models=[
                ModelSpec(
                    name="bare",
                    litellm_model="my-model",
                    api_base="https://openrouter.ai/api/v1",
                ),
            ]
        )
        model_list = _build_model_list(config)
        assert model_list[0]["litellm_params"]["model"] == "openrouter/my-model"

    def test_adds_openai_prefix_for_unknown_base(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        config = _make_config(
            models=[
                ModelSpec(
                    name="bare",
                    litellm_model="custom-model",
                    api_base="https://custom.example.com/v1",
                ),
            ]
        )
        model_list = _build_model_list(config)
        assert model_list[0]["litellm_params"]["model"] == "openai/custom-model"

    def test_model_count_matches_config(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        config = _make_config()
        model_list = _build_model_list(config)
        assert len(model_list) == len(config.models)

    def test_entries_have_required_keys(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        config = _make_config()
        model_list = _build_model_list(config)
        for entry in model_list:
            assert "model_name" in entry
            assert "litellm_params" in entry
            assert "model" in entry["litellm_params"]
            assert "api_key" in entry["litellm_params"]

    def test_api_key_assigned_per_provider(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvda-key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        config = _make_config()
        model_list = _build_model_list(config)
        nvidia_entry = next(e for e in model_list if e["model_name"] == "cheap-model")
        or_entry = next(e for e in model_list if e["model_name"] == "expensive-model")
        assert nvidia_entry["litellm_params"]["api_key"] == "nvda-key"
        assert or_entry["litellm_params"]["api_key"] == "or-key"


# ---------------------------------------------------------------------------
# Strategy + litellm.Router wiring
# ---------------------------------------------------------------------------


class TestStrategyRouterWiring:
    def test_set_custom_routing_strategy_accepted(self):
        """litellm.Router accepts our ModelRoutingStrategy."""
        from litellm import Router as LiteLLMRouter

        model_list = [
            {
                "model_name": "model-a",
                "litellm_params": {"model": "openai/fake-a", "api_key": "test"},
            },
        ]
        litellm_router = LiteLLMRouter(model_list=model_list)
        stub = StubRouter(["model-a"])
        strategy = ModelRoutingStrategy(stub, tolerance=0.20)
        strategy.set_litellm_router(litellm_router)
        litellm_router.set_custom_routing_strategy(strategy)

    def test_strategy_finds_deployment_from_router(self):
        """After set_litellm_router, the strategy resolves deployments by name."""
        from litellm import Router as LiteLLMRouter

        model_list = [
            {
                "model_name": "model-a",
                "litellm_params": {"model": "openai/fake-a", "api_key": "test"},
            },
            {
                "model_name": "model-b",
                "litellm_params": {"model": "openai/fake-b", "api_key": "test"},
            },
        ]
        litellm_router = LiteLLMRouter(model_list=model_list)
        stub = StubRouter(["model-a", "model-b"])
        strategy = ModelRoutingStrategy(stub, tolerance=0.20)
        strategy.set_litellm_router(litellm_router)

        dep_a = strategy._find_deployment("model-a")
        dep_b = strategy._find_deployment("model-b")
        assert dep_a is not None and dep_a["model_name"] == "model-a"
        assert dep_b is not None and dep_b["model_name"] == "model-b"
        assert strategy._find_deployment("nonexistent") is None

    def test_sync_routing_selects_correct_deployment(self):
        """get_available_deployment returns the deployment for the routed model."""
        from litellm import Router as LiteLLMRouter

        model_list = [
            {
                "model_name": "cheap",
                "litellm_params": {"model": "openai/cheap", "api_key": "k"},
            },
            {
                "model_name": "expensive",
                "litellm_params": {"model": "openai/expensive", "api_key": "k"},
            },
        ]
        litellm_router = LiteLLMRouter(model_list=model_list)
        stub = StubRouter(["cheap", "expensive"], selected="cheap")
        strategy = ModelRoutingStrategy(stub, tolerance=0.20)
        strategy.set_litellm_router(litellm_router)
        litellm_router.set_custom_routing_strategy(strategy)

        dep = strategy.get_available_deployment(
            model="test",
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )
        assert dep["model_name"] == "cheap"
        assert strategy.last_result is not None
        assert strategy.last_result.selected_model == "cheap"
        assert strategy.last_result.metadata == {"source": "stub"}

    @pytest.mark.asyncio
    async def test_async_routing_selects_correct_deployment(self):
        """async_get_available_deployment returns the deployment for the routed model."""
        from litellm import Router as LiteLLMRouter

        model_list = [
            {
                "model_name": "cheap",
                "litellm_params": {"model": "openai/cheap", "api_key": "k"},
            },
            {
                "model_name": "expensive",
                "litellm_params": {"model": "openai/expensive", "api_key": "k"},
            },
        ]
        litellm_router = LiteLLMRouter(model_list=model_list)
        stub = StubRouter(["cheap", "expensive"], selected="expensive")
        strategy = ModelRoutingStrategy(stub, tolerance=0.20)
        strategy.set_litellm_router(litellm_router)

        dep = await strategy.async_get_available_deployment(
            model="test",
            messages=[{"role": "user", "content": "Prove P=NP"}],
        )
        assert dep["model_name"] == "expensive"
        assert strategy.last_result.selected_model == "expensive"

    def test_empty_text_falls_back_to_first_deployment(self):
        """With no extractable user text, strategy falls back to model_list[0]."""
        from litellm import Router as LiteLLMRouter

        model_list = [
            {
                "model_name": "first",
                "litellm_params": {"model": "openai/first", "api_key": "k"},
            },
        ]
        litellm_router = LiteLLMRouter(model_list=model_list)
        stub = StubRouter(["first"])
        strategy = ModelRoutingStrategy(stub, tolerance=0.20)
        strategy.set_litellm_router(litellm_router)

        dep = strategy.get_available_deployment(model="test", messages=[])
        assert dep["model_name"] == "first"
        assert strategy.last_result is None

    def test_input_string_used_when_no_messages(self):
        """Strategy uses the `input` param when messages has no user content."""
        from litellm import Router as LiteLLMRouter

        model_list = [
            {
                "model_name": "m",
                "litellm_params": {"model": "openai/m", "api_key": "k"},
            },
        ]
        litellm_router = LiteLLMRouter(model_list=model_list)
        stub = StubRouter(["m"])
        strategy = ModelRoutingStrategy(stub, tolerance=0.20)
        strategy.set_litellm_router(litellm_router)

        dep = strategy.get_available_deployment(
            model="test",
            messages=[{"role": "system", "content": "You are helpful"}],
            input="fallback question",
        )
        assert dep["model_name"] == "m"
        assert strategy.last_result is not None

    def test_routing_changes_with_selected_model(self):
        """Different StubRouter selections yield different deployments."""
        from litellm import Router as LiteLLMRouter

        model_list = [
            {
                "model_name": "a",
                "litellm_params": {"model": "openai/a", "api_key": "k"},
            },
            {
                "model_name": "b",
                "litellm_params": {"model": "openai/b", "api_key": "k"},
            },
        ]
        litellm_router = LiteLLMRouter(model_list=model_list)

        for selected in ["a", "b"]:
            stub = StubRouter(["a", "b"], selected=selected)
            strategy = ModelRoutingStrategy(stub, tolerance=0.20)
            strategy.set_litellm_router(litellm_router)
            dep = strategy.get_available_deployment(
                model="test",
                messages=[{"role": "user", "content": "hello"}],
            )
            assert dep["model_name"] == selected

    @pytest.mark.asyncio
    async def test_router_acompletion_invokes_custom_strategy(self):
        """litellm.Router.acompletion() flows through our custom strategy.

        set_custom_routing_strategy copies method references via setattr,
        so the spy must be installed BEFORE the strategy is registered.
        We let the API call fail (no real server) and verify the strategy
        was invoked before the error.
        """
        import litellm
        from litellm import Router as LiteLLMRouter

        model_list = [
            {
                "model_name": "model-a",
                "litellm_params": {"model": "openai/fake-a", "api_key": "test"},
            },
            {
                "model_name": "model-b",
                "litellm_params": {"model": "openai/fake-b", "api_key": "test"},
            },
        ]
        litellm_router = LiteLLMRouter(model_list=model_list, num_retries=0)
        stub = StubRouter(["model-a", "model-b"], selected="model-a")
        strategy = ModelRoutingStrategy(stub, tolerance=0.20)
        strategy.set_litellm_router(litellm_router)

        strategy_calls = []
        original_async_get = strategy.async_get_available_deployment
        original_sync_get = strategy.get_available_deployment

        async def spy_async_get(*args, **kwargs):
            strategy_calls.append(("async", kwargs))
            return await original_async_get(*args, **kwargs)

        def spy_sync_get(*args, **kwargs):
            strategy_calls.append(("sync", kwargs))
            return original_sync_get(*args, **kwargs)

        strategy.async_get_available_deployment = spy_async_get
        strategy.get_available_deployment = spy_sync_get

        litellm_router.set_custom_routing_strategy(strategy)

        with pytest.raises(Exception):
            await litellm_router.acompletion(
                model="model-a",
                messages=[{"role": "user", "content": "What is the meaning of life?"}],
            )

        assert len(strategy_calls) > 0, "Strategy was never invoked by litellm.Router"
        assert strategy.last_result is not None
        assert strategy.last_result.selected_model == "model-a"


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_returns_ok_with_model_list(self):
        app = _make_test_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert set(data["models"]) == {"cheap-model", "expensive-model"}

    def test_models_endpoint_returns_all_models(self):
        app = _make_test_app()
        client = TestClient(app)
        resp = client.get("/api/models")
        assert resp.status_code == 200
        models = resp.json()
        assert len(models) == 2
        names = {m["name"] for m in models}
        assert names == {"cheap-model", "expensive-model"}
        for m in models:
            assert "display_name" in m
            assert "cost_per_m_input_tokens" in m
            assert "cost_per_m_output_tokens" in m


class TestCompletionsEndpoint:
    def _post_completion(self, client, app, messages, *, stream=False, **extra):
        """Post to /v1/chat/completions with the strategy invoked before the mock."""
        strategy = app.state.strategy
        litellm_router = app.state.litellm_router
        fake_resp = _fake_litellm_response("mocked answer")

        async def mock_acompletion(**kwargs):
            await strategy.async_get_available_deployment(
                model=kwargs.get("model", ""),
                messages=kwargs.get("messages"),
            )
            return fake_resp

        with patch.object(litellm_router, "acompletion", side_effect=mock_acompletion):
            return client.post(
                "/v1/chat/completions",
                json={"messages": messages, "stream": stream, **extra},
            )

    def test_non_streaming_returns_200(self):
        app = _make_test_app()
        client = TestClient(app)
        resp = self._post_completion(
            client,
            app,
            [{"role": "user", "content": "What is the capital of France?"}],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "mocked answer"

    def test_response_includes_routing_metadata(self):
        app = _make_test_app()
        client = TestClient(app)
        resp = self._post_completion(
            client,
            app,
            [{"role": "user", "content": "What is 6 * 7?"}],
        )
        data = resp.json()
        assert "routing" in data
        assert data["routing"]["selected_model"] == "cheap-model"
        assert "cheap-model" in data["routing"]["confidences"]
        assert "expensive-model" in data["routing"]["confidences"]

    def test_tolerance_override_via_body(self):
        app = _make_test_app()
        client = TestClient(app)
        resp = self._post_completion(
            client,
            app,
            [{"role": "user", "content": "test"}],
            tolerance=0.05,
        )
        assert resp.status_code == 200

    def test_streaming_returns_sse(self):
        app = _make_test_app()
        client = TestClient(app)
        litellm_router = app.state.litellm_router

        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "streamed"
        chunk.model_dump.return_value = {
            "choices": [{"delta": {"content": "streamed"}}],
        }

        async def mock_stream(**kwargs):
            class AsyncChunks:
                def __init__(self):
                    self._items = [chunk]
                    self._idx = 0

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    if self._idx >= len(self._items):
                        raise StopAsyncIteration
                    item = self._items[self._idx]
                    self._idx += 1
                    return item

            return AsyncChunks()

        with patch.object(litellm_router, "acompletion", side_effect=mock_stream):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        assert "streamed" in resp.text


class TestChatEndpoint:
    def test_chat_returns_sse_with_routing_event(self):
        app = _make_test_app()
        client = TestClient(app)
        litellm_router = app.state.litellm_router

        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = "Hello world"

        async def mock_stream(**kwargs):
            class AsyncChunks:
                def __init__(self):
                    self._items = [chunk]
                    self._idx = 0

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    if self._idx >= len(self._items):
                        raise StopAsyncIteration
                    item = self._items[self._idx]
                    self._idx += 1
                    return item

            return AsyncChunks()

        with patch.object(litellm_router, "acompletion", side_effect=mock_stream):
            resp = client.post(
                "/api/chat",
                json={"message": "Hi there", "tolerance": 0.15},
            )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = resp.text

        assert "event: routing" in body
        assert "cheap-model" in body
        assert "event: token" in body
        assert "Hello world" in body
        assert "event: done" in body

    def test_chat_tolerance_is_applied(self):
        app = _make_test_app()
        client = TestClient(app)
        litellm_router = app.state.litellm_router

        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = "ok"

        async def mock_stream(**kwargs):
            class AsyncChunks:
                def __init__(self):
                    self._items = [chunk]
                    self._idx = 0

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    if self._idx >= len(self._items):
                        raise StopAsyncIteration
                    item = self._items[self._idx]
                    self._idx += 1
                    return item

            return AsyncChunks()

        with patch.object(litellm_router, "acompletion", side_effect=mock_stream):
            resp = client.post(
                "/api/chat",
                json={"message": "test", "tolerance": 0.42},
            )

        assert resp.status_code == 200
        body = resp.text
        assert "event: routing" in body

    def test_chat_with_no_tokens_emits_error_on_exception(self):
        app = _make_test_app()
        client = TestClient(app)
        litellm_router = app.state.litellm_router

        async def mock_fail(**kwargs):
            raise ConnectionError("API unreachable")

        with patch.object(litellm_router, "acompletion", side_effect=mock_fail):
            resp = client.post(
                "/api/chat",
                json={"message": "test", "tolerance": 0.10},
            )

        assert resp.status_code == 200
        body = resp.text
        assert "event: routing" in body
        assert "event: error" in body
        assert "API unreachable" in body


# ---------------------------------------------------------------------------
# Collect module — litellm.completion usage
# ---------------------------------------------------------------------------


class TestCollectLiteLLMUsage:
    def test_call_model_uses_litellm_completion(self):
        """collect._call_model dispatches through litellm.completion."""
        from model_router_toolkit.collect import _call_model

        fake_resp = MagicMock()
        fake_resp.choices = [MagicMock()]
        fake_resp.choices[0].message.content = "Paris"
        fake_resp.usage = MagicMock(completion_tokens=5)

        with patch("litellm.completion", return_value=fake_resp) as mock_comp:
            content, tokens = _call_model(
                "openai/gpt-test",
                "What is the capital of France?",
                system_prompt="Be concise.",
            )

        mock_comp.assert_called_once()
        call_kwargs = mock_comp.call_args
        assert call_kwargs.kwargs["model"] == "openai/gpt-test"
        msgs = call_kwargs.kwargs["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert content == "Paris"
        assert tokens == 5

    def test_call_model_without_system_prompt(self):
        """When no system_prompt, only the user message is sent."""
        from model_router_toolkit.collect import _call_model

        fake_resp = MagicMock()
        fake_resp.choices = [MagicMock()]
        fake_resp.choices[0].message.content = "42"
        fake_resp.usage = MagicMock(completion_tokens=1)

        with patch("litellm.completion", return_value=fake_resp) as mock_comp:
            _call_model("openai/gpt-test", "What is 6*7?")

        msgs = mock_comp.call_args.kwargs["messages"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
