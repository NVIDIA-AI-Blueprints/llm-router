"""Unit tests for the external-sidecar LiteLLM hook.

Tests focus on hook orchestration (pin, fallback, circuit breaker). The HTTP
request shape itself is exercised once with a mock transport.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

litellm = pytest.importorskip("litellm")
httpx = pytest.importorskip("httpx")

from model_router_toolkit.adapters.litellm import external_hook as eh  # noqa: E402


def _make_hook(**overrides: Any) -> eh.ExternalRouterHook:
    defaults = dict(
        sidecar_url="http://router.test:8079",
        timeout_s=0.5,
        tolerance=0.20,
        default_model="fallback-model",
        failures_before_open=3,
        open_duration_s=0.05,
    )
    defaults.update(overrides)
    return eh.ExternalRouterHook(**defaults)


def _data(**extra: Any) -> dict:
    base = {"model": "client-asked-for-this", "messages": [{"role": "user", "content": "hi"}]}
    base.update(extra)
    return base


@pytest.mark.asyncio
async def test_routes_to_sidecar_decision(monkeypatch):
    hook = _make_hook()

    async def fake_ask(self, messages, tolerance):
        assert messages == [{"role": "user", "content": "hi"}]
        assert tolerance == 0.20
        return "routed-model"

    monkeypatch.setattr(eh.ExternalRouterHook, "_ask_sidecar", fake_ask)

    out = await hook.async_pre_call_hook(None, None, _data(), "completion")
    assert out["model"] == "routed-model"


@pytest.mark.asyncio
async def test_pin_model_bypasses_sidecar(monkeypatch):
    hook = _make_hook()
    called = False

    async def fake_ask(self, messages, tolerance):
        nonlocal called
        called = True
        return "should-not-be-used"

    monkeypatch.setattr(eh.ExternalRouterHook, "_ask_sidecar", fake_ask)

    data = _data(metadata={"pin_model": "pinned-model"})
    out = await hook.async_pre_call_hook(None, None, data, "completion")
    assert out["model"] == "pinned-model"
    assert called is False


@pytest.mark.asyncio
async def test_per_request_tolerance_override(monkeypatch):
    hook = _make_hook(tolerance=0.20)
    seen_tolerance: list[float] = []

    async def fake_ask(self, messages, tolerance):
        seen_tolerance.append(tolerance)
        return "x"

    monkeypatch.setattr(eh.ExternalRouterHook, "_ask_sidecar", fake_ask)

    await hook.async_pre_call_hook(None, None, _data(metadata={"tolerance": 0.5}), "completion")
    assert seen_tolerance == [0.5]


@pytest.mark.asyncio
async def test_non_completion_call_type_passes_through(monkeypatch):
    hook = _make_hook()

    async def fake_ask(self, messages, tolerance):
        raise AssertionError("should not be called for embeddings")

    monkeypatch.setattr(eh.ExternalRouterHook, "_ask_sidecar", fake_ask)

    data = _data()
    out = await hook.async_pre_call_hook(None, None, data, "embeddings")
    assert out["model"] == "client-asked-for-this"


@pytest.mark.asyncio
async def test_empty_messages_passes_through(monkeypatch):
    hook = _make_hook()

    async def fake_ask(self, messages, tolerance):
        raise AssertionError("should not call sidecar without messages")

    monkeypatch.setattr(eh.ExternalRouterHook, "_ask_sidecar", fake_ask)

    out = await hook.async_pre_call_hook(None, None, {"model": "x", "messages": []}, "completion")
    assert out["model"] == "x"


@pytest.mark.asyncio
async def test_sidecar_failure_falls_back(monkeypatch):
    hook = _make_hook(default_model="fallback-model")

    async def boom(self, messages, tolerance):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(eh.ExternalRouterHook, "_ask_sidecar", boom)

    out = await hook.async_pre_call_hook(None, None, _data(), "completion")
    assert out["model"] == "fallback-model"


@pytest.mark.asyncio
async def test_sidecar_failure_without_default_raises(monkeypatch):
    hook = _make_hook(default_model=None)

    async def boom(self, messages, tolerance):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(eh.ExternalRouterHook, "_ask_sidecar", boom)

    with pytest.raises(httpx.ConnectError):
        await hook.async_pre_call_hook(None, None, _data(), "completion")


@pytest.mark.asyncio
async def test_circuit_breaker_opens_then_closes(monkeypatch):
    hook = _make_hook(failures_before_open=2, open_duration_s=0.05)
    call_count = 0

    async def maybe_fail(self, messages, tolerance):
        nonlocal call_count
        call_count += 1
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(eh.ExternalRouterHook, "_ask_sidecar", maybe_fail)

    # 2 calls exhaust the breaker — both still attempt the sidecar
    for _ in range(2):
        await hook.async_pre_call_hook(None, None, _data(), "completion")
    assert call_count == 2

    # 3rd call: breaker open, sidecar skipped, fallback used
    await hook.async_pre_call_hook(None, None, _data(), "completion")
    assert call_count == 2

    # cooldown expires; next call probes sidecar again
    await asyncio.sleep(0.06)
    await hook.async_pre_call_hook(None, None, _data(), "completion")
    assert call_count == 3


@pytest.mark.asyncio
async def test_breaker_resets_on_success(monkeypatch):
    """A success between failures must reset the consecutive-failure counter.

    With failures_before_open=2, the sequence fail/succeed/fail must NOT open
    the breaker — without the reset, two failures (separated by a success)
    would falsely trip it.
    """
    hook = _make_hook(failures_before_open=2)
    outcomes = iter([("fail", None), ("ok", "ok-model"), ("fail", None)])
    call_idx = 0

    async def flaky(self, messages, tolerance):
        nonlocal call_idx
        kind, model = next(outcomes)
        call_idx += 1
        if kind == "fail":
            raise httpx.ConnectError("boom")
        return model

    monkeypatch.setattr(eh.ExternalRouterHook, "_ask_sidecar", flaky)

    await hook.async_pre_call_hook(None, None, _data(), "completion")  # fail
    out = await hook.async_pre_call_hook(None, None, _data(), "completion")  # success
    assert out["model"] == "ok-model"
    await hook.async_pre_call_hook(None, None, _data(), "completion")  # fail

    assert call_idx == 3  # all three were attempted — breaker stayed closed
    assert hook._breaker.is_open is False


def test_from_env_requires_url(monkeypatch):
    monkeypatch.delenv("ROUTER_SIDECAR_URL", raising=False)
    with pytest.raises(RuntimeError, match="ROUTER_SIDECAR_URL"):
        eh.ExternalRouterHook.from_env()


def test_from_env_reads_overrides(monkeypatch):
    monkeypatch.setenv("ROUTER_SIDECAR_URL", "http://r:8079/")
    monkeypatch.setenv("ROUTER_SIDECAR_TIMEOUT_S", "1.5")
    monkeypatch.setenv("ROUTER_SIDECAR_TOLERANCE", "0.35")
    monkeypatch.setenv("ROUTER_SIDECAR_DEFAULT_MODEL", "fb")
    monkeypatch.setenv("ROUTER_SIDECAR_FAILURES_BEFORE_OPEN", "7")
    monkeypatch.setenv("ROUTER_SIDECAR_OPEN_DURATION_S", "12")

    hook = eh.ExternalRouterHook.from_env()
    assert hook._url == "http://r:8079/v1/route"
    assert hook._timeout_s == 1.5
    assert hook._tolerance == 0.35
    assert hook._default_model == "fb"
    assert hook._breaker._threshold == 7
    assert hook._breaker._cooldown == 12.0


@pytest.mark.asyncio
async def test_http_request_shape(monkeypatch):
    """End-to-end through real httpx with a MockTransport."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        import json as _json

        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json={"selected_model": "from-sidecar"})

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def patched_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_client)

    hook = _make_hook()
    out = await hook.async_pre_call_hook(None, None, _data(), "completion")
    assert out["model"] == "from-sidecar"
    assert captured["url"] == "http://router.test:8079/v1/route"
    assert captured["method"] == "POST"
    assert captured["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["body"]["tolerance"] == 0.20


@pytest.mark.asyncio
async def test_sidecar_returns_invalid_payload_falls_back(monkeypatch):
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["called"] = True
        return httpx.Response(200, json={})  # no selected_model

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def patched_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_client)

    hook = _make_hook(default_model="fb")
    out = await hook.async_pre_call_hook(None, None, _data(), "completion")
    assert out["model"] == "fb"
    assert captured.get("called") is True
