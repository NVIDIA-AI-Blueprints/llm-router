import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from model_router_toolkit.privacy.middleware import (
    RequestRedactionMiddleware,
    maybe_add_request_redaction_middleware,
)


class StaticRedactor:
    def redact_text(self, text: str) -> str:
        return "<PII>"


class FailingRedactor:
    def redact_text(self, text: str) -> str:
        raise RuntimeError("redactor unavailable")


async def echo_json(request: Request) -> JSONResponse:
    return JSONResponse(await request.json())


def build_app(redactor, *, fail_open: bool = True) -> Starlette:
    app = Starlette(
        routes=[Route("/v1/chat/completions", echo_json, methods=["POST"])],
    )
    app.add_middleware(
        RequestRedactionMiddleware,
        redactor=redactor,
        fail_open=fail_open,
    )
    return app


@pytest.mark.asyncio
async def test_request_redaction_middleware_rewrites_chat_body_before_downstream():
    app = build_app(StaticRedactor())
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Email ada@example.com"}]},
        )

    assert response.status_code == 200
    assert response.json() == {"messages": [{"role": "user", "content": "<PII>"}]}


@pytest.mark.asyncio
async def test_request_redaction_middleware_fails_open_by_default():
    app = build_app(FailingRedactor(), fail_open=True)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Email ada@example.com"}]},
        )

    assert response.status_code == 200
    assert response.json() == {
        "messages": [{"role": "user", "content": "Email ada@example.com"}]
    }


def test_maybe_add_request_redaction_middleware_skips_disabled_config():
    app = Starlette()

    installed = maybe_add_request_redaction_middleware(app, env={})

    assert installed is False
    assert app.user_middleware == []


def test_maybe_add_request_redaction_middleware_installs_enabled_config(monkeypatch):
    app = Starlette()
    monkeypatch.setattr(
        "model_router_toolkit.privacy.middleware.create_text_redactor",
        lambda _config: StaticRedactor(),
    )

    installed = maybe_add_request_redaction_middleware(
        app,
        env={"MODEL_ROUTER_REDACTION_ENABLED": "true"},
    )

    assert installed is True
    assert len(app.user_middleware) == 1
