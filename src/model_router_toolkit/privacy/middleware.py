"""ASGI middleware for redacting cloud-bound JSON request bodies."""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable, Mapping

from model_router_toolkit.privacy.redaction import (
    RedactionConfig,
    TextRedactor,
    create_text_redactor,
    redact_json_request_body,
)

logger = logging.getLogger(__name__)

_DEFAULT_PATHS = frozenset({"/v1/chat/completions"})

Scope = dict[str, object]
Message = dict[str, object]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]


class RequestRedactionMiddleware:
    """Redact OpenAI-compatible chat JSON before downstream proxy handling."""

    def __init__(
        self,
        app: Callable[[Scope, Receive, Send], Awaitable[None]],
        *,
        redactor: TextRedactor,
        fail_open: bool = True,
        paths: frozenset[str] = _DEFAULT_PATHS,
    ):
        self.app = app
        self._redactor = redactor
        self._fail_open = fail_open
        self._paths = paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not self._should_redact(scope):
            await self.app(scope, receive, send)
            return

        original_body = await _read_body(receive)
        try:
            redacted_body, changed = redact_json_request_body(original_body, self._redactor)
        except Exception:
            logger.exception("Request redaction failed")
            if not self._fail_open:
                raise
            redacted_body = original_body
            changed = False

        next_body = redacted_body if changed else original_body
        await self.app(_with_content_length(scope, len(next_body)), _receive_once(next_body), send)

    def _should_redact(self, scope: Scope) -> bool:
        if scope.get("type") != "http":
            return False
        if str(scope.get("method", "")).upper() != "POST":
            return False
        if scope.get("path") not in self._paths:
            return False
        return "application/json" in _header_value(scope, b"content-type")


def maybe_add_request_redaction_middleware(
    app,
    *,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Install request redaction middleware when enabled by environment."""

    config = RedactionConfig.from_env(env or os.environ)
    if not config.enabled:
        return False

    redactor = create_text_redactor(config)
    app.add_middleware(
        RequestRedactionMiddleware,
        redactor=redactor,
        fail_open=config.fail_open,
    )
    logger.info("Request redaction middleware enabled with backend %s", config.backend)
    return True


async def _read_body(receive: Receive) -> bytes:
    chunks: list[bytes] = []
    more_body = True
    while more_body:
        message = await receive()
        body = message.get("body", b"")
        if isinstance(body, bytes):
            chunks.append(body)
        more_body = bool(message.get("more_body", False))
    return b"".join(chunks)


def _receive_once(body: bytes) -> Receive:
    sent = False

    async def receive() -> Message:
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


def _header_value(scope: Scope, header_name: bytes) -> str:
    headers = scope.get("headers") or []
    if not isinstance(headers, list):
        return ""
    for key, value in headers:
        if key.lower() == header_name and isinstance(value, bytes):
            return value.decode("latin1").lower()
    return ""


def _with_content_length(scope: Scope, body_length: int) -> Scope:
    next_scope = dict(scope)
    next_scope["headers"] = _replace_content_length(scope.get("headers", []), body_length)
    return next_scope


def _replace_content_length(
    headers: object,
    body_length: int,
) -> list[tuple[bytes, bytes]]:
    if not isinstance(headers, list):
        headers = []
    next_headers = [(key, value) for key, value in headers if key.lower() != b"content-length"]
    next_headers.append((b"content-length", str(body_length).encode("ascii")))
    return next_headers
