"""Webhook authentication middleware for enterprise gateway integrations.

Supports HMAC-SHA256 signature verification and bearer token auth.
Used by Portkey, TrueFoundry, Cloudflare, and custom enterprise webhooks.

Enable by passing a secret to the app factory or setting
ROUTER_WEBHOOK_SECRET environment variable.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class WebhookAuthMiddleware(BaseHTTPMiddleware):
    """Authenticate incoming webhook requests via HMAC or bearer token.

    If no secret is configured, all requests pass through (backward compat).
    """

    HMAC_HEADER = "X-Webhook-Signature"
    BEARER_PREFIX = "Bearer "

    def __init__(self, app, *, secret: str | None = None):
        super().__init__(app)
        self.secret = secret or os.environ.get("ROUTER_WEBHOOK_SECRET", "")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.secret:
            return await call_next(request)

        if request.url.path == "/health":
            return await call_next(request)

        hmac_sig = request.headers.get(self.HMAC_HEADER)
        if hmac_sig:
            body = await request.body()
            expected = hmac.new(
                self.secret.encode(), body, hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(hmac_sig, expected):
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid webhook signature"},
                )
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith(self.BEARER_PREFIX):
            token = auth_header[len(self.BEARER_PREFIX):]
            if hmac.compare_digest(token, self.secret):
                return await call_next(request)
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid bearer token"},
            )

        return JSONResponse(
            status_code=401,
            content={"error": "Missing authentication. Provide X-Webhook-Signature or Authorization: Bearer <token>"},
        )
