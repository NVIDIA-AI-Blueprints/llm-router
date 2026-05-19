"""SSE chat endpoint for model-router-toolkit."""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel
from starlette.responses import StreamingResponse

router = APIRouter()


def _sse_event(event: str, data: Any) -> str:
    payload = json.dumps(data) if not isinstance(data, str) else data
    return f"event: {event}\ndata: {payload}\n\n"


def _get_field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_text_value(item) for item in value]
        return "".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("text", "content", "reasoning", "reasoning_content"):
            text = _text_value(value.get(key))
            if text:
                return text
    return ""


def _first_choice(chunk: Any) -> Any:
    choices = _get_field(chunk, "choices")
    if not choices:
        return None
    return choices[0]


def _delta_texts(chunk: Any) -> tuple[str, str]:
    choice = _first_choice(chunk)
    if choice is None:
        return "", ""

    delta = _get_field(choice, "delta")
    if delta is None:
        return "", ""

    content = _text_value(_get_field(delta, "content"))

    reasoning = (
        _text_value(_get_field(delta, "reasoning_content"))
        or _text_value(_get_field(delta, "reasoning"))
    )

    provider_fields = _get_field(delta, "provider_specific_fields")
    if not reasoning and provider_fields:
        reasoning = (
            _text_value(_get_field(provider_fields, "reasoning_content"))
            or _text_value(_get_field(provider_fields, "reasoning"))
        )

    additional_kwargs = _get_field(delta, "additional_kwargs")
    if not reasoning and additional_kwargs:
        reasoning = (
            _text_value(_get_field(additional_kwargs, "reasoning_content"))
            or _text_value(_get_field(additional_kwargs, "reasoning"))
        )

    return content, reasoning


class ChatRequest(BaseModel):
    message: str
    tolerance: float = 0.10
    enabled_models: list[str] | None = None


async def _chat_stream(request: Request, req: ChatRequest):
    litellm_router = request.app.state.litellm_router
    strategy = request.app.state.strategy
    config = request.app.state.config

    tolerance = max(0.0, min(1.0, req.tolerance))

    messages = [{"role": "user", "content": req.message}]

    t0 = time.perf_counter()
    result = strategy.router.route(
        req.message,
        tolerance=tolerance,
        models=req.enabled_models,
    )
    route_ms = (time.perf_counter() - t0) * 1000

    selected = result.selected_model

    model_spec = config.get_model(selected)
    system_prompt = model_spec.system_prompt if model_spec else ""
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    yield _sse_event(
        "routing",
        {
            "selected_model": selected,
            "model_names": result.model_names,
            "confidences": result.confidences,
            "metadata": result.metadata,
            "route_ms": round(route_ms, 2),
        },
    )

    t0 = time.perf_counter()
    tokens_sent = False
    try:
        stream = await litellm_router.acompletion(
            model=selected,
            messages=full_messages,
            stream=True,
        )
        async for chunk in stream:
            content, reasoning = _delta_texts(chunk)
            if reasoning:
                yield _sse_event("reasoning", {"text": reasoning})
                tokens_sent = True
            if content:
                yield _sse_event("token", {"text": content})
                tokens_sent = True
    except Exception as e:
        yield _sse_event("error", {"message": str(e)[:300], "partial": tokens_sent})
        return

    latency_ms = (time.perf_counter() - t0) * 1000
    yield _sse_event("done", {"latency_ms": round(latency_ms, 2)})


@router.post("/chat")
async def chat(request: Request, req: ChatRequest):
    return StreamingResponse(
        _chat_stream(request, req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
