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


class ChatRequest(BaseModel):
    message: str
    tolerance: float = 0.10
    enabled_models: list[str] | None = None


async def _chat_stream(request: Request, req: ChatRequest):
    litellm_router = request.app.state.litellm_router
    strategy = request.app.state.strategy
    config = request.app.state.config

    enabled = set(req.enabled_models) if req.enabled_models else {m.name for m in config.models}
    strategy.tolerance = req.tolerance

    messages = [{"role": "user", "content": req.message}]

    t0 = time.perf_counter()
    result = strategy.router.route(req.message, tolerance=req.tolerance)
    route_ms = (time.perf_counter() - t0) * 1000

    filtered = [
        (n, c, ct)
        for n, c, ct in zip(result.model_names, result.confidences, result.costs)
        if n in enabled
    ]
    if not filtered:
        filtered = list(zip(result.model_names, result.confidences, result.costs))

    p_max = max(c for _, c, _ in filtered)
    threshold = p_max - req.tolerance
    cost_sorted = sorted(filtered, key=lambda x: x[2].estimated_total_cost)
    selected = cost_sorted[-1][0]
    for name, conf, _ in cost_sorted:
        if conf >= threshold:
            selected = name
            break

    model_spec = config.get_model(selected)
    system_prompt = model_spec.system_prompt if model_spec else ""
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    yield _sse_event("routing", {
        "selected_model": selected,
        "model_names": result.model_names,
        "confidences": result.confidences,
        "metadata": result.metadata,
        "route_ms": round(route_ms, 2),
    })

    t0 = time.perf_counter()
    tokens_sent = False
    try:
        stream = await litellm_router.acompletion(
            model=selected,
            messages=full_messages,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None) or ""
                if content:
                    yield _sse_event("token", {"text": content})
                    tokens_sent = True
    except Exception as e:
        if not tokens_sent:
            yield _sse_event("error", {"message": str(e)[:300]})
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
