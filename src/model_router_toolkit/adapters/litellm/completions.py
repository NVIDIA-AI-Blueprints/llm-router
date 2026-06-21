"""OpenAI-compatible chat completions endpoint."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from model_router_toolkit.router import extract_user_text

router = APIRouter()


async def _handle_completion(request: Request, body: dict) -> JSONResponse | StreamingResponse:
    litellm_router = request.app.state.litellm_router
    strategy = request.app.state.strategy
    config = request.app.state.config

    stream = body.get("stream", False)

    if "tolerance" in body:
        strategy.set_request_tolerance(float(body.get("tolerance", 0.20)))

    # Use the first model name from config as the LiteLLM model group.
    # The routing strategy intercepts the call and picks the actual deployment.
    model_group = config.models[0].name if config.models else "default"

    # Forward every OpenAI-compatible field untouched so that agents which rely
    # on `tools` / `tool_choice` / `response_format` / `top_p` / `seed` / `stop`
    # etc. actually reach the upstream model. Only routing-specific keys are
    # stripped:
    #   - `tolerance`, `models`: consumed by the router strategy above
    #   - `model`: replaced with the litellm model group so the strategy can
    #     intercept the call (the original value is ignored, matching the prior
    #     behavior of this endpoint)
    #   - `metadata`: reserved for the router's internal use (we build a fresh
    #     dict containing the `models` filter below); passing through a
    #     client-supplied `metadata` could collide with litellm.Router internal
    #     keys (trace_id, tags, callback context, ...), so it is dropped to
    #     keep the prior surface unchanged
    routing_only_keys = {"tolerance", "models", "model", "metadata"}
    kwargs: dict[str, Any] = {
        k: v for k, v in body.items() if k not in routing_only_keys
    }
    kwargs["model"] = model_group
    kwargs.setdefault("temperature", 0.7)
    kwargs.setdefault("max_tokens", 4096)

    if "models" in body:
        kwargs["metadata"] = {"models": body["models"]}

    if stream:

        async def sse_stream():
            response_stream = await litellm_router.acompletion(**kwargs)
            async for chunk in response_stream:
                data = chunk.model_dump(exclude_none=True)
                yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            sse_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    response = await litellm_router.acompletion(**kwargs)

    result_data = response.model_dump(exclude_none=True)

    for choice in result_data.get("choices", []):
        msg = choice.get("message", {})
        if "content" not in msg:
            msg["content"] = ""

    if strategy.last_result:
        result_data["routing"] = {
            "selected_model": strategy.last_result.selected_model,
            "confidences": dict(
                zip(
                    strategy.last_result.model_names,
                    strategy.last_result.confidences,
                )
            ),
            "metadata": strategy.last_result.metadata,
        }

    from model_router_toolkit import telemetry

    if telemetry.enabled() and strategy.last_result:
        user_text = extract_user_text(body.get("messages", []))
        telemetry.log_chat(
            session_id=None,
            question=user_text,
            selected_model=strategy.last_result.selected_model,
        )

    return JSONResponse(content=result_data)


@router.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    return await _handle_completion(request, body)
