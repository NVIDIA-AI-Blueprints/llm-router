"""OpenAI-compatible chat completions endpoint."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

router = APIRouter()


async def _handle_completion(request: Request, body: dict) -> JSONResponse | StreamingResponse:
    litellm_router = request.app.state.litellm_router
    strategy = request.app.state.strategy
    config = request.app.state.config

    messages = body.get("messages", [])
    stream = body.get("stream", False)
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 4096)

    if "tolerance" in body:
        strategy.set_request_tolerance(float(body.get("tolerance", 0.20)))

    # Use the first model name from config as the LiteLLM model group.
    # The routing strategy intercepts the call and picks the actual deployment.
    model_group = config.models[0].name if config.models else "default"

    kwargs: dict[str, Any] = {
        "model": model_group,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }

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
            "confidences": dict(zip(
                strategy.last_result.model_names,
                strategy.last_result.confidences,
            )),
            "metadata": strategy.last_result.metadata,
        }

    return JSONResponse(content=result_data)


@router.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    return await _handle_completion(request, body)
