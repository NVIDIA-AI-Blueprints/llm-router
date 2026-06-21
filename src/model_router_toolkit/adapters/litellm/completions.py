"""OpenAI-compatible chat completions endpoint."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from model_router_toolkit.router import extract_user_text

router = APIRouter()


# Modality routing tables.
#
# The prefill router uses a text-only encoder, so it has no signal that a
# request carries non-text content. Without a hint it can pick a model whose
# upstream provider will 4xx on image_url / audio / video blocks. To avoid
# that, we read every message's content list, map each non-text block to a
# capability tag (image / audio / video / ...), and narrow the candidate pool
# to models known to support every required capability.
#
# Extending to a new modality is two edits:
#   1. Map the OpenAI content block `type` to a capability in
#      `_BLOCK_TYPE_TO_CAPABILITY`.
#   2. List the model slugs that can serve that capability in
#      `_CAPABILITY_TO_MODELS`.
#
# Model slugs below match the canonical v3-prefill 9-model pool. Pools with
# different names simply fall through — the shim only restricts to the
# intersection with the actually configured pool, so it is safe by default.
_BLOCK_TYPE_TO_CAPABILITY: dict[str, str] = {
    "image_url": "image",
    "input_image": "image",
    "input_audio": "audio",
    "audio": "audio",
    "video_url": "video",
    "input_video": "video",
}

_CAPABILITY_TO_MODELS: dict[str, frozenset[str]] = {
    # Vision: Gemini, Sonnet, and Opus all consume image_url / input_image
    # blocks via OpenRouter (probe-verified against the v3-prefill pool).
    "image": frozenset({"gemini-3-5-flash", "claude-sonnet-4-6", "claude-opus-4-8"}),
    # Audio: of the canonical 9-model pool, only Gemini 3.5 Flash accepts
    # input_audio blocks (probe-verified with a 1s silent WAV; non-Gemini
    # upstreams return 404 "No endpoints found that support input audio").
    "audio": frozenset({"gemini-3-5-flash"}),
    # Video: same finding as audio — only Gemini 3.5 Flash accepts
    # video_url blocks (probe-verified with a 1s 64x64 solid-red MP4).
    "video": frozenset({"gemini-3-5-flash"}),
}


def _required_capabilities(messages: list[dict]) -> set[str]:
    """Collect modality capability tags needed to serve every block in messages."""
    required: set[str] = set()
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            cap = _BLOCK_TYPE_TO_CAPABILITY.get(block.get("type"))
            if cap:
                required.add(cap)
    return required


def _modality_filtered_pool(messages: list[dict], pool: set[str]) -> list[str] | None:
    """Narrow `pool` to models capable of every required modality.

    Returns:
        Sorted list of candidate model names, or None when the request has no
        special modality requirements (caller should leave routing untouched).
        Returns an empty list when no configured model satisfies the required
        capabilities — in that case the caller should also leave routing
        untouched so the existing flow surfaces the upstream "unsupported"
        error rather than forcing the strategy to choose from an empty set.
    """
    required = _required_capabilities(messages)
    if not required:
        return None
    candidates = pool
    for cap in required:
        candidates = candidates & _CAPABILITY_TO_MODELS.get(cap, frozenset())
    return sorted(candidates)


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

    # Modality shim: when the client did not already restrict the candidate
    # pool via `models`, narrow it to models capable of every required
    # modality (see _BLOCK_TYPE_TO_CAPABILITY / _CAPABILITY_TO_MODELS).
    # Empty results are intentionally NOT injected so the upstream "no
    # endpoints support X" error surfaces unchanged for unsupported
    # modalities, rather than forcing the strategy to choose from an
    # empty set.
    if "models" not in body:
        configured_pool = {m.name for m in config.models}
        filtered = _modality_filtered_pool(messages, configured_pool)
        if filtered:
            body["models"] = filtered

    # Use the first model name from config as the LiteLLM model group.
    # The routing strategy intercepts the call and picks the actual deployment.
    model_group = config.models[0].name if config.models else "default"

    metadata: dict[str, Any] = {}
    if "models" in body:
        metadata["models"] = body["models"]

    kwargs: dict[str, Any] = {
        "model": model_group,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if metadata:
        kwargs["metadata"] = metadata

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
        user_text = extract_user_text(messages)
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
