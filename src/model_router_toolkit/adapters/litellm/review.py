"""Auto-review endpoint: judge answer correctness and compare models.

Streams SSE events: judging -> verdict -> (comparing -> model-result* -> comparison-done).
Uses litellm directly (not through the router strategy) so we can target specific models.
"""

from __future__ import annotations

import json
import os
from typing import Any

import litellm
from fastapi import APIRouter, Request
from pydantic import BaseModel
from starlette.responses import StreamingResponse

router = APIRouter()

litellm.suppress_debug_info = True


def _sse_event(event: str, data: Any) -> str:
    payload = json.dumps(data) if not isinstance(data, str) else data
    return f"event: {event}\ndata: {payload}\n\n"


class ReviewRequest(BaseModel):
    question: str
    answer: str
    selected_model: str
    enabled_models: list[str] | None = None


def _get_litellm_params(litellm_router: Any, model_name: str) -> dict | None:
    """Extract litellm params for a named model from the router's model_list."""
    for dep in litellm_router.model_list:
        if isinstance(dep, dict) and dep.get("model_name") == model_name:
            return dict(dep.get("litellm_params", {}))
    return None


def _pick_judge(config: Any, litellm_router: Any) -> tuple[str, dict] | None:
    """Select the most expensive model as judge, return (name, litellm_params)."""
    if not config.models:
        return None
    best = max(config.models, key=lambda m: m.cost_per_m_output_tokens)
    params = _get_litellm_params(litellm_router, best.name)
    if params:
        return best.name, params
    return None


async def _call_model(params: dict, messages: list[dict], **kwargs: Any):
    """Call a model directly via litellm, bypassing the routing strategy."""
    return await litellm.acompletion(
        model=params["model"],
        api_key=params.get("api_key", ""),
        messages=messages,
        **kwargs,
    )


def _parse_json_response(text: str) -> dict:
    """Best-effort JSON extraction from an LLM response."""
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts[1:]:
            candidate = part.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
    try:
        start = cleaned.index("{")
        end = cleaned.rindex("}") + 1
        return json.loads(cleaned[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"correct": None, "confidence": "low", "explanation": cleaned[:200]}


JUDGE_PROMPT = """\
You are an expert evaluator. Given a question and an AI model's answer, \
determine if the answer is correct, accurate, and reasonably complete.

Question: {question}

Answer from {model}:
{answer}

Respond with ONLY valid JSON in this exact format (no other text):
{{"correct": true, "confidence": "high", "explanation": "brief reason"}}

Use true/false for correct, "high"/"medium"/"low" for confidence."""

COMPARE_PROMPT = """Is this answer correct and complete?

Question: {question}

Answer from {model}:
{answer}

Respond with ONLY valid JSON: {{"correct": true, "explanation": "brief reason"}}"""


async def _review_stream(request: Request, req: ReviewRequest):
    config = request.app.state.config
    litellm_router = request.app.state.litellm_router

    judge_info = _pick_judge(config, litellm_router)
    if judge_info is None:
        yield _sse_event("error", {"message": "No judge model available"})
        return

    judge_name, judge_params = judge_info
    judge_display = config.get_model(judge_name)
    judge_label = judge_display.display_name if judge_display else judge_name

    yield _sse_event("judging", {"status": f"Evaluating with {judge_label}..."})

    prompt = JUDGE_PROMPT.format(
        question=req.question,
        model=req.selected_model,
        answer=req.answer[:3000],
    )

    try:
        response = await _call_model(
            judge_params,
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        verdict_text = response.choices[0].message.content or ""
        verdict = _parse_json_response(verdict_text)
        yield _sse_event("verdict", verdict)

        if verdict.get("correct") is not False:
            return

        enabled = set(req.enabled_models) if req.enabled_models else {m.name for m in config.models}
        other_models = [m for m in enabled if m != req.selected_model]
        if not other_models:
            return

        yield _sse_event("comparing", {"status": f"Testing {len(other_models)} other model(s)..."})

        any_correct = False
        for model_name in other_models:
            model_params = _get_litellm_params(litellm_router, model_name)
            if not model_params:
                continue

            display = config.get_model(model_name)
            display_name = display.display_name if display else model_name

            try:
                model_resp = await _call_model(
                    model_params,
                    [{"role": "user", "content": req.question}],
                    temperature=0.7,
                    max_tokens=2048,
                )
                model_answer = model_resp.choices[0].message.content or ""

                compare_prompt = COMPARE_PROMPT.format(
                    question=req.question,
                    model=model_name,
                    answer=model_answer[:2000],
                )
                judge_resp = await _call_model(
                    judge_params,
                    [{"role": "user", "content": compare_prompt}],
                    temperature=0.1,
                    max_tokens=150,
                )
                judge_text = judge_resp.choices[0].message.content or ""
                model_verdict = _parse_json_response(judge_text)

                if model_verdict.get("correct"):
                    any_correct = True

                yield _sse_event(
                    "model-result",
                    {
                        "model": model_name,
                        "display_name": display_name,
                        "correct": model_verdict.get("correct"),
                        "explanation": model_verdict.get("explanation", ""),
                    },
                )
            except Exception as e:
                yield _sse_event(
                    "model-result",
                    {
                        "model": model_name,
                        "display_name": display_name,
                        "correct": None,
                        "explanation": f"Error: {str(e)[:100]}",
                    },
                )

        summary = (
            "Other models answered correctly — routing may have been suboptimal"
            if any_correct
            else "No other model answered correctly — routing was reasonable"
        )
        yield _sse_event("comparison-done", {"any_correct": any_correct, "summary": summary})

    except Exception as e:
        yield _sse_event("error", {"message": str(e)[:300]})


@router.post("/review")
async def review(request: Request, req: ReviewRequest):
    has_key = bool(os.environ.get("OPENROUTER_API_KEY") or os.environ.get("NVIDIA_API_KEY"))
    if not has_key:

        async def _unavailable():
            yield _sse_event(
                "error",
                {
                    "message": "Auto-review requires an API key "
                    "(OPENROUTER_API_KEY or NVIDIA_API_KEY)"
                },
            )

        return StreamingResponse(
            _unavailable(),
            media_type="text/event-stream",
            status_code=503,
        )

    return StreamingResponse(
        _review_stream(request, req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
