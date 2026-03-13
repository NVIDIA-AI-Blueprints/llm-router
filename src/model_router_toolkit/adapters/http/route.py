"""Router-only endpoint: returns routing decisions without LLM inference."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from model_router_toolkit.router import RoutingResult, extract_user_text

router = APIRouter()


class RouteRequest(BaseModel):
    messages: list[dict[str, str]] | None = None
    question: str | None = None
    model: str | None = None
    tolerance: float = Field(default=0.20, ge=0.0, le=1.0)


class RouteResponse(BaseModel):
    selected_model: str
    model_names: list[str]
    confidences: dict[str, float]
    costs: list[dict[str, Any]]
    metadata: dict[str, Any]


def _extract_question(req: RouteRequest) -> str:
    """Pull the question text from messages or the question field."""
    if req.question:
        return req.question
    return extract_user_text(req.messages)


def _result_to_response(result: RoutingResult) -> RouteResponse:
    return RouteResponse(
        selected_model=result.selected_model,
        model_names=result.model_names,
        confidences=dict(zip(result.model_names, result.confidences)),
        costs=[
            {
                "model": name,
                "estimated_total_cost": c.estimated_total_cost,
                "cost_per_m_input_tokens": c.cost_per_m_input_tokens,
                "cost_per_m_output_tokens": c.cost_per_m_output_tokens,
                "median_output_tokens": c.median_output_tokens,
            }
            for name, c in zip(result.model_names, result.costs)
        ],
        metadata=result.metadata,
    )


@router.post("/route", response_model=RouteResponse)
async def route(request: Request, req: RouteRequest):
    app_router = request.app.state.router
    config = request.app.state.config

    # Model-name bypass: if a specific model is requested and it exists
    # in the pool, return it directly without ML inference.
    if req.model and app_router.has_model(req.model):
        t0 = time.perf_counter()
        result = app_router.resolve(req.model)
        route_ms = (time.perf_counter() - t0) * 1000
        response = _result_to_response(result)
        response.metadata["route_ms"] = round(route_ms, 2)
        return response

    question = _extract_question(req)
    if not question:
        return RouteResponse(
            selected_model=config.models[0].name if config.models else "",
            model_names=config.model_names,
            confidences={m: 0.0 for m in config.model_names},
            costs=[],
            metadata={"error": "no question text provided"},
        )

    t0 = time.perf_counter()
    result = app_router.route(question, tolerance=req.tolerance)
    route_ms = (time.perf_counter() - t0) * 1000

    response = _result_to_response(result)
    response.metadata["route_ms"] = round(route_ms, 2)
    return response
