# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Router-Only API Function

Returns routing decisions without wrapping in a ChatCompletion response.
Uses a purpose-built response format following OpenAI conventions for metadata
(id, object, created) combined with HuggingFace-style classification output.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.component_ref import FunctionRef

from nat_sfc_router.schema.openai_chat_request import OpenAIChatRequest

logger = logging.getLogger(__name__)


class ClassificationScore(BaseModel):
    """A single classification result with label and score."""
    label: str = Field(..., description="The model name")
    score: float = Field(..., description="Confidence score for this model")


class RoutingDecision(BaseModel):
    """
    Router-only response format.

    Follows OpenAI conventions (id, object, created) with HuggingFace-style
    classification scores.
    """
    id: str = Field(..., description="Unique identifier for this routing decision")
    object: str = Field(default="routing.decision", description="Object type identifier")
    created: int = Field(..., description="Unix timestamp of when the decision was made")
    selected_model: str = Field(..., description="The model selected for routing")
    classifications: List[ClassificationScore] = Field(
        ...,
        description="All models with their confidence scores, sorted by score descending"
    )
    selection_reason: Optional[str] = Field(
        default=None,
        description="Reason for model selection (e.g., 'cost_optimized', 'highest_probability', 'threshold_fallback')"
    )


class RouterOnlyConfig(FunctionBaseConfig, name="router_only"):
    """Configuration for router-only function."""
    objective_fn: FunctionRef = Field(..., description="The objective function to use for routing")


@register_function(config_type=RouterOnlyConfig)
async def router_only(config: RouterOnlyConfig, builder: Builder):
    """
    Router-Only Function

    Returns routing decisions in a purpose-built format without ChatCompletion wrapping.
    Useful when you need just the routing decision and probabilities without making
    an actual completion request.
    """

    objective_fn = builder.get_function(config.objective_fn)

    async def _response_fn(chat_request: OpenAIChatRequest) -> Dict[str, Any]:
        """Process router-only request and return routing decision."""

        try:
            result = await objective_fn.acall_invoke(chat_request)

            # Handle both (model, probabilities) and (model, probabilities, reason) returns
            if isinstance(result, tuple):
                if len(result) >= 2:
                    model, probabilities = result[0], result[1]
                    selection_reason = result[2] if len(result) > 2 else None
                else:
                    model = result[0]
                    probabilities = {}
                    selection_reason = None
            else:
                model = result
                probabilities = {}
                selection_reason = None

        except Exception as e:
            logger.warning(f"router_only objective fn returned unexpected format: {e}", exc_info=True)
            model = await objective_fn.acall_invoke(chat_request)
            probabilities = {}
            selection_reason = None

        # Build classification scores sorted by score descending
        classifications = [
            ClassificationScore(label=label, score=score)
            for label, score in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        ]

        # If no probabilities returned, add the selected model with score 1.0
        if not classifications:
            classifications = [ClassificationScore(label=model, score=1.0)]

        response = RoutingDecision(
            id=f"routing-{int(time.time() * 1000)}",
            object="routing.decision",
            created=int(time.time()),
            selected_model=model,
            classifications=classifications,
            selection_reason=selection_reason
        )

        return response.model_dump()

    yield FunctionInfo.from_fn(
        _response_fn,
        description="Router-only function that returns routing decisions without ChatCompletion wrapping"
    )
