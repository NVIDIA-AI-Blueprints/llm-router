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

import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.component_ref import FunctionRef

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SfcRouterConfig(FunctionBaseConfig, name="sfc_router"):
    """Configuration for router function."""
    objective_fn: FunctionRef = Field(..., description="The function to be used for the objective function.")

@register_function(
    config_type=SfcRouterConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN]
)
async def sfc_router(config: SfcRouterConfig, builder: Builder):
    """Simple Router Function"""

    import time
    import json

    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.completion_usage import CompletionUsage
    from openai.types.chat import ChatCompletion

    from nat_sfc_router.schema.openai_chat_request import OpenAIChatRequest

    objective_fn = builder.get_function(config.objective_fn)

    async def _response_fn(chat_request: OpenAIChatRequest) -> ChatCompletion:
        """Process SFC router request."""

        try: 
            model, probabilities = await objective_fn.acall_invoke(chat_request)
        except Exception as e:
            logger.warning(f"sfc router objective fn failed, trying again to see if it returns just a model insteaed of model, probabilities: {e}", exc_info=True)
            model = await objective_fn.acall_invoke(chat_request)

        return ChatCompletion(id="chatcmpl-" + str(int(time.time())),
                              object="chat.completion",
                              created=int(time.time()),
                              model=config.objective_fn,
                              choices=[
                                  Choice(index=0,
                                         message=ChatCompletionMessage(
                                             role="assistant", content=model),
                                         finish_reason="stop")
                              ],
                              usage=CompletionUsage(prompt_tokens=0,
                                                    completion_tokens=0,
                                                    total_tokens=0))

    yield FunctionInfo.from_fn(_response_fn,
                               description="Simple Router Function")
