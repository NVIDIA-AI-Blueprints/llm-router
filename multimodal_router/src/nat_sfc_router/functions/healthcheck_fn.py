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


from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class HealthCheckConfig(FunctionBaseConfig, name="healthcheck"):
    """Health check function for workflow."""
    pass

@register_function(config_type=HealthCheckConfig)
async def healthcheck_fn(_config: HealthCheckConfig, _builder: Builder):
    """Health check function for workflow.
    Args:
        _config: The configuration for the health check function.
        _builder: The builder for the health check function.
    Returns:
        A generator of functions that return a healthy status.
    """

    async def _response_fn(unused: str | None = None) -> str:
        """Health check function for workflow.
        Args:
            unused: Unused parameter.
        Returns:
            A healthy status.
        """
        return "healthy"

    yield FunctionInfo.from_fn(
        _response_fn,
        description="Health check function for workflow.")
