# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pydantic import BaseModel
import random
from metrics import request_counter, latency_gauge, tokens_counter, error_counter

# Test messages
MESSAGES = [
    "Write a song about space exploration",
    "Explain quantum computing to a 5-year old",
    "Create a short story about a magical library"
]

class UsageTokens(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionTask:
    def __init__(self, client):
        self.client = client

    def execute(self):
        prompt = random.choice(MESSAGES)
        message = {"role": "user", "content": prompt}
        
        data = {
            "model": "",
            "messages": [message],
            "nim-llm-router": {
                "routing_strategy": "triton",
                "policy": "task_router",
                "threshold": 0.2,
            }
        }

        with self.client.post("/v1/chat/completions", json=data, catch_response=True) as response:
            self.handle_response(response)

    def handle_response(self, response):
        try:
            if response.status_code == 200:
                chat_completion = response.json()
                usage = chat_completion.get("usage")
                model = chat_completion.get("model")
                
                usage_tokens = UsageTokens(
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    total_tokens=usage.get("total_tokens")
                )
                
                # Update metrics
                tokens_counter.labels(model=model, type="prompt").inc(usage_tokens.prompt_tokens)
                tokens_counter.labels(model=model, type="completion").inc(usage_tokens.completion_tokens)
                tokens_counter.labels(model=model, type="total").inc(usage_tokens.total_tokens)
                request_counter.labels(endpoint="/v1/chat/completions", model=model).inc()
                latency_gauge.labels(endpoint="/v1/chat/completions").set(response.elapsed.total_seconds())
            else:
                error_msg = f"Request failed with status {response.status_code}"
                try:
                    error_msg += f": {response.json()}"
                except:
                    error_msg += f": {response.text}"
                response.failure(error_msg)
                error_counter.labels(endpoint="/v1/chat/completions", error_type=str(response.status_code)).inc()
        except Exception as e:
            error_msg = f"Exception during request: {str(e)}"
            response.failure(error_msg)