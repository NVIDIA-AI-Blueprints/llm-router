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

import logging

from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class LLMClient:
    def __init__(self, api_key, base_url):
        self.client = OpenAI(
            api_key=api_key,
            max_retries=0,
        )
        self.client.base_url = base_url

    def predict(self, message, history, routing_strategy, policy, model):
        logging.info("start of predict()")
        logging.info(f"message: {message}")
        logging.info(f"history: {history}")
        history_openai_format = []
        for human, assistant in history:
            # remove model name from history
            assistant = assistant.split("] ", maxsplit=1)[1]

            history_openai_format.append({"role": "user", "content": human})
            history_openai_format.append({"role": "assistant", "content": assistant})

        history_openai_format.append({"role": "user", "content": f"{message}"})
        logging.info(history_openai_format)

        extra_body = {
            "nim-llm-router": {
                "routing_strategy": routing_strategy,
                # "threshold": 0.2, # Implement this in the future
                "policy": policy
            }
        }
        if routing_strategy == "manual":
            extra_body['nim-llm-router']["model"] = model
        
        logging.info(extra_body)
        logging.info("self.client.chat.completions.create")

        try:
            response = self.client.chat.completions.with_raw_response.create(
                model="",
                messages=history_openai_format,
                temperature=0.5,
                top_p=1,
                max_tokens=1024,
                stream=True,
                stream_options={"include_usage": True},
                extra_body=extra_body,
            )
        except Exception as e:
            logging.error(f"API error: {str(e)}")
            return
        # Get headers before parsing the stream
        chosen_classifier = response.headers.get('X-Chosen-Classifier')
        if chosen_classifier:
            logging.info(f"Chosen classifier from header: {chosen_classifier}")

        # Get the stream from the raw response
        stream = response.parse()

        partial_message = ""
        model_that_was_used = None
        for chunk in stream:
            if (len(chunk.choices) > 0 and  # Check if choices exists and has elements
                hasattr(chunk.choices[0], 'delta') and 
                hasattr(chunk.choices[0].delta, 'content') and 
                chunk.choices[0].delta.content is not None):
                if model_that_was_used is None:
                    logging.info(
                        f"setting model from {model_that_was_used} to: {chunk.model}"
                    )
                    model_that_was_used = chunk.model
                partial_message = partial_message + chunk.choices[0].delta.content
                output = f"[**{model_that_was_used}**|{chosen_classifier}] {partial_message}"
                yield output