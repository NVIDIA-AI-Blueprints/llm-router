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

from locust import HttpUser, task, between, SequentialTaskSet
from tasks import ChatCompletionTask
from metrics import start_metrics_server

class ChatWorkflow(SequentialTaskSet):
    @task(1)
    def chat_completion(self):
        ChatCompletionTask(self.client).execute()

class APIUser(HttpUser):
    wait_time = between(1, 3)

    # Define different tasks
    tasks = {
        ChatWorkflow
    }

# Start Prometheus metrics server
start_metrics_server()
