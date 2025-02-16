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

from prometheus_client import start_http_server, Gauge, Counter

# Prometheus metrics
request_counter = Counter('http_requests_total', 'Total requests made', ['endpoint', 'model'])
latency_gauge = Gauge('request_latency_seconds', 'Request latency', ['endpoint'])
tokens_counter = Counter('tokens_total', 'Total tokens used', ['model', 'type'])
error_counter = Counter('http_errors_total', 'Total error responses', ['endpoint', 'error_type'])

def start_metrics_server():
    start_http_server(4000, addr='0.0.0.0') 