// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use lazy_static::lazy_static;
use prometheus::{
    register_histogram, register_histogram_vec, register_int_counter, register_int_counter_vec,
    Histogram, HistogramVec, IntCounter, IntCounterVec,
};
use serde_json::Value;

lazy_static! {
    pub static ref NUM_REQUESTS: IntCounter =
        register_int_counter!("num_requests", "Total number of requests")
            .expect("Failed to create num_requests counter");

    pub static ref REQUESTS_PER_POLICY: IntCounterVec = register_int_counter_vec!(
        "requests_per_policy",
        "Total number of requests per policy",
        &["policy"]
    )
    .expect("Failed to create requests_per_policy counter vector");

    pub static ref REQUESTS_PER_MODEL: IntCounterVec = register_int_counter_vec!(
        "requests_per_model",
        "Total number of requests per model",
        &["model"]
    )
    .expect("Failed to create requests_per_model counter vector");

    pub static ref REQUEST_LATENCY: Histogram = register_histogram!(
        "request_latency_seconds",
        "Latency of processing requests in seconds"
    )
    .expect("Failed to create request_latency histogram");

    pub static ref REQUEST_SUCCESS: IntCounter =
        register_int_counter!("request_success_total", "Total successful requests")
            .expect("Failed to create request_success counter");

    pub static ref REQUEST_FAILURE: IntCounterVec = register_int_counter_vec!(
        "request_failure_total",
        "Total failed requests, broken down by error type (4XX, 5XX, other)",
        &["error_type"]
    )
    .expect("Failed to create request_failure counter vector");

    pub static ref ROUTING_POLICY_USAGE: IntCounterVec = register_int_counter_vec!(
        "routing_policy_usage",
        "Number of times each routing policy was used",
        &["routing_policy"]
    )
    .expect("Failed to create routing_policy_usage counter vector");

    pub static ref MODEL_SELECTION_TIME: Histogram = register_histogram!(
        "model_selection_time_seconds",
        "Time (in seconds) taken for model selection (e.g., by Triton)"
    )
    .expect("Failed to create model_selection_time histogram");

    pub static ref LLM_RESPONSE_TIME: HistogramVec = register_histogram_vec!(
        "llm_response_time_seconds",
        "Response time (in seconds) for each LLM",
        &["llm"]
    )
    .expect("Failed to create llm_response_time histogram vector");

    pub static ref TOKEN_USAGE: IntCounterVec = register_int_counter_vec!(
        "llm_token_usage",
        "Token usage per LLM category",
        &["llm_name", "category"]
    )
    .unwrap();

    pub static ref PROXY_OVERHEAD_LATENCY: Histogram = register_histogram!(
        "proxy_overhead_latency_seconds",
        "Overhead latency of the proxy, calculated as overall latency minus model selection and LLM response time"
    )
    .expect("Failed to create proxy_overhead_latency histogram");
}

pub fn track_token_usage(json: &Value, llm_name: &str) {
    if let Some(usage) = json.get("usage") {
        if let Some(prompt) = usage["prompt_tokens"].as_u64() {
            TOKEN_USAGE
                .with_label_values(&[llm_name, "prompt"])
                .inc_by(prompt);
        }
        if let Some(completion) = usage["completion_tokens"].as_u64() {
            TOKEN_USAGE
                .with_label_values(&[llm_name, "completion"])
                .inc_by(completion);
        }
        if let Some(total) = usage["total_tokens"].as_u64() {
            TOKEN_USAGE
                .with_label_values(&[llm_name, "total"])
                .inc_by(total);
        }
    }
}
