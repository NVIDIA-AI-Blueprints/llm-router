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

//! Proxy
use crate::config::{Policy, RouterConfig};
use crate::error::{GatewayApiError, IntoResponse};
use crate::metrics::{
    track_token_usage, LLM_RESPONSE_TIME, MODEL_SELECTION_TIME, NUM_REQUESTS,
    PROXY_OVERHEAD_LATENCY, REQUESTS_PER_MODEL, REQUESTS_PER_POLICY, REQUEST_FAILURE,
    REQUEST_LATENCY, REQUEST_SUCCESS, ROUTING_POLICY_USAGE,
};
use crate::stream::ReqwestStreamAdapter;
use crate::triton::{InferInputTensor, InferInputs, Output};
use bytes::Bytes;
use http::StatusCode;
use http_body_util::{combinators::BoxBody, BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Method, Request, Response, Uri};
use log::{debug, error, info};
use prometheus::{gather, Encoder, TextEncoder};
use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

fn print_config(config: &RouterConfig) {
    debug!("{:#?}", config);
}

fn extract_forward_uri_path_and_query(req: &Request<Incoming>) -> Result<Uri, GatewayApiError> {
    let uri = req
        .uri()
        .path_and_query()
        .map(|x| x.as_str())
        .unwrap_or("")
        .to_string()
        .parse::<Uri>()
        .map_err(|e| GatewayApiError::InvalidRequest {
            message: format!("Invalid URI: {}", e),
        })?;

    Ok(uri)
}

#[derive(Serialize, Deserialize, Debug)]
struct Message {
    role: String,
    content: String,
}

type Messages = Vec<Message>;

fn extract_messages(value: &Value) -> Option<Messages> {
    value
        .get("messages")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
}

fn convert_messages_to_text_input(messages: &Messages) -> String {
    let text_input = serde_json::to_string(messages).unwrap_or_default();
    shorten_string(&text_input, 2000)
}

fn get_last_message_for_triton(messages: &Messages) -> String {
    messages
        .last()
        .map(|msg| msg.content.clone())
        .unwrap_or_default()
}

fn shorten_string(s: &str, max_length: usize) -> String {
    let len = s.len();
    if len <= max_length {
        s.to_string()
    } else {
        s[len - max_length..].to_string()
    }
}

async fn choose_model(
    policy: &Policy,
    client: &reqwest::Client,
    text_input: &str,
    _threshold: f64,
) -> Result<usize, GatewayApiError> {
    info!("Using policy: {}", &policy.name);
    info!("Triton input text: {:#?}", &text_input);
    let text_tensor = InferInputTensor {
        name: "INPUT".to_string(),
        datatype: "BYTES".to_string(),
        shape: vec![1, 1],
        data: vec![vec![text_input.to_string()]],
    };

    let data = InferInputs {
        inputs: vec![text_tensor],
    };

    let url = policy.url.clone();
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    let response = client
        .post(url)
        .headers(headers)
        .json(&data)
        .send()
        .await
        .map_err(|e| {
            error!("Failed to reach Triton server: {:?}", e);
            GatewayApiError::TritonServiceError {
                status_code: 503,
                message: "Triton server is unreachable".to_string(),
            }
        })?;
    info!("Triton classification response: {:#?}", response);

    if !response.status().is_success() {
        let status = response.status();
        let error_body = response.bytes().await?;
        error!(
            "Triton error response: {}",
            String::from_utf8_lossy(&error_body)
        );

        return Err(GatewayApiError::TritonServiceError {
            status_code: status.as_u16(),
            message: format!(
                "Triton service error: {}",
                String::from_utf8_lossy(&error_body)
            ),
        });
    }

    // Parse successful response
    let response: Output = response.json().await.map_err(|e| {
        error!("Failed to parse Triton response: {:?}", e);
        GatewayApiError::TritonServiceError {
            status_code: 500,
            message: format!("Invalid Triton response: {}", e),
        }
    })?;

    info!("Triton Output: {:#?}", response);

    let output_tensor =
        response
            .outputs
            .first()
            .ok_or_else(|| GatewayApiError::TritonServiceError {
                status_code: 500,
                message: "No outputs returned from the Triton response".to_string(),
            })?;

    let model_index = output_tensor
        .data
        .iter()
        .enumerate()
        .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .ok_or_else(|| {
            error!("Invalid probability distribution from Triton");
            GatewayApiError::TritonServiceError {
                status_code: 500,
                message: "Could not determine model selection from probability distribution"
                    .to_string(),
            }
        })?;

    info!("model_index chosen by classifier: {:#?}", model_index);
    Ok(model_index)
}

fn modify_model(value: Value, model: &str) -> Result<Value, GatewayApiError> {
    let mut json = value.clone();
    json["model"] = Value::String(model.to_string());
    Ok(json)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
enum RoutingStrategy {
    Manual,
    Triton,
}

#[derive(Serialize, Deserialize, Debug)]
struct NimLlmRouterParams {
    policy: String,
    routing_strategy: Option<RoutingStrategy>,
    model: Option<String>,
    threshold: Option<f64>,
}

fn extract_nim_llm_router_params(value: &Value) -> Option<NimLlmRouterParams> {
    value
        .get("nim-llm-router")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
}

fn remove_nim_llm_router_params(mut value: Value) -> Value {
    value
        .as_object_mut()
        .map(|map| map.remove("nim-llm-router"));
    value
}

// This might break response if the stream_options is not supported by the model,
// if you want to use this function, please make sure the model supports it.
// fn include_usage(mut value: Value) -> Value {
//     if let Some(obj) = value.as_object_mut() {
//         // Only add stream_options if not already present
//         if !obj.contains_key("stream_options") && obj.contains_key("stream") {
//             obj.insert(
//                 "stream_options".to_string(),
//                 serde_json::json!({ "include_usage": true }),
//             );
//             info!("Added stream_options to request");
//         }
//     }
//     value
// }

pub fn config(
    config: RouterConfig,
) -> Result<Response<BoxBody<Bytes, GatewayApiError>>, GatewayApiError> {
    let config = config.sanitized();
    let json_vec = serde_json::to_vec(&config).expect("Serialization to JSON should succeed.");
    let body_bytes = Bytes::from(json_vec);

    let full_body = Full::from(body_bytes)
        .map_err(|never| match never {})
        .boxed();

    let client_res = Response::builder().status(200).body(full_body)?;

    info!("/config: {client_res:#?}");
    Ok(client_res)
}

pub fn health() -> Result<Response<BoxBody<Bytes, GatewayApiError>>, GatewayApiError> {
    let body = serde_json::json!({ "status": "OK" });
    let json_vec = serde_json::to_vec(&body).expect("Serialization to JSON should succeed.");
    let body_bytes = Bytes::from(json_vec);

    let full_body = Full::from(body_bytes)
        .map_err(|never| match never {})
        .boxed();

    let client_res = Response::builder().status(200).body(full_body)?;

    info!("/health: {client_res:#?}");
    Ok(client_res)
}

pub fn metrics() -> Result<Response<BoxBody<Bytes, GatewayApiError>>, GatewayApiError> {
    let encoder = TextEncoder::new();
    let metric_families = gather();
    let mut buffer = Vec::new();

    if let Err(err) = encoder.encode(&metric_families, &mut buffer) {
        error!("Metric encoding error: {:?}", err);
        let body = String::from("Failed to encode metrics");
        let json_vec = serde_json::to_vec(&body).expect("Serialization to JSON should succeed.");
        let body_bytes = Bytes::from(json_vec);

        let full_body = Full::from(body_bytes)
            .map_err(|never| match never {})
            .boxed();
        let client_res = Response::builder().status(500).body(full_body)?;
        return Ok(client_res);
    }

    let body_bytes = Bytes::from(buffer);
    let full_body = Full::from(body_bytes)
        .map_err(|never| match never {})
        .boxed();

    let client_res = Response::builder()
        .header("Content-Type", encoder.format_type())
        .status(200)
        .body(full_body)?;

    info!("/metrics: {client_res:#?}");
    Ok(client_res)
}

pub fn unavailable() -> Result<Response<BoxBody<Bytes, GatewayApiError>>, GatewayApiError> {
    let body = serde_json::json!({ "path": "Unavailable" });
    let json_vec = serde_json::to_vec(&body).expect("Serialization to JSON should succeed.");
    let body_bytes = Bytes::from(json_vec);

    let full_body = Full::from(body_bytes)
        .map_err(|never| match never {})
        .boxed();

    let client_res = Response::builder().status(404).body(full_body)?;

    info!("/: {client_res:#?}");
    Ok(client_res)
}

pub async fn handler(
    req: Request<Incoming>,
    cfg: RouterConfig,
) -> Result<Response<BoxBody<Bytes, GatewayApiError>>, GatewayApiError> {
    let uri_path = req.uri().path();
    info!("Received request for URI: {}", uri_path);

    match uri_path {
        "/config" => {
            info!("Routing to config handler");
            config(cfg)
        }
        "/health" => {
            info!("Routing to health handler");
            health()
        }
        "/metrics" => {
            info!("Routing to metrics handler");
            metrics()
        }
        "/v1/chat/completions" | "/completions" => {
            info!("Routing to proxy handler");
            proxy(req, cfg).await
        }
        _ => {
            info!("Routing to Unavailable Path");
            unavailable()
        }
    }
}

pub async fn proxy(
    req: Request<Incoming>,
    config: RouterConfig,
) -> Result<Response<BoxBody<Bytes, GatewayApiError>>, GatewayApiError> {
    let overall_start = Instant::now();
    let mut model_selection_time = 0.0;
    let llm_resp_time_holder = Arc::new(Mutex::new(0.0));

    NUM_REQUESTS.inc();

    let result = (async {
        print_config(&config);

        let forward_uri_path_and_query = extract_forward_uri_path_and_query(&req)?;
        info!("forward_uri_path_and_query: {forward_uri_path_and_query:#?}");

        let (parts, body) = req.into_parts();
        info!("parts: {parts:#?}");

        let body_bytes = body.collect().await?.to_bytes();
        info!("body_bytes: {body_bytes:#?}");

        let body_str = String::from_utf8_lossy(&body_bytes);
        info!("body_str: {:#?}", &body_str);
        let json: Value = serde_json::from_str(&body_str).unwrap_or(Value::Null);
        info!("json: {:#?}", &json);

        let is_stream = if parts.method == Method::POST
            && parts
                .headers
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                == Some("application/json")
        {
            json["stream"].as_bool().unwrap_or(false)
        } else {
            false
        };
        info!("is_stream: {is_stream:#?}");

        let messages = extract_messages(&json).unwrap_or_default();
        info!("messages: {:#?}", &messages);
        let text_input = convert_messages_to_text_input(&messages);
        info!("text_input: {:#?}", &text_input);

        let client = reqwest::Client::new();

        let policy = if let Some(nim_llm_router_params) = extract_nim_llm_router_params(&json) {
            match config.get_policy_by_name(nim_llm_router_params.policy.as_str()) {
                Some(policy) => policy,
                None => {
                    let error = GatewayApiError::PolicyNotFound(nim_llm_router_params.policy.clone());
                    return Ok(error.into_response());
                }
            }
        } else {
            let error = GatewayApiError::InvalidRequest {
                message: "Missing required 'nim-llm-router' parameters in request body. Expected format: { 'nim-llm-router': { 'policy': 'string', 'routing_strategy': 'manual|triton', 'model': 'string' (for manual strategy) } }".to_string(),
            };
            return Ok(error.into_response());
        };

        REQUESTS_PER_POLICY
            .with_label_values(&[policy.name.as_str()])
            .inc();

        let routing_strategy =
            extract_nim_llm_router_params(&json).and_then(|params| params.routing_strategy);

        let model_index = match routing_strategy {
            Some(RoutingStrategy::Manual) => {
                ROUTING_POLICY_USAGE.with_label_values(&["manual"]).inc();
                if let Some(nim_llm_router_params) = extract_nim_llm_router_params(&json) {
                    let model = nim_llm_router_params.model.ok_or_else(|| {
                        GatewayApiError::InvalidRequest {
                            message: "No model specified for manual routing".to_string(),
                        }
                    })?;
                    match policy.llms.iter().position(|llm| llm.name == model) {
                        Some(index) => index,
                        None => {
                            let error_body = format!("Model not found: {}", model);
                            let body = Full::from(error_body.into_bytes())
                                .map_err(|never| match never {})
                                .boxed();

                            let error_response = Response::builder()
                                .status(StatusCode::NOT_FOUND)
                                .header(CONTENT_TYPE, "application/json")
                                .body(body)?;

                            return Ok(error_response);
                        }
                    }
                } else {
                    return Err(GatewayApiError::InvalidRequest {
                        message: "Manual routing strategy requires nim-llm-router params"
                            .to_string(),
                    });
                }
            }
            Some(RoutingStrategy::Triton) => {
                ROUTING_POLICY_USAGE.with_label_values(&["triton"]).inc();
                let selection_start = Instant::now();
                let threshold = extract_nim_llm_router_params(&json)
                    .and_then(|params| params.threshold)
                    .unwrap_or(0.5);
                let triton_text = get_last_message_for_triton(&messages);
                match choose_model(&policy, &client, &triton_text, threshold).await {
                    Ok(index) => {
                        model_selection_time = selection_start.elapsed().as_secs_f64();
                        MODEL_SELECTION_TIME.observe(model_selection_time);
                        index
                    }
                    Err(e) => match e {
                        GatewayApiError::TritonServiceError {
                            status_code,
                            message,
                        } => {
                            let body = Full::from(message.into_bytes())
                                .map_err(|never| match never {})
                                .boxed();

                            let error_response = Response::builder()
                                .status(
                                    StatusCode::from_u16(status_code)
                                        .unwrap_or(StatusCode::SERVICE_UNAVAILABLE),
                                )
                                .header(CONTENT_TYPE, "application/json")
                                .body(body)?;

                            return Ok(error_response);
                        }
                        _ => return Err(e),
                    },
                }
            }
            None => {
                return Err(GatewayApiError::InvalidRequest {
                    message: "No routing strategy specified".to_string(),
                });
            }
        };

        let chosen_llm = policy.get_llm_by_index(model_index).ok_or_else(|| {
            GatewayApiError::ModelNotFound(format!("LLM not found at index {}", model_index))
        })?;

        let chosen_classifier = policy.get_llm_name_by_index(model_index).ok_or_else(|| {
            GatewayApiError::ModelNotFound(format!("LLM not found at index {}", model_index))
        })?;

        info!("Chosen Classifier: {:#?}", &chosen_classifier);

        REQUESTS_PER_MODEL
            .with_label_values(&[chosen_llm.name.as_str()])
            .inc();

        let api_base = &chosen_llm.api_base;
        let api_key = &chosen_llm.api_key;
        let model = &chosen_llm.model;

        info!("api_base: {:#?}", api_base);
        info!("model: {:#?}", model);

        let json = remove_nim_llm_router_params(json);
        info!("json after removing nim llm router params: {json:?}");

        let json = modify_model(json, model)?;
        debug!("json after modifying model: {:#?}", &json);

        // Turn on this line if you want to include usage options in the request
        // let json = if is_stream { include_usage(json) } else { json };
        // info!("json after including usage options: {:#?}", &json);

        let method = http::Method::POST;
        let mut headers = http::HeaderMap::new();
        headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", api_key))?,
        );

        let uri = format!("{}{}", api_base, forward_uri_path_and_query);
        let mut reqwest_request = client.request(method, uri).json(&json);
        info!("reqwest_request: {reqwest_request:#?}");

        for (name, value) in headers.iter() {
            reqwest_request = reqwest_request.header(name, value);
        }

        let llm_req_start = Instant::now();
        let reqwest_response = reqwest_request.send().await.map_err(|e| {
            error!("Failed to reach LLM server: {:?}", e);
            GatewayApiError::LlmServiceError {
                status: StatusCode::SERVICE_UNAVAILABLE,
                message: "LLM server is unreachable".to_string(),
                provider: chosen_llm.name.clone(),
                details: None,
            }
        })?;
        let current_llm_resp = llm_req_start.elapsed().as_secs_f64();
        {
            let mut guard = llm_resp_time_holder.lock().await;
            *guard = current_llm_resp;
        }
        LLM_RESPONSE_TIME
            .with_label_values(&[chosen_llm.name.as_str()])
            .observe(current_llm_resp);

        let status = reqwest_response.status();
        let headers = reqwest_response.headers().clone();

        // If status is not successful, pass through the error response
        if !status.is_success() {
            let error_body = reqwest_response.bytes().await?;
            let status_code = status.as_u16();
            info!("status_code: {status_code:#?}");

            // Create a response that directly uses the error body
            let body = Full::from(error_body)
                .map_err(|never| match never {})
                .boxed();

            let mut error_response = Response::builder()
                .status(status)
                .header(CONTENT_TYPE, "application/json")
                .body(body)?;

            // Add the original headers and classifier
            *error_response.headers_mut() = headers;
            error_response.headers_mut().insert(
                "X-Chosen-Classifier",
                HeaderValue::from_str(&chosen_classifier).unwrap(),
            );

            error!("error_response: {error_response:#?}");
            return Ok(error_response);
        }

        if is_stream {
            let stream = reqwest_response.bytes_stream();
            let body = ReqwestStreamAdapter {
                inner: Box::pin(stream),
                llm_name: chosen_llm.name.clone(),
            };
            let boxed_body = BoxBody::new(body);

            let mut client_res = Response::new(boxed_body);
            *client_res.status_mut() = status;
            *client_res.headers_mut() = headers;
            client_res.headers_mut().insert(
                "X-Chosen-Classifier",
                HeaderValue::from_str(&chosen_classifier).unwrap(),
            );
            Ok(client_res)
        } else {
            let body_bytes = reqwest_response.bytes().await?;
            let body_clone = body_bytes.clone();
            // Parse and track token usage for non-streaming response
            if let Ok(json) = serde_json::from_slice::<Value>(&body_clone) {
                track_token_usage(&json, &chosen_llm.name);
            }
            let body = Full::from(body_bytes)
                .map_err(|never| match never {}) // never happens
                .boxed();

            let mut client_res = Response::builder().status(status).body(body)?;
            *client_res.headers_mut() = headers;
            client_res.headers_mut().insert(
                "X-Chosen-Classifier",
                HeaderValue::from_str(&chosen_classifier).unwrap(),
            );
            info!("client_res: {client_res:#?}");
            Ok(client_res)
        }
    })
    .await;

    let overall_latency = overall_start.elapsed().as_secs_f64();
    REQUEST_LATENCY.observe(overall_latency);

    let llm_resp_time = *llm_resp_time_holder.lock().await;
    let proxy_overhead = overall_latency - llm_resp_time - model_selection_time;
    PROXY_OVERHEAD_LATENCY.observe(proxy_overhead);

    match &result {
        Ok(response) => {
            if response.status().is_success() {
                REQUEST_SUCCESS.inc();
            } else {
                let status_code = response.status().as_u16();
                let error_type = if (400..500).contains(&status_code) {
                    "4xx"
                } else if (500..600).contains(&status_code) {
                    "5xx"
                } else {
                    "other"
                };
                REQUEST_FAILURE.with_label_values(&[error_type]).inc();
            }
        }
        Err(_err) => {
            // Handle system-level errors (non-HTTP errors)
            REQUEST_FAILURE.with_label_values(&["system"]).inc();
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Llm;
    use hyper::body::Body;
    use hyper::Request;
    use serde_json::json;

    fn create_test_config() -> RouterConfig {
        RouterConfig {
            policies: vec![Policy {
                name: "test_policy".to_string(),
                url: "http://triton:8000".to_string(),
                llms: vec![
                    Llm {
                        name: "Brainstroming".to_string(),
                        api_base: "https://integrate.api.nvidia.com".to_string(),
                        api_key: "test-key".to_string(),
                        model: "meta/llama-3.1-8b-instruct".to_string(),
                    },
                    Llm {
                        name: "Code Generation".to_string(),
                        api_base: "https://integrate.api.nvidia.com".to_string(),
                        api_key: "test-key".to_string(),
                        model: "meta/llama-3.1-8b-instruct".to_string(),
                    },
                ],
            }],
        }
    }

    #[tokio::test]
    async fn test_missing_nim_llm_router_params() {
        let config = create_test_config();
        let body = json!({
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(serde_json::to_vec(&body).unwrap())))
            .expect("Failed to create request");

        let response = proxy(req, config).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_policy_not_found() {
        let config = create_test_config();
        let body = json!({
            "messages": [{"role": "user", "content": "Hello"}],
            "nim-llm-router": {
                "policy": "nonexistent_policy",
                "routing_strategy": "manual",
                "model": "meta/llama-3.1-8b-instruct"
            }
        });

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .expect("Failed to create request");

        let response = proxy(req, config).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_model_not_found() {
        let config = create_test_config();
        let body = json!({
            "messages": [{"role": "user", "content": "Hello"}],
            "nim-llm-router": {
                "policy": "test_policy",
                "routing_strategy": "manual",
                "model": "nonexistent-model"
            }
        });

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(hyper::Body::from(serde_json::to_vec(&body).unwrap()))
            .expect("Failed to create request");

        let response = proxy(req, config).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
