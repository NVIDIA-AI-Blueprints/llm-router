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

use http::header::InvalidHeaderValue;
use http::{Response, StatusCode};
use http_body_util::{combinators::BoxBody, BodyExt, Full};
use hyper::body::Bytes;
use serde_json::{json, Value};
use std::convert::Infallible;
use thiserror::Error;

pub trait IntoResponse {
    fn into_response(self) -> Response<BoxBody<Bytes, GatewayApiError>>;
}

#[derive(Debug, Clone, PartialEq)]
pub enum ErrorSource {
    Triton,
    LlmProvider,
    Router,
    Client,
    Infrastructure,
}

#[derive(Debug, Error)]
pub enum GatewayApiError {
    // Triton specific errors
    #[error("Triton Error: {message}")]
    TritonError {
        message: String,
        code: u16,
        details: Option<String>,
    },

    // LLM Provider errors
    #[error("LLM Service Error: {message}")]
    LlmServiceError {
        status: StatusCode,
        message: String,
        provider: String,
        details: Option<Value>,
    },

    // Router errors
    #[error("Routing Error: {message}")]
    RoutingError {
        message: String,
        error_type: RoutingErrorType,
    },

    // Client errors
    #[error("Client Error: {message}")]
    ClientError {
        status: StatusCode,
        message: String,
        error_type: String,
    },

    // Infrastructure errors
    #[error("Infrastructure Error: {0}")]
    Infrastructure(String),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    // #[error(transparent)]
    // InvalidUri(#[from] http::uri::InvalidUri),
    #[error(transparent)]
    Http(#[from] http::Error),

    #[error(transparent)]
    Hyper(#[from] hyper::Error),

    #[error("Invalid request: {message}")]
    InvalidRequest { message: String },

    #[error("Triton service error ({status_code}): {message}")]
    TritonServiceError { status_code: u16, message: String },

    #[error("Unexpected error: {message}")]
    UnexpectedError { message: String },

    #[error("Policy not found: {0}")]
    PolicyNotFound(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("No policy specified in nim-llm-router params")]
    MissingPolicy,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Missing field '{field}' in policy '{policy}'")]
    MissingPolicyField { policy: String, field: String },
    #[error("Missing field '{field}' in LLM '{llm}'")]
    MissingLlmField { llm: String, field: String },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Yaml(#[from] serde_yaml::Error),
}

#[derive(Debug, Clone, PartialEq)]
pub enum RoutingErrorType {
    PolicyNotFound,
    ModelNotFound,
    NoRoutingStrategy,
    InvalidConfiguration,
    TritonUnavailable,
}

impl GatewayApiError {
    pub fn error_source(&self) -> ErrorSource {
        match self {
            Self::TritonError { .. } => ErrorSource::Triton,
            Self::LlmServiceError { .. } => ErrorSource::LlmProvider,
            Self::RoutingError { .. } => ErrorSource::Router,
            Self::ClientError { .. } => ErrorSource::Client,
            _ => ErrorSource::Infrastructure,
        }
    }

    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::TritonError { code, .. } => {
                StatusCode::from_u16(*code).unwrap_or(StatusCode::SERVICE_UNAVAILABLE)
            }
            Self::LlmServiceError { status, .. } => *status,
            Self::ClientError { status, .. } => *status,
            Self::RoutingError { error_type, .. } => match error_type {
                RoutingErrorType::PolicyNotFound => StatusCode::BAD_REQUEST,
                RoutingErrorType::ModelNotFound => StatusCode::NOT_FOUND,
                RoutingErrorType::NoRoutingStrategy => StatusCode::BAD_REQUEST,
                RoutingErrorType::InvalidConfiguration => StatusCode::INTERNAL_SERVER_ERROR,
                RoutingErrorType::TritonUnavailable => StatusCode::SERVICE_UNAVAILABLE,
            },
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub fn to_response(&self) -> Result<Response<BoxBody<Bytes, Self>>, Self> {
        let error_response = match self {
            Self::LlmServiceError {
                status,
                message,
                provider,
                details,
            } => json!({
                "error": {
                    "type": "llm_service_error",
                    "message": message,
                    "status": status.as_u16(),
                    "provider": provider,
                    "details": details,
                    "source": "llm_provider"
                }
            }),
            Self::TritonError {
                message,
                code,
                details,
            } => json!({
                "error": {
                    "type": "triton_error",
                    "message": message,
                    "code": code,
                    "details": details,
                    "source": "triton"
                }
            }),
            Self::RoutingError {
                message,
                error_type,
            } => json!({
                "error": {
                    "type": format!("routing_error_{:?}", error_type).to_lowercase(),
                    "message": message,
                    "status": self.status_code().as_u16(),
                    "source": "router"
                }
            }),
            Self::ClientError {
                message,
                error_type,
                status,
            } => json!({
                "error": {
                    "type": error_type,
                    "message": message,
                    "status": status.as_u16(),
                    "source": "client"
                }
            }),
            _ => json!({
                "error": {
                    "type": "internal_error",
                    "message": self.to_string(),
                    "status": self.status_code().as_u16(),
                    "source": "infrastructure"
                }
            }),
        };

        let body_bytes = Bytes::from(serde_json::to_vec(&error_response)?);
        let boxed_body = Full::from(body_bytes)
            .map_err(|never| match never {})
            .boxed();

        Ok(Response::builder()
            .status(self.status_code())
            .header("Content-Type", "application/json")
            .body(boxed_body)?)
    }

    // Constructor methods
    pub fn triton_error(message: impl Into<String>, code: u16) -> Self {
        Self::TritonError {
            message: message.into(),
            code,
            details: None,
        }
    }

    pub fn llm_error(
        status: StatusCode,
        message: impl Into<String>,
        provider: impl Into<String>,
    ) -> Self {
        Self::LlmServiceError {
            status,
            message: message.into(),
            provider: provider.into(),
            details: None,
        }
    }

    pub fn routing_error(message: impl Into<String>, error_type: RoutingErrorType) -> Self {
        Self::RoutingError {
            message: message.into(),
            error_type,
        }
    }

    pub fn client_error(
        status: StatusCode,
        message: impl Into<String>,
        error_type: impl Into<String>,
    ) -> Self {
        Self::ClientError {
            status,
            message: message.into(),
            error_type: error_type.into(),
        }
    }
}

impl From<reqwest::Error> for GatewayApiError {
    fn from(error: reqwest::Error) -> Self {
        if let Some(status) = error.status() {
            Self::client_error(status, error.to_string(), "http_client_error")
        } else {
            Self::Infrastructure(error.to_string())
        }
    }
}

impl From<Infallible> for GatewayApiError {
    fn from(err: Infallible) -> Self {
        match err {}
    }
}

impl From<InvalidHeaderValue> for GatewayApiError {
    fn from(err: InvalidHeaderValue) -> Self {
        GatewayApiError::InvalidRequest {
            message: format!("Invalid header value: {}", err),
        }
    }
}

impl IntoResponse for GatewayApiError {
    fn into_response(self) -> Response<BoxBody<Bytes, GatewayApiError>> {
        let (status, message) = match &self {
            GatewayApiError::InvalidRequest { message } => {
                (StatusCode::BAD_REQUEST, message.clone())
            }
            GatewayApiError::PolicyNotFound(policy) => (
                StatusCode::NOT_FOUND,
                format!("Policy '{}' not found", policy),
            ),
            _ => (self.status_code(), self.to_string()),
        };

        let error_json = json!({
            "error": {
                "message": message,
                "status": status.as_u16()
            }
        });

        let body = Full::from(Bytes::from(
            serde_json::to_vec(&error_json).unwrap_or_default(),
        ))
        .map_err(|never| match never {})
        .boxed();

        Response::builder()
            .status(status)
            .header("Content-Type", "application/json")
            .body(body)
            .unwrap_or_else(|_| {
                Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .body(
                        Full::from(Bytes::from("Internal Server Error"))
                            .map_err(|never| match never {})
                            .boxed(),
                    )
                    .expect("Failed to create error response")
            })
    }
}

impl From<()> for GatewayApiError {
    fn from(_: ()) -> Self {
        GatewayApiError::UnexpectedError {
            message: "Empty error conversion".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_llm_service_error() {
        let error = GatewayApiError::llm_error(
            StatusCode::PAYMENT_REQUIRED,
            "Insufficient credits",
            "OpenAI",
        );
        let response = error.to_response().unwrap();

        assert_eq!(response.status(), StatusCode::PAYMENT_REQUIRED);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["error"]["type"], "llm_service_error");
        assert_eq!(json["error"]["provider"], "OpenAI");
        assert_eq!(json["error"]["status"], 402);
        assert_eq!(json["error"]["source"], "llm_provider");
    }

    #[tokio::test]
    async fn test_triton_error() {
        let error = GatewayApiError::triton_error("Model loading failed", 503);
        let response = error.to_response().unwrap();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["error"]["type"], "triton_error");
        assert_eq!(json["error"]["code"], 503);
        assert_eq!(json["error"]["source"], "triton");
    }

    #[tokio::test]
    async fn test_routing_error() {
        let error = GatewayApiError::routing_error(
            "No appropriate model found",
            RoutingErrorType::ModelNotFound,
        );
        let response = error.to_response().unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["error"]["type"], "routing_error_model_not_found");
        assert_eq!(json["error"]["source"], "router");
    }

    #[tokio::test]
    async fn test_client_error() {
        let error = GatewayApiError::client_error(
            StatusCode::BAD_REQUEST,
            "Invalid request parameters",
            "validation_error",
        );
        let response = error.to_response().unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["error"]["type"], "validation_error");
        assert_eq!(json["error"]["source"], "client");
    }
}
