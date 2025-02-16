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

//! Triton
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct InferInputs {
    pub inputs: Vec<InferInputTensor>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Output {
    pub model_name: String,
    pub model_version: String,
    pub parameters: Parameters,
    pub outputs: Vec<InferOutputTensor>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Parameters {
    pub sequence_id: i64,
    pub sequence_start: bool,
    pub sequence_end: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct InferInputTensor {
    pub name: String,
    pub datatype: String,
    pub shape: Vec<i64>,
    pub data: Vec<Vec<String>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct InferOutputTensor {
    pub name: String,
    pub datatype: String,
    pub shape: Vec<i64>,
    pub data: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use crate::error::GatewayApiError;
    use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
    use tokio;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    #[tokio::test]
    async fn test_triton() -> Result<(), GatewayApiError> {
        // Start a mock server
        let mock_server = MockServer::start().await;

        // Create a mock POST request to some Triton URL like e.g.
        // "/v2/models/bert_ensemble/infer" and create the input we
        // might send
        let text_tensor = InferInputTensor {
            name: "INPUT".to_string(),
            datatype: "BYTES".to_string(),
            shape: vec![1, 1],
            data: vec![vec!["Hello world!".to_string()]],
        };

        let data = InferInputs {
            inputs: vec![text_tensor],
        };

        // Create create the output we might expect back
        let output_tensor = InferOutputTensor {
            name: "logits".to_string(),
            datatype: "FP32".to_string(),
            shape: vec![1, 3],
            data: vec![0.09, -0.45, 0.69],
        };
        let parameters = Parameters {
            sequence_id: 0,
            sequence_start: false,
            sequence_end: false,
        };
        let output = Output {
            model_name: "bert".to_string(),
            model_version: "1".to_string(),
            parameters,
            outputs: vec![output_tensor],
        };
        let mock_response = Mock::given(method("POST"))
            .and(path("/v2/models/bert_ensemble/infer"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&output));

        // Register the mock response we expect with the server
        mock_server.register(mock_response).await;

        // Create the reqwest client
        let client = reqwest::Client::new();

        // Create headers
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        // Create url
        let url = format!("{}/v2/models/bert_ensemble/infer", &mock_server.uri());

        // Call the function with the mock server URL
        let response = client
            .post(url)
            .headers(headers)
            .json(&data)
            .send()
            .await
            .unwrap();

        let response: Output = response.json().await.unwrap();

        // Assert the response
        assert_eq!(response.model_name, "bert");

        Ok(())
    }
}
