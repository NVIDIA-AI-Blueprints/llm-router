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

//! Stream
use crate::error::GatewayApiError;
use crate::metrics::track_token_usage;
use bytes::Bytes;
use futures_util::Stream;
use http_body::Frame;
use log::{debug, info, warn};
use pin_project_lite::pin_project;
use serde_json::Value;
use std::pin::Pin;

pin_project! {
    pub struct ReqwestStreamAdapter {
        #[pin]
        pub inner: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send + Sync>>,
        pub llm_name: String,
    }
}

impl http_body::Body for ReqwestStreamAdapter {
    type Data = Bytes;
    type Error = GatewayApiError;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.project();
        match this.inner.poll_next(cx) {
            std::task::Poll::Ready(Some(Ok(chunk))) => {
                let chunk_str = String::from_utf8_lossy(&chunk);
                for event in chunk_str.split("\n\n") {
                    let cleaned_event = event.trim().strip_prefix("data: ").unwrap_or(event);

                    if cleaned_event.is_empty() || cleaned_event == "[DONE]" {
                        continue;
                    }

                    debug!("Processing event: {}", cleaned_event);

                    match serde_json::from_str::<Value>(cleaned_event) {
                        Ok(json) => {
                            // Handle final usage statistics
                            if let Some(finish_reason) =
                                json["choices"][0]["finish_reason"].as_str()
                            {
                                if finish_reason == "stop" {
                                    if let Some(usage) = json.get("usage") {
                                        let prompt = usage["prompt_tokens"].as_u64().unwrap_or(0);
                                        let completion =
                                            usage["completion_tokens"].as_u64().unwrap_or(0);
                                        let total = usage["total_tokens"].as_u64().unwrap_or(0);
                                        info!(
                                            "Usage statistics: prompt={}, completion={}, total={}",
                                            prompt, completion, total
                                        );
                                        track_token_usage(&json, this.llm_name);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            warn!("Failed to parse JSON: {} in {}", e, cleaned_event);
                        }
                    }
                }
                std::task::Poll::Ready(Some(Ok(Frame::data(chunk))))
            }
            std::task::Poll::Ready(Some(Err(e))) => {
                std::task::Poll::Ready(Some(Err(GatewayApiError::from(e))))
            }
            std::task::Poll::Ready(None) => std::task::Poll::Ready(None),
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}
