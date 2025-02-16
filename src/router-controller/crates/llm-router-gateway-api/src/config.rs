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

//! Config
use crate::error::ConfigError;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RouterConfig {
    pub policies: Vec<Policy>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Policy {
    pub name: String,
    pub url: String,
    pub llms: Vec<Llm>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Llm {
    pub name: String,
    pub api_base: String,
    pub api_key: String,
    pub model: String,
}

impl RouterConfig {
    pub fn load_config(path: &str) -> Result<RouterConfig> {
        let content = std::fs::read_to_string(path)?;
        let config: RouterConfig = serde_yaml::from_str(&content)?;
        validate_config(&config)?;
        Ok(config)
    }

    pub fn get_policy_by_name(&self, name: &str) -> Option<Policy> {
        self.policies
            .iter()
            .find(|policy| policy.name.trim() == name.trim())
            .cloned()
    }

    pub fn get_policy_by_index(&self, index: usize) -> Option<Policy> {
        self.policies.get(index).cloned()
    }

    pub fn sanitized(&self) -> Self {
        let sanitized_policies = self
            .policies
            .iter()
            .map(|policy| {
                let sanitized_llms = policy
                    .llms
                    .iter()
                    .map(|llm| Llm {
                        api_key: "[REDACTED]".to_string(),
                        ..llm.clone()
                    })
                    .collect();
                Policy {
                    llms: sanitized_llms,
                    ..policy.clone()
                }
            })
            .collect();

        RouterConfig {
            policies: sanitized_policies,
        }
    }
}

impl Policy {
    pub fn get_llm_by_name(&self, name: &str) -> Option<Llm> {
        self.llms
            .iter()
            .find(|llm| llm.name.trim() == name.trim())
            .cloned()
    }

    pub fn get_llm_by_index(&self, index: usize) -> Option<Llm> {
        self.llms.get(index).cloned()
    }

    pub fn get_llm_name_by_index(&self, index: usize) -> Option<String> {
        self.llms.get(index).map(|llm| llm.name.clone())
    }
}

pub type Result<T> = std::result::Result<T, ConfigError>;

fn validate_config(config: &RouterConfig) -> Result<()> {
    for policy in &config.policies {
        if policy.name.is_empty() {
            return Err(ConfigError::MissingPolicyField {
                policy: policy.name.clone(),
                field: "name".to_string(),
            });
        }

        for llm in &policy.llms {
            if llm.api_base.is_empty() {
                return Err(ConfigError::MissingLlmField {
                    llm: llm.name.clone(),
                    field: "api_base".to_string(),
                });
            }
            if llm.model.is_empty() {
                return Err(ConfigError::MissingLlmField {
                    llm: llm.name.clone(),
                    field: "model".to_string(),
                });
            }
            if llm.api_key.is_empty() {
                return Err(ConfigError::MissingLlmField {
                    llm: llm.name.clone(),
                    field: "api_key".to_string(),
                });
            }
        }
    }
    Ok(())
}
