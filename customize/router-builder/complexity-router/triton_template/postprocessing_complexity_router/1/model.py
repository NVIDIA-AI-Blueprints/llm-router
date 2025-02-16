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

import numpy as np
import triton_python_backend_utils as pb_utils
from logits_processor import LogitsProcessor
from transformers import AutoConfig

class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.config = AutoConfig.from_pretrained("nvidia/prompt-task-and-complexity-classifier")
        
        # Load the necessary configurations
        self.target_sizes = self.config.target_sizes
        self.task_type_map = self.config.task_type_map
        self.weights_map = self.config.weights_map
        self.divisor_map = self.config.divisor_map
        
        # Initialize the LogitsProcessor
        self.processor = LogitsProcessor(
            task_type_map=self.task_type_map,
            weights_map=self.weights_map,
            divisor_map=self.divisor_map)
        
        # complexity metrics from config
        self.complexity_metrics = []
        for metric in self.weights_map.keys():
            if metric != "task_type" and metric in self.divisor_map:
                self.complexity_metrics.append(metric)
        
        self.logger.log_info(f"Using complexity metrics: {self.complexity_metrics}")
        self.logger.log_info("Initialization complete")

    def execute(self, requests):
        self.logger.log_info(f"Executing {len(requests)} requests")
        responses = []
        
        for request in requests:
            logits = pb_utils.get_input_tensor_by_name(request, "logits").as_numpy()
            self.logger.log_info(f"Logits Shape: {logits.shape}")
            
            # Process the logits into separate outputs
            processed_results = self.process_results(logits, self.target_sizes.values())
            self.logger.log_info(f"Processed Results Shape: {processed_results}")
            result = self.processor.process_logits(processed_results)
            self.logger.log_info(f"Final result: {result}")
            # Calculate complexity scores
            complexity_scores = {}
            for metric in self.complexity_metrics:
                if metric in result:
                    complexity_scores[metric] = float(result[metric][0])
            # Find the highest complexity metric
            highest_complexity = max(complexity_scores.items(), key=lambda x: x[1])[0]
            
            # Create one-hot encoded vector for complexity metrics
            one_hot = np.zeros(len(self.complexity_metrics))
            metric_index = self.complexity_metrics.index(highest_complexity)
            one_hot[metric_index] = 1
            
            self.logger.log_info(f"Complexity scores: {complexity_scores}")
            self.logger.log_info(f"Prompt complexity score: {result['prompt_complexity_score'][0]}")
            self.logger.log_info(f"Highest complexity: {highest_complexity}")
            self.logger.log_info(f"Encoded one hot vector: {one_hot}")
            
            # Create output tensor
            output_tensor = pb_utils.Tensor("OUTPUT", one_hot.astype(np.float32))
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)
        
        return responses


    def process_results(self, output_tensor, target_sizes):
        results = []
        start_idx = 0
        for size in target_sizes:
            end_idx = start_idx + size
            result = output_tensor[:, start_idx:end_idx]
            results.append(result)
            start_idx = end_idx
        return results

    def finalize(self):
        self.logger.log_info("Finalizing TritonPythonModel")
