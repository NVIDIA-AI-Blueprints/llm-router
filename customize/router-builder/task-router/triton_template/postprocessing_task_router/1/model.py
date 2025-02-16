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
import json
from logits_processor import LogitsProcessor
from transformers import AutoConfig


class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.config = AutoConfig.from_pretrained("nvidia/prompt-task-and-complexity-classifier")
        # args_str = json.dumps(args, indent=2)
        # self.logger.log_info(f"Initializing TritonPythonModel with args: {args_str}")
        # model_config = json.loads(args['model_config'])
        
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
        
        self.logger.log_info("Initialization complete")

    def execute(self, requests):
        self.logger.log_info(f"Executing {len(requests)} requests")
        responses = []
        for i, request in enumerate(requests):
            self.logger.log_info(f"Processing request {i+1}")
            logits = pb_utils.get_input_tensor_by_name(request, "logits").as_numpy()
            self.logger.log_info(f"Logits Shape: {logits.shape}")
            # Process the logits
            # This is to reformat the logits into the appropriate shapes before processing
            processed_results = self.process_results(logits, self.target_sizes.values())
            self.logger.log_info(f"Processed Results Shape: {processed_results}")
            result = self.processor.process_logits(processed_results)
            self.logger.log_info(f"Final result: {result}")
            # Create one-hot encoded vector for task types
            task_types = list(self.task_type_map.values())
            one_hot = np.zeros(len(task_types))
            task_type_index = task_types.index(result['task_type_1'][0])
            one_hot[task_type_index] = 1
            self.logger.log_info(f"Encoded one hot vector: {one_hot}")

            # Create output tensor
            output_tensor = pb_utils.Tensor("OUTPUT", one_hot.astype(np.float32))
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)
            
            
            # # Convert the result dictionary to a JSON string
            # output_data = json.dumps(result).encode('utf-8')
            # output_tensor = pb_utils.Tensor("OUTPUT", np.array([output_data]))
            # inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            # responses.append(inference_response)
        self.logger.log_info("Execution complete")
        return responses

    # it's a concatenated tensor of multiple outputs, so we need to separate it
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