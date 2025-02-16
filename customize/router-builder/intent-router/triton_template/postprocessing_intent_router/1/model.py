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
import os

class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.logger.log_info("Initializing TritonPythonModel")
        
        #Load the label encoder
        current_dir = os.path.dirname(__file__)
        labels_path = os.path.join(current_dir, 'labels.json')
        with open(labels_path, 'r') as f:
            self.label_map = json.load(f)
        self.logger.log_info(f"Loaded labels {self.label_map}")
        # self.label_map= {"0": "account_management", "1": "billing", "2": "customer_education", "3": "dispute_resolution", "4": "financial_planning", "5": "international_services", "6": "product_information", "7": "security_and_fraud_prevention", "8": "technical_support", "9": "transaction_support"}
        self.num_labels = len(self.label_map)        
        self.logger.log_info("Initialization complete")

    def execute(self, requests):
        self.logger.log_info(f"Executing {len(requests)} requests")
        responses = []
        for i, request in enumerate(requests):
            self.logger.log_info(f"Processing request {i+1}")
            logits = pb_utils.get_input_tensor_by_name(request, "logits").as_numpy()
            self.logger.log_info(f"Logits Shape: {logits.shape}")
            
            # Process the logits
            predicted_class_id = np.argmax(logits, axis=1)
            self.logger.log_info(f"Predicted class ID: {predicted_class_id}")
            
            # Create one-hot encoded vector
            one_hot = np.zeros((logits.shape[0], self.num_labels), dtype=np.float32)
            one_hot[np.arange(logits.shape[0]), predicted_class_id] = 1
            self.logger.log_info(f"One-hot encoded shape: {one_hot.shape}")
            
            # Get the predicted class label
            predicted_classes = [self.label_map[str(id)] for id in predicted_class_id]
            self.logger.log_info(f"Predicted classes: {predicted_classes}")
            
            # Create output tensors
            one_hot_tensor = pb_utils.Tensor("OUTPUT", one_hot)
            
            inference_response = pb_utils.InferenceResponse(output_tensors=[one_hot_tensor])
            responses.append(inference_response)
        
        self.logger.log_info("Execution complete")
        return responses

    def finalize(self):
        self.logger.log_info("Finalizing TritonPythonModel")
