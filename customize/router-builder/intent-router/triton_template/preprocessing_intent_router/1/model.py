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

import json
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import DebertaV2Tokenizer

class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        args_str = json.dumps(args, indent=2)
        self.logger.log_info(f"Initializing the model with args: {args_str}")
        self.model_config = json.loads(args['model_config'])
        
        # Initialize the tokenizer
        self.tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/DeBERTa-v3-base')
        self.logger.log_info("Tokenizer initialized with model: microsoft/DeBERTa-v3-base")
        
        output0_config = pb_utils.get_output_config_by_name(self.model_config, "preprocessed_input_ids")
        output1_config = pb_utils.get_output_config_by_name(self.model_config, "preprocessed_attention_mask")
        
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config['data_type'])
        self.output1_dtype = pb_utils.triton_string_to_numpy(output1_config['data_type'])
        self.logger.log_info(f"Output configurations set: output0_dtype={self.output0_dtype}, output1_dtype={self.output1_dtype}")

    def execute(self, requests):
        responses = []
        for request in requests:
            input_text = pb_utils.get_input_tensor_by_name(request, "INPUT").as_numpy()[0][0].decode('utf-8')
            self.logger.log_info(f"Input text: {input_text}")

            # Tokenize the input
            encoded = self.tokenizer(
                input_text,
                add_special_tokens=True,
                max_length=512,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='np',
            )
            self.logger.log_info(f"Encoded input: {encoded}")

            input_ids = encoded['input_ids'].astype(self.output0_dtype)
            attention_mask = encoded['attention_mask'].astype(self.output1_dtype)
            
            output0 = pb_utils.Tensor("preprocessed_input_ids", input_ids)
            output1 = pb_utils.Tensor("preprocessed_attention_mask", attention_mask)
            
            inference_response = pb_utils.InferenceResponse(output_tensors=[output0, output1])
            responses.append(inference_response)
        
        return responses

    def finalize(self):
        self.logger.log_info("Finalizing preprocessing model")
