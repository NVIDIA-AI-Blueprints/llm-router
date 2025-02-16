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

import os
import yaml

def update_yaml_with_env_var(yaml_file_path, env_var_name):
  """
  This function updates a YAML file with the value of an environment variable.

  Args:
      yaml_file_path (str): Path to the YAML file.
      env_var_name (str): Name of the environment variable containing the API key.
      replacement_value (str, optional): The string to replace with the environment variable value. Defaults to "api_key: API_KEY_VALUE".
  """
  # Get the API key from the environment variable
  api_key = os.getenv(env_var_name)

  # Check if the environment variable is set
  if not api_key:
    print(f"Environment variable '{env_var_name}' not found. Skipping update.")
    return

  # Read the YAML file
  with open(yaml_file_path, 'r') as f:
    data = yaml.safe_load(f)

  # Update the llms section with the API key
  for policy_id in range(0, len(data['policies'])):
    for llm_id in range(0, len(data['policies'][policy_id]['llms'])): 
        data['policies'][policy_id]['llms'][llm_id]['api_key'] = f"{api_key}"

  # Write the updated data back to the YAML file
  with open(yaml_file_path, 'w') as f:
    yaml.dump(data, f)

  print(f"Successfully updated '{yaml_file_path}' with API key from '{env_var_name}'.")


yaml_file_path = "src/router-controller/config.yaml"
env_var_name = "NVIDIA_API_KEY"

update_yaml_with_env_var(yaml_file_path, env_var_name)