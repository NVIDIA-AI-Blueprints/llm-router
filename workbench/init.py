import yaml
import os
import sys

sys.dont_write_bytecode = True

# Load the YAML file
with open('/project/src/router-controller/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Function to replace 'api_key' fields with the environment variable
def replace_api_keys(data):
    for i in range(len(data["policies"])):
        if "llms" in data["policies"][i]:
            for j in range(len(data["policies"][i]["llms"])):
                if "api_key" in data["policies"][i]["llms"][j]:
                    data["policies"][i]["llms"][j]["api_key"] = os.environ.get('NVIDIA_API_KEY', '')
    return data

# Replace 'api_key' fields
new_config = replace_api_keys(config)

# Dump the modified YAML back to a file
with open('/project/src/router-controller/config.yaml', 'w') as file:
    yaml.dump(new_config, file, default_flow_style=False)