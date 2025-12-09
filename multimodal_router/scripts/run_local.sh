#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the NAT router service
nat serve --config_file "$SCRIPT_DIR/../configs/config.yml" --host 0.0.0.0 --port 8000
