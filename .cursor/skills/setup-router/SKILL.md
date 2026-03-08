# Set Up Router

Trigger: "set up router", "configure router", "deploy router", "model-router setup"

## Steps
1. Run `model-router setup` in the terminal
2. The wizard detects GPUs, asks for routing method and API keys
3. Generates configs/generated.yaml
4. Run `model-router serve --config configs/generated.yaml`
5. Access UI at http://localhost:8000/
6. Access API at http://localhost:8000/v1/chat/completions

## Manual Setup
If you prefer manual config:
1. Copy configs/cloud-only.yaml (or openrouter-kmeans.yaml, local-prefill.yaml)
2. Edit model pool and API settings
3. Set NVIDIA_API_KEY or OPENROUTER_API_KEY in environment
4. Run `model-router serve --config your-config.yaml`
