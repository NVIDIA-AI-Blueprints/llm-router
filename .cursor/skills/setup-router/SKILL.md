# Set Up Router

Trigger: "set up router", "configure router", "deploy router", "model-router setup", "start server"

## Quick start (use existing checkpoint)

```bash
pip install -e '.[prefill]'
model-router serve --config configs/prefill-qwen08b.yaml --port 8000
```

Access playground UI at `http://localhost:8000/`, API at `http://localhost:8000/v1/chat/completions`.

Requires: `OPENROUTER_API_KEY` (or `NVIDIA_API_KEY` depending on config).

## Interactive setup

```bash
model-router setup
```

The wizard detects GPUs, asks for routing method and API keys, generates `configs/generated.yaml`.

## Manual config

1. Copy an example config:
   - `configs/prefill-qwen08b.yaml` -- prefill routing, 4 models, Qwen3.5-0.8B encoder
   - `configs/smoke-test.yaml` -- minimal 2-model config for testing
   - `configs/cloud-only.yaml` -- kmeans routing, cloud embeddings
2. Edit the model pool and API settings
3. Set API keys: `export OPENROUTER_API_KEY=...` or `export NVIDIA_API_KEY=...`
4. Start: `model-router serve --config your-config.yaml`

## Connect your app

Any OpenAI-compatible client works:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="routed",
    messages=[{"role": "user", "content": "Hello"}]
)
```

For existing LiteLLM setups:
```python
from model_router_toolkit import ModelRoutingStrategy
strategy = ModelRoutingStrategy.from_config("configs/prefill-qwen08b.yaml")
litellm.set_custom_routing_strategy(strategy)
```
