# Application Integration

The Model Router Toolkit exposes an OpenAI-compatible API at `/v1/chat/completions`. Any application that speaks the OpenAI protocol can use it as a drop-in replacement.

## Quick Setup

Start the server:
```bash
model-router serve --config configs/cloud-only.yaml --port 8000
```

## OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="routed",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
print(response.choices[0].message.content)
```

## OpenClaw

Set the environment variable before starting OpenClaw:
```bash
export OPENAI_API_BASE=http://localhost:8000/v1
```

## OpenCode

Set the environment variable:
```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
```

Then configure OpenCode to use the "routed" model.

## cURL

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "routed",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

## LiteLLM SDK Integration

For applications already using LiteLLM's Router, you can add routing without running a separate server:

```python
from litellm import Router
from model_router_toolkit import ModelRoutingStrategy

# Your existing model list
model_list = [
    {"model_name": "nem-think", "litellm_params": {"model": "nvidia_nim/nvidia/nvidia/Nemotron-3-Nano-30B-A3B"}},
    {"model_name": "gpt-5.2", "litellm_params": {"model": "nvidia_nim/openai/openai/gpt-5.2"}},
]

router = Router(model_list=model_list)
strategy = ModelRoutingStrategy.from_config("configs/cloud-only.yaml")
router.set_custom_routing_strategy(strategy)

# Now every call is intelligently routed
response = await router.acompletion(
    model="nem-think",
    messages=[{"role": "user", "content": "Prove sqrt(2) is irrational"}],
)

# Access routing metadata
print(strategy.last_result.selected_model)
print(strategy.last_result.confidences)
```

## LiteLLM Proxy

For existing LiteLLM proxy deployments, the ModelRoutingStrategy can be registered as a custom routing strategy. This feature is planned but not yet available via YAML config. Use the SDK integration above in the meantime.
