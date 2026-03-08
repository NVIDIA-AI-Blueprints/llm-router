# Application Integration

The Model Router Toolkit exposes an OpenAI-compatible API. Any application that speaks the OpenAI protocol can use it as a drop-in replacement.

## API Keys

The server and LiteLLM SDK integration call model providers at inference time, so an API key is required:

```bash
export OPENROUTER_API_KEY=sk-or-...   # for OpenRouter-based configs
# or
export NVIDIA_API_KEY=nvapi-...       # for NVIDIA NIM-based configs
```

Set before starting the server or initializing the strategy.

## Server Mode

Start the server:

```bash
model-router serve --config configs/prefill-qwen08b.yaml --port 8000
```

### OpenAI Python SDK

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

### Environment Variable (LLM-powered tools, coding assistants, etc.)

```bash
export OPENAI_API_BASE=http://localhost:8000/v1
```

Any tool that reads `OPENAI_API_BASE` or `OPENAI_BASE_URL` will route through the toolkit.

### cURL

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "routed",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Playground UI

Open `http://localhost:8000/` in a browser for the interactive playground with routing cards, probability bars, tolerance slider, and model toggles.

## LiteLLM SDK Integration (No Server)

For applications already using LiteLLM, add routing without running a separate server:

```python
from litellm import Router
from model_router_toolkit import ModelRoutingStrategy

model_list = [
    {"model_name": "nem-think", "litellm_params": {"model": "openrouter/nvidia/nemotron-3-nano-30b-a3b"}},
    {"model_name": "nem-nothink", "litellm_params": {"model": "openrouter/nvidia/nemotron-3-nano-30b-a3b"}},
]

router = Router(model_list=model_list)
strategy = ModelRoutingStrategy.from_config("configs/prefill-qwen08b.yaml")
router.set_custom_routing_strategy(strategy)

response = await router.acompletion(
    model="nem-think",
    messages=[{"role": "user", "content": "Prove sqrt(2) is irrational"}],
)
```

Or with an existing LiteLLM Router:

```python
from litellm import Router
from model_router_toolkit import ModelRoutingStrategy

model_list = [
    {"model_name": "nem-think", "litellm_params": {"model": "openrouter/nvidia/nemotron-3-nano-30b-a3b"}},
    {"model_name": "nem-nothink", "litellm_params": {"model": "openrouter/nvidia/nemotron-3-nano-30b-a3b"}},
]

router = Router(model_list=model_list)
strategy = ModelRoutingStrategy.from_config("configs/prefill-qwen08b.yaml")
router.set_custom_routing_strategy(strategy)

response = await router.acompletion(
    model="nem-think",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Direct Python Library

Use the router directly without LiteLLM or a server:

```python
from model_router_toolkit.config import load_config, build_router_from_config

config = load_config("configs/prefill-qwen08b.yaml")
router = build_router_from_config(config)

result = router.route("What is the capital of France?", tolerance=0.20)
print(result.selected_model)       # cheapest model above threshold
print(result.confidences)          # P(correct) per model
print(result.selected_cost)        # estimated cost
print(result.metadata)             # routing metadata (p_max, threshold)
```

## Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat (streaming + non-streaming) |
| `/api/chat` | POST | SSE chat endpoint for the playground UI |
| `/api/models` | GET | Model pool with cost data |
| `/api/config` | GET | Server config (routing method, available features) |
| `/api/review` | POST | Auto-review: judges answer correctness (when API key available) |
| `/health` | GET | Health check |
| `/` | GET | Interactive playground UI |
