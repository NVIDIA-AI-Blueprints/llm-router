# Application Integration

The Model Router Toolkit provides four integration paths, from a standalone server with a UI down to a direct Python API. All paths that serve HTTP perform both routing and LLM inference — the toolkit picks the model, then LiteLLM dispatches the request to the provider.

| Path | Best for | Routing | Inference | Config files |
|------|----------|---------|-----------|-------------|
| [Standalone Server](#standalone-server) | Demos, development, exploring | Yes | Yes | 1 (pool config) |
| [LiteLLM Proxy](#litellm-proxy) | Production, existing LiteLLM stacks | Yes | Yes | 2 (pool + litellm proxy) |
| [LiteLLM SDK](#litellm-sdk-integration-no-server) | Embedding in your own app | Yes | Yes | 1 (pool config) |
| [Direct Python Library](#direct-python-library) | Routing decisions only, no inference | Yes | No | 1 (pool config) |

## API Keys

Both the server and proxy modes call model providers at inference time, so an API key is required:

```bash
export OPENROUTER_API_KEY=sk-or-...   # for OpenRouter-based configs
# or
export NVIDIA_API_KEY=nvapi-...       # for NVIDIA NIM-based configs
```

Set before starting the server, proxy, or initializing the SDK strategy. **Not needed** for the Direct Python Library path (routing only, no inference).

---

## Standalone Server

A lightweight FastAPI server with a built-in playground UI. Good for demos, local development, and quick deployments.

```bash
model-router serve --config configs/prefill-qwen08b.yaml --port 8000
```

### Connecting your app

**OpenAI Python SDK:**

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

**Environment variable** (works with any tool that reads `OPENAI_API_BASE`):

```bash
export OPENAI_API_BASE=http://localhost:8000/v1
```

**cURL:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "routed",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

**Playground UI:** Open `http://localhost:8000/` in a browser for the interactive playground with routing cards, probability bars, tolerance slider, and model toggles.

### Server endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat (streaming + non-streaming) |
| `/api/chat` | POST | SSE chat endpoint for the playground UI |
| `/api/models` | GET | Model pool with cost data |
| `/api/config` | GET | Server config (routing method, available features) |
| `/api/review` | POST | Auto-review: judges answer correctness (when API key available) |
| `/health` | GET | Health check |
| `/` | GET | Interactive playground UI |

---

## LiteLLM Proxy

Starts the full **LiteLLM Proxy server** with the routing strategy injected at startup. Use this when you want LiteLLM's production features (auth, rate limiting, spend tracking, caching, virtual keys) or when you already run a LiteLLM proxy and want to add intelligent routing.

### Setup

Generate the LiteLLM proxy config from your pool config, then start the proxy:

```bash
# Generate litellm config (one-time)
model-router proxy-config \
    --config configs/prefill-qwen08b.yaml \
    --output configs/litellm-proxy.yaml

# Start the proxy
model-router proxy \
    --litellm-config configs/litellm-proxy.yaml \
    --router-config configs/prefill-qwen08b.yaml \
    --port 4000
```

The proxy requires **two config files**:

| File | What it controls |
|------|-----------------|
| LiteLLM proxy config (`litellm-proxy.yaml`) | `model_list` with provider endpoints and API keys, `router_settings`, auth, caching |
| Pool config (`prefill-qwen08b.yaml`) | Routing method, checkpoint path, tolerance, encoder, model costs |

The `proxy-config` command bridges the two — it reads your pool config and generates a matching LiteLLM config with the correct model names and API key references.

### Connecting your app

The proxy exposes the same OpenAI-compatible API on port 4000:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000/v1",
    api_key="not-needed",  # or your LiteLLM virtual key if auth is configured
)

response = client.chat.completions.create(
    model="nem-think",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
```

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nem-think",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Docker

For containerized deployment, use the Docker compose setup:

```bash
# Set API keys
export OPENROUTER_API_KEY=sk-or-...

# Start (proxy mode, CPU)
docker compose -f docker/docker-compose.yaml up
```

Or build directly:

```bash
# CPU (KMeans routing)
docker build -f docker/Dockerfile --target proxy -t model-router:proxy .

# GPU (prefill routing with local encoder)
docker build -f docker/Dockerfile --target proxy-gpu -t model-router:gpu .
```

### Serve vs Proxy — when to use which

| Scenario | Use | Why |
|----------|-----|-----|
| Trying out the toolkit for the first time | `serve` | Playground UI, single config, minimal setup |
| Local development and debugging | `serve` | Playground UI shows routing decisions visually |
| Already running a LiteLLM proxy | `proxy` | Drop-in — keeps your existing auth, spend tracking, caching |
| Production without existing LiteLLM | `proxy` | Gets you auth, rate limiting, virtual keys out of the box |
| Containerized / Kubernetes | Docker `proxy` or `proxy-gpu` | Standard container with health checks |

---

## LiteLLM SDK Integration (No Server)

For applications already using the LiteLLM Python SDK, add routing with three lines of code — no separate server needed:

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

Per-request tolerance override:

```python
strategy.set_request_tolerance(0.10)  # tighter tolerance for this request
response = await router.acompletion(model="nem-think", messages=messages)
```

Access routing metadata after a call:

```python
if strategy.last_result:
    print(strategy.last_result.selected_model)
    print(strategy.last_result.confidences)
```

---

## Direct Python Library

Use the router directly without LiteLLM or a server. Returns routing decisions only — no LLM inference. Useful for building custom dispatchers or evaluating routing behavior.

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

No API keys required — the router runs the encoder locally and scores with the trained MLP checkpoint.
