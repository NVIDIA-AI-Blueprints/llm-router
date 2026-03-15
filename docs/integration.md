# Application Integration

The Model Router Toolkit provides seven integration paths. Each path trades off simplicity, control, and infrastructure requirements.

## Topology Overview

| Path | Topology | Routing | Inference | Extras needed | Config files |
|------|----------|---------|-----------|--------------|-------------|
| [LiteLLM SDK](#litellm-sdk) | Embedded | In-process | In-process (litellm) | `[litellm]` | 1 (pool) |
| [LiteLLM Proxy](#litellm-proxy) | Gateway | In-proxy | In-proxy (litellm) | `[proxy]` | 2 (pool + litellm) |
| [Standalone Server](#standalone-server) | Server | In-server | In-server (litellm) | `[litellm]` | 1 (pool) |
| [Router Sidecar](#router-sidecar) | Sidecar | Sidecar | External | `[server]` | 1 (pool) |
| [Webhook Integration](#webhook-integration) | Sidecar | Sidecar + auth | External | `[server]` | 1 (pool) |
| [OpenClaw Plugin](#openclaw-plugin) | Gateway Plugin | Sidecar | Gateway | `[server]` | 1 (pool) + plugin config |
| [Direct Python](#direct-python) | Embedded | In-process | None | *(core only)* | 1 (pool) |

## API Keys

Paths that perform LLM inference require a provider API key:

```bash
export OPENROUTER_API_KEY=sk-or-...   # for OpenRouter-based configs
# or
export NVIDIA_API_KEY=nvapi-...       # for NVIDIA NIM-based configs
```

**Not needed** for: Router Sidecar, Webhook Integration, OpenClaw Plugin, and Direct Python (these return routing decisions only).

---

## LiteLLM SDK

Embed routing directly in your Python application. No server, no extra process. Best when you already use `litellm.Router`.

```bash
pip install 'model-router-toolkit[litellm]'
```

```python
from litellm import Router
from model_router_toolkit import ModelRoutingStrategy

model_list = [
    {"model_name": "nem-think", "litellm_params": {"model": "openrouter/nvidia/nemotron-3-nano-30b-a3b"}},
    {"model_name": "nem-nothink", "litellm_params": {"model": "openrouter/nvidia/nemotron-3-nano-30b-a3b"}},
]

router = Router(model_list=model_list)
strategy = ModelRoutingStrategy.from_config("configs/v1-9models-qwen08b.yaml")
strategy.set_litellm_router(router)
router.set_custom_routing_strategy(strategy)

response = await router.acompletion(
    model="nem-think",
    messages=[{"role": "user", "content": "Prove sqrt(2) is irrational"}],
)
```

**Per-request tolerance override:**

```python
strategy.set_request_tolerance(0.10)  # tighter for this request
response = await router.acompletion(model="nem-think", messages=messages)
```

**Access routing metadata:**

```python
if strategy.last_result:
    print(strategy.last_result.selected_model)
    print(strategy.last_result.confidences)
```

**Source:** `adapters/litellm/strategy.py`

---

## LiteLLM Proxy

Starts the full **LiteLLM Proxy server** with the routing strategy injected at startup. Use this when you want LiteLLM's production features (auth, rate limiting, spend tracking, caching, virtual keys).

```bash
pip install 'model-router-toolkit[proxy]'
```

### Setup

Generate the LiteLLM proxy config from your pool config, then start:

```bash
model-router proxy-config \
    --config configs/v1-9models-qwen08b.yaml \
    --output configs/litellm-proxy.yaml

model-router proxy \
    --litellm-config configs/litellm-proxy.yaml \
    --router-config configs/v1-9models-qwen08b.yaml \
    --port 4000
```

Two config files are required:

| File | Controls |
|------|---------|
| LiteLLM proxy config (`litellm-proxy.yaml`) | `model_list`, provider endpoints, API keys, auth, caching |
| Pool config (`v1-9models-qwen08b.yaml`) | Routing method, checkpoint, tolerance, encoder, costs |

The `proxy-config` command bridges the two — reads pool config and generates a matching LiteLLM config.

### Connecting

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:4000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="nem-think",
    messages=[{"role": "user", "content": "Hello"}],
)
```

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "nem-think", "messages": [{"role": "user", "content": "Hello"}]}'
```

**Source:** `adapters/litellm/proxy.py`, `adapters/litellm/config_bridge.py`

---

## Standalone Server

A full FastAPI server with routing, inference, and a playground UI. Best for demos, development, and quick deployments.

```bash
pip install 'model-router-toolkit[litellm]'
model-router serve --config configs/v1-9models-qwen08b.yaml --port 8000
```

### Connecting

**OpenAI SDK:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="routed",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
```

**Environment variable** (works with any tool reading `OPENAI_API_BASE`):

```bash
export OPENAI_API_BASE=http://localhost:8000/v1
```

**cURL:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "routed", "messages": [{"role": "user", "content": "Hello"}]}'
```

**Playground UI:** Open `http://localhost:8000/` for routing cards, probability bars, tolerance slider, and model toggles.

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat (streaming + non-streaming) |
| `/api/chat` | POST | SSE chat endpoint for the playground UI |
| `/api/models` | GET | Model pool with cost data |
| `/api/config` | GET | Server config (routing method, features) |
| `/api/review` | POST | Auto-review: judges answer correctness |
| `/health` | GET | Health check |
| `/` | GET | Interactive playground UI |

**Source:** `adapters/litellm/app.py`, `adapters/litellm/completions.py`, `adapters/litellm/chat.py`, `adapters/litellm/review.py`

---

## Router Sidecar

A lightweight HTTP server that returns routing decisions **without performing LLM inference**. No litellm dependency — only FastAPI + uvicorn. Deploy alongside your existing inference stack.

```bash
pip install 'model-router-toolkit[server]'
model-router serve-router --config configs/v1-9models-qwen08b.yaml --port 8079
```

### Calling the route endpoint

```bash
curl -X POST http://localhost:8079/v1/route \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "tolerance": 0.20
  }'
```

Response:

```json
{
  "selected_model": "nem-nothink",
  "model_names": ["nem-think", "nem-nothink"],
  "confidences": {"nem-think": 0.92, "nem-nothink": 0.88},
  "costs": [
    {"model": "nem-think", "estimated_total_cost": 0.0004},
    {"model": "nem-nothink", "estimated_total_cost": 0.0001}
  ],
  "metadata": {"p_max": 0.92, "threshold": 0.72, "route_ms": 45.2}
}
```

You can also pass a plain question:

```bash
curl -X POST http://localhost:8079/v1/route \
  -d '{"question": "What is 2+2?", "tolerance": 0.15}'
```

#### Model-name bypass (pin mode)

To pin a specific model without ML inference, pass a `model` field matching a pool model name:

```bash
curl -X POST http://localhost:8079/v1/route \
  -d '{"model": "nem-think"}'
```

Response:

```json
{
  "selected_model": "nem-think",
  "model_names": ["nem-think", "nem-nothink"],
  "confidences": {"nem-think": 1.0, "nem-nothink": 0.0},
  "costs": [...],
  "metadata": {"pinned": true, "route_ms": 0.01}
}
```

Use this for **router-per-subagent** flows: route once normally, then pass the `selected_model` back as `model` on subsequent calls to lock the model for the subagent's lifetime.

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/route` | POST | Routing decision (no inference) |
| `/api/models` | GET | Model pool metadata |
| `/health` | GET | Health check |

**Source:** `adapters/http/app.py`, `adapters/http/route.py`

---

## Webhook Integration

The router sidecar supports HMAC-SHA256 and bearer token authentication for enterprise webhook pipelines (Portkey, TrueFoundry, Cloudflare, custom gateways).

```bash
export ROUTER_WEBHOOK_SECRET=my-shared-secret
model-router serve-router --config configs/v1-9models-qwen08b.yaml --port 8079
```

### Calling with HMAC

```python
import hashlib, hmac, json, requests

body = json.dumps({"question": "Explain quantum computing", "tolerance": 0.20})
signature = hmac.new(b"my-shared-secret", body.encode(), hashlib.sha256).hexdigest()

resp = requests.post(
    "http://localhost:8079/v1/route",
    data=body,
    headers={
        "Content-Type": "application/json",
        "X-Webhook-Signature": signature,
    },
)
```

### Calling with bearer token

```bash
curl -X POST http://localhost:8079/v1/route \
  -H "Authorization: Bearer my-shared-secret" \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain quantum computing"}'
```

When no secret is configured, all requests pass through (backward compatible).

**Source:** `adapters/http/auth.py`

---

## OpenClaw Plugin

TypeScript plugin for the OpenClaw gateway. Uses the `before_model_resolve` hook to call the router sidecar before each LLM request, overriding OpenClaw's model selection with the router's cost-aware decision.

### Setup

1. Start the router sidecar:

```bash
model-router serve-router --config configs/v1-9models-qwen08b.yaml --port 8079
```

2. Install the plugin in your OpenClaw configuration:

```json
{
  "plugins": {
    "entries": {
      "model-router": {
        "enabled": true,
        "config": {
          "sidecarUrl": "http://127.0.0.1:8079",
          "tolerance": 0.20,
          "enabled": true,
          "timeoutMs": 5000,
          "pool": [
            {"routerName": "nem-think", "provider": "openrouter", "model": "nvidia/nemotron-3-nano-30b-a3b"},
            {"routerName": "nem-nothink", "provider": "openrouter", "model": "nvidia/nemotron-3-nano-30b-a3b"}
          ]
        }
      }
    }
  }
}
```

The `pool` array maps router model names to OpenClaw provider/model references. When the router selects a model, the plugin translates that to an OpenClaw `modelOverride` + `providerOverride`.

**Graceful degradation:** If the sidecar is unreachable or returns an error, the plugin returns `{}` and OpenClaw uses its default model selection.

**Source:** `plugins/openclaw/index.ts`, `plugins/openclaw/openclaw.plugin.json`

---

## Direct Python

Use the router as a library — routing decisions only, no inference, no API keys needed.

```python
from model_router_toolkit.config import load_config, build_router_from_config

config = load_config("configs/v1-9models-qwen08b.yaml")
router = build_router_from_config(config)

result = router.route("What is the capital of France?", tolerance=0.20)
print(result.selected_model)       # cheapest model above threshold
print(result.confidences)          # P(correct) per model
print(result.selected_cost)        # estimated cost
print(result.metadata)             # routing metadata (p_max, threshold)
```

No server, no API keys — the router runs the encoder locally and scores with the trained MLP checkpoint.

### Pin mode (router-per-subagent)

```python
# Check if a model is in the pool
router.has_model("nem-think")  # True

# Pin a model without ML inference
pinned = router.resolve("nem-think")
print(pinned.selected_model)   # "nem-think"
print(pinned.metadata)         # {"pinned": True}
```

**Source:** `config.py` (`load_config`, `build_router_from_config`), `router.py` (`BaseRouter`)

---

## Decision Matrix

| Scenario | Recommended path | Why |
|----------|-----------------|-----|
| First time trying the toolkit | Standalone Server | Playground UI, single config, visual routing |
| Local development and debugging | Standalone Server | Playground shows routing decisions |
| Already running LiteLLM Proxy | LiteLLM Proxy | Drop-in — keeps auth, spend tracking, caching |
| Production without existing gateway | LiteLLM Proxy | Auth, rate limiting, virtual keys out of the box |
| Existing Python app with litellm | LiteLLM SDK | 4 lines, no extra server |
| API gateway (OpenClaw, Portkey) | Router Sidecar + Plugin | Route-only, no inference duplication |
| Enterprise webhook pipeline | Router Sidecar + Webhook Auth | HMAC or bearer token validation |
| Custom dispatcher, evaluation scripts | Direct Python | Routing decisions only |
| Air-gapped / no API keys | Direct Python + Prefill | Local encoder, no network calls |
