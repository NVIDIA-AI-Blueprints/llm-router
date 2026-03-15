# Serving & Deployment Guide

This guide covers every way to deploy the router — from a single Python function call to a production LiteLLM Proxy. Each deployment mode serves a different use case; pick the one that fits your infrastructure.

---

## Table of Contents

- [Deployment Modes at a Glance](#deployment-modes-at-a-glance)
- [Direct Python (No Server)](#direct-python-no-server)
- [Standalone Server](#standalone-server)
- [Router-Only Sidecar](#router-only-sidecar)
- [LiteLLM Proxy](#litellm-proxy)
- [LiteLLM SDK Embedding](#litellm-sdk-embedding)
- [OpenClaw Gateway Plugin](#openclaw-gateway-plugin)
- [Choosing a Deployment Mode](#choosing-a-deployment-mode)
- [Production Considerations](#production-considerations)

---

## Deployment Modes at a Glance

| Mode | What it does | Dependencies | API Keys Needed | Inference |
|------|-------------|-------------|-----------------|-----------|
| **Direct Python** | Route in-process, you handle inference | `[prefill]` | No | You handle it |
| **Standalone Server** | Full server: route + infer + playground | `[prefill,litellm]` | Yes | Server handles it |
| **Router Sidecar** | Route-only HTTP API | `[prefill,server]` | No | You handle it |
| **LiteLLM Proxy** | Drop into existing LiteLLM Proxy | `[prefill,proxy]` | Yes | Proxy handles it |
| **LiteLLM SDK** | Embed in any litellm.Router | `[prefill,litellm]` | Yes | litellm handles it |
| **OpenClaw Plugin** | Gateway plugin (TypeScript) | Sidecar running | No (for plugin) | Gateway handles it |

---

## Direct Python (No Server)

Use the router as a Python library. No server, no API keys. You get a model name back and handle inference yourself.

### Setup

```bash
git lfs install && git lfs pull   # fetch checkpoint files (required once after clone)
pip install -e '.[prefill]'
```

### Basic usage

```python
from model_router_toolkit.config import load_config, build_router_from_config

config = load_config("configs/v1-9models-qwen08b.yaml")
router = build_router_from_config(config)

result = router.route("What is the capital of France?", tolerance=0.20)

print(result.selected_model)        # "nemotron-3-nano-reasoning"
print(result.selected_confidence)   # 0.92
print(result.model_names)           # ["nemotron-3-nano-reasoning", "gpt-oss-20b-high", ...]
print(result.confidences)           # [0.92, 0.89, ...]
print(result.selected_cost)         # CostEstimate(median_output_tokens=150, ...)
```

### Restricting models

```python
result = router.route(
    "Explain quantum entanglement",
    tolerance=0.15,
    models=["nemotron-3-nano-reasoning", "gpt-oss-120b-high", "claude-opus-4-6-high"],
)
```

Only the specified models are considered for selection. Confidences for all models are still computed.

### Model pinning

For multi-turn conversations where you want the same model throughout:

```python
# First turn: route normally
result = router.route("Explain quantum computing")
model = result.selected_model  # "gpt-oss-120b-high"

# Subsequent turns: pin to the same model (instant, no encoder run)
result = router.resolve(model)
print(result.metadata)  # {"pinned": True}
```

### Cleanup

```python
router.unload()  # frees encoder memory
```

### The RoutingResult object

| Field | Type | Description |
|-------|------|-------------|
| `selected_model` | `str` | The chosen model name |
| `model_names` | `list[str]` | All models in the pool |
| `confidences` | `dict[str, float]` | P(correct) per model (model name → probability) |
| `costs` | `list[CostEstimate]` | Cost estimates for each model |
| `metadata` | `dict` | Additional info (e.g., `pinned`, `raw_scores`) |
| `selected_confidence` | `float` | P(correct) for the selected model (property) |
| `selected_cost` | `CostEstimate` | Cost estimate for the selected model (property) |

### The CostEstimate object

| Field | Type | Description |
|-------|------|-------------|
| `median_output_tokens` | `int` | Median output tokens (from training data) |
| `cost_per_m_input_tokens` | `float` | Input cost per million tokens |
| `cost_per_m_output_tokens` | `float` | Output cost per million tokens |
| `estimated_input_tokens` | `int` | Estimated input tokens for this query |
| `estimated_output_cost` | `float` | Estimated output cost |
| `estimated_input_cost` | `float` | Estimated input cost |
| `estimated_total_cost` | `float` | Estimated total cost |

### When to use

- You have your own inference pipeline (custom HTTP client, internal API, etc.)
- You want routing decisions without any server overhead
- Air-gapped environments with no network access
- Embedding routing in a larger Python application

---

## Standalone Server

Full-featured server with routing, inference (via LiteLLM), and an interactive playground UI.

### Setup

```bash
pip install -e '.[prefill,litellm]'
export OPENROUTER_API_KEY=your-key
```

### Start the server

```bash
model-router serve --config configs/v1-9models-qwen08b.yaml --port 8000
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | (required) | Pool config YAML |
| `--port` | 8000 | Server port |
| `--models` | all | Comma-separated model subset |

### Endpoints

#### `GET /` — Playground UI

Interactive web UI for testing routing. Features: chat interface, tolerance slider, model toggles, routing visualization, cost tracking, auto-review.

#### `GET /health` — Health check

```json
{"status": "ok", "mode": "full", "method": "prefill", "models": ["nem-think", "nem-nano", "..."]}
```

#### `GET /api/models` — List models

```json
[
  {"name": "nemotron-3-nano-reasoning", "display_name": "Nemotron 3 Nano (Reasoning)"},
  {"name": "gpt-oss-20b-high", "display_name": "GPT-OSS 20B High"},
  ...
]
```

#### `GET /api/config` — Current config

Returns the pool configuration as JSON.

#### `POST /v1/chat/completions` — OpenAI-compatible chat

Standard OpenAI chat completions API. The router selects the model; the server calls it via LiteLLM and returns the response.

**Request**:

```json
{
  "model": "routed",
  "messages": [{"role": "user", "content": "What is 2+2?"}],
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 1024
}
```

The `model` field is ignored for routing (any value works). Optional fields `tolerance` and `models` can override routing behavior per request.

**Response**: Standard OpenAI chat completion response with an additional `routing` field:

```json
{
  "id": "chatcmpl-...",
  "choices": [{"message": {"role": "assistant", "content": "4"}}],
  "routing": {
    "selected_model": "nemotron-3-nano-reasoning",
    "confidences": {"nemotron-3-nano-reasoning": 0.92, ...},
    "metadata": {}
  }
}
```

#### `POST /api/chat` — SSE streaming (playground)

Used by the playground UI. Returns Server-Sent Events:

| Event | Data | When |
|-------|------|------|
| `routing` | Model selection, confidences | Before inference starts |
| `token` | Streamed token | During inference |
| `done` | Final stats | After inference completes |
| `error` | Error message | On failure |

#### `POST /api/review` — Auto-judge

Sends the question and answer to the most expensive model in the pool for correctness judgment. Returns SSE events with the verdict and optional comparisons with other models.

### When to use

- Demos and local development
- Small teams wanting a quick, self-contained deployment
- Testing routing quality interactively via the playground

---

## Router-Only Sidecar

Returns routing decisions via HTTP. No inference, no API keys, minimal dependencies.

### Setup

```bash
pip install -e '.[prefill,server]'
```

### Start the sidecar

```bash
model-router serve-router --config configs/v1-9models-qwen08b.yaml --port 8079
```

### Endpoints

#### `POST /v1/route` — Get routing decision

**Request** (question text):

```json
{"question": "What is 2+2?", "tolerance": 0.20}
```

**Request** (OpenAI messages):

```json
{"messages": [{"role": "user", "content": "What is 2+2?"}]}
```

**Request** (model pin):

```json
{"model": "nemotron-3-nano-reasoning"}
```

**Request** (with model filter):

```json
{
  "question": "What is 2+2?",
  "tolerance": 0.15,
  "models": ["nemotron-3-nano-reasoning", "gpt-oss-120b-high"]
}
```

**Response**:

```json
{
  "selected_model": "nemotron-3-nano-reasoning",
  "model_names": ["nemotron-3-nano-reasoning", "gpt-oss-20b-high", "..."],
  "confidences": {"nemotron-3-nano-reasoning": 0.92, "gpt-oss-20b-high": 0.89, "...": 0.0},
  "costs": [{"median_output_tokens": 150, "cost_per_m_input_tokens": 0.05, "...": 0}],
  "metadata": {}
}
```

#### `GET /health` — Health check

```json
{"status": "ok", "mode": "router-only", "method": "prefill", "models": ["nem-think", "nem-nano", "..."]}
```

### Authentication

The sidecar supports webhook-style auth via the `ROUTER_WEBHOOK_SECRET` environment variable:

```bash
export ROUTER_WEBHOOK_SECRET=my-shared-secret
model-router serve-router --config configs/v1-9models-qwen08b.yaml --port 8079
```

Clients authenticate with either:

**HMAC-SHA256 signature**:
```bash
curl -X POST http://localhost:8079/v1/route \
  -H "Content-Type: application/json" \
  -H "X-Webhook-Signature: sha256=<hmac_hex>" \
  -d '{"question": "What is 2+2?"}'
```

**Bearer token**:
```bash
curl -X POST http://localhost:8079/v1/route \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-shared-secret" \
  -d '{"question": "What is 2+2?"}'
```

The `/health` endpoint is exempt from auth.

> **Interpreting Confidence Values**
>
> Confidence values in routing responses are *relative rankings*, not calibrated probabilities.
> The routing algorithm selects the cheapest model whose score is within `tolerance` of the best.
> A confidence of 0.05 vs 0.02 is meaningful — it means the first model is more likely correct
> — even though both numbers look "low" in absolute terms. Use them to compare models within a
> single request, not as absolute accuracy estimates.

### When to use

- Gateway integrations (OpenClaw, Portkey, custom)
- Microservice architectures where routing and inference are separate services
- When you don't want litellm as a dependency
- When the inference layer already exists and you just need model selection

---

## LiteLLM Proxy

Inject intelligent routing into an existing LiteLLM Proxy deployment. You get all of LiteLLM's features (auth, rate limiting, spend tracking, caching) plus routing.

### Setup

```bash
pip install -e '.[prefill,proxy]'
```

### Step 1: Generate LiteLLM config

```bash
model-router proxy-config \
  --config configs/v1-9models-qwen08b.yaml \
  --output configs/litellm-proxy.yaml
```

This generates a LiteLLM-compatible YAML from your pool config, mapping model names to provider endpoints and setting API keys from environment variables.

### Step 2: Validate alignment

The `proxy-config` command prints any mismatches between the pool config and the generated LiteLLM config. Common issues:
- Models in the pool config missing from LiteLLM config
- API key environment variables not set

### Step 3: Start the proxy

```bash
model-router proxy \
  --litellm-config configs/litellm-proxy.yaml \
  --router-config configs/v1-9models-qwen08b.yaml \
  --host 0.0.0.0 \
  --port 4000
```

The proxy starts as a standard LiteLLM Proxy with the routing strategy injected. All normal LiteLLM Proxy features work (auth, rate limiting, spend tracking, etc.).

### Using the proxy

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-litellm-key" \
  -d '{"model": "nemotron-3-nano-reasoning", "messages": [{"role": "user", "content": "Hello"}]}'
```

The `model` field in the request triggers routing. The proxy intercepts the request, runs the router, and forwards to the selected model.

### When to use

- Production deployments needing auth, rate limiting, and spend tracking
- Teams already using LiteLLM Proxy
- When you want a drop-in upgrade to an existing proxy

---

## LiteLLM SDK Embedding

Embed routing directly in any Python app that uses `litellm.Router`.

### Setup

```bash
pip install -e '.[prefill,litellm]'
```

### Usage

```python
from litellm import Router
from model_router_toolkit import ModelRoutingStrategy

# Standard LiteLLM model list
model_list = [
    {
        "model_name": "nemotron-3-nano-reasoning",
        "litellm_params": {
            "model": "openrouter/nvidia/nemotron-3-nano-30b-a3b",
            "api_key": "sk-or-...",
        },
    },
    {
        "model_name": "gpt-oss-120b-high",
        "litellm_params": {
            "model": "openrouter/openai/gpt-oss-120b",
            "api_key": "sk-or-...",
        },
    },
]

# Create LiteLLM router
litellm_router = Router(model_list=model_list)

# Create and attach routing strategy
strategy = ModelRoutingStrategy.from_config("configs/v1-9models-qwen08b.yaml")
strategy.set_litellm_router(litellm_router)
litellm_router.set_custom_routing_strategy(strategy)

# Every call is now automatically routed
response = await litellm_router.acompletion(
    model="nemotron-3-nano-reasoning",
    messages=[{"role": "user", "content": "What is quantum computing?"}],
)

# Access routing result
print(strategy.last_result.selected_model)
print(strategy.last_result.confidences)
```

### Per-request tolerance

```python
# Override tolerance for a single request
strategy.set_request_tolerance(0.05)  # strict for this request
response = await litellm_router.acompletion(...)
```

### Model pinning

```python
response = await litellm_router.acompletion(
    model="nemotron-3-nano-reasoning",
    messages=[{"role": "user", "content": "Follow-up question"}],
    metadata={"pin_model": "gpt-oss-120b-high"},  # bypass routing
)
```

### When to use

- Existing Python apps already using litellm
- Minimal integration effort (4 lines of code)
- When you want routing + inference in the same process

---

## OpenClaw Gateway Plugin

TypeScript plugin for the OpenClaw API gateway. Calls the router sidecar and overrides model selection.

### Prerequisites

- Router sidecar running (see [Router-Only Sidecar](#router-only-sidecar))
- OpenClaw gateway

### Setup

1. Copy the plugin files from `src/model_router_toolkit/plugins/openclaw/` to your OpenClaw plugins directory.
2. Configure the plugin in OpenClaw's config:

```json
{
  "sidecarUrl": "http://localhost:8079",
  "tolerance": 0.20,
  "enabled": true,
  "timeoutMs": 5000,
  "pool": [
    {
      "routerName": "nemotron-3-nano-reasoning",
      "provider": "nvidia",
      "model": "nemotron-3-nano-30b-a3b"
    },
    {
      "routerName": "gpt-oss-120b-high",
      "provider": "openai",
      "model": "gpt-oss-120b"
    }
  ]
}
```

The `pool` array maps router model names to OpenClaw's provider/model pairs.

### How it works

1. OpenClaw receives a chat request
2. The plugin's `before_model_resolve` hook fires
3. The plugin calls `POST /v1/route` on the sidecar
4. The sidecar returns the selected model
5. The plugin maps the router's model name to an OpenClaw provider/model via the `pool` config
6. OpenClaw routes the request to the selected provider

If the sidecar is unreachable or returns an error, the plugin returns an empty override and OpenClaw falls back to its default model selection.

### When to use

- Teams using OpenClaw as their API gateway
- When you want routing decisions at the gateway level

---

## Choosing a Deployment Mode

| Scenario | Recommended Mode | Why |
|----------|-----------------|-----|
| Local development / demos | Standalone Server | Playground UI, single command, fast iteration |
| Existing LiteLLM stack | LiteLLM Proxy | Drop-in; keeps auth, rate limiting, spend tracking |
| Production without LiteLLM | LiteLLM Proxy | Best out-of-box production features |
| Gateway integration | Router Sidecar + Plugin | Route-only, no inference duplication |
| Existing Python app | LiteLLM SDK or Direct Python | Minimal integration; no server needed |
| Custom dispatcher | Direct Python | Full control; routing decisions only |
| Air-gapped / no API keys | Direct Python | Local encoder, no network calls |
| Multi-service architecture | Router Sidecar | Decoupled routing service |

---

## Production Considerations

### Latency

The encoder forward pass is the latency bottleneck:

| Encoder | CPU | GPU |
|---------|-----|-----|
| Qwen3.5-0.8B | ~5s | ~100ms |
| Qwen3.5-35B-A3B | N/A | ~200ms |

For production, GPU is strongly recommended. The 100ms routing overhead is negligible compared to typical LLM inference times (1–30s).

### Memory

The encoder stays loaded in memory after the first request:

| Encoder | RAM (CPU, fp32) | VRAM (GPU, fp16) |
|---------|-----------------|------------------|
| Qwen3.5-0.8B | ~3.2 GB | ~1.6 GB |

Plan your container/instance sizing accordingly.

### Scaling

- **Standalone Server / Sidecar**: Single process. For horizontal scaling, run multiple instances behind a load balancer. Each instance loads its own encoder.
- **LiteLLM Proxy**: LiteLLM's built-in scaling applies. The routing strategy runs in-process.
- **Direct Python**: Scale with your application's scaling strategy.

### Health checks

All server modes expose `/health`:
- Returns HTTP 200 with routing config info when healthy
- Use as a liveness/readiness probe in Kubernetes

### CORS

Set `CORS_ORIGINS` for browser clients:

```bash
export CORS_ORIGINS="https://myapp.example.com,http://localhost:3000"
```

Default is `*` (allow all origins).

### Warm-up

Both server modes warm up the router on startup (a dummy route call to load the encoder). The server responds to `/health` immediately but the first real route call may be slightly slower while the encoder's CUDA kernels compile (GPU only).
