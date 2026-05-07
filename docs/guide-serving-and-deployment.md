# Serving & Deployment Guide

This guide covers every way to deploy the router — from a single Python function call to a production LiteLLM Proxy. Each deployment mode serves a different use case; pick the one that fits your infrastructure.

---

## Table of Contents

- [Deployment Modes at a Glance](#deployment-modes-at-a-glance)
- [Direct Python (No Server)](#direct-python-no-server)
- [Standalone Server](#standalone-server)
- [Router-Only Sidecar](#router-only-sidecar)
- [LiteLLM Proxy](#litellm-proxy)
- [LiteLLM Proxy + External Sidecar Hook](#litellm-proxy--external-sidecar-hook)
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
| **LiteLLM Proxy** | Drop routing into LiteLLM Proxy (in-process encoder) | `[prefill,proxy]` | Yes | Proxy handles it |
| **Proxy + External Sidecar** | LiteLLM Proxy delegates routing to a separate Router Sidecar via callback | proxy: core; sidecar: `[prefill,server]` | Yes | Proxy handles it |
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

## LiteLLM Proxy + External Sidecar Hook

Same client-facing behavior as the in-process LiteLLM Proxy mode, but the encoder runs in a separate Router Sidecar process. The proxy stays vanilla — it doesn't install `[prefill]`, doesn't load torch, and doesn't need GPU memory. The router can be scaled, restarted, and GPU-pinned independently of the proxy.

### When to choose this over the in-process proxy

| Concern | In-process Proxy | Proxy + External Sidecar |
|---------|------------------|--------------------------|
| Encoder lives in | The proxy process | A dedicated sidecar process |
| Proxy install size | ~5 GB (torch + transformers + encoder weights) | <100 MB (vanilla LiteLLM + the hook module) |
| Independent scaling | No — proxy and router scale together | Yes — many proxies can share one router |
| GPU pinning | Proxy host needs the GPU | Only the sidecar host needs the GPU |
| Per-request overhead | None (in-process call) | One HTTP roundtrip (typically 1–10 ms LAN) |
| Failure isolation | Router OOM crashes the proxy | Sidecar can fail; hook fails-open or trips a circuit breaker |

### Setup

```bash
# Sidecar host: full prefill stack
pip install -e '.[prefill,server]'

# Proxy host: core only — gets the hook module without the encoder deps
pip install -e '.'
pip install 'litellm[proxy]'
```

### Step 1: Start the Router Sidecar

```bash
model-router serve-router \
  --config configs/v1-9models-qwen08b.yaml \
  --port 8079
```

The sidecar exposes `POST /v1/route` and `GET /health`. See [Router-Only Sidecar](#router-only-sidecar) for the full reference.

### Step 2: Configure the proxy

Create the LiteLLM Proxy `config.yaml`. The `model_name` entries must match the `name:` field of every model in the router's pool config — that's how the rewritten model gets resolved by LiteLLM.

```yaml
model_list:
  - model_name: nemotron-3-nano-reasoning
    litellm_params:
      model: openrouter/nvidia/nemotron-3-nano-30b-a3b
  - model_name: gpt-oss-120b-high
    litellm_params:
      model: openrouter/openai/gpt-oss-120b
  - model_name: claude-opus-4-6-high
    litellm_params:
      model: openrouter/anthropic/claude-opus-4-6
  # ... one entry per pool model

litellm_settings:
  callbacks: model_router_toolkit.adapters.litellm.external_hook.external_router_hook
  fallbacks:
    - gpt-oss-120b-high: [nemotron-3-super]
  default_fallbacks: [gpt-oss-120b-high]
  num_retries: 2
  request_timeout: 30
```

### Step 3: Set environment and start

```bash
export ROUTER_SIDECAR_URL=http://router-sidecar:8079
export ROUTER_SIDECAR_DEFAULT_MODEL=gpt-oss-120b-high   # fail-open target
export ROUTER_SIDECAR_TOLERANCE=0.20

litellm --config config.yaml --port 4000
```

The proxy's `external_router_hook` callback is built from `ROUTER_SIDECAR_URL` at import time. If the variable is unset the singleton is `None` and LiteLLM's callback loader will reject it.

### Using the proxy

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-your-litellm-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "any-pool-model", "messages": [{"role": "user", "content": "Hello"}]}'
```

The hook intercepts each request, calls the sidecar for a routing decision, and rewrites `data["model"]` before LiteLLM dispatches. From the client's perspective the proxy works exactly like a normal LiteLLM Proxy.

### Failure handling

The hook composes with LiteLLM's own fallback chain:

1. **Sidecar unreachable / timeout / non-2xx**: hook logs a warning and either rewrites to `ROUTER_SIDECAR_DEFAULT_MODEL` (if set) or re-raises the error.
2. **Repeated failures**: an in-memory circuit breaker trips after `ROUTER_SIDECAR_FAILURES_BEFORE_OPEN` consecutive failures and skips the sidecar entirely for `ROUTER_SIDECAR_OPEN_DURATION_S` seconds, going straight to the default. This prevents the proxy from paying the timeout on every request when the sidecar is flapping.
3. **Upstream provider failure**: LiteLLM's `fallbacks` / `default_fallbacks` / `num_retries` apply to the dispatched call as usual — independent of the routing layer.

For the full env-var reference, see [Adapters & Plugins → External Sidecar Hook](guide-adapters-and-plugins.md#external-sidecar-hook-external_hookpy).

### When to use

- Production deployments where the router and proxy have different scaling profiles
- Multi-proxy fleets sharing a single router
- Environments where you want to avoid pulling torch / transformers into the proxy host
- Cases where you want to update or roll back the router independently of proxy releases

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

TypeScript plugin for the OpenClaw API gateway. Calls the router sidecar's `POST /v1/route` endpoint before each LLM call and overrides model selection based on the routing decision.

### Quick Start

```bash
# 1. Start the sidecar (in one terminal)
model-router serve-router --config configs/v1-9models-qwen08b.yaml --port 8079

# 2. Install the plugin
mkdir -p ~/.openclaw/extensions/model-router
cp src/model_router_toolkit/plugins/openclaw/* ~/.openclaw/extensions/model-router/

# 3. Add plugin config to ~/.openclaw/openclaw.json (see below)

# 4. Restart the gateway
openclaw gateway restart

# 5. Verify — model should be the router's choice, not the default
openclaw agent --agent main --message "What is 2+2?" --json
```

Add to `~/.openclaw/openclaw.json`:

```json5
{
  "plugins": {
    "allow": ["model-router"],
    "entries": {
      "model-router": {
        "enabled": true,
        "config": {
          "sidecarUrl": "http://127.0.0.1:8079",
          "tolerance": 0.20,
          "enabled": true,
          "timeoutMs": 15000,
          "pool": [
            { "routerName": "nemotron-3-nano-reasoning", "provider": "openrouter", "model": "nvidia/nemotron-3-nano-30b-a3b" },
            { "routerName": "nemotron-3-super", "provider": "openrouter", "model": "nvidia/nemotron-3-super-120b-a12b:free" }
          ]
        }
      }
    }
  }
}
```

The `pool` maps router model names (from your pool config YAML) to OpenClaw provider/model pairs. Adjust `provider` and `model` to match your OpenClaw setup (OpenRouter, Ollama, etc.).

### Prerequisites

- OpenClaw 2026.2+ installed (requires Node.js 22+)
- Router sidecar running (see [Router-Only Sidecar](#router-only-sidecar))

### How it works

1. OpenClaw receives a chat request
2. The plugin's `before_model_resolve` hook fires with `event.prompt`
3. The plugin calls `POST /v1/route` on the sidecar with the prompt text
4. The sidecar returns `{ selected_model: "nemotron-3-nano-reasoning", ... }`
5. The plugin maps the router's model name to an OpenClaw provider/model via the `pool` config
6. Returns `{ modelOverride, providerOverride }` — OpenClaw uses the overridden model

If the sidecar is unreachable or returns an error, the plugin returns `{}` and OpenClaw falls back to its default model selection.

### When to use

- Teams using OpenClaw as their API gateway
- When you want routing decisions at the gateway level

For full configuration reference and troubleshooting, see the [OpenClaw Plugin](guide-adapters-and-plugins.md#openclaw-plugin) section in the Adapters & Plugins guide.

---

## Choosing a Deployment Mode

| Scenario | Recommended Mode | Why |
|----------|-----------------|-----|
| Local development / demos | Standalone Server | Playground UI, single command, fast iteration |
| Existing LiteLLM stack, single proxy | LiteLLM Proxy (in-process) | Drop-in; keeps auth, rate limiting, spend tracking |
| LiteLLM stack, fleet of proxies sharing a router | Proxy + External Sidecar | One router, many proxies; independent scaling and GPU pinning |
| Production without LiteLLM | LiteLLM Proxy (in-process) | Best out-of-box production features |
| Proxy host should stay torch-free | Proxy + External Sidecar | Encoder lives in the sidecar; proxy install stays small |
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
