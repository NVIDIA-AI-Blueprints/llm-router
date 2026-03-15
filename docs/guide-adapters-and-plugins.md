# Adapters & Plugins Guide

This guide covers every adapter and plugin in the toolkit, their APIs, configuration, and how to write your own.

---

## Table of Contents

- [What Are Adapters?](#what-are-adapters)
- [LiteLLM Adapter](#litellm-adapter)
  - [ModelRoutingStrategy](#modelroutingstrategy)
  - [Standalone Server (app.py)](#standalone-server-apppy)
  - [LiteLLM Proxy Injection (proxy.py)](#litellm-proxy-injection-proxypy)
  - [Config Bridge (config_bridge.py)](#config-bridge-config_bridgepy)
- [HTTP Adapter](#http-adapter)
  - [Router Sidecar (app.py)](#router-sidecar-apppy)
  - [Route Endpoint (route.py)](#route-endpoint-routepy)
  - [Webhook Auth (auth.py)](#webhook-auth-authpy)
- [OpenClaw Plugin](#openclaw-plugin)
  - [How It Works](#how-it-works)
  - [Configuration](#configuration)
  - [Deployment](#deployment)
- [Writing a Custom Adapter](#writing-a-custom-adapter)
- [API Reference: /v1/route](#api-reference-v1route)

---

## What Are Adapters?

The routing engine (`BaseRouter` → `RoutingResult`) is a pure function: question in, model selection out. It has no knowledge of HTTP, LiteLLM, gateways, or any platform.

**Adapters** bridge the gap between the routing engine and specific platforms. Each adapter translates between a platform's interface and the `BaseRouter` API.

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Platform    │ ──► │     Adapter      │ ──► │  BaseRouter  │
│  (LiteLLM,  │     │  (translates     │     │  (pure       │
│   FastAPI,   │ ◄── │   request/       │ ◄── │   routing    │
│   Gateway)   │     │   response)      │     │   engine)    │
└──────────────┘     └──────────────────┘     └──────────────┘
```

The core package never imports adapter dependencies. Each adapter brings its own extras (`[litellm]`, `[server]`, `[proxy]`).

---

## LiteLLM Adapter

Location: `src/model_router_toolkit/adapters/litellm/`

Three integration patterns, all sharing the same `ModelRoutingStrategy`:

### ModelRoutingStrategy

**File**: `strategy.py`

The heart of the LiteLLM integration. Wraps any `BaseRouter` as a `CustomRoutingStrategyBase` that plugs into `litellm.Router`.

#### Construction

```python
from model_router_toolkit import ModelRoutingStrategy

# From config file (recommended)
strategy = ModelRoutingStrategy.from_config("configs/v1-9models-qwen08b.yaml")

# From an existing router
from model_router_toolkit.config import load_config, build_router_from_config
config = load_config("configs/v1-9models-qwen08b.yaml")
router = build_router_from_config(config)
strategy = ModelRoutingStrategy(router=router, tolerance=0.20)
```

#### Attaching to litellm.Router

```python
from litellm import Router

litellm_router = Router(model_list=my_models)
strategy.set_litellm_router(litellm_router)
litellm_router.set_custom_routing_strategy(strategy)
```

#### Key methods and properties

| Method/Property | Description |
|----------------|-------------|
| `from_config(config_path, **kwargs)` | Class method. Builds strategy from a pool config YAML. |
| `set_litellm_router(router)` | Sets the LiteLLM Router instance. Must be called before routing. |
| `async_get_available_deployment()` | LiteLLM callback. Called by litellm.Router on each request to pick a deployment. |
| `get_available_deployment()` | Sync version of the above. |
| `set_request_tolerance(value)` | Override tolerance for the current request (uses contextvars). |
| `effective_tolerance` | Returns per-request tolerance if set, otherwise default. |
| `last_result` | The most recent `RoutingResult`. Available after each routing call. |

#### Model pinning

The strategy supports pinning via `metadata.pin_model` in the request:

```python
response = await litellm_router.acompletion(
    model="any-pool-model",
    messages=[{"role": "user", "content": "..."}],
    metadata={"pin_model": "gpt-oss-120b-high"},
)
```

When `pin_model` is set and the model exists in the pool, `router.resolve()` is called instead of `router.route()` — instant, no ML inference.

#### Model filtering

Restrict routing to a subset of models for a single request:

```python
response = await litellm_router.acompletion(
    model="any-pool-model",
    messages=[{"role": "user", "content": "..."}],
    metadata={"models": ["nemotron-3-nano-reasoning", "gpt-oss-120b-high"]},
)
```

#### Error handling

If routing fails (exception in the encoder or MLP), the strategy falls back to the first deployment in the LiteLLM model list. This ensures requests never silently fail due to routing errors.

---

### Standalone Server (app.py)

**File**: `app.py`

A FastAPI application that combines routing, LiteLLM inference, and a web playground.

#### Creating the app

```python
from model_router_toolkit.adapters.litellm.app import create_app

app = create_app("configs/v1-9models-qwen08b.yaml", warmup=True, models=None)
```

Or via CLI:

```bash
model-router serve --config configs/v1-9models-qwen08b.yaml --port 8000
```

#### Internal architecture

```
create_app()
  │
  ├── Loads pool config
  ├── Creates ModelRoutingStrategy
  ├── Creates litellm.Router with model_list from config
  ├── Attaches strategy to litellm.Router
  ├── Mounts endpoint handlers:
  │   ├── /v1/chat/completions  (completions.py)
  │   ├── /api/chat             (chat.py)
  │   ├── /api/review           (review.py)
  │   ├── /health, /api/models, /api/config  (_shared.py)
  │   └── /                     (static files)
  └── Optionally warms up the router
```

#### API key resolution

The app resolves API keys from the `litellm_model` prefix:
- `openrouter/...` → `OPENROUTER_API_KEY`
- `nvidia_nim/...` → `NVIDIA_API_KEY`
- Other → `OPENAI_API_KEY`

---

### LiteLLM Proxy Injection (proxy.py)

**File**: `proxy.py`

Injects `ModelRoutingStrategy` into a running LiteLLM Proxy.

#### How it works

1. Validates that `litellm[proxy]` is installed (min version 1.50.0)
2. Sets `LITELLM_CONFIG_FILE_PATH` environment variable
3. Imports `litellm.proxy.proxy_server`
4. Creates `ModelRoutingStrategy` from the router config
5. Patches the proxy's internal `Router` with the strategy
6. Starts the proxy via uvicorn

#### Usage

```bash
model-router proxy \
  --litellm-config configs/litellm-proxy.yaml \
  --router-config configs/v1-9models-qwen08b.yaml \
  --host 0.0.0.0 \
  --port 4000
```

---

### Config Bridge (config_bridge.py)

**File**: `config_bridge.py`

Generates LiteLLM Proxy config from a pool config.

#### Functions

**`generate_litellm_config(pool_config, output=None)`**

Creates a LiteLLM-compatible config dict:
- Maps each `ModelSpec` to a LiteLLM model list entry
- Sets API keys as `os.environ/VARIABLE_NAME` references
- Sets `router_settings.routing_strategy: simple-shuffle` (the routing strategy is injected at runtime)

```python
from model_router_toolkit.adapters.litellm.config_bridge import generate_litellm_config
from model_router_toolkit.config import load_config

config = load_config("configs/v1-9models-qwen08b.yaml")
litellm_config = generate_litellm_config(config, output="configs/litellm-proxy.yaml")
```

**`validate_model_alignment(litellm_config_path, pool_config_path)`**

Checks for mismatches between the LiteLLM config and pool config:
- Models in pool but missing from LiteLLM config
- Model name mismatches
- Returns a list of warning strings

---

## HTTP Adapter

Location: `src/model_router_toolkit/adapters/http/`

Lightweight route-only sidecar. No LiteLLM dependency — only needs FastAPI and uvicorn.

### Router Sidecar (app.py)

**File**: `app.py`

A minimal FastAPI application that exposes routing decisions via HTTP.

#### Creating the app

```python
from model_router_toolkit.adapters.http.app import create_app

app = create_app("configs/v1-9models-qwen08b.yaml", warmup=True, models=None)
```

Or via CLI:

```bash
model-router serve-router --config configs/v1-9models-qwen08b.yaml --port 8079
```

#### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/route` | POST | Routing decision |
| `/health` | GET | Health check |
| `/api/models` | GET | List models |

#### Authentication

If `ROUTER_WEBHOOK_SECRET` is set, all endpoints (except `/health`) require authentication.

---

### Route Endpoint (route.py)

**File**: `route.py`

The core route endpoint handler.

#### Request format (RouteRequest)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `messages` | `list[dict]` | No | OpenAI-style messages (extracts last user message) |
| `question` | `str` | No | Plain question text (alternative to messages) |
| `model` | `str` | No | Model name to pin (bypasses routing) |
| `models` | `list[str]` | No | Restrict routing to these models |
| `tolerance` | `float` | No | Override default tolerance (default: 0.20) |

At least one of `messages`, `question`, or `model` must be provided.

#### Response format (RouteResponse)

| Field | Type | Description |
|-------|------|-------------|
| `selected_model` | `str` | The chosen model name |
| `model_names` | `list[str]` | All models in the pool |
| `confidences` | `list[float]` | P(correct) per model |
| `costs` | `list[dict]` | Cost estimates per model |
| `metadata` | `dict` | Additional info (e.g., `{"pinned": true}`) |

#### Routing logic

1. If `model` is present and the model exists in the pool → pin (return directly, no ML)
2. Otherwise, extract question text from `question` or `messages`
3. Call `router.route(question, tolerance=tolerance, models=models)`
4. Return the `RoutingResult` as JSON

---

### Webhook Auth (auth.py)

**File**: `auth.py`

ASGI middleware for authenticating requests from API gateways and webhooks.

#### Supported auth methods

**HMAC-SHA256 signature**:
- Header: `X-Webhook-Signature: sha256=<hex_digest>`
- The HMAC is computed over the raw request body using the shared secret

**Bearer token**:
- Header: `Authorization: Bearer <secret>`
- The token must exactly match the shared secret

#### Configuration

```bash
export ROUTER_WEBHOOK_SECRET=my-shared-secret
```

Or programmatically:

```python
from model_router_toolkit.adapters.http.auth import WebhookAuthMiddleware

app.add_middleware(WebhookAuthMiddleware, secret="my-shared-secret")
```

#### Exempt paths

`/health` is always exempt from authentication (for load balancer health checks).

---

## OpenClaw Plugin

Location: `src/model_router_toolkit/plugins/openclaw/`

A TypeScript plugin for the OpenClaw API gateway.

### How It Works

1. OpenClaw receives a chat completion request
2. The plugin's `before_model_resolve` hook fires
3. The plugin sends `POST /v1/route` to the router sidecar with the user's messages
4. The sidecar returns `{ selected_model: "..." }`
5. The plugin maps the router's model name to an OpenClaw provider/model via its `pool` config
6. Returns `{ modelOverride, providerOverride }` to OpenClaw
7. OpenClaw routes the request to the selected provider and model

On failure (sidecar unreachable, timeout, error), the plugin returns `{}` — OpenClaw falls back to its default model selection.

### Configuration

Plugin config in OpenClaw:

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
    },
    {
      "routerName": "claude-opus-4-6-high",
      "provider": "anthropic",
      "model": "claude-opus-4-6"
    }
  ]
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sidecarUrl` | string | (required) | URL of the router sidecar |
| `tolerance` | number | `0.20` | Routing tolerance |
| `enabled` | boolean | `true` | Enable/disable the plugin |
| `timeoutMs` | number | `5000` | HTTP timeout for sidecar calls |
| `pool` | array | (required) | Maps router names to OpenClaw providers |

The `pool` array is the critical mapping. Each entry maps a router model name (`routerName`) to an OpenClaw provider and model name. This decouples the router's naming from the gateway's naming.

### Deployment

1. Start the router sidecar:
   ```bash
   model-router serve-router --config configs/v1-9models-qwen08b.yaml --port 8079
   ```

2. Copy plugin files to OpenClaw's plugins directory:
   ```
   plugins/
     model-router/
       index.ts
       openclaw.plugin.json
       package.json
   ```

3. Configure the plugin in OpenClaw's settings with the sidecar URL and pool mapping.

4. The plugin calls `GET /health` on the sidecar at gateway startup to verify connectivity.

---

## Writing a Custom Adapter

If you need to integrate the router with a platform not covered by the built-in adapters, here's the pattern.

### Step 1: Use BaseRouter directly

Every adapter ultimately calls `BaseRouter.route()` or `BaseRouter.resolve()`:

```python
from model_router_toolkit.config import load_config, build_router_from_config
from model_router_toolkit.router import BaseRouter, RoutingResult, extract_user_text

config = load_config("configs/my-pool.yaml")
router = build_router_from_config(config)

# Route a question
result: RoutingResult = router.route("What is 2+2?", tolerance=0.20)

# Pin a model
result: RoutingResult = router.resolve("nemotron-3-nano-reasoning")

# Extract question from messages
question = extract_user_text([{"role": "user", "content": "What is 2+2?"}])
```

### Step 2: Map platform requests to route() calls

Your adapter needs to:
1. Extract the question text from the platform's request format
2. Determine if this is a routing request or a pin request
3. Call `router.route()` or `router.resolve()` accordingly
4. Map the `RoutingResult` back to the platform's response format

### Step 3: Handle errors

Always handle routing errors gracefully. If the encoder or MLP fails, fall back to a sensible default (e.g., the cheapest model, or the most expensive one for safety).

### Example: Custom webhook adapter

```python
from fastapi import FastAPI, Request
from model_router_toolkit.config import load_config, build_router_from_config
from model_router_toolkit.router import extract_user_text

app = FastAPI()
config = load_config("configs/my-pool.yaml")
router = build_router_from_config(config)

@app.post("/webhook/route")
async def webhook_route(request: Request):
    body = await request.json()

    messages = body.get("messages", [])
    question = extract_user_text(messages)

    try:
        result = router.route(question, tolerance=body.get("tolerance", 0.20))
        return {
            "model": result.selected_model,
            "confidence": result.selected_confidence,
        }
    except Exception:
        return {"model": config.models[0].name, "confidence": 0.0}
```

---

## API Reference: /v1/route

The `/v1/route` endpoint is the standard routing API, used by the HTTP sidecar and consumed by the OpenClaw plugin. Any custom integration should target this API.

### Request

```
POST /v1/route
Content-Type: application/json
```

**Body**:

```json
{
  "messages": [{"role": "user", "content": "What is 2+2?"}],
  "question": "What is 2+2?",
  "model": "nemotron-3-nano-reasoning",
  "models": ["nemotron-3-nano-reasoning", "gpt-oss-120b-high"],
  "tolerance": 0.20
}
```

All fields are optional, but at least one of `messages`, `question`, or `model` must be provided.

**Priority**:
1. If `model` is set and exists in the pool → pin (no ML inference)
2. If `question` is set → use it directly
3. If `messages` is set → extract last user message

### Response

```json
{
  "selected_model": "nemotron-3-nano-reasoning",
  "model_names": ["nemotron-3-nano-reasoning", "gpt-oss-20b-high", "..."],
  "confidences": [0.92, 0.89, "..."],
  "costs": [
    {
      "median_output_tokens": 150,
      "cost_per_m_input_tokens": 0.05,
      "cost_per_m_output_tokens": 0.20,
      "estimated_input_tokens": 0,
      "estimated_output_cost": 0.0,
      "estimated_input_cost": 0.0,
      "estimated_total_cost": 0.0
    }
  ],
  "metadata": {}
}
```

### Error responses

| Status | When |
|--------|------|
| 400 | No question, messages, or model provided |
| 404 | Pinned model not found in pool |
| 500 | Internal routing error |

### Authentication

When `ROUTER_WEBHOOK_SECRET` is set:

```bash
# HMAC-SHA256
curl -X POST http://localhost:8079/v1/route \
  -H "Content-Type: application/json" \
  -H "X-Webhook-Signature: sha256=$(echo -n '{"question":"What is 2+2?"}' | openssl dgst -sha256 -hmac 'my-secret' | cut -d' ' -f2)" \
  -d '{"question": "What is 2+2?"}'

# Bearer token
curl -X POST http://localhost:8079/v1/route \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-secret" \
  -d '{"question": "What is 2+2?"}'
```
