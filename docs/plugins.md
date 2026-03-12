# Plugins Guide

Plugins integrate model-router-toolkit into API gateways and platforms that have their own model selection hooks. Unlike adapters (which run inside the toolkit's process), plugins run inside the **host platform's process** and call the toolkit's HTTP sidecar for routing decisions.

```
┌─────────────────┐     POST /v1/route     ┌──────────────────────┐
│  Host Platform   │ ────────────────────>  │  Router Sidecar      │
│  (OpenClaw, etc.)│ <────────────────────  │  (model-router       │
│  + Plugin        │     RouteResponse      │   serve-router)      │
└─────────────────┘                         └──────────────────────┘
```

## Part 1: OpenClaw Plugin

### Overview

The OpenClaw plugin hooks into `before_model_resolve` — a lifecycle event that fires before each LLM request. It calls the router sidecar, gets a routing decision, and overrides OpenClaw's model/provider selection.

### Prerequisites

1. A running model-router sidecar:

```bash
pip install 'model-router-toolkit[server,prefill]'
model-router serve-router --config configs/prefill-qwen08b.yaml --port 8079
```

2. OpenClaw gateway with plugin support

### Install

Copy the plugin directory into your OpenClaw plugins folder:

```bash
cp -r src/model_router_toolkit/plugins/openclaw/ /path/to/openclaw/plugins/model-router/
```

Or reference it from your OpenClaw config.

### Configure

Add to your OpenClaw configuration:

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
            {
              "routerName": "nem-think",
              "provider": "openrouter",
              "model": "nvidia/nemotron-3-nano-30b-a3b"
            },
            {
              "routerName": "nem-nothink",
              "provider": "openrouter",
              "model": "nvidia/nemotron-3-nano-30b-a3b"
            }
          ]
        }
      }
    }
  }
}
```

#### Config Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sidecarUrl` | `string` | `"http://127.0.0.1:8079"` | URL of the model-router sidecar |
| `tolerance` | `number` | `0.20` | Accuracy-cost tradeoff [0.0–1.0] |
| `enabled` | `boolean` | `true` | Enable/disable routing (disabled = OpenClaw default) |
| `timeoutMs` | `number` | `5000` | Sidecar request timeout in ms |
| `pool` | `PoolEntry[]` | `[]` | Maps router model names to OpenClaw provider/model |

#### Pool Mapping

The `pool` array bridges the router's model names to OpenClaw's provider/model references:

| Pool Field | Description | Example |
|-----------|-------------|---------|
| `routerName` | Model name in the router's pool config | `"nem-think"` |
| `provider` | OpenClaw provider ID | `"openrouter"`, `"anthropic"`, `"ollama"` |
| `model` | OpenClaw model ID within that provider | `"nvidia/nemotron-3-nano-30b-a3b"` |

### Architecture

```
User Request
    │
    ▼
OpenClaw Gateway
    │
    ├── before_model_resolve hook fires
    │       │
    │       ▼
    │   Model Router Plugin
    │       │
    │       ├── POST /v1/route to sidecar
    │       │       │
    │       │       ▼
    │       │   Router Sidecar
    │       │   (BaseRouter.route())
    │       │       │
    │       │       ▼
    │       │   { selected_model: "nem-nothink" }
    │       │
    │       ▼
    │   Map "nem-nothink" → { provider: "openrouter", model: "..." }
    │   Return { modelOverride, providerOverride }
    │
    ▼
OpenClaw dispatches to overridden model
```

### Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Plugin returns `{}`, no routing | Sidecar not running | Start: `model-router serve-router --config pool.yaml --port 8079` |
| Plugin returns `{}`, sidecar is running | `routerName` mismatch | Ensure pool `routerName` values match names in pool config |
| Timeout errors | Encoder warmup on first request | Increase `timeoutMs` to 30000 for first request, or pre-warm sidecar |
| All requests go to same model | `tolerance` too high or too low | Try `0.20` (default). Check sidecar logs for confidence values. |
| 401 from sidecar | Webhook auth enabled | Set `ROUTER_WEBHOOK_SECRET` or add auth headers |

### Plugin Files

| File | Purpose |
|------|---------|
| `index.ts` | Plugin entry point — registers `before_model_resolve` and `gateway_start` hooks |
| `openclaw.plugin.json` | Plugin manifest with ID, name, description, config schema |
| `package.json` | NPM package metadata |

---

## Part 2: Writing Plugins for Other Platforms

Any API gateway or LLM framework with a pre-request hook can integrate with model-router-toolkit via the `/v1/route` HTTP endpoint. Here's the general pattern.

### Step 1: Find the hook

Identify your platform's pre-request or model-selection hook:

| Platform | Hook / Extension Point |
|----------|----------------------|
| OpenClaw | `before_model_resolve` event |
| Portkey | Pre-request webhook |
| TrueFoundry | Routing middleware |
| LangChain | Custom router class |
| Cloudflare AI Gateway | Worker middleware |
| Kong | Plugin (Lua or Go) |
| Envoy | External processing filter |

### Step 2: Call `/v1/route`

From your hook, make an HTTP POST to the sidecar:

```
POST http://127.0.0.1:8079/v1/route
Content-Type: application/json

{
  "messages": [{"role": "user", "content": "What is 2+2?"}],
  "tolerance": 0.20
}
```

Or with a plain question:

```json
{
  "question": "What is 2+2?",
  "tolerance": 0.20
}
```

### Step 3: Map the response

The sidecar returns:

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

Map `selected_model` to your platform's model/provider identifiers. Use a lookup table (like the OpenClaw plugin's `pool` config).

### Step 4: Handle failures gracefully

Always fall back to the platform's default behavior if the sidecar is unreachable:

```typescript
try {
    const res = await fetch(`${sidecarUrl}/v1/route`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: prompt, tolerance }),
        signal: AbortSignal.timeout(timeoutMs),
    });
    if (!res.ok) return defaultBehavior();
    const route = await res.json();
    return mapToMyPlatform(route.selected_model);
} catch {
    return defaultBehavior();
}
```

```python
import requests

try:
    resp = requests.post(
        f"{sidecar_url}/v1/route",
        json={"question": prompt, "tolerance": tolerance},
        timeout=timeout_s,
    )
    resp.raise_for_status()
    return map_to_platform(resp.json()["selected_model"])
except (requests.RequestException, KeyError):
    return default_behavior()
```

### Step 5: Add health checking

On startup, verify the sidecar is reachable:

```python
try:
    resp = requests.get(f"{sidecar_url}/health", timeout=5)
    if resp.ok:
        logger.info("Model router sidecar is healthy")
except requests.RequestException:
    logger.warning(f"Model router sidecar not reachable at {sidecar_url}")
```

### Example: Portkey Webhook Plugin

```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)
SIDECAR_URL = "http://127.0.0.1:8079"
POOL_MAP = {
    "nem-think": "openrouter/nvidia/nemotron-3-nano-30b-a3b",
    "nem-nothink": "openrouter/nvidia/nemotron-3-nano-30b-a3b",
}

@app.post("/portkey-webhook")
def route_webhook():
    payload = request.json
    prompt = payload.get("prompt", "")

    try:
        resp = requests.post(
            f"{SIDECAR_URL}/v1/route",
            json={"question": prompt, "tolerance": 0.20},
            timeout=5,
        )
        resp.raise_for_status()
        selected = resp.json()["selected_model"]
        return jsonify({"model": POOL_MAP.get(selected, selected)})
    except Exception:
        return jsonify({})  # Fall back to Portkey default
```

### Example: LangChain Custom Router

```python
from langchain.llms import BaseLLM
import requests

class ModelRouterLLM(BaseLLM):
    sidecar_url: str = "http://127.0.0.1:8079"
    tolerance: float = 0.20
    model_map: dict = {}  # router name -> LangChain LLM instance

    def _call(self, prompt: str, **kwargs) -> str:
        try:
            resp = requests.post(
                f"{self.sidecar_url}/v1/route",
                json={"question": prompt, "tolerance": self.tolerance},
                timeout=5,
            )
            selected = resp.json()["selected_model"]
            llm = self.model_map.get(selected)
            if llm:
                return llm._call(prompt, **kwargs)
        except Exception:
            pass
        # Fallback
        return next(iter(self.model_map.values()))._call(prompt, **kwargs)
```

---

## Part 3: `/v1/route` API Reference

### Request

**URL:** `POST /v1/route`

**Content-Type:** `application/json`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `messages` | `list[dict]` | One of `messages` or `question` | — | OpenAI-format messages. Last user message is extracted. |
| `question` | `string` | One of `messages` or `question` | — | Plain text question. Takes priority over `messages`. |
| `tolerance` | `number` | No | `0.20` | Accuracy-cost tradeoff [0.0–1.0] |

### Response

**Content-Type:** `application/json`

```json
{
  "selected_model": "nem-nothink",
  "model_names": ["nem-think", "nem-nothink"],
  "confidences": {
    "nem-think": 0.92,
    "nem-nothink": 0.88
  },
  "costs": [
    {
      "model": "nem-think",
      "estimated_total_cost": 0.0004,
      "cost_per_m_input_tokens": 0.20,
      "cost_per_m_output_tokens": 0.20,
      "median_output_tokens": 150
    },
    {
      "model": "nem-nothink",
      "estimated_total_cost": 0.0001,
      "cost_per_m_input_tokens": 0.04,
      "cost_per_m_output_tokens": 0.16,
      "median_output_tokens": 150
    }
  ],
  "metadata": {
    "p_max": 0.92,
    "threshold": 0.72,
    "route_ms": 45.2
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `selected_model` | `string` | Model chosen by the router |
| `model_names` | `string[]` | All models in the pool |
| `confidences` | `dict[string, number]` | P(correct) per model |
| `costs` | `object[]` | Cost estimates per model |
| `costs[].model` | `string` | Model name |
| `costs[].estimated_total_cost` | `number` | Total estimated cost (USD) for this request |
| `costs[].cost_per_m_input_tokens` | `number` | USD per million input tokens |
| `costs[].cost_per_m_output_tokens` | `number` | USD per million output tokens |
| `costs[].median_output_tokens` | `number` | Median output tokens from training data |
| `metadata` | `object` | Routing internals |
| `metadata.p_max` | `number` | Highest P(correct) among all models |
| `metadata.threshold` | `number` | `p_max - tolerance` |
| `metadata.route_ms` | `number` | Routing latency in milliseconds |

### Health Check

**URL:** `GET /health`

```json
{
  "status": "ok",
  "mode": "router-only",
  "method": "prefill",
  "models": ["nem-think", "nem-nothink"]
}
```

### Models List

**URL:** `GET /api/models`

```json
[
  {
    "name": "nem-think",
    "display_name": "Nemotron Think (30B)",
    "cost_per_m_input_tokens": 0.20,
    "cost_per_m_output_tokens": 0.20
  },
  {
    "name": "nem-nothink",
    "display_name": "Nemotron Fast (30B)",
    "cost_per_m_input_tokens": 0.04,
    "cost_per_m_output_tokens": 0.16
  }
]
```
