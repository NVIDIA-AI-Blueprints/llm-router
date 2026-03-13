# Architecture

The Model Router Toolkit is built around a **BaseRouter** abstraction. All routing methods inherit from BaseRouter and produce a model selection with confidence scores. Platform integrations live in the **adapters/** and **plugins/** directories, keeping the core routing engine free of framework dependencies.

## Class Hierarchy

```
BaseRouter (abstract)
    |   route(question, tolerance)  -> RoutingResult  # ML-based model selection
    |   has_model(model_name)       -> bool           # pool membership check
    |   resolve(model_name)         -> RoutingResult   # pin a model without ML inference
    |
    +-- KMeansRouter      # Embedding-based clustering; no GPU required
    |
    +-- PrefillRouter     # Prefill complexity scoring; CPU or GPU

Adapters (platform integrations)
    |
    +-- adapters/litellm/
    |       +-- ModelRoutingStrategy   # LiteLLM custom routing strategy
    |       +-- app.py                 # Full serve mode (routing + inference + UI)
    |       +-- proxy.py               # LiteLLM Proxy injection
    |       +-- completions.py         # /v1/chat/completions endpoint
    |       +-- chat.py                # /api/chat SSE endpoint
    |       +-- review.py              # /api/review auto-judge endpoint
    |       +-- config_bridge.py       # Pool config → LiteLLM config generator
    |
    +-- adapters/http/
    |       +-- app.py                 # Router-only sidecar (no inference)
    |       +-- route.py               # POST /v1/route endpoint
    |       +-- auth.py                # Webhook HMAC/bearer auth middleware
    |       +-- _shared.py             # Warmup, health, models helpers
    |
    +-- plugins/openclaw/
            +-- index.ts               # before_model_resolve hook
            +-- openclaw.plugin.json   # Plugin manifest + config schema
```

## Dependency Graph

The core package has **zero framework dependencies** — only pydantic, numpy, scikit-learn, requests, and PyYAML. Each adapter brings its own extras:

```
┌──────────────────────────────────────────────────────┐
│  Core (pip install model-router-toolkit)             │
│  router.py, config.py, checkpoint.py, kmeans/,      │
│  prefill/, train.py, evaluate.py, collect.py         │
│  Deps: pydantic, numpy, scikit-learn, requests, yaml │
└──────────────┬───────────────────────┬───────────────┘
               │                       │
    ┌──────────▼──────────┐  ┌────────▼─────────┐
    │  [server] extra     │  │  [litellm] extra  │
    │  adapters/http/     │  │  adapters/litellm/ │
    │  FastAPI, uvicorn   │  │  litellm, FastAPI  │
    └─────────────────────┘  └────────┬──────────┘
                                      │
                             ┌────────▼──────────┐
                             │  [proxy] extra     │
                             │  litellm[proxy]    │
                             │  adapters/litellm/ │
                             │  proxy.py          │
                             └───────────────────┘

    ┌─────────────────────┐  ┌────────────────────┐
    │  [prefill] extra    │  │  [training] extra   │
    │  torch, transformers│  │  litellm            │
    │  accelerate, tqdm   │  │  (collect command)  │
    └─────────────────────┘  └────────────────────┘
```

Install only what you need:

| Extra | Installs | Enables |
|-------|----------|---------|
| *(none)* | Core only | `BaseRouter`, `RoutingResult`, config, KMeans routing |
| `[server]` | FastAPI, uvicorn | `adapters/http/` — router-only sidecar |
| `[litellm]` | litellm, FastAPI, uvicorn | `adapters/litellm/` — strategy, serve mode |
| `[proxy]` | litellm[proxy], packaging | LiteLLM Proxy injection (`model-router proxy`) |
| `[prefill]` | torch, transformers, accelerate, tqdm | Prefill routing method |
| `[training]` | litellm | `model-router collect` (calls provider APIs) |
| `[dev]` | pytest, ruff, mypy, httpx | Testing and linting |
| `[all]` | Everything above | Full development setup |

## Adapter Layer

### Why adapters?

The routing engine (BaseRouter → RoutingResult) is a pure function: question in, model selection out. But real deployments need to slot into existing infrastructure — LiteLLM proxies, API gateways, OpenAI-compatible servers, webhook pipelines.

**Adapters** bridge the gap. Each adapter translates between a platform's interface and the BaseRouter API. The core never imports FastAPI, litellm, or any adapter-specific dependency.

### adapters/litellm/

Full integration with the LiteLLM ecosystem. Three deployment patterns:

| Module | Pattern | What it does |
|--------|---------|-------------|
| `strategy.py` | SDK embedding | Wraps BaseRouter as `CustomRoutingStrategyBase` for `litellm.Router` |
| `app.py` | Standalone server | FastAPI app with routing + inference + playground UI |
| `proxy.py` | Proxy injection | Patches LiteLLM Proxy's internal Router at startup |

Supporting modules:

| Module | Purpose |
|--------|---------|
| `completions.py` | `/v1/chat/completions` — OpenAI-compatible endpoint |
| `chat.py` | `/api/chat` — SSE streaming for the playground UI |
| `review.py` | `/api/review` — auto-judge answer correctness |
| `config_bridge.py` | Generates LiteLLM proxy config from pool config |

### adapters/http/

Lightweight router-only sidecar. No LiteLLM dependency — only needs `[server]` (FastAPI + uvicorn). Returns routing decisions via `POST /v1/route` without performing LLM inference.

| Module | Purpose |
|--------|---------|
| `app.py` | FastAPI app factory — `/v1/route`, `/health`, `/api/models` |
| `route.py` | Route endpoint — accepts messages or question text, returns `RouteResponse` |
| `auth.py` | `WebhookAuthMiddleware` — HMAC-SHA256 signature or bearer token verification |
| `_shared.py` | Shared helpers: warmup, health dict, models list |

### plugins/openclaw/

TypeScript plugin for the OpenClaw gateway. Hooks into `before_model_resolve` to call the HTTP sidecar and override model selection. Graceful degradation — if the sidecar is unreachable, falls back to OpenClaw's default.

## Inference Flow

### KMeans Path

```
Question --> Embed API (build.nvidia.com) --> KMeansRouter --> Adapter dispatch
                  |                              |
                  v                              v
          nvidia/llama-nemotron-         checkpoint.pkl
          embed-1b-v2                  (centroids, Platt calibrators)
```

1. Question is embedded via API
2. KMeansRouter assigns to nearest cluster, applies Platt calibration for P(correct) per model
3. Selects cheapest model above tolerance threshold
4. Adapter dispatches to selected provider (or returns decision only)

### Prefill Path

```
Question --> Encoder (Qwen3.5-0.8B) --> PrefillRouter --> Adapter dispatch
                  |                          |
                  v                          v
          hidden states               checkpoint.pt
          (single forward pass)       (PCA + MLP ensemble)
```

1. Question is run through the encoder model (single forward pass, `output_hidden_states=True`)
2. Hidden states at the best layer are extracted (last-token or mean-pooled)
3. Per-model StandardScaler + PCA reduces dimensions
4. Concatenated features go through SharedTrunkNet MLP ensemble
5. Sigmoid outputs give P(correct) per target model
6. Cheapest model with P(correct) within `tolerance` of the best is selected

The encoder (Qwen3.5-0.8B, 0.8B parameters) runs on CPU in ~5s per question. GPU reduces this to <100ms.

## Training Pipeline

Training bypasses the BaseRouter interface and works directly with prefill components for batch efficiency.

```
train.csv --> Load Labels --> Batch Extract Prefill
                                    |
                              Sweep (layer/mode/PCA per target)
                                    |
                              Fit Transforms (StandardScaler + PCA)
                                    |
                              Train SharedTrunkNet Ensemble
                                    |
                              Save .pt Checkpoint + serve.yaml
```

**Sweep**: For each target model, grid-searches over hidden state mode (last-token vs mean-pooled) and PCA dimension, with ternary search over encoder layers. Uses 5-fold CV AUC with logistic regression as the quality metric.

**Trunk training**: BCEWithLogitsLoss with Adam optimizer, early stopping on validation split. Trains N seeds (default 10), keeps the top K by validation loss (default 5). Final ensemble averages sigmoid outputs.

**Checkpoint**: Self-contained `.pt` file with pool config, per-model transforms (scaler, PCA, layer, mode), trunk state dicts, trunk architecture config, and cost table. The same checkpoint is used for both training evaluation and serving inference.

## Evaluation Pipeline

Evaluation also bypasses BaseRouter for batch extraction:

```
test.csv + checkpoint.pt --> Batch Extract --> Apply Transforms --> Run Trunk
                                                                       |
                                                                  Rich Metrics Report
```

Reports per-model AUC, oracle vs router accuracy, lift, headroom captured, routing distribution, agreement zone analysis, near-miss diagnostics, and pairwise confidence win rates.

## Config-Driven Dispatch

All behavior is config-driven:

```yaml
routing:
  method: prefill          # or kmeans
  checkpoint: path/to.pt   # trained checkpoint
  tolerance: 0.20          # accuracy-cost tradeoff
  encoder: Qwen/Qwen3.5-0.8B  # HF encoder for prefill

models:
  - name: nem-think
    litellm_model: openrouter/nvidia/nemotron-3-nano-30b-a3b
    cost_per_m_input_tokens: 0.20
    cost_per_m_output_tokens: 0.20
```

`routing.method` determines which BaseRouter is instantiated. The model pool, costs, and endpoints are all in the config. No code changes needed to add or remove models.

## Model-Name Bypass (Pin Mode)

All adapters support **model-name bypass**: if the caller specifies a model name that exists in the pool, the router returns it directly without running ML inference. This enables two routing workflows:

| Workflow | How it works | When to use |
|----------|-------------|-------------|
| **Per-turn routing** | Every request goes through `route()`. Model can change on each call. | Default. Cost-optimize every LLM call independently. |
| **Router-per-subagent** | First request goes through `route()`. Caller captures `selected_model` and sends it as `model` on subsequent requests. `resolve()` returns instantly. | Multi-turn agent chains where mid-chain model switches would hurt quality. |

### How it works

`BaseRouter` exposes two methods:

- `has_model(name)` — returns `True` if the name is in the model pool
- `resolve(name)` — returns a `RoutingResult` with `selected_model` pinned and `metadata.pinned = True`, without running the encoder or ML model

Each adapter uses a different signal for pinning (because the `model` parameter means different things in different contexts):

| Adapter | Pin signal | Why |
|---------|-----------|-----|
| **LiteLLM strategy** | `metadata.pin_model` in `request_kwargs` | `model` is always a pool name in LiteLLM flows — can't distinguish "route me" from "pin me" by model name alone |
| **HTTP route endpoint** | `model` field in `RouteRequest` body | Separate from `question` — presence of `model` is an unambiguous pin signal |
| **Direct Python** | Caller calls `router.resolve(name)` explicitly | Full programmatic control |

### Example: router-per-subagent via LiteLLM Proxy

```
1. First call:   metadata={}                              → strategy runs route() → selected_model="nem-think"
2. Next calls:   metadata={"pin_model": "nem-think"}      → strategy runs resolve() → instant pin, no ML
```

### Example: router-per-subagent via HTTP sidecar

```bash
# First call: get routing decision
curl -X POST http://localhost:8079/v1/route \
  -d '{"question": "Explain quantum computing"}'
# Response: {"selected_model": "nem-think", "metadata": {...}}

# Subsequent calls: pin the model (no ML inference)
curl -X POST http://localhost:8079/v1/route \
  -d '{"model": "nem-think"}'
# Response: {"selected_model": "nem-think", "metadata": {"pinned": true}}
```

## Deployment Topologies

| Topology | Adapter | Routing | Inference | Latency overhead |
|----------|---------|---------|-----------|-----------------|
| **Embedded SDK** | `adapters/litellm/strategy.py` | In-process | In-process (litellm) | ~0ms network |
| **Standalone Server** | `adapters/litellm/app.py` | In-server | In-server (litellm) | 1 hop |
| **LiteLLM Proxy** | `adapters/litellm/proxy.py` | In-proxy | In-proxy (litellm) | 1 hop |
| **Router Sidecar** | `adapters/http/app.py` | Sidecar | External (caller handles) | 1 hop (route only) |
| **Gateway Plugin** | `plugins/openclaw/` | Sidecar | Gateway handles | 1 hop (route only) |
| **Direct Python** | `config.build_router_from_config()` | In-process | None (caller handles) | 0 |

### Which Topology Should I Use?

| Scenario | Recommended | Why |
|----------|------------|-----|
| Demos, local development | Standalone Server | Playground UI, single config, fast to start |
| Existing LiteLLM stack | LiteLLM Proxy | Drop-in — keeps auth, rate limiting, spend tracking |
| Production without LiteLLM | LiteLLM Proxy | Auth, rate limiting, caching out of the box |
| Gateway integration (OpenClaw, Portkey) | Router Sidecar + Plugin | Route-only, no inference duplication |
| Existing Python app | Embedded SDK | 3 lines, no server needed |
| Custom dispatcher | Direct Python | Routing decisions only, you handle inference |
| Air-gapped / no API keys | Direct Python + Prefill | Local encoder, no network calls |
