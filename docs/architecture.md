# Architecture

The Model Router Toolkit is built around a **BaseRouter** abstraction. All routing methods inherit from BaseRouter and produce a model selection with confidence scores. **ModelRoutingStrategy** wraps any BaseRouter and implements LiteLLM's `CustomRoutingStrategyBase` for drop-in integration.

## Class Hierarchy

```
BaseRouter (abstract)
    |
    +-- KMeansRouter      # Embedding-based clustering; no GPU required
    |
    +-- PrefillRouter     # Prefill complexity scoring; CPU or GPU

ModelRoutingStrategy
    +-- wraps BaseRouter
    +-- implements CustomRoutingStrategyBase (LiteLLM)
```

## Inference Flow

### KMeans Path

```
Question --> Embed API (build.nvidia.com) --> KMeansRouter --> LiteLLM dispatch
                  |                              |
                  v                              v
          nvidia/llama-nemotron-         checkpoint.pkl
          embed-1b-v2                  (centroids, Platt calibrators)
```

1. Question is embedded via API
2. KMeansRouter assigns to nearest cluster, applies Platt calibration for P(correct) per model
3. Selects cheapest model above tolerance threshold
4. LiteLLM dispatches to selected provider

### Prefill Path

```
Question --> Encoder (Qwen3.5-0.8B) --> PrefillRouter --> LiteLLM dispatch
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

## Deployment Modes

The toolkit offers three deployment modes. All three perform routing **and** LLM inference — the routing decision determines which model handles the request, then LiteLLM dispatches it.

### Standalone Server (`model-router serve`)

A custom FastAPI application built into the toolkit. Creates a `litellm.Router` internally and registers `ModelRoutingStrategy` via `set_custom_routing_strategy()`.

```
Client --> FastAPI app (port 8000)
               |
               +-- /v1/chat/completions  (OpenAI-compatible)
               +-- /api/chat             (SSE for playground UI)
               +-- /api/models           (pool info + costs)
               +-- /api/config           (routing method, features)
               +-- /api/review           (answer quality judging)
               +-- /health
               +-- /                     (interactive playground UI)
               |
               v
         ModelRoutingStrategy --> BaseRouter.route()
               |
               v
         litellm.Router.acompletion() --> provider API
```

**Config**: Single pool config YAML (e.g., `configs/prefill-qwen08b.yaml`).

```bash
model-router serve --config configs/prefill-qwen08b.yaml --port 8000
```

### LiteLLM Proxy (`model-router proxy`)

Starts the full **LiteLLM Proxy server** and injects `ModelRoutingStrategy` at startup. The proxy is LiteLLM's production-grade API gateway with built-in auth, rate limiting, spend tracking, caching, virtual keys, and load balancing.

```
Client --> LiteLLM Proxy (port 4000)
               |
               +-- /v1/chat/completions   (OpenAI-compatible)
               +-- /v1/completions        (legacy completions)
               +-- /v1/embeddings         (embedding passthrough)
               +-- /health
               +-- LiteLLM's full endpoint set (auth, spend, etc.)
               |
               v
         ModelRoutingStrategy injected at startup
               |
               v
         litellm.proxy.Router --> provider API
```

**Config**: Two config files — a LiteLLM proxy config (`model_list` + `router_settings`) and a pool config (routing method + checkpoint).

```bash
# Generate the LiteLLM proxy config from your pool config
model-router proxy-config --config configs/prefill-qwen08b.yaml --output configs/litellm-proxy.yaml

# Start the proxy
model-router proxy \
    --litellm-config configs/litellm-proxy.yaml \
    --router-config configs/prefill-qwen08b.yaml \
    --port 4000
```

### Docker

The Dockerfile provides multi-stage builds for containerized deployment:

| Target | Extras installed | Use case |
|--------|-----------------|----------|
| `proxy` | `.[proxy]` (CPU only) | KMeans routing or prefill with remote encoder |
| `proxy-gpu` | `.[proxy,prefill]` (torch + transformers) | Prefill routing with local encoder on GPU |

Both targets run `model-router proxy` via `docker/entrypoint.sh`.

```bash
# CPU (KMeans or remote encoder) — run from repo root
docker build -f docker/Dockerfile --target proxy -t model-router:proxy .

# GPU (local prefill encoder) — run from repo root
docker build -f docker/Dockerfile --target proxy-gpu -t model-router:gpu .

# Or via compose (proxy-gpu target, prefill routing)
docker compose -f docker/docker-compose.yaml up
```

### Which Mode Should I Use?

| Scenario | Recommended mode | Why |
|----------|-----------------|-----|
| Demos, local development, exploring routing | `serve` | Includes playground UI, single config file, fast to start |
| Already using LiteLLM Proxy in your stack | `proxy` | Drop-in replacement — keeps your existing LiteLLM auth, rate limiting, and spend tracking |
| Production deployment without existing LiteLLM | `proxy` | LiteLLM Proxy provides auth, rate limiting, caching, and virtual keys out of the box |
| Containerized / Kubernetes deployment | Docker (`proxy` or `proxy-gpu`) | Standard container with health checks, non-root user |
| Adding routing to an existing LiteLLM SDK setup | Neither — use SDK integration | 3 lines of Python, no server needed (see [integration guide](integration.md)) |
