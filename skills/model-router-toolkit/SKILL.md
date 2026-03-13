---
name: model-router-toolkit
description: >
  LLM routing toolkit that learns which model handles which queries best, then routes
  each request to the cheapest model above an accuracy threshold. Use when the user asks
  about model routing, training a router, evaluating routing quality, serving a router,
  collecting training data, deploying with LiteLLM proxy, integrating routing
  into an application, configuring routing tolerance/models/checkpoints, or anything
  involving the `model-router` CLI. Also use when the user discusses cost-quality
  tradeoffs across multiple LLMs, intelligent model selection, or prefill/KMeans
  routing strategies.
---

# Model Router Toolkit

## What It Is

The Model Router Toolkit is an LLM routing system that learns per-query model strengths from labeled data, then routes each incoming request to the cheapest model that meets an accuracy threshold. It replaces static model selection with data-driven routing — saving cost on easy queries (route to a small model) while preserving quality on hard ones (route to a large model).

**Core idea:** Given a pool of LLMs with different costs and capabilities, the router predicts P(correct) for each model on each incoming question, then picks the cheapest model whose confidence is within `tolerance` of the best.

**Two routing methods:**

- **Prefill routing** (primary) — Runs a small encoder (Qwen3.5-0.8B, 0.8B params) on the question, extracts hidden-state features, and scores them through a trained MLP ensemble. Best accuracy. Requires CPU or GPU.
- **KMeans routing** — Embeds the question via API, assigns to a learned cluster, and uses Platt-calibrated per-cluster accuracy. No GPU needed. Simpler but less precise.

**Key properties:**
- OpenAI-compatible API (`/v1/chat/completions`) — drop-in replacement
- Config-driven — all behavior controlled by a single YAML file
- LiteLLM under the hood — works with any provider (OpenRouter, NVIDIA NIM, OpenAI, Anthropic, local vLLM)
- Full pipeline via CLI: `collect` → `train` → `evaluate` → `serve`
- Multiple deployment modes: standalone server with playground UI, LiteLLM Proxy (production), SDK integration

**Package:** `model-router-toolkit` (PyPI name). CLI command: `model-router`. Python 3.10+.

## Installation

```bash
# Recommended (includes prefill routing with local encoder)
pip install -e '.[prefill]'

# KMeans-only (no GPU, no torch)
pip install -e .

# Development (adds pytest, ruff, mypy)
pip install -e '.[dev,prefill]'

# LiteLLM Proxy mode
pip install -e '.[proxy]'

# Everything
pip install -e '.[dev,prefill,proxy]'
```

The `model-router` CLI is available after install.

## Environment Variables

| Variable | When needed | Purpose |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | `serve`, `collect`, OpenRouter configs | API key for OpenRouter provider |
| `NVIDIA_API_KEY` | `serve`, `collect`, NVIDIA NIM configs, notebooks | API key for build.nvidia.com |
| `OPENAI_API_KEY` | Fallback | Used if neither OpenRouter nor NVIDIA key matches |
| `CORS_ORIGINS` | `serve` | Comma-separated allowed origins (default: `*`) |

**Not needed** for `train` or `evaluate` — these run offline with a local encoder and pre-collected CSV.

```bash
export OPENROUTER_API_KEY=sk-or-...
# or
export NVIDIA_API_KEY=nvapi-...
```

## Configuration Reference

All commands use a single YAML pool config. Two top-level sections: `routing` and `models`.

### Routing Section

| Field | Type | Required | Method | Description |
|-------|------|----------|--------|-------------|
| `method` | string | Yes | both | `prefill` or `kmeans` |
| `checkpoint` | string | Yes | both | Path to trained checkpoint (`.pt` for prefill, `.pkl` for kmeans) |
| `tolerance` | float | No | both | Accuracy-cost tradeoff, 0.0–1.0 (default: 0.20). Lower = cheaper, higher = more accurate |
| `encoder` | string | Yes | prefill | HuggingFace model ID (e.g. `Qwen/Qwen3.5-0.8B`) |
| `encoder_backend` | string | No | prefill | `transformers` (local, default) or `server` (remote) |
| `encoder_server` | string | No | prefill | URL of remote encoder server (when `encoder_backend: server`) |
| `training_mode` | string | No | prefill | `auto` (default), `per_model`, or `single` |
| `embed_model` | string | Yes | kmeans | Embedding model ID (e.g. `nvidia/llama-nemotron-embed-1b-v2`) |
| `embed_mode` | string | No | kmeans | `api` (remote, default) or `local` (in-process) |
| `embed_api_base` | string | No | kmeans | Embedding API base URL |

### Models Section

Each entry in the `models` list:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique ID — must match CSV `model` column for training |
| `display_name` | string | No | Human-readable label |
| `litellm_model` | string | Yes | LiteLLM model ID. Prefix with provider: `openrouter/...`, `nvidia_nim/...`, `openai/...` |
| `cost_per_m_input_tokens` | float | Yes | USD per 1M input tokens |
| `cost_per_m_output_tokens` | float | Yes | USD per 1M output tokens |
| `system_prompt` | string | No | System prompt prepended when calling this model |
| `chat_template_kwargs` | object | No | Template kwargs (e.g. `enable_thinking: true`, `reasoning_effort: high`) |
| `api_base` | string | No | Override API base URL for this model |

### Example: Prefill Config

```yaml
routing:
  method: prefill
  checkpoint: checkpoints/prefill_qwen08b.pt
  tolerance: 0.20
  encoder: Qwen/Qwen3.5-0.8B
  encoder_backend: transformers

models:
  - name: nem-think
    display_name: Nemotron 3 Nano Think
    litellm_model: openrouter/nvidia/nemotron-3-nano-30b-a3b
    cost_per_m_input_tokens: 0.20
    cost_per_m_output_tokens: 0.20
    system_prompt: Think step-by-step before answering.
    chat_template_kwargs:
      enable_thinking: true

  - name: nem-nothink
    display_name: Nemotron 3 Nano
    litellm_model: openrouter/nvidia/nemotron-3-nano-30b-a3b
    cost_per_m_input_tokens: 0.04
    cost_per_m_output_tokens: 0.16
    system_prompt: Answer directly and concisely.
```

### Example: KMeans Config

```yaml
routing:
  method: kmeans
  checkpoint: checkpoints/kmeans_c100_db.pkl
  tolerance: 0.20
  embed_model: nvidia/llama-nemotron-embed-1b-v2
  embed_mode: api
  embed_api_base: https://integrate.api.nvidia.com/v1

models:
  - name: nem-think
    litellm_model: nvidia_nim/nvidia/nvidia/Nemotron-3-Nano-30B-A3B
    cost_per_m_input_tokens: 0.20
    cost_per_m_output_tokens: 0.20
```

### Starter Configs

| Config | Method | Provider | When to use |
|--------|--------|----------|-------------|
| `configs/prefill-qwen08b.yaml` | Prefill | OpenRouter | Default — GPU available, best accuracy |
| `configs/cloud-only.yaml` | KMeans | NVIDIA NIM | No GPU, cloud embeddings |
| `configs/smoke-test.yaml` | Prefill | OpenRouter | Quick 2-model test |
| `configs/local-prefill.yaml` | Prefill | Local | Air-gapped / local-only |
| `configs/openrouter-kmeans.yaml` | KMeans | OpenRouter | OpenRouter + KMeans, no GPU |

Customize by copying: `cp configs/prefill-qwen08b.yaml configs/my-config.yaml`

## Routing Methods

### Prefill (primary, best accuracy)

1. Question runs through encoder (Qwen3.5-0.8B) — single forward pass with `output_hidden_states=True`
2. Hidden states at the optimal layer are extracted (last-token or mean-pooled, per target model)
3. Per-model StandardScaler + PCA reduces dimensions
4. SharedTrunkNet MLP ensemble produces P(correct) per target model
5. Cheapest model with P(correct) within `tolerance` of the best is selected

CPU: ~5s per question. GPU: <100ms.

### KMeans (no GPU needed)

1. Question is embedded via API (e.g. `nvidia/llama-nemotron-embed-1b-v2`)
2. KMeansRouter assigns to nearest cluster
3. Platt calibration produces P(correct) per model
4. Cheapest model above tolerance threshold is selected

### Tolerance

Controls the accuracy-cost tradeoff. Range 0.0–1.0.

- `0.0` — always pick the cheapest model (ignore accuracy)
- `0.20` — pick cheapest model within 20% confidence of the best (default)
- `1.0` — always pick the most accurate model (ignore cost)

The router selects the cheapest model where `P(correct) >= p_max - tolerance`.

## Training Data Format

CSV with columns: `question`, `model`, `isCorrect`, `output_tokens` (optional).

One row per (question, model) pair. The same question appears once per model in the pool. `isCorrect` is `0` or `1`.

```csv
question,model,isCorrect,output_tokens
"What is the capital of France?",nem-think,1,45
"What is the capital of France?",nem-nothink,1,12
"Prove sqrt(2) is irrational",nem-think,1,380
"Prove sqrt(2) is irrational",nem-nothink,0,95
```

---

## Journey 1: Collect Training Data

Runs every model in the pool on each question, judges correctness, writes a labeled CSV.

```bash
model-router collect \
  --config configs/prefill-qwen08b.yaml \
  --questions questions.txt \
  --output data/collected.csv \
  --judge vote
```

### Flags

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--config` | Yes | — | Pool config YAML |
| `--questions` | Yes | — | Questions file, one question per line |
| `--output` | Yes | — | Output CSV path |
| `--judge` | No | `vote` | Judging method: `vote`, `reference`, or `llm` |
| `--references` | No | — | Reference CSV for `--judge reference` |

### Judge Methods

| Method | How it works | When to use |
|--------|-------------|-------------|
| `vote` | Majority vote across model outputs | No ground truth available |
| `reference` | Compares against reference answers CSV | Have ground truth answers |
| `llm` | Not yet implemented | — |

### Workflow

1. Prepare `questions.txt` with one question per line (500+ recommended)
2. Run collect — requires API key since it calls every model
3. Split output: 80% `train.csv`, 20% `test.csv`

Reference judging:
```bash
model-router collect \
  --config configs/prefill-qwen08b.yaml \
  --questions questions.txt \
  --output data/collected.csv \
  --judge reference --references answers.csv
```

---

## Journey 2: Train a Router

Trains a routing model from labeled CSV data. No API key needed — uses local encoder.

```bash
model-router train \
  --config configs/prefill-qwen08b.yaml \
  --data data/train.csv \
  --output-dir checkpoints/
```

### Flags

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--config` | Yes | — | Pool config YAML |
| `--data` | Yes | — | Labeled CSV (question, model, isCorrect) |
| `--output-dir` | No | `checkpoints/` | Output directory for checkpoint |
| `--mode` | No | `auto` | Training mode: `auto`, `single`, or `per_model` |
| `--device` | No | auto-detect | `cpu`, `cuda`, `mps`, or auto |
| `--batch-size` | No | 4 | Encoder extraction batch size |
| `--n-seeds` | No | 10 | Number of ensemble seeds |
| `--n-keep` | No | 5 | Best models to keep from ensemble |
| `--prefill-dir` | No | — | Cache dir for extracted prefill features (saves hours on re-runs) |
| `--epochs` | No | 150 | Max MLP training epochs |
| `--patience` | No | 15 | Early stopping patience |
| `--pca-dims` | No | `50,100,150,200,300` | PCA dimensions to sweep, comma-separated |

### What the Pipeline Does

1. **Load labels** from CSV
2. **Extract prefill features** via encoder (Qwen3.5-0.8B forward pass on each question)
3. **Sweep** layer, pooling mode (last-token vs mean), and PCA dimension per target model — uses 5-fold CV AUC with logistic regression
4. **Fit transforms** — StandardScaler + PCA per target
5. **Train SharedTrunkNet MLP ensemble** — BCEWithLogitsLoss, Adam optimizer, early stopping. Trains `n_seeds` seeds, keeps top `n_keep` by validation loss
6. **Save** `.pt` checkpoint + `serve.yaml`

### Smoke Test (fast verification)

```bash
model-router train \
  --config configs/smoke-test.yaml \
  --data data/smoke-train.csv \
  --output-dir checkpoints/smoke/ \
  --n-seeds 2 --n-keep 1 --device cpu \
  --epochs 5 --patience 3 --pca-dims 10,20
```

### Tips

- Use `--prefill-dir cache/` to cache extracted features — dramatically speeds up re-runs
- `--device cpu` forces CPU when GPU detection causes issues
- Wider `--pca-dims` sweep (e.g. `50,100,150,200,300,400`) can find better transforms at the cost of longer sweep time

---

## Journey 3: Evaluate a Checkpoint

Evaluates a trained checkpoint against test data. No API key needed.

```bash
model-router evaluate \
  --config configs/prefill-qwen08b.yaml \
  --checkpoint checkpoints/prefill_router.pt \
  --data data/test.csv
```

### Flags

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--config` | Yes | — | Pool config YAML |
| `--checkpoint` | Yes | — | Trained checkpoint (`.pt` or `.pkl`) |
| `--data` | Yes | — | Test CSV (question, model, isCorrect) |
| `--device` | No | auto | `cpu`, `cuda`, or `mps` |
| `--batch-size` | No | 4 | Encoder extraction batch size |
| `--prefill-dir` | No | — | Cache dir for extracted features |

### Metrics Reference

| Metric | Meaning | Good value |
|--------|---------|------------|
| **AUC** (per-model) | ROC AUC of P(correct) prediction vs actual | > 0.70 useful, > 0.80 strong |
| **Oracle** | Accuracy if you always picked the best model per question | Theoretical ceiling |
| **Best single** | Accuracy of just using one model for everything | Baseline to beat |
| **Headroom** | Oracle - Best single | Available improvement from routing |
| **Router accuracy** | Accuracy of the router's selections | Should exceed Best single |
| **Lift** | Router accuracy - Best single | Positive = routing helps |
| **Headroom captured** | Percentage of headroom the router captures | > 20% is useful |

### Agreement Zones

| Zone | Description | Significance |
|------|-------------|--------------|
| All correct | Every model answers correctly | Routing doesn't matter — route to cheapest |
| Disagree | Some right, some wrong | Where routing adds value — key accuracy metric |
| All wrong | No model answers correctly | Nothing to save |

### Deep Analysis

- **Near-miss analysis**: For wrong routing decisions in the disagree zone, how close was the confidence gap? Small gaps mean the router nearly got it right.
- **Pairwise win rates**: When model A is correct and B is wrong, how often does A have higher confidence? Values > 0.5 = meaningful signal.

### Interpreting Results

**Good signs:**
- Per-model AUC > 0.70
- Router accuracy > best single model
- Headroom captured > 20%
- Disagree zone accuracy above random (1/n_models)
- Pairwise win rates > 0.5

**Warning signs and fixes:**
- AUC near 0.50 → more training data, different encoder, wider PCA sweep
- Router accuracy < best single → routing is hurting, check data quality
- All traffic to one model → lower tolerance (e.g. 0.10), or insufficient signal
- Low headroom → models are too similar, add a more diverse model

---

## Journey 4: Serve (Standalone Server)

Starts a FastAPI server with routing + LLM inference + playground UI.

```bash
model-router serve --config configs/prefill-qwen08b.yaml --port 8000
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `configs/prefill-qwen08b.yaml` | Pool config YAML |
| `--port` | 8000 | Server port |

Requires an API key for the model provider.

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat (streaming + non-streaming) |
| `/api/chat` | POST | SSE chat endpoint for playground UI |
| `/api/models` | GET | Model pool with cost data |
| `/api/config` | GET | Routing method, features, tolerance |
| `/api/review` | POST | Auto-review: judges answer correctness |
| `/health` | GET | Health check |
| `/` | GET | Interactive playground UI |

### Connecting Your App

**OpenAI Python SDK:**
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="routed",
    messages=[{"role": "user", "content": "Hello"}]
)
```

**Environment variable** (works with any tool that reads `OPENAI_API_BASE`):
```bash
export OPENAI_API_BASE=http://localhost:8000/v1
```

**cURL:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "routed", "messages": [{"role": "user", "content": "Hello"}]}'
```

**Playground:** Open `http://localhost:8000/` for routing cards, probability bars, tolerance slider, and model toggles.

---

## Journey 5: Serve (Router-Only)

Returns routing decisions only — no LLM inference, no API keys needed. Useful for external dispatchers or microservice architectures.

```bash
model-router serve-router --config configs/prefill-qwen08b.yaml --port 8080
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `configs/prefill-qwen08b.yaml` | Pool config YAML |
| `--port` | 8080 | Server port |

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/route` | POST | Routing decision |
| `/health` | GET | Health check |
| `/api/models` | GET | Model pool |

### Route Request

```json
{
  "messages": [{"role": "user", "content": "Prove sqrt(2) is irrational"}],
  "tolerance": 0.20
}
```

Or use the `question` field directly:

```json
{
  "question": "Prove sqrt(2) is irrational",
  "tolerance": 0.20
}
```

### Route Response

```json
{
  "selected_model": "nem-think",
  "model_names": ["nem-think", "nem-nothink", "gptoss-high", "gpt-5.2"],
  "confidences": {"nem-think": 0.85, "nem-nothink": 0.42, "gptoss-high": 0.78, "gpt-5.2": 0.91},
  "costs": [
    {"model": "nem-think", "estimated_total_cost": 0.00012, "cost_per_m_input_tokens": 0.20, "cost_per_m_output_tokens": 0.20, "median_output_tokens": 200}
  ],
  "metadata": {"route_ms": 4.52, "p_max": 0.91, "threshold": 0.71}
}
```

---

## Journey 6: LiteLLM Proxy

Production-grade deployment with auth, rate limiting, spend tracking, caching, and virtual keys via LiteLLM Proxy.

### Step 1: Generate proxy config

```bash
model-router proxy-config \
  --config configs/prefill-qwen08b.yaml \
  --output configs/litellm-proxy.yaml
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--config` | Yes | — | Pool config YAML |
| `--output` | No | stdout | Output path (prints YAML to stdout if omitted) |

### Step 2: Start the proxy

```bash
model-router proxy \
  --litellm-config configs/litellm-proxy.yaml \
  --router-config configs/prefill-qwen08b.yaml \
  --port 4000
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--litellm-config` | Yes | — | LiteLLM proxy config (model_list, router_settings) |
| `--router-config` | Yes | — | Pool config YAML (routing method, checkpoint) |
| `--host` | No | `0.0.0.0` | Bind host |
| `--port` | No | 4000 | Proxy port |

The proxy exposes the OpenAI-compatible API at `http://localhost:4000/v1/chat/completions`.

### Serve vs Proxy Decision

| Scenario | Use | Why |
|----------|-----|-----|
| Trying out the toolkit | `serve` | Playground UI, single config, minimal setup |
| Local development | `serve` | Visual routing decisions in playground |
| Already running LiteLLM | `proxy` | Drop-in with existing auth, spend tracking |
| Production without LiteLLM | `proxy` | Auth, rate limiting, virtual keys out of the box |
---

## Journey 7: SDK and Library Integration

### LiteLLM SDK (no server needed)

Add routing to an existing LiteLLM application with three lines:

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

**Per-request tolerance override:**
```python
strategy.set_request_tolerance(0.10)
response = await router.acompletion(model="nem-think", messages=messages)
```

**Access routing metadata:**
```python
if strategy.last_result:
    print(strategy.last_result.selected_model)
    print(strategy.last_result.confidences)
```

### Direct Python Library (routing decisions only)

No LLM inference, no API keys. Returns routing decisions for custom dispatchers.

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

---

## CLI Quick Reference

| Command | Purpose | API Key |
|---------|---------|---------|
| `model-router collect` | Run models on questions, judge correctness, write CSV | Yes |
| `model-router train` | Train routing model from labeled CSV | No |
| `model-router evaluate` | Evaluate checkpoint against test CSV | No |
| `model-router serve` | Start full server with playground UI | Yes |
| `model-router serve-router` | Start router-only server (no inference) | No |
| `model-router proxy-config` | Generate LiteLLM proxy config from pool config | No |
| `model-router proxy` | Start LiteLLM Proxy with routing | Yes |
| `model-router serve-config` | Generate serve config (coming soon) | No |

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `Error: OPENROUTER_API_KEY not set` | Missing API key | `export OPENROUTER_API_KEY=sk-or-...` |
| `FileNotFoundError: checkpoint` | Checkpoint path wrong or missing | Check `routing.checkpoint` in config points to an existing file |
| All traffic to one model | Tolerance too high or insufficient training signal | Lower `routing.tolerance` (e.g. 0.10); collect more training data |
| AUC near 0.50 | Router can't distinguish model quality | More training data, wider PCA sweep (`--pca-dims 50,100,200,300,400`), different encoder |
| Router accuracy < best single | Routing is actively hurting | Check training data quality; verify CSV `model` names match config `models[].name` |
| Slow prefill on CPU | Encoder running on CPU (~5s/question) | Use `--device cuda` or `--device mps`; or switch to KMeans method |
| `CUDA out of memory` | Encoder too large for GPU | Lower `--batch-size` to 1 or 2; or use `--device cpu` |
| Proxy startup warnings | Model name misalignment | Ensure pool config model names match LiteLLM proxy config `model_name` entries |

## Testing

```bash
# Unit tests (no API keys or checkpoints needed)
pytest tests/ --ignore=tests/integration/ -v

# Integration tests (uses mocks, no API keys needed)
pytest tests/integration/ -v

# All tests with coverage
pytest tests/ --cov=model_router_toolkit --cov-report=term-missing
```
