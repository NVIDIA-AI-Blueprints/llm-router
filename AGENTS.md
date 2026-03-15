# AGENTS.md -- Model Router Toolkit

LLM routing toolkit. Learns which model handles which queries best, routes to the cheapest model above an accuracy threshold. Prefill complexity-based routing via Qwen3.5-0.8B encoder behind a unified BaseRouter interface. Full collect/train/evaluate/serve pipeline via CLI.

## Project Structure

```
src/model_router_toolkit/
├── __init__.py                # Public API exports (lazy imports for optional deps)
├── __main__.py                # CLI entry point (model-router)
├── config.py                  # PoolConfig, ModelSpec, RoutingConfig (pydantic)
├── router.py                  # BaseRouter ABC, RoutingResult, CostEstimate, extract_user_text
├── checkpoint.py              # Checkpoint load/save (.pt)
├── gpu.py                     # GPU detection + VRAM checks
├── train.py                   # Unified training dispatcher
├── evaluate.py                # Unified evaluation (rich metrics)
├── collect.py                 # Data collection (run models + judge correctness)
├── telemetry.py               # SQLite session/chat logging
├── prefill/
│   ├── router.py              # PrefillRouter(BaseRouter) -- inference path
│   ├── scorer.py              # Prefill scoring wrapper (loads checkpoint, runs MLP)
│   ├── extract.py             # Batch prefill extraction, PrefillResult, caching
│   ├── transforms.py          # StandardScaler + PCA pipelines (fit + apply)
│   ├── trunk.py               # SharedTrunkNet MLP, train_mlp, train_ensemble
│   ├── sweep.py               # Layer/mode/PCA grid search with ternary layer search
│   └── train.py               # Full prefill training pipeline
├── adapters/
│   ├── __init__.py            # "Platform adapters for model-router-toolkit."
│   ├── litellm/
│   │   ├── __init__.py        # Exports ModelRoutingStrategy
│   │   ├── strategy.py        # ModelRoutingStrategy (LiteLLM CustomRoutingStrategyBase)
│   │   ├── app.py             # Full-mode FastAPI app (routing + inference + UI)
│   │   ├── proxy.py           # LiteLLM Proxy strategy injection
│   │   ├── config_bridge.py   # Pool config → LiteLLM proxy config generator
│   │   ├── completions.py     # /v1/chat/completions endpoint
│   │   ├── chat.py            # /api/chat SSE endpoint
│   │   ├── review.py          # /api/review auto-judge endpoint
│   │   └── static/            # Playground UI (index.html, playground.js, playground.css)
│   └── http/
│       ├── __init__.py        # "HTTP adapter — router-only sidecar"
│       ├── app.py             # Router-only FastAPI app (no inference)
│       ├── route.py           # POST /v1/route endpoint (RouteRequest/RouteResponse)
│       ├── auth.py            # WebhookAuthMiddleware (HMAC-SHA256 + bearer token)
│       └── _shared.py         # Shared helpers: warmup_router, health_dict, models_list
└── plugins/
    └── openclaw/
        ├── index.ts           # before_model_resolve hook (TypeScript)
        ├── openclaw.plugin.json  # Plugin manifest + config schema
        └── package.json       # NPM package metadata
```

Key directories outside the package:
- `configs/` -- Pool config YAMLs (prefill-qwen08b, v1-9models, etc.)
- `data/` -- Training/test CSVs (gitignored, not tracked)
- `checkpoints/` -- Trained routing checkpoints (gitignored)
- `notebooks/` -- Quickstart notebook (prefill)
- `tests/` -- Unit tests; `tests/integration/` -- API integration tests
- `docs/` -- Architecture, integration, quickstart, configuration, adapters, plugins, extending, training, evaluation

## Tech Stack

- **Python 3.10+**, pydantic, numpy, scikit-learn, requests, PyYAML
- **[server]**: FastAPI, uvicorn
- **[litellm]**: litellm, FastAPI, uvicorn
- **[proxy]**: litellm[proxy], packaging
- **[prefill]**: torch, transformers, accelerate, tqdm
- **[training]**: litellm
- **Dev**: pytest, ruff, mypy

## Setup

```bash
pip install -e '.[prefill,litellm]'   # Recommended: prefill routing + serve
pip install -e '.[all]'              # Full development setup
```

## Key Import Patterns

```python
# Core (always available)
from model_router_toolkit.config import load_config, build_router_from_config, PoolConfig
from model_router_toolkit.router import BaseRouter, RoutingResult, CostEstimate, extract_user_text

# LiteLLM strategy (requires [litellm] extra)
from model_router_toolkit import ModelRoutingStrategy
# or directly:
from model_router_toolkit.adapters.litellm.strategy import ModelRoutingStrategy

# Config bridge (requires [litellm] or [proxy] extra)
from model_router_toolkit.adapters.litellm.config_bridge import generate_litellm_config

# HTTP route types (requires [server] extra)
from model_router_toolkit.adapters.http.route import RouteRequest, RouteResponse

# Webhook auth (requires [server] extra)
from model_router_toolkit.adapters.http.auth import WebhookAuthMiddleware
```

## CLI Reference

### Collect training data
```bash
model-router collect \
  --config configs/prefill-qwen08b.yaml \
  --questions questions.txt \
  --output data/train.csv \
  --judge vote                    # or: reference --references refs.csv
```
Runs every model in the pool on each question, judges correctness. Output CSV: `question, model, isCorrect, output_tokens`.

### Train a router
```bash
model-router train \
  --config configs/prefill-qwen08b.yaml \
  --data data/train.csv \
  --output-dir checkpoints/
```
Pipeline: load labels -> extract prefill features (Qwen3.5-0.8B) -> sweep layer/mode/PCA -> train SharedTrunkNet ensemble -> save .pt checkpoint.

Key options:
- `--device cpu|cuda|mps` -- compute device (auto-detected if omitted)
- `--n-seeds 10 --n-keep 5` -- ensemble size
- `--prefill-dir cache/` -- cache dir for extracted features (default: `cache/`, disable with `--no-cache`)
- `--pca-dims 50,100,200` -- PCA dimensions to sweep
- `--epochs 150 --patience 15` -- MLP training

### Evaluate a checkpoint
```bash
model-router evaluate \
  --config configs/prefill-qwen08b.yaml \
  --checkpoint checkpoints/prefill_router.pt \
  --data data/test.csv
```
Reports: per-model AUC/accuracy, oracle vs router accuracy, lift, headroom captured, routing distribution, agreement zones, near-miss analysis, pairwise confidence win rates.

### Serve (full mode — routing + inference + UI)
```bash
model-router serve --config configs/prefill-qwen08b.yaml --port 8000
```
OpenAI-compatible API at `/v1/chat/completions`. Playground UI at `/`.
Uses `adapters/litellm/app.py`.

### Serve router only (no inference)
```bash
model-router serve-router --config configs/prefill-qwen08b.yaml --port 8079
```
Route-only API at `/v1/route`. No API keys needed.
Uses `adapters/http/app.py`.

### LiteLLM Proxy
```bash
model-router proxy-config --config pool.yaml --output litellm.yaml
model-router proxy --litellm-config litellm.yaml --router-config pool.yaml --port 4000
```
Uses `adapters/litellm/proxy.py` and `adapters/litellm/config_bridge.py`.

### Configuration

| Config | Method | Provider | When to use |
|--------|--------|----------|-------------|
| `prefill-qwen08b.yaml` | Prefill | OpenRouter | Default — best accuracy |
| `v1-9models-qwen08b.yaml` | Prefill | OpenRouter | Full 9-model v1 pool |
| `local-prefill.yaml` | Prefill | Local | Air-gapped / local-only |

## Architecture

One routing method behind `BaseRouter`:
- **PrefillRouter**: encoder forward pass -> per-layer hidden states -> PCA -> SharedTrunkNet MLP -> P(correct) per model -> cheapest above tolerance

Training and evaluation bypass BaseRouter (batch extraction + direct trunk inference). Inference/serving uses BaseRouter.

**Adapter layer**: Platform integrations live in `adapters/` and `plugins/`. Core has no framework deps.
- `adapters/litellm/` — ModelRoutingStrategy, standalone server, proxy injection
- `adapters/http/` — Router-only sidecar, webhook auth middleware
- `plugins/openclaw/` — TypeScript plugin for OpenClaw gateway

## Config Format

```yaml
routing:
  method: prefill
  checkpoint: checkpoints/prefill_router.pt
  tolerance: 0.20              # accuracy-cost tradeoff
  encoder: Qwen/Qwen3.5-0.8B  # HF model for prefill extraction

models:
  - name: nem-think
    litellm_model: openrouter/nvidia/nemotron-3-nano-30b-a3b
    cost_per_m_input_tokens: 0.20
    cost_per_m_output_tokens: 0.20
    system_prompt: Think step-by-step before answering.
```

Training data CSV format: `question, model, isCorrect, output_tokens`.

## Code Style

- **ruff** for linting, **mypy** for type checking
- **Line length**: 100
- **Target**: Python 3.10 -- use `X | Y` union syntax
- **Imports**: sorted via ruff `I` rules
- **Lazy imports**: Heavy deps (torch, litellm, transformers) imported inside functions

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `NVIDIA_API_KEY` | API key for NVIDIA NIM |
| `OPENROUTER_API_KEY` | API key for OpenRouter |
| `OPENAI_API_KEY` | Fallback API key for OpenAI-compatible providers |
| `ROUTER_WEBHOOK_SECRET` | Shared secret for webhook HMAC/bearer auth |
| `CORS_ORIGINS` | Comma-separated CORS origins (default: `*`) |

## Boundaries

### Always do
- Run tests before considering work complete
- Follow BaseRouter abstraction for inference paths
- Config-driven dispatch (method in YAML determines behavior)
- Lazy imports for optional dependencies (torch, litellm, etc.)
- Import guards with helpful error messages for missing extras

### Ask first
- Changing pyproject.toml deps
- Modifying BaseRouter interface
- Adding new CLI subcommands
- Adding new adapters or plugins

### Never do
- Commit API keys
- Bypass the BaseRouter abstraction for inference
- Hardcode routing method selection
- Import heavy deps at module level (use lazy imports)

## Testing

```bash
pytest tests/ --ignore=tests/integration/ -v   # Unit tests
pytest tests/integration/ -v                    # Integration tests
pytest tests/ -v --cov=model_router_toolkit     # With coverage
```
