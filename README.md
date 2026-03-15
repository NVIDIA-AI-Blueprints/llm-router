# Model Router Toolkit

LLM routing toolkit that learns which model handles which queries best, then routes each query to the cheapest model above an accuracy threshold. Prefill complexity-based routing via a lightweight encoder (Qwen3.5-0.8B) behind a unified `BaseRouter` interface.

## Quickstart

```bash
pip install -e '.[prefill,litellm]'

export OPENROUTER_API_KEY=your-key
model-router serve --config configs/v1-9models-qwen08b.yaml --port 8000
```

Open `http://localhost:8000/` for the interactive playground UI, or call the API:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "routed", "messages": [{"role": "user", "content": "Hello"}]}'
```

Or use the router as a Python library (no server, no API keys):

```python
from model_router_toolkit.config import load_config, build_router_from_config

config = load_config("configs/v1-9models-qwen08b.yaml")
router = build_router_from_config(config)

result = router.route("What is the capital of France?", tolerance=0.20)
print(result.selected_model, result.confidences)
```

For a notebook walkthrough, open `notebooks/quickstart-prefill.ipynb`.

## Install

Pick extras based on what you need:

| Extra | What it adds | When you need it |
|-------|-------------|-----------------|
| *(none)* | Core routing engine | Library use only |
| `[server]` | FastAPI, uvicorn | Router-only HTTP sidecar |
| `[litellm]` | litellm, FastAPI, uvicorn | Standalone server, LiteLLM SDK |
| `[proxy]` | litellm[proxy] | LiteLLM Proxy injection |
| `[prefill]` | torch, transformers | Prefill routing (recommended) |
| `[training]` | litellm | Data collection (`model-router collect`) |
| `[dev]` | pytest, ruff, mypy | Testing and linting |
| `[all]` | Everything | Full development setup |

```bash
pip install -e '.[prefill,litellm]'      # Recommended — prefill + serve
pip install -e '.[prefill,server]'       # Router sidecar only (no inference)
pip install -e '.[prefill,proxy]'        # LiteLLM Proxy mode
pip install -e '.[all]'                  # Everything for development
```

Or use the install script (also copies the AI assistant skill):

```bash
bash scripts/install.sh                          # pip install + skill
bash scripts/install.sh --cursor                 # skill → ~/.cursor/skills/
bash scripts/install.sh --extras 'dev,prefill'   # custom extras
```

## API Keys

| Variable | When needed | What for |
|----------|-------------|----------|
| `OPENROUTER_API_KEY` | Serving, collecting | Calls models via OpenRouter (default provider) |
| `NVIDIA_API_KEY` | Notebooks, NIM configs | Calls models and embeddings via build.nvidia.com |

**Not needed** for `train`, `evaluate`, or the direct Python library (these work offline).

## Adapters

Adapters connect the routing engine to different platforms. The core has no framework dependencies — adapters bring their own.

| Adapter | Location | Install | What it does |
|---------|----------|---------|-------------|
| **LiteLLM Strategy** | `adapters/litellm/strategy.py` | `[litellm]` | Embed routing in any `litellm.Router` — 4 lines of Python |
| **Standalone Server** | `adapters/litellm/app.py` | `[litellm]` | Full server with routing + inference + playground UI |
| **LiteLLM Proxy** | `adapters/litellm/proxy.py` | `[proxy]` | Inject routing into LiteLLM Proxy at startup |
| **Router Sidecar** | `adapters/http/app.py` | `[server]` | Route-only HTTP sidecar (POST /v1/route) |
| **Webhook Auth** | `adapters/http/auth.py` | `[server]` | HMAC-SHA256 / bearer token middleware |
| **OpenClaw Plugin** | `plugins/openclaw/` | — | TypeScript plugin for OpenClaw's before_model_resolve |

### LiteLLM SDK Integration

```python
from litellm import Router
from model_router_toolkit import ModelRoutingStrategy

router = Router(model_list=my_models)
strategy = ModelRoutingStrategy.from_config("configs/v1-9models-qwen08b.yaml")
strategy.set_litellm_router(router)
router.set_custom_routing_strategy(strategy)

response = await router.acompletion(
    model="nem-think",
    messages=[{"role": "user", "content": "Hello"}],
)
```

### Router-Only Sidecar

No inference, no API keys — just routing decisions via HTTP:

```bash
model-router serve-router --config configs/v1-9models-qwen08b.yaml --port 8079
```

```bash
curl -X POST http://localhost:8079/v1/route \
  -H "Content-Type: application/json" \
  -d '{"question": "What is 2+2?", "tolerance": 0.20}'
```

See the [adapters guide](docs/adapters.md) for details on all adapters, and the [plugins guide](docs/plugins.md) for gateway plugins.

## Workflow: Collect, Train, Evaluate, Serve

### 1. Collect labeled data

```bash
model-router collect \
  --config configs/v1-9models-qwen08b.yaml \
  --questions questions.txt \
  --output data/collected.csv \
  --judge vote
```

### 2. Train a router

```bash
model-router train \
  --config configs/v1-9models-qwen08b.yaml \
  --data data/train.csv \
  --output-dir checkpoints/
```

Key options: `--device cpu|cuda|mps`, `--n-seeds 10`, `--n-keep 5`, `--prefill-dir cache/`, `--pca-dims 50,100,200`.

### 3. Evaluate

```bash
model-router evaluate \
  --config configs/v1-9models-qwen08b.yaml \
  --checkpoint checkpoints/prefill_router_qwen08b.pt \
  --data data/test.csv
```

### 4. Serve

**Standalone server** (playground UI — best for demos):

```bash
model-router serve --config configs/v1-9models-qwen08b.yaml --port 8000
```

**LiteLLM Proxy** (production — auth, rate limiting, spend tracking):

```bash
model-router proxy-config --config configs/v1-9models-qwen08b.yaml --output configs/litellm-proxy.yaml
model-router proxy \
    --litellm-config configs/litellm-proxy.yaml \
    --router-config configs/v1-9models-qwen08b.yaml \
    --port 4000
```

**Router sidecar** (route-only, no inference):

```bash
model-router serve-router --config configs/v1-9models-qwen08b.yaml --port 8079
```

## Config

```yaml
routing:
  method: prefill
  checkpoint: checkpoints/prefill_router_qwen08b.pt
  tolerance: 0.20
  encoder: Qwen/Qwen3.5-0.8B

models:
  - name: nem-think
    litellm_model: openrouter/nvidia/nemotron-3-nano-30b-a3b
    cost_per_m_input_tokens: 0.20
    cost_per_m_output_tokens: 0.20
  - name: nem-nothink
    litellm_model: openrouter/nvidia/nemotron-3-nano-30b-a3b
    cost_per_m_input_tokens: 0.04
    cost_per_m_output_tokens: 0.16
```

See `configs/` for starter configs and [docs/configuration.md](docs/configuration.md) for the full schema reference.

## Documentation

| Doc | What it covers |
|-----|---------------|
| [Quickstart](docs/quickstart.md) | 5-minute getting started |
| [Architecture](docs/architecture.md) | Class hierarchy, inference flow, training pipeline, deployment topologies |
| [Configuration](docs/configuration.md) | Full YAML schema, environment variables, annotated examples |
| [Integration Guide](docs/integration.md) | All 7 integration paths with code examples |
| [Adapters Guide](docs/adapters.md) | Using bundled adapters, writing custom adapters, API reference |
| [Plugins Guide](docs/plugins.md) | OpenClaw plugin, writing gateway plugins, /v1/route API reference |
| [Extending Guide](docs/extending.md) | Custom routing methods, adding adapters, contributing |
| [Training Guide](docs/training-guide.md) | Data collection, training pipeline, hyperparameters |
| [Evaluation Guide](docs/evaluation-guide.md) | Metrics, interpretation, comparison workflow |
| [Model Pool Reference](docs/model-pool-reference.md) | Provider setup, model configurations |

## AI Assistant Skill

The `skills/model-router-toolkit/` directory contains a comprehensive AI skill that teaches Claude Code and Cursor how to use every `model-router` CLI command.

```bash
bash scripts/install.sh                  # default: pip install + skill → ~/.claude/skills/
bash scripts/install.sh --cursor         # skill → ~/.cursor/skills/
bash scripts/install.sh --skip-skill     # pip install only
```

## Development

```bash
pip install -e '.[all]'
```

```bash
# Unit tests
pytest tests/ --ignore=tests/integration/ -v

# Integration tests
pytest tests/integration/ -v

# All tests with coverage
pytest tests/ -v --cov=model_router_toolkit --cov-report=term-missing

# Lint and format
ruff check src/ tests/
ruff format src/ tests/
mypy src/
```
