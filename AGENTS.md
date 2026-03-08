# AGENTS.md -- Model Router Toolkit

LLM routing toolkit. Learns which model handles which queries best, routes to the cheapest model above an accuracy threshold. Prefill complexity-based routing (primary, via Qwen3.5-0.8B encoder) and KMeans embedding-based routing behind a unified BaseRouter interface. Full collect/train/evaluate/serve pipeline via CLI.

## Project Structure

```
src/model_router_toolkit/
├── __init__.py                # Public API exports
├── __main__.py                # CLI entry point (model-router)
├── config.py                  # PoolConfig, ModelSpec, RoutingConfig (pydantic)
├── router.py                  # BaseRouter ABC, RoutingResult, CostEstimate
├── strategy.py                # ModelRoutingStrategy (LiteLLM integration)
├── checkpoint.py              # Checkpoint load/save (pkl + pt)
├── gpu.py                     # GPU detection + VRAM checks
├── train.py                   # Unified training dispatcher
├── evaluate.py                # Unified evaluation (prefill: rich metrics, kmeans: basic)
├── collect.py                 # Data collection (run models + judge correctness)
├── setup_wizard.py            # Interactive setup CLI
├── telemetry.py               # SQLite session/chat logging
├── kmeans/
│   ├── router.py              # KMeansRouter(BaseRouter)
│   ├── embed.py               # Embedding client (API + local)
│   └── train.py               # KMeans training (NotImplementedError stub)
├── prefill/
│   ├── router.py              # PrefillRouter(BaseRouter) -- inference path
│   ├── scorer.py              # Prefill scoring wrapper (loads checkpoint, runs MLP)
│   ├── extract.py             # Batch prefill extraction, PrefillResult, caching
│   ├── transforms.py          # StandardScaler + PCA pipelines (fit + apply)
│   ├── trunk.py               # SharedTrunkNet MLP, train_mlp, train_ensemble
│   ├── sweep.py               # Layer/mode/PCA grid search with ternary layer search
│   └── train.py               # Full prefill training pipeline
└── server/
    ├── app.py                 # FastAPI app factory
    ├── chat.py                # /api/chat SSE endpoint
    ├── completions.py         # /v1/chat/completions (OpenAI-compat)
    └── static/                # Playground UI assets
```

Key directories outside the package:
- `configs/` -- Pool config YAMLs (prefill-qwen08b, smoke-test, cloud-only, etc.)
- `data/` -- Training/test CSVs (gitignored, not tracked)
- `checkpoints/` -- Trained routing checkpoints (gitignored)
- `notebooks/` -- Quickstart notebooks (kmeans + prefill)
- `tests/` -- Unit tests; `tests/integration/` -- API integration tests
- `docs/` -- Architecture, training guide, evaluation guide

## Tech Stack

- **Python 3.10+**, pydantic, numpy, scikit-learn, requests, litellm, FastAPI + uvicorn, PyYAML
- **Dev**: pytest, ruff, mypy
- **Prefill extra**: torch, transformers, accelerate, tqdm

## Setup

```bash
pip install -e '.[prefill]'   # Prefill routing (recommended)
pip install -e '.[dev]'       # + testing tools
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
- `--prefill-dir cache/` -- cache extracted features to disk
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

Options: `--device`, `--batch-size`, `--prefill-dir` (same as train).

### Serve
```bash
model-router serve --config configs/prefill-qwen08b.yaml --port 8000
```
OpenAI-compatible API at `/v1/chat/completions`. Playground UI at `/`.

### Setup wizard
```bash
model-router setup
```
Interactive: detects GPU, asks for routing method and API keys, generates config YAML.

## Architecture

Two routing methods behind `BaseRouter`:
- **PrefillRouter**: encoder forward pass -> per-layer hidden states -> PCA -> SharedTrunkNet MLP -> P(correct) per model -> cheapest above tolerance
- **KMeansRouter**: embed question -> cluster assignment -> Platt calibration -> P(correct) per model -> cheapest above tolerance

Training and evaluation bypass BaseRouter (batch extraction + direct trunk inference). Inference/serving uses BaseRouter.

`ModelRoutingStrategy` wraps any BaseRouter and implements LiteLLM's `CustomRoutingStrategyBase`. Config determines which method runs.

## Config Format

```yaml
routing:
  method: prefill              # or kmeans
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

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `NVIDIA_API_KEY` | API key for NVIDIA NIM |
| `OPENROUTER_API_KEY` | API key for OpenRouter |

## Boundaries

### Always do
- Run tests before considering work complete
- Follow BaseRouter abstraction for inference paths
- Config-driven dispatch (method in YAML determines behavior)

### Ask first
- Changing pyproject.toml deps
- Modifying BaseRouter interface
- Adding new CLI subcommands

### Never do
- Commit API keys
- Bypass the BaseRouter abstraction for inference
- Hardcode routing method selection

## Testing

```bash
pytest tests/ --ignore=tests/integration/ -v   # Unit tests
pytest tests/integration/ -v                    # Integration (needs API keys)
```
