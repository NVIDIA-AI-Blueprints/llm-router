# AGENTS.md — Model Router Toolkit

Model Router Toolkit is a Python library for intelligent LLM routing. It learns which model handles which types of queries best, then routes each query to the most cost-efficient model above an accuracy threshold. Two routing methods (KMeans embedding-based and prefill complexity-based) behind a unified BaseRouter interface. Primary integration via LiteLLM's set_custom_routing_strategy(). Includes a CLI, FastAPI server with UI, and training/evaluation pipeline.

## Project Structure

```
src/model_router_toolkit/
├── __init__.py                # Public API exports
├── __main__.py                # CLI entry point (model-router)
├── config.py                  # PoolConfig, ModelSpec, pydantic validation
├── router.py                  # BaseRouter ABC, RoutingResult, CostEstimate
├── strategy.py                # ModelRoutingStrategy (LiteLLM integration)
├── checkpoint.py              # Checkpoint load/save (pkl + pt)
├── gpu.py                     # GPU detection + VRAM checks
├── train.py                   # Unified training dispatcher
├── evaluate.py                # Unified evaluation + metrics
├── collect.py                 # Data collection (run models + judge)
├── setup_wizard.py            # Interactive setup CLI
├── telemetry.py               # SQLite session/chat logging
├── kmeans/
│   ├── router.py              # KMeansRouter(BaseRouter)
│   ├── embed.py               # Embedding client (API + local)
│   └── train.py               # KMeans training pipeline
├── prefill/
│   ├── router.py              # PrefillRouter(BaseRouter)
│   ├── scorer.py              # Prefill scoring wrapper
│   └── train.py               # Prefill training pipeline
└── server/
    ├── app.py                 # FastAPI app factory
    ├── chat.py                # /api/chat SSE endpoint
    ├── completions.py         # /v1/chat/completions (OpenAI-compat)
    └── static/                # UI assets
```

Key directories outside the package:
- `configs/` — Example pool_config.yaml files
- `notebooks/` — Quickstart + advanced walkthrough
- `checkpoints/` — Pre-trained routing models
- `tests/` — Unit tests; `tests/integration/` — API integration tests
- `docs/` — Architecture, training guide, evaluation guide, model reference

## Tech Stack

- **Python 3.10+**, pydantic, numpy, scikit-learn, requests, litellm, FastAPI + uvicorn, PyYAML
- **Dev**: pytest, ruff, mypy
- **Prefill extra**: torch, transformers, accelerate

## Setup

```bash
pip install -e .           # Core (kmeans routing)
pip install -e '.[dev]'    # + testing tools
pip install -e '.[prefill]' # + GPU prefill routing
```

## CLI Reference

```bash
model-router setup                    # Interactive environment setup
model-router serve --config X --port 8000  # Start server
model-router train --config X --data Y --output-dir Z  # Train router
model-router evaluate --config X --checkpoint Y --data Z  # Evaluate
model-router collect --config X --questions Y --output Z --judge vote  # Collect data
```

## Architecture

BaseRouter abstraction. Two implementations: KMeansRouter (embed -> cluster -> Platt -> select) and PrefillRouter (prefill -> MLP -> P(correct) -> select). ModelRoutingStrategy wraps any BaseRouter and implements LiteLLM's CustomRoutingStrategyBase. Config determines which method runs. The server creates a litellm.Router internally and calls set_custom_routing_strategy().

## Code Style

- **ruff** for linting, **mypy** for type checking
- **Line length**: 100
- **Target**: Python 3.10 — use `X | Y` union syntax
- **Imports**: sorted via ruff `I` rules

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `NVIDIA_API_KEY` | API key for NVIDIA NIM |
| `OPENROUTER_API_KEY` | API key for OpenRouter |

## Boundaries

### Always do
- Run tests before considering work complete
- Follow BaseRouter abstraction
- Config-driven dispatch

### Ask first
- Changing pyproject.toml deps
- Modifying BaseRouter interface
- Adding new CLI subcommands

### Never do
- Commit API keys
- Bypass the BaseRouter abstraction
- Hardcode routing method selection

## Testing

```bash
pytest tests/ --ignore=tests/integration/ -v   # Unit tests
pytest tests/integration/ -v                    # Integration (needs API keys)
```
