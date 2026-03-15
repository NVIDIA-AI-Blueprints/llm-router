# Quickstart

Get intelligent LLM routing running locally.

## Install

```bash
pip install -e '.[prefill,litellm]'
```

Pick extras based on what you need:

| Extra | What it adds | When you need it |
|-------|-------------|-----------------|
| *(none)* | Core routing engine | Library use only |
| `[server]` | FastAPI, uvicorn | Router-only HTTP sidecar |
| `[litellm]` | litellm, FastAPI, uvicorn | Standalone server, LiteLLM SDK integration |
| `[proxy]` | litellm[proxy] | LiteLLM Proxy injection |
| `[prefill]` | torch, transformers | Prefill routing method (recommended) |
| `[training]` | litellm | Data collection (`model-router collect`) |
| `[dev]` | pytest, ruff, mypy | Development and testing |
| `[all]` | Everything | Full development setup |

Common combos:

```bash
pip install -e '.[prefill,litellm]'      # Recommended — prefill routing + serve
pip install -e '.[prefill,server]'       # Router sidecar only (no inference)
pip install -e '.[prefill,proxy]'        # LiteLLM Proxy mode
pip install -e '.[all]'                  # Everything for development
```

## Reproduce the v1 Checkpoint

No pre-trained checkpoint is included in the repository. Before routing, you need to
reproduce the v1 checkpoint from the provided training data.

The v1 dataset covers three benchmarks — **MMLU Pro**, **LiveCodeBench**, and
**Humanity's Last Exam** — evaluated across a 9-model pool ranging from Nemotron 3 Nano
($0.05/M input) to Claude Opus 4.6 ($2.77/M input).

### Data prerequisites

Training requires label CSVs and pre-extracted prefill features. Two modes are available:

| Mode | Required files | Size |
|------|---------------|------|
| **Lean** (recommended) | `data/train_v1.csv`, `data/test_v1.csv`, `data/v1-9models-lean/train_features.pt`, `data/v1-9models-lean/test_features.pt` | ~90 MB |
| **Full** (sweep + train) | `data/train_v1.csv`, `data/test_v1.csv`, `data/v1-9models-pool/train.pt`, `data/v1-9models-pool/test.pt` | ~2.4 GB |

### Train the checkpoint

Lean mode uses pre-transformed features and runs in ~30 seconds on CPU:

```bash
./scripts/reproduce_v1_checkpoint.sh --lean
```

Full mode runs the hyperparameter sweep (layer, pooling mode, PCA dimension) before training:

```bash
./scripts/reproduce_v1_checkpoint.sh
```

Both produce `checkpoints/prefill_router.pt`. The script also runs evaluation on the
held-out test set so you can verify the checkpoint before using it.

## Route in Python (3 lines)

```python
from model_router_toolkit.config import load_config, build_router_from_config

config = load_config("configs/v1-9models.yaml")
router = build_router_from_config(config)

result = router.route("What is the capital of France?", tolerance=0.20)
print(f"Selected: {result.selected_model}")
print(f"Confidences: {dict(zip(result.model_names, result.confidences))}")
```

No API keys needed — this runs the encoder locally and returns a routing decision.

## Start the HTTP Sidecar

```bash
model-router serve-router --config configs/v1-9models.yaml --port 8079
```

Query it:

```bash
curl -X POST http://localhost:8079/v1/route \
  -H "Content-Type: application/json" \
  -d '{"question": "What is 2+2?", "tolerance": 0.20}'
```

```json
{
  "selected_model": "nemotron-3-nano-reasoning",
  "model_names": [
    "nemotron-3-nano-reasoning", "gpt-oss-20b-high", "nemotron-3-super",
    "gpt-oss-120b-high", "qwen-3-5-35b", "qwen-3-5-122b",
    "gpt-5-2-high", "gpt-5-4-high", "claude-opus-4-6-high"
  ],
  "confidences": {
    "nemotron-3-nano-reasoning": 0.91, "gpt-oss-20b-high": 0.87,
    "nemotron-3-super": 0.93, "gpt-oss-120b-high": 0.90,
    "qwen-3-5-35b": 0.94, "qwen-3-5-122b": 0.95,
    "gpt-5-2-high": 0.96, "gpt-5-4-high": 0.97,
    "claude-opus-4-6-high": 0.98
  },
  "metadata": {"p_max": 0.98, "threshold": 0.78, "route_ms": 85.4}
}
```

## Start the Full Server (with inference)

Full-server mode routes **and** calls the selected model. This requires each model in
your config to have a `litellm_model` field and a valid API key. The `v1-9models.yaml`
config includes cost fields only — add `litellm_model` entries for the models you want
to serve, or see `configs/prefill-qwen08b.yaml` for a ready-to-serve example. See
[Configuration](configuration.md) for the full field reference.

```bash
export OPENROUTER_API_KEY=sk-or-...
model-router serve --config configs/v1-9models.yaml --port 8000
```

Call the OpenAI-compatible API:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "routed", "messages": [{"role": "user", "content": "Hello"}]}'
```

Open `http://localhost:8000/` for the interactive playground UI.

## Embed in Your App (LiteLLM SDK)

Each model in `model_list` needs a `litellm_params.model` matching a LiteLLM provider
string. The example below shows two of the nine v1 pool models — add as many as you
want to serve.

```python
from litellm import Router
from model_router_toolkit import ModelRoutingStrategy

router = Router(model_list=[
    {"model_name": "nemotron-3-nano-reasoning", "litellm_params": {"model": "openrouter/nvidia/nemotron-3-nano-30b-a3b"}},
    {"model_name": "nemotron-3-super", "litellm_params": {"model": "openrouter/nvidia/nemotron-3-super-49b-v1"}},
    {"model_name": "gpt-5-2-high", "litellm_params": {"model": "openrouter/openai/gpt-5.2"}},
])
strategy = ModelRoutingStrategy.from_config("configs/v1-9models.yaml")
strategy.set_litellm_router(router)
router.set_custom_routing_strategy(strategy)

response = await router.acompletion(
    model="nemotron-3-nano-reasoning",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## What's Next?

| Topic | Doc |
|-------|-----|
| Full config reference (every field, type, default) | [Configuration](configuration.md) |
| All integration paths with code examples | [Integration Guide](integration.md) |
| Bundled adapters and writing custom ones | [Adapters Guide](adapters.md) |
| OpenClaw plugin and writing gateway plugins | [Plugins Guide](plugins.md) |
| Custom routing methods and contributing | [Extending Guide](extending.md) |
| Architecture, inference flow, training pipeline | [Architecture](architecture.md) |
| Collect, train, evaluate workflow | [Training Guide](training-guide.md) |
| Evaluation metrics and interpretation | [Evaluation Guide](evaluation-guide.md) |
