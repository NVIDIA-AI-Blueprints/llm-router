# Quickstart

Get intelligent LLM routing running in 5 minutes.

## Install

```bash
pip install -e '.[prefill,litellm]'
```

Pick extras based on what you need:

| Extra | What it adds | When you need it |
|-------|-------------|-----------------|
| *(none)* | Core routing engine | Library use only, KMeans routing |
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

## Route in Python (3 lines)

```python
from model_router_toolkit.config import load_config, build_router_from_config

config = load_config("configs/prefill-qwen08b.yaml")
router = build_router_from_config(config)

result = router.route("What is the capital of France?", tolerance=0.20)
print(f"Selected: {result.selected_model}")
print(f"Confidences: {dict(zip(result.model_names, result.confidences))}")
```

No API keys needed — this runs the encoder locally and returns a routing decision.

## Start the HTTP Sidecar

```bash
model-router serve-router --config configs/prefill-qwen08b.yaml --port 8079
```

Query it:

```bash
curl -X POST http://localhost:8079/v1/route \
  -H "Content-Type: application/json" \
  -d '{"question": "What is 2+2?", "tolerance": 0.20}'
```

```json
{
  "selected_model": "nem-nothink",
  "model_names": ["nem-think", "nem-nothink"],
  "confidences": {"nem-think": 0.92, "nem-nothink": 0.88},
  "metadata": {"p_max": 0.92, "threshold": 0.72, "route_ms": 45.2}
}
```

## Start the Full Server (with inference)

```bash
export OPENROUTER_API_KEY=sk-or-...
model-router serve --config configs/prefill-qwen08b.yaml --port 8000
```

Call the OpenAI-compatible API:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "routed", "messages": [{"role": "user", "content": "Hello"}]}'
```

Open `http://localhost:8000/` for the interactive playground UI.

## Embed in Your App (LiteLLM SDK)

```python
from litellm import Router
from model_router_toolkit import ModelRoutingStrategy

router = Router(model_list=[
    {"model_name": "nem-think", "litellm_params": {"model": "openrouter/nvidia/nemotron-3-nano-30b-a3b"}},
    {"model_name": "nem-nothink", "litellm_params": {"model": "openrouter/nvidia/nemotron-3-nano-30b-a3b"}},
])
strategy = ModelRoutingStrategy.from_config("configs/prefill-qwen08b.yaml")
strategy.set_litellm_router(router)
router.set_custom_routing_strategy(strategy)

response = await router.acompletion(
    model="nem-think",
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
