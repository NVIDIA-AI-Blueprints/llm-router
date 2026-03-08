# Connect Apps to the Router

Trigger: "connect to app", "integrate", "openai sdk", "litellm", "api client"

## OpenAI SDK (any client)

Point any OpenAI-compatible client at the router server:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="routed",
    messages=[{"role": "user", "content": "Hello"}]
)
```

## Environment variable (for tools like OpenClaw, OpenCode, etc.)

```bash
export OPENAI_API_BASE=http://localhost:8000/v1
```

## LiteLLM SDK integration (no server needed)

```python
import litellm
from model_router_toolkit import ModelRoutingStrategy

strategy = ModelRoutingStrategy.from_config("configs/prefill-qwen08b.yaml")
litellm.set_custom_routing_strategy(strategy)
response = litellm.completion(model="model-router/default", messages=[...])
```

## Python library (direct)

```python
from model_router_toolkit.config import load_config, build_router_from_config

config = load_config("configs/prefill-qwen08b.yaml")
router = build_router_from_config(config)
result = router.route("What is the capital of France?", tolerance=0.20)
print(result.selected_model)       # cheapest model above threshold
print(result.confidences)          # P(correct) per model
print(result.selected_cost)        # estimated cost
```
