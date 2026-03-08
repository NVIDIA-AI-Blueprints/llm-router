# Connect Apps to the Router

Trigger: "connect to app", "integrate", "openclaw", "opencode", "openai sdk"

## OpenAI SDK (any client)
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="routed",
    messages=[{"role": "user", "content": "Hello"}]
)
```

## OpenClaw
Set environment: OPENAI_API_BASE=http://localhost:8000/v1

## OpenCode  
Set environment: OPENAI_BASE_URL=http://localhost:8000/v1

## Existing LiteLLM Setup (SDK)
```python
from litellm import Router
from model_router_toolkit import ModelRoutingStrategy

router = Router(model_list=your_models)
strategy = ModelRoutingStrategy.from_config("pool_config.yaml")
router.set_custom_routing_strategy(strategy)
```

## Existing LiteLLM Proxy (YAML)
Not yet supported via YAML config. Use the SDK integration above.
