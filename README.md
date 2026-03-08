# Model Router Toolkit

LLM routing toolkit that learns which model handles which queries best, routes to the most cost-efficient model above an accuracy threshold. Two routing methods (KMeans embedding-based and prefill complexity-based) behind a unified interface. Primary integration via LiteLLM's `set_custom_routing_strategy()`.

## Quick Start

- **Quickstart (5 min)**: Open `notebooks/quickstart.ipynb`, set NVIDIA API key, run cells.
- **Deploy a Router (30 min)**: `pip install -e .` then `model-router setup` then `model-router serve`.
- **Train Your Own (hours)**: `model-router collect` then `model-router train` then `model-router evaluate`.

## LiteLLM Integration

```python
from model_router_toolkit import ModelRoutingStrategy

litellm.set_custom_routing_strategy(ModelRoutingStrategy())
response = litellm.completion(model="model-router/default", messages=[...])
```

## Model Pool

| Model | Description |
|-------|-------------|
| Nemotron 3 Nano Think | Small, fast |
| Nemotron 3 Super | Large, capable |
| GPT-OSS 20B | Open-source 20B |
| GPT-OSS 120B | Open-source 120B |
| Qwen 3.5 122B | Qwen 3.5 122B |
| GPT-5.2 | GPT-5.2 |
| Claude Opus 4.6 | Claude Opus 4.6 |

## Development

```bash
pip install -e '.[dev]'
pytest tests/ -v
```

## License

[Placeholder]
