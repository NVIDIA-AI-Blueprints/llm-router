# Model Router Toolkit

LLM routing toolkit that learns which model handles which queries best, then routes each query to the cheapest model above an accuracy threshold. Prefill complexity-based routing (primary) and KMeans embedding-based routing behind a unified `BaseRouter` interface.

## Install

```bash
pip install -e '.[prefill]'   # Prefill routing (torch + transformers)
pip install -e .               # Core only (kmeans routing)
```

## Workflow: Collect, Train, Evaluate, Serve

### 1. Collect labeled data

Prepare a questions file (one per line), then run every model in the pool and judge correctness:

```bash
model-router collect \
  --config configs/prefill-qwen08b.yaml \
  --questions questions.txt \
  --output data/collected.csv \
  --judge vote
```

Output CSV has columns: `question, model, isCorrect, output_tokens`. Split into train/test sets.

### 2. Train a router

```bash
model-router train \
  --config configs/prefill-qwen08b.yaml \
  --data data/train.csv \
  --output-dir checkpoints/
```

The pipeline: loads labels, extracts prefill hidden states via the encoder model (Qwen3.5-0.8B), sweeps layer/mode/PCA per target, trains a SharedTrunkNet MLP ensemble, saves a `.pt` checkpoint.

Key options: `--device cpu|cuda|mps`, `--n-seeds 10`, `--n-keep 5`, `--prefill-dir cache/` (cache extracted features), `--pca-dims 50,100,200`.

### 3. Evaluate

```bash
model-router evaluate \
  --config configs/prefill-qwen08b.yaml \
  --checkpoint checkpoints/prefill_router.pt \
  --data data/test.csv
```

Prints per-model AUC, oracle vs router accuracy, routing distribution, agreement zone analysis, near-miss diagnostics, and pairwise win rates.

### 4. Serve

```bash
model-router serve --config configs/prefill-qwen08b.yaml --port 8000
```

OpenAI-compatible API at `http://localhost:8000/v1/chat/completions`. Interactive playground at `http://localhost:8000/`.

## Config

```yaml
routing:
  method: prefill
  checkpoint: checkpoints/prefill_router.pt
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

See `configs/` for full examples.

## LiteLLM SDK Integration

```python
from model_router_toolkit import ModelRoutingStrategy

strategy = ModelRoutingStrategy.from_config("configs/prefill-qwen08b.yaml")
litellm.set_custom_routing_strategy(strategy)
response = litellm.completion(model="model-router/default", messages=[...])
```

## Development

```bash
pip install -e '.[dev,prefill]'
pytest tests/ --ignore=tests/integration/ -v
```
