# Model Router Toolkit

LLM routing toolkit that learns which model handles which queries best, then routes each query to the cheapest model above an accuracy threshold. Prefill complexity-based routing (primary) and KMeans embedding-based routing behind a unified `BaseRouter` interface.

## Quickstart

```bash
pip install -e '.[prefill]'
export OPENROUTER_API_KEY=your-key    # needed for serving and data collection
model-router serve --config configs/prefill-qwen08b.yaml --port 8000
```

Open `http://localhost:8000/` for the interactive playground UI, or call the API:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "routed", "messages": [{"role": "user", "content": "Hello"}]}'
```

For notebooks, open `notebooks/quickstart.ipynb` (KMeans) or `notebooks/quickstart-prefill.ipynb` (prefill). These are standalone -- no package install required, just an `NVIDIA_API_KEY`.

## API Keys

| Variable | When needed | What for |
|----------|-------------|----------|
| `OPENROUTER_API_KEY` | Serving, collecting | Calls models via OpenRouter (default provider in configs) |
| `NVIDIA_API_KEY` | Notebooks, NVIDIA NIM configs | Calls models and embeddings via build.nvidia.com |

Set before running `serve`, `collect`, or the notebooks. **Not needed** for `train` or `evaluate` (these work offline with a local encoder + pre-collected CSV data).

```bash
export OPENROUTER_API_KEY=sk-or-...
# or
export NVIDIA_API_KEY=nvapi-...
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

No API key needed -- training uses a local encoder model (Qwen3.5-0.8B, downloaded from HuggingFace on first run).

```bash
model-router train \
  --config configs/prefill-qwen08b.yaml \
  --data data/train.csv \
  --output-dir checkpoints/
```

The pipeline: loads labels, extracts prefill hidden states via the encoder, sweeps layer/mode/PCA per target, trains a SharedTrunkNet MLP ensemble, saves a `.pt` checkpoint.

Key options: `--device cpu|cuda|mps`, `--n-seeds 10`, `--n-keep 5`, `--prefill-dir cache/` (cache extracted features), `--pca-dims 50,100,200`.

### 3. Evaluate

No API key needed -- evaluation works offline with the checkpoint and test CSV.

```bash
model-router evaluate \
  --config configs/prefill-qwen08b.yaml \
  --checkpoint checkpoints/prefill_router.pt \
  --data data/test.csv
```

Prints per-model AUC, oracle vs router accuracy, routing distribution, agreement zone analysis, near-miss diagnostics, and pairwise win rates.

### 4. Serve

Requires an API key for the model provider (see API Keys above).

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

See `configs/` for full examples and `configs/schema.md` for the config reference. See `docs/` for the [training guide](docs/training-guide.md), [evaluation guide](docs/evaluation-guide.md), [architecture](docs/architecture.md), [integration guide](docs/integration.md), and [model pool reference](docs/model-pool-reference.md).

## LiteLLM SDK Integration

```python
from litellm import Router
from model_router_toolkit import ModelRoutingStrategy

router = Router(model_list=my_models)
strategy = ModelRoutingStrategy.from_config("configs/prefill-qwen08b.yaml")
router.set_custom_routing_strategy(strategy)

response = await router.acompletion(
    model="nem-think",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Development

```bash
pip install -e '.[dev,prefill]'
```

### Running Tests

```bash
# Unit tests (no API keys or checkpoints needed)
pytest tests/ --ignore=tests/integration/ -v

# Integration tests (no API keys needed; tests use mocks)
pytest tests/integration/ -v

# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=model_router_toolkit --cov-report=term-missing
```

Note: Some KMeans unit tests are skipped if `checkpoints/kmeans_c100_db.pkl` is not present.
