# Training Guide

How to train a prefill router from scratch: collect labeled data, train a checkpoint, evaluate it, and deploy.

## Prerequisites

```bash
pip install -e '.[prefill]'
```

You need:
- A pool config YAML defining your models and encoder (see `configs/prefill-qwen08b.yaml`)
- A set of evaluation questions (500+ recommended)

### API Keys

| Step | API key needed? | Why |
|------|----------------|-----|
| **Collect** | Yes | Calls each model in the pool via LiteLLM |
| **Train** | No | Uses local encoder (Qwen3.5-0.8B, auto-downloaded from HuggingFace) |
| **Evaluate** | No | Runs offline with checkpoint + test CSV |
| **Serve** | Yes | Routes to model providers at inference time |

Set before collecting or serving:
```bash
export OPENROUTER_API_KEY=sk-or-...   # for OpenRouter configs
# or
export NVIDIA_API_KEY=nvapi-...       # for NVIDIA NIM configs
```

## 1. Collect Labeled Data

Requires an API key (see Prerequisites). Prepare a `questions.txt` file with one question per line. Then run every model in the pool on each question:

```bash
model-router collect \
  --config configs/prefill-qwen08b.yaml \
  --questions questions.txt \
  --output data/collected.csv \
  --judge vote
```

This calls each model via LiteLLM and judges correctness by majority vote. Output CSV format:

| Column | Description |
|--------|-------------|
| `question` | The question text |
| `model` | Target model name (matches config) |
| `isCorrect` | 1 if correct, 0 otherwise |
| `output_tokens` | Number of output tokens used |

For reference-based judging (when you have ground truth answers):

```bash
model-router collect \
  --config configs/prefill-qwen08b.yaml \
  --questions questions.txt \
  --output data/collected.csv \
  --judge reference --references answers.csv
```

The references CSV needs columns: `question`, `answer`.

### Splitting Data

Split the collected data into training and test sets (80/20). Ensure the same question doesn't appear in both:

```python
import csv
from collections import defaultdict

rows_by_q = defaultdict(list)
with open("data/collected.csv") as f:
    for row in csv.DictReader(f):
        rows_by_q[row["question"]].append(row)

questions = list(rows_by_q.keys())
split = int(len(questions) * 0.8)
train_qs, test_qs = set(questions[:split]), set(questions[split:])

for name, qs in [("train.csv", train_qs), ("test.csv", test_qs)]:
    with open(f"data/{name}", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question", "model", "isCorrect", "output_tokens"])
        w.writeheader()
        for q in qs:
            w.writerows(rows_by_q[q])
```

## 2. Train

```bash
model-router train \
  --config configs/prefill-qwen08b.yaml \
  --data data/train.csv \
  --output-dir checkpoints/
```

### What Happens

The pipeline has 6 steps:

1. **Load labels**: Reads CSV, normalizes questions, computes per-model accuracy and output token statistics
2. **Extract prefill**: Loads the encoder model (e.g., Qwen3.5-0.8B), runs a forward pass per question, extracts per-layer hidden states
3. **Sweep**: For each target model, searches over layer (ternary search), pooling mode (last-token vs mean), and PCA dimension to find the best configuration by 5-fold CV AUC
4. **Fit transforms**: Fits StandardScaler + PCA on training data for each target's best configuration
5. **Train trunk**: Trains a SharedTrunkNet MLP ensemble (10 seeds, keeps best 5) with BCEWithLogitsLoss and early stopping
6. **Save checkpoint**: Writes a self-contained `.pt` file

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | auto | `cpu`, `cuda`, or `mps` |
| `--n-seeds` | 10 | Number of ensemble seeds to train |
| `--n-keep` | 5 | Number of best seeds to keep |
| `--prefill-dir` | none | Directory to cache extracted prefill features |
| `--pca-dims` | 50,100,150,200,300 | PCA dimensions to sweep (comma-separated) |
| `--epochs` | 150 | Maximum MLP training epochs |
| `--patience` | 15 | Early stopping patience |
| `--batch-size` | 4 | Encoder extraction batch size |

### Caching Prefill Features

Extraction is the slowest step (~5s/question on CPU). Use `--prefill-dir` to cache features:

```bash
model-router train \
  --config configs/prefill-qwen08b.yaml \
  --data data/train.csv \
  --output-dir checkpoints/ \
  --prefill-dir cache/
```

Subsequent runs with the same encoder and questions skip extraction entirely.

### Quick Smoke Test

Verify the pipeline works before running on full data:

```bash
model-router train \
  --config configs/smoke-test.yaml \
  --data data/smoke-train.csv \
  --output-dir checkpoints/smoke/ \
  --n-seeds 2 --n-keep 1 --device cpu \
  --epochs 5 --patience 3 --pca-dims 10,20
```

## 3. Evaluate

See [Evaluation Guide](evaluation-guide.md) for details.

```bash
model-router evaluate \
  --config configs/prefill-qwen08b.yaml \
  --checkpoint checkpoints/prefill_router.pt \
  --data data/test.csv
```

## 4. Deploy

Update your config to point to the new checkpoint:

```yaml
routing:
  checkpoint: checkpoints/prefill_router.pt
```

Start the server:

```bash
model-router serve --config configs/prefill-qwen08b.yaml --port 8000
```

The trained checkpoint is self-contained -- it includes the pool config, transforms, and MLP weights. The server loads the encoder model on first request, then caches it for subsequent calls.

## Config Reference

```yaml
routing:
  method: prefill                    # routing method
  checkpoint: checkpoints/router.pt  # trained checkpoint path
  tolerance: 0.20                    # accuracy-cost tradeoff (0.0 = always best, 0.20 = allow 20pp drop for cheaper model)
  encoder: Qwen/Qwen3.5-0.8B        # HuggingFace encoder for prefill extraction

models:
  - name: nem-think                  # name (must match CSV model column)
    litellm_model: openrouter/nvidia/nemotron-3-nano-30b-a3b  # LiteLLM model ID
    cost_per_m_input_tokens: 0.20    # cost per million input tokens
    cost_per_m_output_tokens: 0.20   # cost per million output tokens
    system_prompt: "Think step-by-step."  # optional system prompt
    chat_template_kwargs:            # optional template kwargs
      enable_thinking: true
```
