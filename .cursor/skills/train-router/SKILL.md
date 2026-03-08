# Train a Custom Router

Trigger: "train router", "customize router", "collect training data", "model-router train", "evaluate router"

## End-to-End Workflow

### 1. Collect labeled data

Prepare a `questions.txt` file with one question per line (500+ recommended for good results).

```bash
model-router collect \
  --config configs/prefill-qwen08b.yaml \
  --questions questions.txt \
  --output data/collected.csv \
  --judge vote
```

Each model in the pool answers every question. Majority vote determines correctness. Output CSV: `question, model, isCorrect, output_tokens`.

For reference-based judging (when you have ground truth):
```bash
model-router collect \
  --config configs/prefill-qwen08b.yaml \
  --questions questions.txt \
  --output data/collected.csv \
  --judge reference --references answers.csv
```

Split the output: 80% `train.csv`, 20% `test.csv`.

### 2. Train

```bash
model-router train \
  --config configs/prefill-qwen08b.yaml \
  --data data/train.csv \
  --output-dir checkpoints/custom/
```

What happens: loads labels, extracts prefill features via Qwen3.5-0.8B encoder, sweeps layer/mode/PCA per target model, trains a SharedTrunkNet MLP ensemble (10 seeds, keeps best 5), saves `.pt` checkpoint + `serve.yaml`.

Useful options:
- `--device cpu` -- force CPU (default: auto-detect GPU/MPS/CPU)
- `--prefill-dir cache/` -- cache extracted features (saves hours on re-runs)
- `--n-seeds 10 --n-keep 5` -- ensemble configuration
- `--pca-dims 50,100,200` -- PCA dimensions to sweep (default: 50,100,150,200,300)
- `--epochs 150 --patience 15` -- MLP training hyperparameters

### 3. Evaluate

```bash
model-router evaluate \
  --config configs/prefill-qwen08b.yaml \
  --checkpoint checkpoints/custom/prefill_router.pt \
  --data data/test.csv
```

What to look for in the report:
- **Per-model AUC > 0.70**: the router can distinguish when each model is correct
- **Router accuracy > best single model**: routing adds value over just using one model
- **Headroom captured > 0%**: the router is capturing some of the oracle improvement
- **Disagree zone accuracy**: where routing matters most (models disagree on correctness)

### 4. Deploy

Update your config to point to the new checkpoint:
```yaml
routing:
  checkpoint: checkpoints/custom/prefill_router.pt
```

Then start the server:
```bash
model-router serve --config configs/prefill-qwen08b.yaml --port 8000
```

### Quick smoke test (2 minutes with cached prefill)

```bash
model-router train \
  --config configs/smoke-test.yaml \
  --data data/smoke-train.csv \
  --output-dir checkpoints/smoke/ \
  --n-seeds 2 --n-keep 1 --device cpu \
  --epochs 5 --patience 3 --pca-dims 10,20

model-router evaluate \
  --config configs/smoke-test.yaml \
  --checkpoint checkpoints/smoke/prefill_router.pt \
  --data data/smoke-test.csv --device cpu
```

## Config reference

The training config is the same YAML used for serving. Key fields:

| Field | Purpose |
|-------|---------|
| `routing.method` | Must be `prefill` |
| `routing.encoder` | HuggingFace encoder model (e.g., `Qwen/Qwen3.5-0.8B`) |
| `models[].name` | Target model name (must match CSV `model` column) |
| `models[].cost_per_m_input_tokens` | Input token cost per million |
| `models[].cost_per_m_output_tokens` | Output token cost per million |

## Training data format

CSV with columns: `question`, `model`, `isCorrect`, `output_tokens` (optional).

One row per (question, model) pair. Same question appears once per model in the pool. `isCorrect` is 0 or 1.
