# Training & Evaluation Guide

This guide covers the full training pipeline — from labeled CSV to a production-ready checkpoint — and the evaluation framework that measures routing quality. Training and evaluation share infrastructure (feature extraction, transforms), so they're covered together.

---

## Table of Contents

- [Overview](#overview)
- [Training](#training)
  - [Running Training](#running-training)
  - [What Happens Under the Hood](#what-happens-under-the-hood)
  - [All Training Options](#all-training-options)
  - [Quick Verification Run](#quick-verification-run)
- [Evaluation](#evaluation)
  - [Running Evaluation](#running-evaluation)
  - [All Evaluation Options](#all-evaluation-options)
  - [Understanding the Report](#understanding-the-report)
  - [Interpreting Results](#interpreting-results)
  - [What to Do When Results Are Poor](#what-to-do-when-results-are-poor)
- [End-to-End Workflow](#end-to-end-workflow)

---

## Overview

The lifecycle:

```
Labeled CSV ──> Train ──> Checkpoint (.pt) ──> Evaluate ──> Deploy
```

- **Training** is fully offline. No API keys, no network. The encoder model (Qwen3.5-0.8B) is downloaded from HuggingFace on first run and cached locally.
- **Evaluation** is also fully offline. It uses the same encoder and checkpoint.
- The output is a self-contained `.pt` checkpoint that includes everything the router needs at inference time.

> **Git LFS:** Checkpoints and data files are tracked with Git LFS. Run `git lfs install && git lfs pull` after cloning to fetch the actual files.

---

## Training

### Running Training

```bash
pip install -e '.[prefill]'

model-router train \
  --config configs/v1-9models-qwen08b.yaml \
  --data data/train.csv \
  --output-dir checkpoints/
```

This takes ~10–30 minutes on CPU (dominated by encoder extraction) or ~2–5 minutes on GPU.

### What Happens Under the Hood

The training pipeline has six stages:

#### Stage 1: Load Labels

Reads the CSV and builds a label matrix.

- Normalizes questions (lowercase, collapse whitespace)
- Deduplicates questions
- Builds `Y` matrix: shape `(n_questions, n_models)`, where `Y[i][j] = 1` if model j answered question i correctly
- Computes output token statistics per model (median, mean, p25, p75) for cost estimation
- Prints per-model accuracy summary

#### Stage 2: Extract Prefill Features

Runs the encoder on all training questions.

- Loads the HuggingFace encoder (e.g., Qwen3.5-0.8B)
- Processes questions in batches (default: batch_size=4)
- Extracts hidden states from the second half of the encoder's layers
- For each layer, stores both last-token and mean-pooled representations
- Results are cached to disk (default: `cache/`) so re-runs skip this step

This is the slowest stage. On CPU with 500 questions, it takes ~40 minutes. On GPU, ~2 minutes.

#### Stage 3: Sweep Hyperparameters

For each target model in the pool, finds the best (layer, mode, PCA dimension) configuration.

- **Layer search**: Ternary search across the extracted layers, assuming AUC is unimodal (peaks at one layer and decreases on both sides)
- **Mode search**: Grid search over `last` and `mean` pooling
- **PCA dimension search**: Grid search over `[50, 100, 150, 200, 300]` (configurable)
- **Quality metric**: 5-fold cross-validated AUC with logistic regression

For each (mode, PCA dim) combination, ternary search finds the best layer. The overall best (layer, mode, PCA dim) is selected for each model.

Example output:

```
  Sweep: nemotron-3-nano-reasoning
    Best: layer=18, mode=mean, pca_dim=100, cv_auc=0.782
  Sweep: gpt-oss-20b-high
    Best: layer=19, mode=last, pca_dim=150, cv_auc=0.814
```

#### Stage 4: Fit Transforms

Fits StandardScaler + PCA on training data for each model's best configuration.

- Extracts raw hidden states at the selected (layer, mode) for each model
- Fits StandardScaler on training data (zero mean, unit variance)
- Fits PCA on training data (reduce to `pca_dim` components)
- Transforms all data (train + any held-out) through the fitted pipeline
- Concatenates per-model features into a shared feature matrix

#### Fixed All-Layer Meanpool PCA-200

Training can bypass the layer/mode/PCA sweep with an explicit feature recipe:

```yaml
routing:
  method: prefill
  encoder: Qwen/Qwen3.6-35B-A3B
  features:
    aggregation: all_layers_concat
    layers: all
    pooling: mean
    pca_dim: 200
    hidden_state_indexing: direct
```

This path mean-pools every saved encoder state over non-padding tokens,
concatenates the states in numeric order, and fits a train-only
`StandardScaler` and randomized PCA-200 transform. For Qwen 3.6 35B's 40
saved states and hidden width 2,048, the raw feature width is 81,920. The
concatenated buffer is scaled in place to limit peak host-memory use.

`hidden_state_indexing: direct` means logical layer `L` reads
`outputs.hidden_states[L]`. Logical layer 0 is therefore the embedding state.
The convention is saved in the generated checkpoint and reused during
evaluation and serving.

The transformed 200-dimensional feature block enters the multi-output shared
trunk once, regardless of how many target models are trained. The resulting
architecture is `200 -> 256 -> 128 -> n_targets`. The scaler and PCA are
fitted once and shared by all target outputs.

Run it through the normal training command:

```bash
model-router train \
  --config configs/qwen36-35b-all-layers-mean-pca200.yaml \
  --data data/train.csv \
  --output-dir checkpoints/
```

This configuration does not inspect test labels and does not run the
layer/mode/PCA sweep. All-layer extraction and the 81,920-wide raw transform
have substantially higher memory and runtime requirements than the default
single-layer path.

The repository provides the training code and example configuration, not a
trained checkpoint or Qwen prefill artifact. The command writes generated
artifacts to the selected output and cache directories.

#### Stage 5: Train MLP Ensemble

Trains the SharedTrunkNet with multiple seeds and keeps the best.

- **Architecture**: Linear(d_in, 256) → ReLU → Dropout(0.3) → Linear(256, 128) → ReLU → Dropout(0.2) → Linear(128, n_models)
- **Loss**: BCEWithLogitsLoss (multi-label binary cross-entropy)
- **Optimizer**: Adam, lr=1e-3, weight_decay=1e-4
- **Split**: 85% train / 15% validation (within training data)
- **Early stopping**: Monitors validation loss, patience=15 epochs
- **Ensemble**: Trains `n_seeds` models (default 10), keeps `n_keep` with lowest validation loss (default 5)

Example output:

```
  Training trunk ensemble (10 seeds, keeping best 5)...
    Seed 0: val_loss=0.4823 (epoch 87)
    Seed 1: val_loss=0.4801 (epoch 92)
    ...
    Kept seeds: [1, 4, 7, 3, 9] (val_loss: 0.4801, 0.4812, 0.4819, 0.4825, 0.4831)
```

#### Stage 6: Save Checkpoint

Builds and saves the `.pt` checkpoint containing:
- Model names and pool config
- Per-model transforms (encoder, layer, mode, fitted scaler, fitted PCA, chat_template_kwargs)
- Trunk ensemble state dicts
- Trunk architecture config
- Cost table (output token statistics)

### All Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | (required) | Pool config YAML path |
| `--data` | (required) | Training CSV path |
| `--output-dir` | (required) | Directory to save checkpoint |
| `--device` | auto-detect | `cpu`, `cuda`, or `mps` (experimental) |
| `--batch-size` | 4 | Encoder extraction batch size |
| `--n-seeds` | 10 | Number of MLP seeds to train |

> **Apple Silicon (MPS) Warning:** MPS GPU support is experimental and may cause
> silent crashes (SIGSEGV) during training or evaluation. If you encounter crashes,
> use `--device cpu` instead. CPU is slower but stable.
| `--n-keep` | 5 | Number of best seeds to keep in ensemble |
| `--pca-dims` | 50,100,150,200,300 | PCA dimensions to sweep (comma-separated) |
| `--epochs` | 150 | Max MLP training epochs |
| `--patience` | 15 | Early stopping patience (epochs without improvement) |
| `--models` | all | Comma-separated model subset to train on |
| `--mode` | auto | Training mode override |

### Quick Verification Run

Before running the full pipeline on a large dataset, verify everything works with minimal settings:

```bash
model-router train \
  --config configs/v1-9models-qwen08b.yaml \
  --data data/train.csv \
  --output-dir checkpoints/ \
  --n-seeds 2 --n-keep 1 \
  --epochs 5 --patience 3 \
  --pca-dims 10,20 \
  --device cpu
```

This runs in a few minutes and validates the full pipeline without the full hyperparameter search.

---

## Evaluation

### Running Evaluation

```bash
model-router evaluate \
  --config configs/v1-9models-qwen08b.yaml \
  --checkpoint checkpoints/prefill_router_qwen08b.pt \
  --data data/test.csv
```

### All Evaluation Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | (required) | Pool config YAML path |
| `--checkpoint` | (required) | Checkpoint `.pt` path |
| `--data` | (required) | Test CSV path |
| `--device` | auto-detect | `cpu`, `cuda`, or `mps` (experimental) |
| `--batch-size` | 4 | Encoder extraction batch size |
| `--models` | all | Model subset to evaluate |
| `--pricing` | (none) | Pricing CSV override (`model,cost_per_m_input_tokens`) |

### Understanding the Report

The evaluation prints a multi-section report. Here's what each section means.

#### Transforms

Shows the best (layer, mode, PCA dimension) found during training for each model:

```
  Transforms:
    nemotron-3-nano-reasoning : L18 mean PCA100 (Qwen3.5-0.8B)
    gpt-oss-20b-high          : L19 last PCA150 (Qwen3.5-0.8B)
    nemotron-3-super          : L17 mean PCA100 (Qwen3.5-0.8B)
```

This tells you which encoder features are most informative for predicting each model's correctness.

#### Per-Model Metrics

| Metric | What it means |
|--------|---------------|
| **Accuracy** | Fraction of test questions the model answers correctly (ground truth from CSV) |
| **AUC** | ROC AUC of the router's predicted P(correct) vs actual correctness |

```
  Model                      Accuracy      AUC
  -------------------------  --------  -------
  nemotron-3-nano-reasoning    0.6200   0.7823
  gpt-oss-20b-high             0.6800   0.8145
  claude-opus-4-6-high         0.8900   0.8567
```

**AUC interpretation**: The AUC measures how well the router's confidence scores separate correct from incorrect predictions for each model. An AUC of 0.50 is random (no signal). Above 0.70 is useful. Above 0.80 is strong.

#### Accuracy Summary

```
  Oracle:       0.9200
  Best single:  0.8900 (claude-opus-4-6-high)
  Headroom:     3.0pp

  Router (argmax):
    Accuracy:     0.9050 (+1.5pp, 50.0% headroom captured)
```

| Metric | Definition |
|--------|-----------|
| **Oracle** | Accuracy if you always picked the model that's actually correct (theoretical maximum) |
| **Best single** | Accuracy of always using the single best-performing model |
| **Headroom** | Oracle − Best single. The maximum possible improvement from routing. |
| **Router accuracy** | Accuracy when the router picks the argmax-confidence model |
| **Lift** | Router accuracy − Best single. Positive means routing is helping. |
| **Headroom captured** | Lift / Headroom × 100. What fraction of the possible improvement the router captures. |

#### Routing Distribution

```
  Distribution:
    nemotron-3-nano-reasoning : 312 ( 52.0%)  acc_when_chosen=0.8654
    gpt-oss-20b-high          :  96 ( 16.0%)  acc_when_chosen=0.8125
    nemotron-3-super          :  48 (  8.0%)  acc_when_chosen=0.8542
    claude-opus-4-6-high      : 144 ( 24.0%)  acc_when_chosen=0.9167
```

This shows how traffic splits across models and the accuracy of each model on the questions it's chosen for. A good router sends straightforward questions to lightweight models (maintaining accuracy) and complex questions to more capable ones.

#### Agreement Zones

Questions are grouped by how many models answer correctly:

| Zone | Description | Why it matters |
|------|-------------|----------------|
| **All correct** | Every model in the pool gets it right | Routing to the most efficient model is optimal. Maximum efficiency, no accuracy tradeoff. |
| **Disagree** | Some models right, some wrong | Where routing adds value. Router accuracy here is the key metric. |
| **All wrong** | No model answers correctly | Nothing to save. Router accuracy is 0 by definition. |

```
  By agreement zone:
    All correct     ( 380,  63.3%): acc=1.0000
    Disagree        ( 120,  20.0%): acc=0.7917
    All wrong       ( 100,  16.7%): acc=0.0000
```

#### Near-Miss Analysis

For wrong routing decisions in the disagree zone: how close was the router to getting it right?

```
  Near-miss (disagree zone, wrong routing):
    Wrong decisions: 25/120 (20.8%)
    Confidence gap: mean=0.0442
    Flippable (gap < 0.05): 12 (48.0%)
```

- **Confidence gap**: The difference between the chosen (wrong) model's confidence and the best correct model's confidence. Smaller gaps mean the router was close.
- **Flippable**: Decisions where the gap is less than 0.05 — a slightly different tolerance or more training data might flip these to correct.

#### Pairwise Win Rates

When model A is correct and model B is wrong, how often does the router give A higher confidence?

```
  Pairwise confidence win rates (disagree zone):
    When A correct & B wrong, P(conf_A > conf_B):
    nemotron-3-nano vs gpt-oss-20b: 0.720
    gpt-oss-20b vs nemotron-3-nano: 0.680
```

Values above 0.50 mean the router has learned meaningful signal. Values near 1.0 mean the router almost always correctly identifies which model will succeed.

#### P-AUCCC Metrics

Cost-coverage analysis at different tolerance levels:

- **Model Pareto**: The Pareto frontier of (cost, accuracy) across models
- **Router P-AUCCC**: Area under the router's cost-coverage curve
- **MDP-AUCCC**: Maximum possible cost-coverage (using oracle decisions)
- **PDP-AUCCC**: The router's fraction of the maximum possible cost-coverage

### Interpreting Results

#### Good signs

- Per-model AUC > 0.70 for most models
- Router accuracy > best single model
- Headroom captured > 20%
- Disagree zone accuracy > 1/n_models (random chance)
- Pairwise win rates > 0.50
- Distribution is not concentrated on a single model

#### Warning signs

- AUC near 0.50 for most models → the encoder can't distinguish difficulty for those models
- Router accuracy ≤ best single → routing is not helping (or hurting)
- All traffic to one model → tolerance is too high, or the router can't differentiate
- Pairwise win rates near 0.50 → the router can't tell which model will succeed

### What to Do When Results Are Poor

| Problem | Likely cause | What to try |
|---------|-------------|-------------|
| Low AUC across all models | Not enough training data | Collect more questions, especially in the "disagree" zone |
| Low AUC for one model | That model is unpredictable from the question alone | Remove it from the pool, or accept lower routing quality for it |
| Zero lift over best single | Models are too similar, or headroom is near zero | Add more diverse models to the pool, or check if the cheap models are actually good enough |
| All traffic to cheapest model | Tolerance too high | Lower tolerance (e.g., 0.10 instead of 0.20) |
| All traffic to most expensive | Tolerance too low, or cheap models have very low AUC | Raise tolerance, or check if cheap models are genuinely bad |
| High "all wrong" zone | Pool lacks a strong enough model for hard questions | Add a more capable model |
| High "all correct" zone | Questions are too easy | The router is working perfectly — it routes to the most efficient model. But metrics look flat. |
| Many near-miss flippable errors | Router is close but not quite there | More training data, especially in the difficulty range where models disagree |

---

## Live Quality Review

When the standalone server is running (`model-router serve`), the `/api/review` endpoint provides live auto-judging: it sends the question and model answer to the most expensive model in the pool for a correctness verdict. This complements offline evaluation by letting you spot-check routing quality on real traffic. See the [Serving & Deployment Guide](guide-serving-and-deployment.md#post-apireview--auto-judge) for details.

---

## End-to-End Workflow

Here's the complete workflow from scratch:

```bash
# 1. Install
pip install -e '.[prefill,training]'

# 2. Set up API key
export OPENROUTER_API_KEY=your-key

# 3. Prepare questions (one per line)
# questions.txt should have 500+ diverse questions

# 4. Collect labeled data
model-router collect \
  --config configs/v1-9models-qwen08b.yaml \
  --questions questions.txt \
  --output data/collected.csv \
  --judge vote

# 5. Split into train/test
model-router split --data data/collected.csv --train-output data/train.csv --test-output data/test.csv

# 6. Train
model-router train \
  --config configs/v1-9models-qwen08b.yaml \
  --data data/train.csv \
  --output-dir checkpoints/

# 7. Evaluate
model-router evaluate \
  --config configs/v1-9models-qwen08b.yaml \
  --checkpoint checkpoints/prefill_router.pt \
  --data data/test.csv

# 8. Update config to point to new checkpoint
# Edit configs/v1-9models-qwen08b.yaml: checkpoint: checkpoints/prefill_router.pt

# 9. Serve
export OPENROUTER_API_KEY=your-key
model-router serve --config configs/v1-9models-qwen08b.yaml --port 8000
```
