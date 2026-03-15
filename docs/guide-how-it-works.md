# How Routing Works

This guide explains the complete routing algorithm — from a raw question string to a model selection decision. It covers the encoder, hidden state extraction, dimensionality reduction, the MLP classifier, and the selection logic.

If you just want to use the router, you don't need to read this. But if you want to understand what's happening under the hood, train on your own data, or extend the system, this is the place.

---

## Table of Contents

- [The Big Picture](#the-big-picture)
- [Step 1: Question Encoding](#step-1-question-encoding)
- [Step 2: Hidden State Extraction](#step-2-hidden-state-extraction)
- [Step 3: Dimensionality Reduction (PCA)](#step-3-dimensionality-reduction-pca)
- [Step 4: MLP Classification](#step-4-mlp-classification)
- [Step 5: Model Selection](#step-5-model-selection)
- [The Checkpoint Format](#the-checkpoint-format)
- [Training vs Inference Path](#training-vs-inference-path)
- [Performance Characteristics](#performance-characteristics)

---

## The Big Picture

The core question the router answers: **given a question, which models in my pool will answer it correctly?**

It does this by encoding the question with a small language model and using the internal representations to predict correctness. The full pipeline:

```
Question (string)
    │
    ▼
Encoder (Qwen3.5-0.8B)                 ← forward pass, output_hidden_states=True
    │
    ▼
Hidden States (per-layer tensors)       ← one vector per layer
    │
    ▼
Per-Model Transform:                    ← each target model uses a different layer/mode
  Layer Selection (e.g., layer 18)
  Pooling Mode (last-token or mean)
  StandardScaler → PCA
    │
    ▼
Concatenated Feature Vector             ← all model features joined
    │
    ▼
SharedTrunkNet MLP Ensemble             ← n_keep=5 models, averaged sigmoid
    │
    ▼
P(correct) per model                    ← e.g., [0.92, 0.89, 0.95, ...]
    │
    ▼
Selection: cheapest above threshold     ← threshold = max(P) - tolerance
    │
    ▼
RoutingResult                           ← selected_model, confidences, costs
```

---

## Step 1: Question Encoding

The router uses a small causal language model as its encoder. The default is **Qwen3.5-0.8B** — a 0.8-billion parameter model that runs comfortably on CPU.

The encoder is not generating text. It's being used purely as a feature extractor: we feed it the question and read its internal hidden states.

### What happens

1. The question is formatted as a chat message using the model's chat template:
   ```
   <|im_start|>user
   What is the capital of France?<|im_end|>
   ```

2. The formatted text is tokenized and passed through the encoder with `output_hidden_states=True`.

3. The encoder produces hidden states at every layer — one tensor per layer, with shape `(sequence_length, hidden_dim)`.

### Why this works

Language models build increasingly abstract representations of input text as data flows through layers. Early layers capture surface-level features (syntax, word identity), while later layers capture semantic meaning (topic, complexity, reasoning requirements).

Different target models have different strengths. A cheap model might handle questions where the answer is in the surface-level phrasing (factual recall), while an expensive model is needed when the question requires multi-step reasoning. The hidden states at the right layer capture exactly these distinctions.

### Chat template kwargs

Each model in the pool can specify `chat_template_kwargs` in the config. These are passed to the encoder's tokenizer when formatting the question. For example, `enable_thinking: true` activates a chain-of-thought template variant in the encoder, which changes the hidden state representations.

During training, the sweep finds which template kwargs (if any) produce the best features for each target model. These are stored in the checkpoint and applied automatically at inference.

### Encoder options

| Encoder | Params | Hidden Dim | Layers | Speed (CPU) | Speed (GPU) |
|---------|--------|------------|--------|-------------|-------------|
| Qwen3.5-0.8B | 0.8B | 1536 | 24 | ~5s/query | ~100ms/query |
| Qwen3.5-35B-A3B | 35B (MoE, ~3B active) | 2560 | 64 | Not recommended | ~200ms/query |

The 0.8B encoder is recommended for most use cases. The 35B encoder produces higher AUC but requires a GPU.

---

## Step 2: Hidden State Extraction

After the encoder forward pass, we extract a single vector per layer from the full hidden state tensor.

### Pooling modes

Each layer produces a tensor of shape `(sequence_length, hidden_dim)`. We need to reduce this to a single vector of shape `(hidden_dim,)`. Two modes are available:

| Mode | What it does | When it's better |
|------|-------------|-----------------|
| `last` | Takes the hidden state at the last token position | Captures the model's "summary" of the full input. Often better for factual/short queries. |
| `mean` | Averages hidden states across all token positions | Captures distributed information. Often better for longer, more complex queries. |

### Layer selection

Not all layers are equally informative. The training sweep uses ternary search to find the single best layer for each target model. Typically:

- **Middle layers** (12–18 in a 24-layer model) capture broad semantic features
- **Later layers** (18–23) capture more task-specific features

Different target models may use different best layers. For example:
- A cheap reasoning model might be best predicted by layer 18 (broad complexity)
- An expensive model might be best predicted by layer 22 (fine-grained difficulty)

The sweep optimizes layer selection per model using 5-fold cross-validated AUC with logistic regression.

### Extraction scope

By default, hidden states are extracted from the second half of the encoder's layers (layers `n_layers/2` through `n_layers-1`). Early layers are skipped because they typically carry low-level lexical features that don't discriminate between model capabilities.

---

## Step 3: Dimensionality Reduction (PCA)

Raw hidden states are high-dimensional (1536 for Qwen3.5-0.8B, 2560 for 35B). Most of this dimensionality is redundant for routing. PCA reduces it to a compact, informative representation.

### Pipeline (per target model)

1. **StandardScaler**: Centers and scales features to zero mean, unit variance. Fitted on the training set.
2. **PCA**: Projects into the top principal components. Fitted on the training set.

The number of PCA components is a hyperparameter searched during training (default candidates: 50, 100, 150, 200, 300).

### Why per-model transforms

Each target model has its own (layer, mode, PCA dim) configuration because the features that best predict "can nemotron-nano answer this?" are different from those that predict "can claude-opus answer this?".

After PCA, each model produces a reduced feature vector. These are concatenated into a single "shared feature vector" that goes into the MLP:

```
Model A features (PCA_A dims) ─┐
Model B features (PCA_B dims) ─┼─> Concatenated shared vector (sum of all PCA dims)
Model C features (PCA_C dims) ─┘
```

---

## Step 4: MLP Classification

The SharedTrunkNet is a small multi-layer perceptron that takes the concatenated feature vector and outputs logits for each target model.

### Architecture

```
Input (d_shared) → Linear(d_shared, 256) → ReLU → Dropout(0.3)
                 → Linear(256, 128)       → ReLU → Dropout(0.2)
                 → Linear(128, n_models)  → logits
```

- **d_shared**: Sum of all per-model PCA dimensions
- **n_models**: Number of target models in the pool
- **Output**: Raw logits (sigmoid is applied at inference, BCEWithLogitsLoss at training)

### Ensemble

Multiple MLPs are trained with different random seeds. The default trains 10 seeds and keeps the 5 with the lowest validation loss. At inference, all 5 produce sigmoid outputs, which are averaged:

```
P(correct | model_i) = mean(sigmoid(logit_1_i), sigmoid(logit_2_i), ..., sigmoid(logit_5_i))
```

Ensembling smooths out noise and improves calibration.

### Training details

- **Loss**: BCEWithLogitsLoss (binary cross-entropy on multi-label targets)
- **Optimizer**: Adam, lr=1e-3, weight_decay=1e-4
- **Split**: 85% train, 15% validation
- **Early stopping**: Patience=15 epochs on validation loss
- **Max epochs**: 150
- **Batch size**: 512

---

## Step 5: Model Selection

Given P(correct) for each model and each model's cost, the selection algorithm picks the cheapest model that meets the accuracy threshold.

### Algorithm

```python
p_max = max(confidences)           # best confidence across all models
threshold = p_max - tolerance      # minimum acceptable confidence

candidates = [m for m in models if confidence[m] >= threshold]
selected = min(candidates, key=lambda m: cost[m])
```

### Tolerance explained

The `tolerance` parameter is the maximum acceptable drop in P(correct) below the best model's confidence, in exchange for a cheaper model:

| Tolerance | Behavior |
|-----------|----------|
| 0.00 | Always pick the model with the highest P(correct) — pure accuracy maximization |
| 0.05 | Allow models within 5pp of the best — mild cost savings |
| 0.20 | Allow models within 20pp of the best — aggressive cost savings (default) |
| 0.50 | Very aggressive — large accuracy drops acceptable for cost |
| 1.00 | Always pick the cheapest model regardless of confidence |

### Cost ranking

Models are ranked by `cost_per_m_input_tokens` from the pool config. The cost estimates in `RoutingResult` also include median output token counts (learned during training) for more accurate total-cost estimates.

### Model filtering

When `models` is passed to `route()`, only those models are considered for selection. Confidences are still computed for the full pool (because the MLP always outputs all models), but the selection step only considers the allowed subset.

---

## The Checkpoint Format

The `.pt` checkpoint is a self-contained PyTorch file with everything needed for inference:

```python
{
    "version": 2,
    "model_names": ["nemotron-3-nano-reasoning", "gpt-oss-20b-high", ...],

    "transforms": {
        "nemotron-3-nano-reasoning": {
            "encoder": "Qwen/Qwen3.5-0.8B",
            "layer": 18,
            "mode": "mean",
            "scaler": <fitted StandardScaler>,
            "pca": <fitted PCA>,
            "chat_template_kwargs": {"enable_thinking": true},
        },
        # ... one entry per model
    },

    "shared_trunk": [<state_dict_1>, <state_dict_2>, ...],  # ensemble
    "trunk_config": {"d_in": 850, "n_outputs": 9, "hidden": [256, 128], "dropout": [0.3, 0.2]},

    "cost_table": {
        "nemotron-3-nano-reasoning": {
            "median_output_tokens": 150,
            "mean_output_tokens": 180,
            "p25_output_tokens": 80,
            "p75_output_tokens": 250,
        },
        # ... per model
    },

    "pool_config": {
        "nemotron-3-nano-reasoning": {
            "cost_per_m_input_tokens": 0.05,
            "cost_per_m_output_tokens": 0.20,
        },
        # ... per model
    },
}
```

The checkpoint includes the pool config at the time of training. When serving, cost information from the checkpoint is used for ranking. If you update costs in your YAML config, you need to retrain or manually update the checkpoint.

---

## Training vs Inference Path

### Inference (online, per-query)

Uses the `BaseRouter` interface:

```
PrefillRouter.route(question)
  → PrefillScorer.score(question)
    → PrefillExtractor.extract(question)  # single question
    → build_features(result, ...)         # per-model PCA
    → predict_proba(trunk_nets, feats)    # MLP ensemble
  → selection logic
  → RoutingResult
```

The encoder model is loaded once on first call and cached in memory.

### Training (offline, batch)

Bypasses `BaseRouter` for efficiency:

```
train_prefill(config, data_path, ...)
  → _load_labels(csv)
  → run_extraction(encoder, all_questions)  # batch extraction
  → sweep_model(prefill_result, labels)     # per-model hyperparameter search
  → fit_pca_pipeline(raw, train_mask, pca_dim)  # fit transforms
  → train_ensemble(SharedTrunkNet, X, y)    # train MLP ensemble
  → _build_checkpoint(...)                  # save everything
```

Batch extraction is much more efficient — questions are batched, padded, and processed together. The sweep and MLP training operate on numpy arrays and torch tensors directly, without going through the router abstraction.

### Evaluation (offline, batch)

Also bypasses `BaseRouter`:

```
_run_prefill_evaluate(checkpoint, data, ...)
  → extract_from_checkpoint(ckpt, questions)  # batch extraction
  → _build_shared_features(ckpt, prefill_results)  # apply saved transforms
  → predict_proba(trunk_nets, shared_feats)   # MLP ensemble
  → _print_eval_report(...)                   # rich metrics
```

---

## Performance Characteristics

### Latency

| Component | CPU | GPU (CUDA) | GPU (MPS) |
|-----------|-----|------------|-----------|
| Encoder forward pass (0.8B) | ~5s | ~100ms | ~200ms |
| PCA + MLP inference | <1ms | <1ms | <1ms |
| **Total per-query** | **~5s** | **~100ms** | **~200ms** |

The encoder is the bottleneck. On GPU, routing adds negligible latency to the overall request.

### Memory

| Component | Memory |
|-----------|--------|
| Qwen3.5-0.8B (fp16) | ~1.6 GB |
| Qwen3.5-0.8B (fp32) | ~3.2 GB |
| Checkpoint (MLP + transforms) | ~5 MB |
| PCA + scaler per model | ~1 MB each |

The encoder model is loaded once and stays in memory. On GPU, it uses ~1.6 GB VRAM in half precision. CPU inference uses more RAM but doesn't require a GPU.

### Batch extraction (training/eval)

Batch extraction processes multiple questions in parallel with padding:

| Batch size | GPU (0.8B) | CPU (0.8B) |
|------------|------------|------------|
| 4 (default) | ~40ms/question | ~4s/question |
| 8 | ~25ms/question | ~3.5s/question |
| 16 | ~20ms/question | ~3s/question |

Larger batch sizes improve throughput but require more memory. The default of 4 balances memory and speed.

### Caching

Extracted features are cached to disk by default (in `cache/` or `--prefill-dir`). Cache keys are based on the encoder name, chat template kwargs, and the set of questions. This means:

- Rerunning training with the same questions skips extraction entirely
- Changing the encoder or template kwargs invalidates the cache
- Adding new questions requires re-extraction (the cache stores all questions together)
