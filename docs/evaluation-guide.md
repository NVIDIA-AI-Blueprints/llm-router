# Evaluation Guide

How to evaluate a trained router checkpoint and interpret the results.

## Running Evaluation

```bash
model-router evaluate \
  --config configs/v1-9models-qwen08b.yaml \
  --checkpoint checkpoints/prefill_router_qwen08b.pt \
  --data data/test.csv
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | auto | `cpu`, `cuda`, or `mps` |
| `--batch-size` | 4 | Encoder extraction batch size |
| `--prefill-dir` | `cache/` | Cache dir for extracted features |
| `--no-cache` | off | Disable automatic prefill caching |

Prefill caching is **enabled by default** — extracted features are saved to `cache/` and reused automatically. Use `--no-cache` to disable.

## Test Data Format

Same CSV format as training data: `question, model, isCorrect, output_tokens`.

Each question must have one row per model in the checkpoint's target pool. The `model` column must match the model names in the checkpoint.

## Reading the Report

The evaluation prints several sections:

### Transforms

Shows the best layer, pooling mode, and PCA dimension found during training for each target:

```
  Transforms:
    nem-think           : L18 mean PCA10 (Qwen3.5-0.8B)
    nem-nothink         : L19 last PCA10 (Qwen3.5-0.8B)
```

### Per-Model Metrics

| Metric | What it means | Good value |
|--------|---------------|------------|
| **Accuracy** | Raw fraction of questions the model answers correctly | Higher is better; varies by model |
| **AUC** | ROC AUC of the router's P(correct) prediction vs actual correctness | > 0.70 means useful signal; > 0.80 is strong |

```
  Model                 Accuracy      AUC
  --------------------  --------  -------
  nem-think               0.5500   0.6667
  nem-nothink             0.4500   0.8036
```

### Accuracy Summary

| Metric | What it means |
|--------|---------------|
| **Oracle** | Accuracy if you always picked the best model per question (theoretical ceiling) |
| **Best single** | Accuracy of just using the single best model for everything (baseline) |
| **Headroom** | Oracle minus best single -- the maximum possible improvement from routing |
| **Router accuracy** | Accuracy of the router's argmax selection |
| **Lift** | Router accuracy minus best single (positive = routing helps) |
| **Headroom captured** | What percentage of the available headroom the router captures |

```
  Oracle:       0.6000
  Best single:  0.5500 (nem-think)
  Headroom:     5.0pp

  Router (argmax):
    Accuracy:     0.5500 (+0.00pp, 0.0% headroom captured)
```

### Routing Distribution

Shows how often each model is selected and its accuracy when chosen:

```
    Distribution:
      nem-think           :   19 ( 95.0%)  acc_when_chosen=0.5263
      nem-nothink         :    1 (  5.0%)  acc_when_chosen=1.0000
```

A heavily skewed distribution may indicate the router doesn't have enough signal to differentiate, or the tolerance is too high (always picks cheapest).

### Agreement Zones

Questions are grouped by how many models get them right:

| Zone | Description | Why it matters |
|------|-------------|----------------|
| **All correct** | Every model in the pool answers correctly | Routing doesn't matter -- any choice is correct. Route to cheapest. |
| **Disagree** | Some models right, some wrong | This is where routing adds value. Router accuracy here is the key metric. |
| **All wrong** | No model answers correctly | Nothing to save. Router accuracy is 0 by definition. |

```
  By agreement zone:
    All correct     (   8,  40.0%): acc=1.0000
    Disagree        (   4,  20.0%): acc=0.7500
    All wrong       (   8,  40.0%): acc=0.0000
```

### Deep Analysis

Printed when there are disagreements between models:

**Near-miss analysis**: For wrong routing decisions in the disagree zone, how close was the confidence gap between the chosen (wrong) model and the best correct model? Small gaps mean the router was close to getting it right.

```
  Near-miss (disagree zone, wrong routing):
    Wrong decisions: 1/4 (25.0%)
    Confidence gap: mean=0.0885
    Flippable (gap < 0.05): 0 (0.0%)
```

**Pairwise win rates**: When model A is correct and model B is wrong, how often does A have higher confidence than B? Values > 0.5 mean the router has learned meaningful signal.

```
  Pairwise confidence win rates (disagree zone):
    When A correct & B wrong, P(conf_A > conf_B):
    nem-think   vs nem-nothink: 1.000
    nem-nothink vs nem-think  : 0.000
```

## Interpreting Results

### Good signs
- Per-model AUC > 0.70
- Router accuracy > best single model
- Headroom captured > 20%
- Disagree zone accuracy > random chance (1/n_models)
- Pairwise win rates > 0.5

### Warning signs
- AUC near 0.50 (no better than random)
- Router accuracy < best single (routing is hurting)
- All traffic going to one model (tolerance too high, or not enough signal)
- Pairwise win rates near 0.5 (router can't distinguish)

### What to try
- **Low AUC**: Try more training data, different encoder, wider PCA sweep
- **Skewed distribution**: Lower tolerance (e.g., 0.10 instead of 0.20)
- **Low headroom**: The models in the pool may be too similar -- try adding a more diverse model
- **High "all wrong" zone**: The pool needs a stronger model for hard questions
