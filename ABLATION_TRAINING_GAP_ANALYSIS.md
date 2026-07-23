# Qwen 3.6 35B All-Layer PCA Ablation to Blueprint v3

## Scope

This note compares:

- `swesmith_80_20_ablation/exp_prompt_only_ultra_overlap/qwen36_35b_all_layers_pca_shared_trunk`
- `swesmith_80_20_ablation/exp_prompt_only_ultra_overlap/qwen36_35b_all_layers_pca_shared_trunk/exp_meanpool_pca200_single_block`
- NVIDIA `llm-router` branch `v3` at commit
  `655993e24266acb5a94c4132638f24c81c7f1963`

The objective is to port the all-layer training method into the blueprint
without depending on the experiment's synthetic `.pt` wrapper or local
shape-inference workarounds.

## Exact Ablation Flow

### 1. Inputs and split

The experiment consumes two pre-extracted Qwen 3.6 35B artifacts:

- first half: layers 0 through 19
- second half: layers 20 through 39

Together the artifacts cover 8,977 rows and hidden width 2,048. Their `query_ids`
must match in the same order.

The label artifacts define a fixed prompt-only SWE-Smith split:

- train: 7,181 unique query IDs
- test: 1,796 unique query IDs
- train/test overlap: zero

Rows are joined to prefills by `query_id`, not by normalized question text.

### 2. All-layer feature construction

For each pooling mode, the probe concatenates every saved layer in numeric
order:

```text
last:
  concat(layer_0, ..., layer_39)

meanpool:
  concat(layer_0_meanpool, ..., layer_39_meanpool)
```

Each raw feature matrix is therefore:

```text
8,977 x (40 * 2,048) = 8,977 x 81,920
```

The current extraction convention in both the research package and blueprint
uses `outputs.hidden_states[li]`. Consequently, `layer_0` is the embedding
state and `layer_39` is the output after decoder block 38. The output after
decoder block 39 is not represented. This convention must be preserved for
checkpoint reproduction or changed under an explicit feature-schema version.

### 3. Probe PCA

For each pooling mode:

1. Fit `StandardScaler` on the 4,619 training rows.
2. Transform the complete raw matrix.
3. Fit randomized PCA-200 on the training rows.
4. Transform all rows once.
5. Treat prefixes of that PCA-200 representation as PCA dimensions
   25, 50, 100, and 200.

This is not the same implementation as independently fitting PCA at each
candidate dimension.

### 4. Probe models and metrics

For every `(pooling, PCA dimension, target model)` tuple:

- LR:
  `StandardScaler -> LogisticRegression(max_iter=1000)`
- MLP:
  `StandardScaler -> MLPClassifier(hidden=(256, 64), max_iter=400,
  early_stopping=True, n_iter_no_change=20)`
- CV:
  stratified 5-fold AUC over the training split
- test:
  fit on the complete training split and report AUC on the held-out test split

The extra scaler after PCA affects regularized LR and MLP training. Blueprint
v3 currently feeds PCA values directly to LR during its sweep.

### 5. Configuration selection

The selected production comparison is `exp_meanpool_pca200_single_block`,
which pins meanpool plus PCA-200 and passes one reduced block to the trunk.

This is useful as a research comparison but is test-set selection leakage. A
blueprint trainer must not automate this behavior. For production training:

- choose by train-only CV, or
- pin meanpool/PCA-200 explicitly when reproducing the research checkpoint.

On the saved probe table:

- best global macro LR test configuration: meanpool/PCA-200
- best global macro LR CV configuration: meanpool/PCA-100
- best per-target LR CV configurations are not all identical

### 6. Shared-trunk fitting

The experiment creates a synthetic one-layer prefill:

```text
layer_0          = concat(all 48 last-token layer vectors)
layer_0_meanpool = concat(all 48 mean-pooled layer vectors)
```

It then calls the normal research trainer with a pinned layer/mode and one
PCA dimension. The trainer refits scaler and PCA on training rows; it does not
reuse the probe's fitted PCA object.

Final trunk settings:

- architecture: `d_in -> 256 -> 128 -> 4`
- activations: ReLU
- dropout: 0.3, then 0.2
- loss: `BCEWithLogitsLoss`
- optimizer: Adam, learning rate `1e-3`, weight decay `1e-4`
- internal split: random 85% train / 15% validation per seed
- maximum epochs: 150
- early-stopping patience: 15
- seeds: 10
- retained ensemble members: best 5 by validation loss

These core trunk settings already match blueprint v3.

The research trainer constructs one transform per target and horizontally
stacks them. Because all four targets are pinned to the same encoder,
aggregation, pooling, and PCA:

- PCA-50 produces trunk input width 200, not 50.
- PCA-200 produces trunk input width 800, not 200.

This is four copies of the same deterministic feature block. It should be
treated as a research artifact, not preserved in the blueprint trainer. The
selected production design retrains the trunk on one shared 200-dimensional
feature block.

## Blueprint v3 Gaps

| Area | Ablation | Blueprint v3 | Required change |
|---|---|---|---|
| Data identity | Explicit `query_id` join | Normalized question-text key | Accept/persist an optional query ID and validate uniqueness/completeness |
| Existing prefills | Loads experiment `.pt` files | Extracts or loads internal cache only | Add a supported pre-extracted prefill input or conversion command |
| Extracted layers | All 40 saved states | Second half by default | Make required layers part of the feature specification |
| Raw feature | Concatenation across layers | One selected layer | Add `all_layers_concat` aggregation |
| Pooling candidates | Last and meanpool | Last and mean | Naming can map directly |
| PCA search | Fit max dimension once, use prefixes | Refit PCA per dimension | Add a max-PCA-prefix search strategy |
| Probe normalization | Scales PCA scores again | No post-PCA scaling | Add the post-PCA probe scaler for faithful search |
| Selection | Research run used test AUC | Per-target train CV | Keep train CV as default; use explicit pinning for reproduction |
| Selection scope | One global setting was pinned | Per-target setting | Support global macro-CV selection and existing per-target selection |
| Final transform | Synthetic layer-0 wrapper | Physical layer transform | Store aggregation and layer list explicitly in checkpoint |
| Shared input | Repeats identical block per target | Also stacks per-target blocks | New fixed path uses one shared 200-dimensional block |
| Serving | Local scorer infers all-layer need from scaler width | Extracts only checkpoint layer IDs | Extract the union of explicit layer lists and aggregate identically |
| Cache key | Experiment artifact is explicit | Encoder/template/question set | Include feature schema/layer set; also fix row-order safety |
| Artifacts | All-seed checkpoint and `seed_aucs.csv` | Retained ensemble only | Add optional all-seed and seed-metric outputs |

## Important Correctness Issues

### Do not port test-set selection

The test set should only produce final metrics. The blueprint should select:

- a global feature spec by macro train CV AUC, or
- a feature spec per target by that target's train CV AUC.

An explicit pinned feature spec can reproduce meanpool/PCA-200 without making
the held-out labels visible to training.

### Make layer semantics explicit

The checkpoint must record whether logical layer `L` means:

- `hidden_states[L]`, or
- decoder output `hidden_states[L + 1]`.

Changing this silently would make newly extracted serving features
incompatible with the saved ablation prefills.

### Do not infer aggregation from scaler width

The research scorer detects an all-layer transform when:

```text
scaler.n_features_in_ == hidden_size * num_hidden_layers
```

That is a useful experiment workaround but an unsafe production contract.
The checkpoint should explicitly store aggregation, ordered layers, pooling,
and hidden-state indexing.

### Fix cache identity

Blueprint cache names hash a sorted question set, while cached tensors retain
the input row order. Passing the same questions in another order can load
misordered features. Either:

- make the hash order-sensitive, or
- persist stable IDs/questions and realign on load.

The cache key must also include the required layer set and feature-schema
version so a second-half cache cannot satisfy an all-layer request.

### Validate label completeness

Both trainers currently turn a missing target label into zero. The port should
fail or explicitly mask missing labels rather than silently treating a missing
row as an incorrect answer.

## Recommended Blueprint Design

### Feature specification

Add a typed feature-search section, for example:

```yaml
routing:
  feature_search:
    aggregation: all_layers_concat
    layers: all
    pooling: [last, mean]
    pca_dims: [25, 50, 100, 200]
    pca_strategy: max_dim_prefix
    probe: logistic_regression
    post_pca_standardize: true
    selection_scope: global
    selection_metric: cv_auc
```

For exact checkpoint reproduction, allow a pinned specification:

```yaml
routing:
  feature_search:
    pinned:
      aggregation: all_layers_concat
      layers: all
      pooling: mean
      pca_dim: 200
```

### Checkpoint schema

Bump the checkpoint schema and store the transform contract explicitly:

```text
feature_spec:
  aggregation: concat
  layers: [0, ..., 47]
  pooling: mean
  hidden_state_indexing: direct
scaler: ...
pca: ...
pca_dim: 200
encoder: Qwen/Qwen3.6-35B-A3B
```

The selected blueprint design defines one shared feature group:

```text
feature_groups:
  - id: qwen36_35b_all_mean_pca200
    transform: ...
    targets: [qwen-122b, nemotron-3-super, opus-4.7, gpt-5.5]
```

The trunk input is one PCA block shared by all target outputs. This requires
retraining and is intentionally incompatible with the repeated-input research
checkpoint.

## Implementation Order

1. Add `FeatureSpec` plus concatenation helpers in `prefill/transforms.py`.
2. Add required-layer extraction and cache-key metadata in
   `prefill/extract.py`.
3. Extend `prefill/sweep.py` with all-layer candidates, max-dimension PCA
   prefixes, post-PCA probe scaling, and global macro-CV selection.
4. Extend `prefill/train.py` to accept pinned/search feature specs, fit the
   final train-only transform, and save the explicit feature schema.
5. Update `prefill/scorer.py` and `evaluate.py` to extract the union of
   required layers and apply the exact saved aggregation.
6. Preserve checkpoint-v2 loading while writing a new schema version.
7. Add optional per-seed metrics and all-seed checkpoint output.
8. Expose the new options through `config.py`, the CLI, and the training guide.

## Required Tests

- Concatenation is ordered numerically and has expected shape.
- Last and mean pooling select the correct tensors.
- Layer indexing convention is explicit and stable.
- Scaler/PCA are fit only on training rows.
- PCA-prefix candidates use one fitted max-dimension PCA.
- Global selection uses only CV scores, never test labels.
- Query IDs align prefills and labels regardless of source row order.
- Missing IDs, duplicate IDs, and missing target labels fail clearly.
- Cache keys differ for different required layer sets and feature schemas.
- Train-time and scorer-time feature matrices are numerically equal.
- Checkpoint-v2 scoring remains backward compatible.
- Synthetic all-layer end-to-end train/evaluate/score smoke test.
- Optional private regression test reproduces the saved ablation probe scores
  within an agreed numerical tolerance.

## Recommended Port Boundary

The first blueprint change should reproduce the research model family while
keeping selection evaluation-clean:

```text
all saved layers
-> concatenate by pooling mode
-> train-only scaler
-> max-dimension PCA and candidate prefixes
-> train-only LR CV selection
-> refit selected transform on all training rows
-> shared trunk ensemble
-> held-out evaluation
```

Do not include the experiment-specific hard-coded paths, synthetic layer-0
artifact, test-AUC selection, or scaler-width inference in the blueprint.
