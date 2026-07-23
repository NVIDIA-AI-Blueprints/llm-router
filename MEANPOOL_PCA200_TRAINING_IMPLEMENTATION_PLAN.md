# All-Layer Meanpool PCA-200 Training Implementation Plan

## Objective

Extend the existing NVIDIA LLM Router v3 training pipeline to support the
feature configuration selected by the research ablation:

```text
all saved encoder layers
-> mean-pool each layer over non-padding tokens
-> concatenate layers in numeric order
-> StandardScaler fit on training rows
-> PCA-200 fit on training rows
-> existing SharedTrunkNet ensemble
```

This work ships training code, configuration, tests, and documentation. It
does not commit a trained router checkpoint, Qwen prefill artifact, fitted
scaler, or fitted PCA artifact to the repository. Users create those artifacts
by running the existing `model-router train` command.

## Branch and Merge Target

Implementation will be developed on:

```text
feature/all-layer-meanpool-pca200-training
```

The branch was created directly from:

```text
base branch: v3
base commit: 655993e24266acb5a94c4132638f24c81c7f1963
PR target:   v3
```

Before final review, rebase or merge the latest `v3` into the feature branch,
run the full test suite, and confirm that the diff contains no generated
checkpoints, prefills, PCA artifacts, or research dataset files.

## Required Behavior

The initial implementation will reproduce the final ablation configuration:

```yaml
aggregation: all_layers_concat
layers: all
pooling: mean
pca_dim: 200
hidden_state_indexing: direct
```

For Qwen 3.6 35B:

```text
number of saved states: 40
hidden width:           2,048
raw feature width:      40 * 2,048 = 81,920
reduced feature width:  200
```

The existing extraction convention is `outputs.hidden_states[layer]`.
Therefore, logical layer 0 is the embedding state. This convention must remain
explicit so training and inference cannot silently disagree.

The final method is fixed to meanpool/PCA-200. The research LR and MLP probe
sweep is not part of this implementation.

## Existing User Workflow

The feature configuration must integrate into the current training command:

```bash
model-router train \
  --config configs/qwen36-35b-all-layers-mean-pca200.yaml \
  --data data/train.csv \
  --output-dir checkpoints/
```

No standalone experiment trainer, synthetic `layer_0` builder, or
experiment-specific absolute paths will be added.

Running the command will produce a local checkpoint in `--output-dir`, as the
existing trainer does. No generated checkpoint will be checked into the
blueprint repository.

## Configuration

### Files

- `src/model_router_toolkit/config.py`
- `configs/qwen36-35b-all-layers-mean-pca200.yaml`

### Changes

Add a typed feature configuration under `routing`:

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

Validation requirements:

- `aggregation` initially accepts `single_layer` and `all_layers_concat`.
- `pooling` accepts `last` and `mean`.
- `layers` accepts `all` or an explicit ordered list of non-negative integers.
- `pca_dim` must be positive.
- `hidden_state_indexing` initially accepts `direct`.
- `all_layers_concat` requires at least one resolved layer.

Backward compatibility:

- Existing configs without `routing.features` retain the current
  layer/mode/PCA sweep.
- Existing config fields and CLI commands remain valid.

## Feature-Aware Extraction

### File

- `src/model_router_toolkit/prefill/extract.py`

### Changes

Pass the requested feature specification into extraction so the extractor
knows:

- which hidden-state indexes are required;
- which pooling modes are required;
- whether all layers must be retained.

For the selected configuration:

1. Resolve `layers: all` to `range(model.config.num_hidden_layers)`.
2. Run the existing prefill forward pass with `output_hidden_states=True`.
3. For every requested layer, compute the mean over valid tokens only.
4. Do not retain last-token tensors when the feature configuration requests
   only mean pooling.
5. Save enough metadata in `PrefillResult` to validate the layer convention,
   resolved layer list, pooling modes, encoder, and row identity.

Mean pooling must exclude padding:

```text
mean(hidden_state[:sequence_length], axis=token)
```

### Cache correctness

Extend the prefill cache identity to include:

- encoder;
- chat-template arguments;
- ordered or stably identified input rows;
- resolved layer list;
- pooling modes;
- hidden-state indexing convention;
- feature-schema version.

The cache must not reuse a second-half-layer artifact for an all-layer request.

The existing question-set hash is order-insensitive while saved tensor rows
retain input order. Correct this by either:

- making the cache hash order-sensitive; or
- persisting stable row IDs and realigning tensors on load.

The second option is preferred when a `query_id` column is present.

## Native All-Layer Feature Construction

### File

- `src/model_router_toolkit/prefill/transforms.py`

### Changes

Introduce an explicit feature builder driven by the saved feature
specification.

For all-layer mean pooling:

```python
raw = np.concatenate(
    [prefill.hidden_mean[layer] for layer in resolved_layers],
    axis=1,
)
```

Requirements:

- Layers are concatenated in the exact order stored in the feature spec.
- Missing layers fail with an error listing requested and available layers.
- Row counts and hidden widths must match across layers.
- The resulting width must equal
  `len(resolved_layers) * prefill.hidden_dim`.
- No scaler-width inference is used to guess that concatenation is required.

The existing single-layer `raw_hidden` behavior remains available for legacy
training and checkpoints.

## Training Integration

### File

- `src/model_router_toolkit/prefill/train.py`

### Fixed-feature path

When `routing.features` specifies all-layer meanpool/PCA-200:

1. Load labels through the existing training path.
2. Validate that every selected target has a label for every training query.
3. Extract or load prefills using the feature-aware cache.
4. Build the 81,920-wide all-layer meanpool matrix.
5. Fit `StandardScaler` on training rows only.
6. Transform the training matrix.
7. Fit PCA with `n_components=200` and `random_state=42` on training rows only.
8. Transform the training matrix to 200 dimensions.
9. Skip the layer/mode/PCA sweep.
10. Train the existing shared-trunk ensemble.
11. Save the generated checkpoint to the user-provided output directory.

The fitted transform is label-independent and identical for every target.
Fit it once and reuse it rather than refitting four equivalent PCA objects.

### Shared-once trunk layout

The production trainer passes the transformed feature block to the
multi-output shared trunk once:

```text
shared_input = pca_features
```

The trunk input is 200-dimensional regardless of target count:

```text
200 -> 256 -> 128 -> n_targets
```

This intentionally differs from the research run, which repeated the same
PCA block once per target. The research checkpoint is not compatible with
this layout and must not be reused. The blueprint implementation retrains the
trunk with the shared-once representation.

### Existing trunk settings

Do not change the current blueprint defaults:

- hidden dimensions: `(256, 128)`;
- activations: ReLU;
- dropout: `(0.3, 0.2)`;
- loss: `BCEWithLogitsLoss`;
- optimizer: Adam;
- learning rate: `1e-3`;
- weight decay: `1e-4`;
- trunk batch size: 512;
- internal train/validation split: 85/15;
- maximum epochs: 150;
- early-stopping patience: 15;
- ensemble seeds: 10;
- retained ensemble members: 5 selected by validation loss.

The CLI extraction `--batch-size` remains separate from the trunk batch size.

## Generated Checkpoint Contract

### Files

- `src/model_router_toolkit/prefill/train.py`
- `src/model_router_toolkit/checkpoint.py`

The training command must record enough information in its generated
checkpoint for evaluation and serving to reconstruct the feature exactly:

```yaml
feature_spec:
  aggregation: all_layers_concat
  layers: [0, 1, ..., 47]
  pooling: mean
  pca_dim: 200
  hidden_state_indexing: direct
  encoder: Qwen/Qwen3.6-35B-A3B
```

The checkpoint must also contain the fitted scaler and PCA, as current
checkpoints do.

Store the fitted all-layer transform once and reference it from every target.
The trunk input layout must record that the transformed block is consumed
once:

```yaml
trunk_config:
  d_in: 200
  feature_layout: shared_once
  feature_width: 200
```

Legacy checkpoint support:

- Existing checkpoints without `feature_spec` continue through the current
  single-layer transform path.
- New checkpoints use an explicit schema version.
- Loading a new checkpoint with inconsistent feature dimensions fails before
  encoder inference.

No checkpoint file is added to source control.

## Scoring and Evaluation

### Files

- `src/model_router_toolkit/prefill/scorer.py`
- `src/model_router_toolkit/evaluate.py`

### Changes

Both paths must use the same feature builder as training:

1. Read the generated checkpoint's feature specification.
2. Determine the union of required encoder layers.
3. Extract all required mean-pooled layer states in one encoder pass.
4. Concatenate layers in saved order.
5. Apply the saved scaler and PCA.
6. Pass the 200-dimensional feature to the shared trunk once.
7. Run the existing trunk ensemble.

Training, batch evaluation, and single-query scoring must not contain separate
implementations of all-layer concatenation.

## CLI

### Files

- `src/model_router_toolkit/__main__.py`
- `src/model_router_toolkit/train.py`

The standard command remains unchanged. The feature behavior comes from the
YAML configuration.

Optional CLI overrides may be added only if they map directly to typed config
fields. The YAML remains the source of truth for reproducible training.

No command should accept the held-out test set during training.

## Tests

### Unit tests

Update or add tests under:

- `tests/test_config.py`
- `tests/test_transforms.py`
- `tests/test_sweep.py`
- `tests/test_trunk.py`
- `tests/test_checkpoint.py`

Required coverage:

1. Parse and validate the all-layer meanpool/PCA-200 config.
2. Resolve `layers: all` in numeric order.
3. Mean pooling excludes padding.
4. All-layer concatenation has the expected shape and ordering.
5. Missing or inconsistent layers fail clearly.
6. Scaler and PCA are fit only on training rows.
7. Fixed-feature mode bypasses the sweep.
8. The fitted transform is computed once and reused across targets.
9. Target count does not change the 200-wide trunk input.
10. Cache keys differ when layer sets, pooling, or indexing semantics differ.
11. Cache loads preserve row alignment.
12. Legacy checkpoints continue to load.
13. New checkpoints preserve the complete feature specification.

### Integration tests

Add a synthetic end-to-end test:

```text
synthetic multi-layer PrefillResult
-> all-layer mean concatenation
-> scaler/PCA
-> shared-trunk training
-> generated checkpoint
-> batch evaluation
-> single-query scoring
```

Assertions:

- train-time, evaluation-time, and scorer-time features are numerically equal;
- output probabilities have one column per target;
- no test labels are available to the trainer;
- no binary artifact is required from the research repository.

Add an optional slow test using a small public causal language model. Do not
require Qwen 3.6 35B or a GPU for default CI.

## Documentation

### Files

- `docs/guide-training-and-evaluation.md`
- example config under `configs/`

Document:

- the all-layer feature equation;
- direct hidden-state indexing semantics;
- memory and runtime implications;
- train-only scaler/PCA fitting;
- the fixed meanpool/PCA-200 choice;
- the shared-once 200-dimensional trunk input;
- the unchanged `model-router train` workflow;
- that no pretrained checkpoint for this configuration ships with the code.

## Implementation Sequence

1. Add and test the typed feature configuration.
2. Add feature-aware extraction and correct cache identity.
3. Add the native all-layer transform builder.
4. Add the fixed-feature branch to the existing trainer.
5. Add the generated checkpoint schema and legacy fallback.
6. Update scorer and evaluation to share the transform implementation.
7. Add unit and synthetic end-to-end tests.
8. Add the example YAML and training documentation.
9. Run the default blueprint test suite.
10. Run an optional private regression using the research data without adding
    that data or any generated artifacts to the blueprint repository.

## Acceptance Criteria

The implementation is complete when:

- the normal `model-router train` command can select this behavior through
  configuration;
- the trainer natively constructs all-layer meanpool features without a
  synthetic prefill wrapper;
- PCA is fixed to 200 and fitted only on training rows;
- the existing shared-trunk training procedure is unchanged;
- a generated checkpoint can be evaluated and served with identical feature
  construction;
- legacy configurations and checkpoints still work;
- default CI does not download or load Qwen 3.6 35B;
- no trained checkpoint, prefill tensor, or fitted transform is committed;
- the test set has no role in feature selection or training.

## Non-Goals

The first implementation does not include:

- the LR/MLP research probe sweep;
- automatic selection among last and mean pooling;
- automatic PCA-dimension selection;
- a last-token all-layer production configuration;
- compatibility with the repeated 800-wide research checkpoint;
- research dataset paths or SWE-Smith-specific data loaders;
- a committed trained checkpoint or prefill artifact.
