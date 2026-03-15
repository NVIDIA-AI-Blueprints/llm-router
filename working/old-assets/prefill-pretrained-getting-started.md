# Pretrained Prefill Router — Getting Started Flow

**Author:** Nave Algarici
**Created:** March 13, 2026
**Status:** Planning

## Problem

The current getting-started flow requires users to run the full training pipeline (collect → train → evaluate → serve), which means:

1. Collecting labeled data by calling every model via API (requires API keys, $$, time)
2. Running encoder extraction (~5s/question on CPU, needs torch + the encoder model)
3. Running the sweep (many CV iterations across layer/mode/PCA combos)
4. Training the MLP and saving a checkpoint

For users who just want to try the router, this is a high barrier. We should ship **pre-extracted encoder outputs and sweep results** so users can skip steps 1–3 and jump straight to fitting PCA + training the MLP on bundled data.

## What Ships in the Repo

| Artifact | Path | Description |
|----------|------|-------------|
| Labeled CSV (train) | `data/bundled-train.csv` | Pre-collected training data (question, model, isCorrect, output_tokens) |
| Labeled CSV (test) | `data/bundled-test.csv` | Held-out test data |
| Prefill cache | `data/prefill-cache/prefill_Qwen_Qwen3.5-0.8B_<hash>.pt` | Extracted hidden states for all bundled questions |
| Sweep results | `data/bundled-sweep.json` | Best (layer, mode, pca_dim) per target model |
| Pool config | `configs/prefill-qwen08b.yaml` | Already exists |

The prefill cache is the largest artifact (~100s of MB depending on question count). Consider Git LFS or a download script if it exceeds reasonable repo size.

## User Journey: "Train a Router in 2 Minutes"

```bash
# 1. Install
pip install -e '.[prefill]'

# 2. Train from bundled data (no API keys, no encoder download, no GPU)
model-router train \
    --config configs/prefill-qwen08b.yaml \
    --data data/bundled-train.csv \
    --output-dir checkpoints/ \
    --prefill-dir data/prefill-cache/ \
    --sweep-config data/bundled-sweep.json   # <-- NEW FLAG

# 3. Evaluate
model-router evaluate \
    --config configs/prefill-qwen08b.yaml \
    --checkpoint checkpoints/prefill_router.pt \
    --data data/bundled-test.csv \
    --prefill-dir data/prefill-cache/

# 4. Serve (needs API key for inference, not for routing)
export OPENROUTER_API_KEY=sk-or-...
model-router serve --config configs/prefill-qwen08b.yaml --port 8000
```

Step 2 skips extraction (loads from cache) and skips the sweep (loads from JSON). Only PCA fitting + MLP training run, which takes ~10 seconds on CPU.

## Implementation Plan

### 1. Add SweepResult serialization

**File:** `src/model_router_toolkit/prefill/sweep.py`

Add `to_dict()` / `from_dict()` to `SweepResult` and top-level `save_sweep_results()` / `load_sweep_results()` functions:

```python
@dataclass
class SweepResult:
    layer: int
    mode: str
    pca_dim: int
    cv_auc: float

    def to_dict(self) -> dict:
        return {"layer": self.layer, "mode": self.mode,
                "pca_dim": self.pca_dim, "cv_auc": self.cv_auc}

    @classmethod
    def from_dict(cls, d: dict) -> SweepResult:
        return cls(layer=d["layer"], mode=d["mode"],
                   pca_dim=d["pca_dim"], cv_auc=d["cv_auc"])

def save_sweep_results(results: dict[str, SweepResult], path: str | Path) -> None:
    """Save sweep results to JSON."""
    import json
    with open(path, "w") as f:
        json.dump({k: v.to_dict() for k, v in results.items()}, f, indent=2)

def load_sweep_results(path: str | Path) -> dict[str, SweepResult]:
    """Load sweep results from JSON."""
    import json
    with open(path) as f:
        data = json.load(f)
    return {k: SweepResult.from_dict(v) for k, v in data.items()}
```

### 2. Add `--sweep-config` parameter to `train_prefill()`

**File:** `src/model_router_toolkit/prefill/train.py`

Add optional `sweep_config` parameter. When provided, load sweep results from JSON instead of running the sweep:

```python
def train_prefill(
    config: PoolConfig,
    data_path: str | Path,
    output_dir: str | Path,
    *,
    # ... existing params ...
    sweep_config: str | Path | None = None,   # <-- NEW
) -> Path:
```

In the sweep step:

```python
if sweep_config:
    print("  [3/6] Loading sweep results from", sweep_config)
    sweep_results = load_sweep_results(sweep_config)
    # Validate that sweep covers all model_names
    missing = set(model_names) - set(sweep_results.keys())
    if missing:
        raise ValueError(f"Sweep config missing models: {missing}")
else:
    print("  [3/6] Sweeping layer/mode/PCA per target...")
    # ... existing sweep logic ...
```

### 3. Add `--sweep-config` CLI flag

**File:** `src/model_router_toolkit/__main__.py`

Add to the train subcommand parser:

```python
train_parser.add_argument(
    "--sweep-config", type=str, default=None,
    help="Path to pre-computed sweep results JSON (skips sweep step)",
)
```

Pass through to `train_prefill()`.

### 4. Add `--save-sweep` flag for sweep export

**File:** `src/model_router_toolkit/prefill/train.py` and `__main__.py`

When running a full training (with sweep), optionally save the sweep results to a file for others to reuse:

```python
train_parser.add_argument(
    "--save-sweep", type=str, default=None,
    help="Save sweep results to this JSON path after sweep completes",
)
```

In `train_prefill()`, after the sweep step:

```python
if save_sweep and sweep_results:
    save_sweep_results(sweep_results, save_sweep)
    print(f"         Saved sweep results: {save_sweep}")
```

### 5. Bundle data artifacts

Generate and commit the bundled artifacts:

```bash
# Run full pipeline once to generate artifacts
model-router train \
    --config configs/prefill-qwen08b.yaml \
    --data data/train.csv \
    --output-dir checkpoints/ \
    --prefill-dir data/prefill-cache/ \
    --save-sweep data/bundled-sweep.json

# Copy train/test CSVs
cp data/train.csv data/bundled-train.csv
cp data/test.csv data/bundled-test.csv
```

Consider whether the prefill cache should be:
- **In-repo (Git LFS)** — simplest, but large
- **Downloaded on demand** — add a `model-router download-data` command or check+download in train
- **Regenerated** — if user has GPU, regeneration is fast; skip cache for CPU-only quickstart and include the trained checkpoint instead

### 6. Update docs and quickstart

**Files:** `docs/quickstart.md`, `docs/training-guide.md`, `README.md`

Add a "Quick Train (Bundled Data)" section to the quickstart that shows the 2-minute flow. Update the training guide with a section on using pre-computed sweep results.

### 7. Tests

- Unit test for `SweepResult.to_dict()` / `from_dict()` / `save_sweep_results()` / `load_sweep_results()`
- Unit test for `train_prefill()` with `sweep_config` parameter (mock the sweep, verify it's skipped)
- Integration test: full train with `--prefill-dir` + `--sweep-config` using smoke data

## File Changes Summary

| File | Change |
|------|--------|
| `prefill/sweep.py` | Add `to_dict`, `from_dict`, `save_sweep_results`, `load_sweep_results` |
| `prefill/train.py` | Add `sweep_config` and `save_sweep` params, conditional sweep skip |
| `__main__.py` | Add `--sweep-config` and `--save-sweep` CLI flags |
| `docs/quickstart.md` | Add "Quick Train" section |
| `docs/training-guide.md` | Add "Using Pre-computed Sweep Results" section |
| `README.md` | Update getting started flow |
| `data/` | Bundle train CSV, test CSV, prefill cache, sweep JSON |
| `tests/test_sweep.py` | Add serialization tests |

## Open Questions

1. **Prefill cache size** — How large is the cache for the full training set? If >50 MB, Git LFS or a download script is needed. The `.gitattributes` already tracks `*.mhtml` via LFS; adding `*.pt` in `data/` would follow the same pattern.

2. **Should we also ship a pre-trained checkpoint?** — If the goal is "try routing fast", shipping `checkpoints/prefill_router.pt` directly lets users skip training entirely and go straight to serve. The train-from-bundled-data flow is for users who want to understand/customize the pipeline.

3. **Sweep results are encoder-specific** — The bundled sweep JSON is only valid for the same encoder (Qwen3.5-0.8B) and the same training data. Should `load_sweep_results` validate against the current config's encoder?

4. **Prefill cache naming** — The cache filename includes an encoder+template hash. If users change the config encoder, the cache won't match. Document this clearly.
