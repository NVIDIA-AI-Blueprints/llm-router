# User Journey Review -- Model Router Toolkit

> **Date:** March 13, 2026 | **Reviewer:** Automated Agent | **Scope:** Journey 1 (Explore & Evaluate) | **Duration:** ~5 min

---

## Executive Summary

**Overall Readiness Score: 6.5 / 10**

Journey 1's two notebooks (KMeans and Prefill) are technically functional — every step executes without errors, API calls succeed, and cost savings are displayed. However, the demo has a critical narrative failure: **both the easy and hard example questions route to the same cheapest model in both notebooks**, undermining the core value proposition of "different models for different complexity." The default tolerance of 0.20 is too generous for the chosen example questions, so the router never escalates to a stronger model. A user completing this journey would see 90% cost savings but zero routing differentiation, leaving them unable to answer: "Why do I need a router instead of just using the cheapest model?" The fix is straightforward — better example questions or a lower demo tolerance — making this a high-impact, low-effort improvement.

### Top Blockers

| # | Blocker | Journeys Affected | Severity |
|---|---------|-------------------|----------|
| 1 | Demo questions don't trigger model differentiation — both route to cheapest model in both notebooks | J1 (both tracks) | Major |

### Top Quick Wins

| # | Quick Win | Impact | Effort | Journeys |
|---|-----------|--------|--------|----------|
| 1 | Use lower tolerance (0.05) or find questions that trigger escalation at tolerance 0.20 | High | S | J1 |
| 2 | Add cross-notebook navigation cells ("Try the other notebook") | Medium | S | J1 |
| 3 | Fix incomplete LiteLLM code in "What's Next" sections (missing `set_litellm_router`) | Medium | S | J1 |
| 4 | Suppress/contextualize confusing warnings in prefill notebook | Medium | S | J1 Track B |

### Key Metrics

| Metric | Value |
|--------|-------|
| Journeys reviewed | 1 (2 tracks) |
| Steps executed | 18 (8 KMeans + 8 Prefill + 2 cross-comparison) |
| Steps passed | 16 (89%) |
| Steps passed with workaround | 0 (0%) |
| Steps failed | 0 (0%) |
| Steps with issues noted | 2 (11%) — routing differentiation + cross-comparison |
| Implementation plans produced | 4 |

### Resources Available

| Resource | Status | Notes |
|----------|--------|-------|
| NVIDIA_API_KEY | Set | build.nvidia.com APIs working |
| OPENROUTER_API_KEY | Set | Not needed for J1 |
| GPU | CPU only | No CUDA, no MPS; prefill runs ~5s/query on CPU |
| Docker | Not tested | Not needed for J1 |
| Checkpoints | All present | `kmeans_c100_db.pkl` (845KB), `prefill_qwen08b.pt` (7.6MB) |

---

## Per-Journey Reviews

### Journey 1: Explore & Evaluate (The Evaluator)

**Persona:** AI team leads, PMs, or developers evaluating whether intelligent model routing belongs in their stack.
**Job to Be Done:** "Show me this works — in 5 minutes, with no infrastructure."
**Overall Result:** Partial | **Duration:** ~3 min (KMeans) + ~2.5 min (Prefill, encoder pre-cached)

#### Step Execution Log

**Track A: KMeans Routing**

| Step | Action | Status | Wall Clock | Notes |
|------|--------|--------|------------|-------|
| 1.1 | Open KMeans quickstart | Pass | -- | Notebook loads cleanly in Jupyter |
| 1.2 | Enter NVIDIA API key | Pass | -- | Auto-detected from env var, 70 chars confirmed |
| 1.3 | Load pre-trained KMeans router | Pass | 11.3s | 100 clusters, 3 models. Slow pickle load for 845KB file (sklearn deserialization overhead) |
| 1.4 | Embed a question via API | Pass | 1.3s | 2048-dim vector returned from `nvidia/llama-nemotron-embed-1b-v2` |
| 1.5 | See KMeans routing decision | Pass | 38ms | nem-nothink selected (Nemotron 3 Nano), probabilities displayed |
| 1.6 | Call selected model | Pass | 0.4s (easy), 1.1s (hard) | Correct responses returned |
| 1.7 | Compare easy vs. hard question | **Issue** | -- | Both questions route to the same model (nem-nothink). No differentiation visible. |
| 1.8 | Review cost savings | Pass | -- | 90% savings displayed, but savings are trivial since both use cheapest model |

**Track B: Prefill Routing**

| Step | Action | Status | Wall Clock | Notes |
|------|--------|--------|------------|-------|
| 1.9 | Open prefill quickstart | Pass | -- | Notebook loads cleanly |
| 1.10 | Enter API key(s) | Pass | -- | NVIDIA_API_KEY auto-detected |
| 1.11 | Load encoder model | Pass | 17.1s | Qwen3.5-0.8B loads on CPU. 3 confusing warnings emitted (see Issues) |
| 1.12 | Extract hidden states | Pass | ~5.8s | Hidden state tensor extracted on CPU |
| 1.13 | See prefill routing decision | Pass | <1ms (MLP) | nem-nothink selected, P(correct) values displayed |
| 1.14 | Call selected model | Pass | 1.6s (easy), 2.5s (hard) | Correct responses returned |
| 1.15 | Compare easy vs. hard question | **Issue** | -- | Both route to nem-nothink. Probabilities differ (0.999 vs 0.885) but tolerance=0.20 doesn't trigger escalation. |
| 1.16 | Review cost savings | Pass | -- | 90% savings — same issue as Track A |

**Cross-Method Comparison**

| Step | Action | Status | Wall Clock | Notes |
|------|--------|--------|------------|-------|
| 1.17 | Compare both methods | **Missing** | -- | No cross-comparison cell or summary exists in either notebook |
| 1.18 | Understand tradeoffs | **Missing** | -- | No method comparison table in notebooks (exists only in user-journeys doc) |
| 1.19 | Decide next step | Pass | -- | Both notebooks have "What's Next" sections with 3 clear paths |

#### What Worked Well

- **Zero-dependency KMeans notebook** (`notebooks/quickstart.ipynb`): Requires only `requests`, `numpy`, `scikit-learn` — no torch, no GPU. True "5-minute" experience for Track A. The install cell (`%pip install -q requests numpy scikit-learn`) covers everything needed.

- **Self-contained design**: Both notebooks work without importing `model_router_toolkit`. All routing logic is inline, making the demo portable and debuggable. A user can understand the full pipeline from the notebook alone.

- **API key handling** (`notebooks/quickstart.ipynb` cell 5, `notebooks/quickstart-prefill.ipynb` cell 5): Clean pattern — checks env var first, falls back to `input()` with a direct link to `build.nvidia.com`. No unnecessary friction.

- **Visual probability bars** (cell 13 in both notebooks): The Unicode block bar chart (`█░`) effectively communicates relative confidence. Users can immediately see which model the router is most confident about.

- **Accurate latency documentation** (`notebooks/quickstart-prefill.ipynb` cell 6): The note "On GPU, the encoder runs in under 200ms per query. On CPU, expect ~5 seconds per routing decision" is accurate — we measured 5.0-5.8s on CPU. This sets expectations correctly.

- **Memory cleanup cell** (`notebooks/quickstart-prefill.ipynb` cell 23): Properly deletes encoder and tokenizer, calls `gc.collect()`, and clears CUDA cache if available. Good practice for a notebook that loads a 0.8B model.

- **Checkpoint bundled in repo**: Both `kmeans_c100_db.pkl` and `prefill_qwen08b.pt` are present in `checkpoints/`. No external download required. This is critical for the "zero friction" promise.

- **Both API endpoints work**: NVIDIA embedding API (`nvidia/llama-nemotron-embed-1b-v2`) and chat completions API (`nvidia/nemotron-3-nano-30b-a3b`, `openai/gpt-oss-20b`) all return valid responses. Endpoints are live.

#### Documentation Gaps and Workarounds

| Gap | Missing Info | Workaround Used | What Docs Should Say |
|-----|-------------|-----------------|---------------------|
| No cross-notebook navigation | Neither notebook mentions the other exists | N/A — user must discover independently | Add a cell at the top: "This notebook shows KMeans routing. For higher-accuracy prefill routing, see `quickstart-prefill.ipynb`." (and vice versa) |
| Incomplete LiteLLM code in "What's Next" | Both notebooks show 3-line LiteLLM integration that's missing `strategy.set_litellm_router(router)` | N/A — not executed in notebook | Add the missing line: `strategy.set_litellm_router(router)` before `router.set_custom_routing_strategy(strategy)` |
| No tolerance guidance | Tolerance=0.20 used throughout with no explanation of what different values do | N/A | Add a cell: "Try different tolerance values: 0.05 (prefer accuracy), 0.20 (balanced), 0.40 (prefer cost)" with a loop showing how model selection changes |
| Checkpoint download instructions missing | Checkpoints are bundled in the repo but the notebook doesn't explain where they come from or how to obtain them | N/A — already present | Add note: "Pre-trained checkpoints are included in the repo at `checkpoints/`. To train your own, see the Training Guide." |
| KMeans notebook "What's Next" serve command | References `configs/cloud-only.yaml` which is correct for KMeans, but a user might try the default `configs/prefill-qwen08b.yaml` from the README instead | N/A | Clarify: "Use `cloud-only.yaml` for KMeans routing (this notebook's method). For prefill routing, use `prefill-qwen08b.yaml`." |

#### Issues and Bugs

| Issue | Severity | Error | Reproduction | File |
|-------|----------|-------|--------------|------|
| Both demo questions route to same model in both notebooks | Major | N/A (logical issue) | Run cells 14-15 in either notebook — both select `nem-nothink` | `notebooks/quickstart.ipynb` cell 14-15, `notebooks/quickstart-prefill.ipynb` cell 14-15 |
| Prefill notebook emits "fast path not available" warning | Minor | `The fast path is not available because one of the required library is not installed.` | Run cell 7 in prefill notebook | `notebooks/quickstart-prefill.ipynb` cell 7 |
| Prefill notebook emits "unauthenticated HF Hub" warning | Minor | `Warning: You are sending unauthenticated requests to the HF Hub.` | Run cell 7 without `HF_TOKEN` set | `notebooks/quickstart-prefill.ipynb` cell 7 |
| Prefill notebook emits cache permission errors | Minor | `Could not cache non-existence of file. Will ignore error and continue. Error: [Errno 1] Operation not permitted` | Run cell 7 in sandboxed environment | `notebooks/quickstart-prefill.ipynb` cell 7 |
| KMeans checkpoint load takes 11s for 845KB file | Minor | N/A (performance) | `pickle.load()` of `kmeans_c100_db.pkl` | `notebooks/quickstart.ipynb` cell 7 |
| KMeans notebook model order in display differs from probabilities | Minor | N/A (cosmetic) | Cell 13 displays `gptoss-high, nem-think, nem-nothink` (expensive-to-cheap) but `MODELS_BY_COST` is cheap-to-expensive | `notebooks/quickstart.ipynb` cell 13 |

#### Workaround Log

No steps required workarounds — all steps executed successfully. The routing differentiation issue is a design/demo issue, not a runtime failure.

#### Recommendations

| # | Recommendation | Impact | Effort | Plan |
|---|---------------|--------|--------|------|
| 1 | Fix demo questions to show routing differentiation | High | S | [Plan 1](#implementation-plan-1-fix-demo-routing-differentiation) |
| 2 | Add cross-notebook comparison and navigation | Medium | S | [Plan 2](#implementation-plan-2-add-cross-notebook-navigation-and-comparison) |
| 3 | Fix LiteLLM "What's Next" code examples | Medium | S | [Plan 3](#implementation-plan-3-fix-whats-next-litellm-code) |
| 4 | Suppress/contextualize confusing prefill warnings | Medium | S | [Plan 4](#implementation-plan-4-clean-up-prefill-notebook-warnings) |
| 5 | Add tolerance sensitivity demonstration cell | Low | S | -- |
| 6 | Add "convinced moment" summary cell quantifying value | Low | S | -- |

#### Scorecard

| Dimension | Score (1-5) | Notes |
|-----------|-------------|-------|
| Completeness | 3 | All steps execute, but cross-comparison (Steps 1.17-1.18) is completely missing. The demo doesn't demonstrate the key feature (routing differentiation). |
| Clarity | 4 | Well-written markdown, good visual bars, clear latency note in prefill notebook. Pipeline diagrams are excellent. Minor gap: no tolerance explanation. |
| Executability | 4 | Every command works first try. API keys handled cleanly. Checkpoints present. One slow step (encoder load 17s) is documented. |
| Error Handling | 3 | API key check is good. But 3 confusing warnings in prefill notebook are not addressed. No guidance on what to do if API calls fail. |
| Continuity | 3 | "What's Next" sections exist with 3 paths each, but: no link between notebooks, no cross-comparison, LiteLLM code example is incomplete. |

#### JTBD Verification

| JTBD | Documented Status | Verified Status | Notes |
|------|-------------------|-----------------|-------|
| Understand what routing does | Partial | Confirmed Partial | Pipeline diagrams are excellent, but the demo doesn't show routing actually working (different models for different queries) |
| See a live KMeans routing decision | Complete | Confirmed | Routing decision displayed with probabilities and selected model |
| See a live prefill routing decision | Complete | Confirmed | Routing decision displayed with P(correct) per model |
| Compare both routing methods | Partial | Confirmed Partial | Two separate notebooks, no shared comparison. Both use same example questions by coincidence. |
| Verify cost savings | Complete | **Downgraded to Partial** | Savings shown (90%), but misleading — savings come from always using cheapest model, not from intelligent routing. Router never escalates. |
| Understand accuracy tradeoffs | Partial | Confirmed Partial | Tolerance parameter present but not explained. No visualization of tolerance impact. |
| Understand infrastructure tradeoffs | Partial | Confirmed Partial | Implicit from running both (KMeans=fast+cloud, Prefill=slow+local) but not explicitly compared |
| Share results with team | Partial | Confirmed Partial | Notebook output is shareable but no export mechanism |

---

## Cross-Journey Analysis

### Common Patterns

- **Routing differentiation gap**: Both notebooks suffer from the same issue — the demo questions don't trigger model escalation at the default tolerance. This is systemic: the 3-model pool's probability distributions are too close together at tolerance=0.20 for these particular questions.

- **Self-contained notebooks**: Both notebooks avoid importing `model_router_toolkit`, implementing routing logic inline. This is good for portability but means the notebook code doesn't match what a user would write in production (using `build_router_from_config()`).

- **Consistent API patterns**: Both notebooks use the same `build.nvidia.com` API, same model names, same cost structure. No naming inconsistencies between notebooks.

### Shared Infrastructure Gaps

| Gap | Journeys Affected | Single Fix |
|-----|-------------------|------------|
| No cross-notebook comparison | J1 Track A, J1 Track B | Add a "Method Comparison" section to both notebooks with a shared comparison cell |
| Incomplete LiteLLM code in "What's Next" | J1 Track A, J1 Track B | Add `strategy.set_litellm_router(router)` to both "What's Next" cells |
| No tolerance sensitivity demo | J1 Track A, J1 Track B | Add a cell in each notebook that routes at 3 different tolerances |

### Documentation Consistency

| Issue | Files | Description |
|-------|-------|-------------|
| Model order mismatch | `quickstart.ipynb` cell 2 vs cell 13 | Cell 2 lists models Nano → Think → GPT-OSS (cheap to expensive). Cell 13 displays them GPT-OSS → Think → Nano (expensive to cheap). Inconsistent ordering. |
| Prefill notebook mentions "plus one more internally" | `quickstart-prefill.ipynb` cell 2 | Checkpoint has 4 models but only 3 are exposed. The "one more internally" note (gpt-5.2) is cryptic and may confuse evaluators. |
| Train command output path differs | `quickstart.ipynb` cell 22, `quickstart-prefill.ipynb` cell 22 | KMeans: `checkpoints/custom/router.pkl`, Prefill: `checkpoints/custom/prefill.pt`. Different naming conventions. |

### Journey Progression

| From | To | Transition Quality | Friction |
|------|----|-------------------|----------|
| J1 Track A | J1 Track B | Rough | No link from KMeans notebook to prefill notebook. User must discover the other notebook independently. |
| J1 Track B | J1 Track A | Rough | Same — no cross-reference |
| J1 | J2 (Deploy) | Smooth | "What's Next" provides clear `model-router serve` command with config path |
| J1 | J3 (Train) | Smooth | "What's Next" provides collect → train → evaluate pipeline commands |
| J1 | J5 (LiteLLM) | Rough | Code example is incomplete (missing `set_litellm_router`). Would fail if copy-pasted. |

---

## Implementation Plans

### Implementation Plan 1: Fix Demo Routing Differentiation

**Source:** Journey 1, Steps 1.7 and 1.15
**Impact:** High | **Effort:** S | **Priority:** P0
**Category:** UX Improvement

#### Problem Statement

The core value proposition of the router is "different models for different complexity queries." But in both demo notebooks, the easy question ("What is the capital of France?") and the hard question ("Prove that the square root of 2 is irrational") both route to the cheapest model (Nemotron 3 Nano / nem-nothink) at the default tolerance of 0.20. A first-time evaluator sees 90% cost savings but zero routing intelligence — they'd reasonably conclude "just use the cheapest model always."

#### Current Behavior

KMeans (tolerance=0.20):
```
Easy:  nem-nothink  (probs: gptoss-high=0.641, nem-think=0.764, nem-nothink=0.651)
Hard:  nem-nothink  (probs: gptoss-high=0.556, nem-think=0.722, nem-nothink=0.604)
```

Prefill (tolerance=0.20):
```
Easy:  nem-nothink  (probs: gptoss-high=0.999, nem-think=1.000, nem-nothink=0.999)
Hard:  nem-nothink  (probs: gptoss-high=0.947, nem-think=0.981, nem-nothink=0.885)
```

Both questions → same model. No differentiation visible.

#### Expected Behavior

Easy question → cheapest model (nem-nothink). Hard question → stronger model (nem-think or gptoss-high). The demo should visually demonstrate the router making different decisions for different complexity levels.

#### Proposed Solution

**Option A (Preferred): Use a lower demo tolerance + better example questions.**

Find a pair of questions where the prefill router's probability spread is wider than 0.20, so the demo works at the default tolerance. Alternatively, use tolerance=0.05 for the demo cells (which would route the hard question to nem-think in both notebooks based on observed spreads).

**`notebooks/quickstart.ipynb`** and **`notebooks/quickstart-prefill.ipynb`**:
- Change the demo tolerance from 0.20 to 0.05, OR
- Find example questions where the probability spread naturally exceeds 0.20
- Add a tolerance explanation: "We use tolerance=0.05 for this demo to show clear differentiation. In production, higher tolerances (0.15-0.25) aggressively optimize cost."

**Option B (Complementary): Add a tolerance sweep cell.**

After the two example questions, add a cell that shows how tolerance affects model selection:

```python
print("How tolerance affects model selection:")
print(f"{'Tolerance':>10}  {'Easy Question':>20}  {'Hard Question':>20}")
for tol in [0.01, 0.05, 0.10, 0.20, 0.30]:
    r_easy = route(emb_easy, tolerance=tol)  # or route(question, tol) for prefill
    r_hard = route(emb_hard, tolerance=tol)
    print(f"{tol:>10.2f}  {MODEL_CONFIG[r_easy['selected_model']]['display']:>20}  {MODEL_CONFIG[r_hard['selected_model']]['display']:>20}")
```

#### Files to Modify

| File | Change Type | Description |
|------|------------|-------------|
| `notebooks/quickstart.ipynb` | Edit | Change demo tolerance and/or example questions in cells 14-15 |
| `notebooks/quickstart-prefill.ipynb` | Edit | Same changes in cells 14-15 |

#### Testing

1. Run both notebooks end-to-end
2. Verify that the easy question routes to a cheaper model than the hard question
3. Verify that the "Results" summary cell shows different models for easy vs. hard

#### Acceptance Criteria

- [ ] Easy and hard demo questions route to different models in KMeans notebook
- [ ] Easy and hard demo questions route to different models in Prefill notebook
- [ ] Cost savings comparison shows the value of routing (not just "always use cheapest")
- [ ] Existing tests still pass: `pytest tests/ --ignore=tests/integration/ -v`

#### Dependencies

None — this plan can be implemented independently.

#### Effort Breakdown

| Task | Estimate |
|------|----------|
| Test tolerance values and candidate questions against both checkpoints | 20 min |
| Update notebook cells with new questions/tolerance | 10 min |
| Re-run both notebooks to capture new outputs | 10 min |
| **Total** | **~40 min** |

---

### Implementation Plan 2: Add Cross-Notebook Navigation and Comparison

**Source:** Journey 1, Steps 1.17-1.18
**Impact:** Medium | **Effort:** S | **Priority:** P1
**Category:** Documentation Gap

#### Problem Statement

The two notebooks (KMeans and Prefill) exist as isolated documents. An evaluator completing Track A has no idea Track B exists, and vice versa. Steps 1.17-1.18 (cross-method comparison) are completely missing — there's no cell in either notebook that compares the two methods or links to the other notebook.

#### Current Behavior

Each notebook stands alone with no reference to the other. A user completing the KMeans notebook would need to independently discover the prefill notebook by browsing the `notebooks/` directory.

#### Expected Behavior

- Each notebook has a "See Also" cell at the top linking to the other notebook
- Each notebook has a "Method Comparison" cell at the bottom (before "What's Next") that summarizes the tradeoffs

#### Proposed Solution

**Add to both notebooks:**

1. A cell after cell 0 (intro) in each notebook:

```markdown
> **Two routing methods available:**
> - **This notebook**: KMeans routing — cloud-only, no GPU needed, ~100ms routing latency
> - **Also available**: [`quickstart-prefill.ipynb`](quickstart-prefill.ipynb) — encoder-based routing, higher accuracy, requires torch
```

(And the reverse for the prefill notebook.)

2. A "Method Comparison" cell before "What's Next" in each notebook:

```markdown
## KMeans vs. Prefill Comparison

| Dimension | KMeans (this notebook) | Prefill |
|-----------|----------------------|---------|
| Dependencies | requests, numpy, sklearn | + torch, transformers |
| GPU required | No | No (but recommended) |
| Routing latency | ~100ms (API call) | ~200ms GPU / ~5s CPU |
| Accuracy | Good | Higher |
| Best for | Quick eval, cloud-only | Production, domain tuning |
```

#### Files to Modify

| File | Change Type | Description |
|------|------------|-------------|
| `notebooks/quickstart.ipynb` | Edit | Add cross-reference cell after intro, add comparison table before "What's Next" |
| `notebooks/quickstart-prefill.ipynb` | Edit | Same changes, reversed perspective |

#### Testing

1. Open each notebook and verify the cross-reference links work
2. Verify the comparison table is accurate

#### Acceptance Criteria

- [ ] KMeans notebook links to prefill notebook
- [ ] Prefill notebook links to KMeans notebook
- [ ] Both notebooks contain a method comparison table
- [ ] Links use relative paths that work in Jupyter

#### Dependencies

None — this plan can be implemented independently.

#### Effort Breakdown

| Task | Estimate |
|------|----------|
| Write cross-reference cells | 10 min |
| Write comparison table | 10 min |
| Add cells to both notebooks | 10 min |
| **Total** | **~30 min** |

---

### Implementation Plan 3: Fix "What's Next" LiteLLM Code

**Source:** Journey 1, Step 1.19 (Continuity to J5)
**Impact:** Medium | **Effort:** S | **Priority:** P1
**Category:** Bug Fix (Documentation)

#### Problem Statement

Both notebooks' "What's Next" sections include a LiteLLM SDK integration example that's missing a required line. If a user copy-pastes this code, it will fail because the strategy doesn't have a reference to the LiteLLM router.

#### Current Behavior

```python
from litellm import Router
from model_router_toolkit import ModelRoutingStrategy

router = Router(model_list=my_models)
strategy = ModelRoutingStrategy.from_config("pool_config.yaml")
router.set_custom_routing_strategy(strategy)
```

#### Expected Behavior

```python
from litellm import Router
from model_router_toolkit import ModelRoutingStrategy

router = Router(model_list=my_models)
strategy = ModelRoutingStrategy.from_config("configs/prefill-qwen08b.yaml")
strategy.set_litellm_router(router)
router.set_custom_routing_strategy(strategy)
```

#### Proposed Solution

Add the missing `strategy.set_litellm_router(router)` line. Also update the config path from generic `pool_config.yaml` to a real example path (`configs/prefill-qwen08b.yaml` or `configs/cloud-only.yaml` depending on the notebook).

#### Files to Modify

| File | Change Type | Description |
|------|------------|-------------|
| `notebooks/quickstart.ipynb` | Edit | Fix LiteLLM code in "What's Next" cell (cell 22). Use `configs/cloud-only.yaml`. |
| `notebooks/quickstart-prefill.ipynb` | Edit | Fix LiteLLM code in "What's Next" cell (cell 22). Use `configs/prefill-qwen08b.yaml`. |

#### Acceptance Criteria

- [ ] LiteLLM code in both notebooks includes `strategy.set_litellm_router(router)`
- [ ] Config paths reference real config files from the repo

#### Dependencies

None — this plan can be implemented independently.

#### Effort Breakdown

| Task | Estimate |
|------|----------|
| Update cell 22 in both notebooks | 5 min |
| **Total** | **~5 min** |

---

### Implementation Plan 4: Clean Up Prefill Notebook Warnings

**Source:** Journey 1, Step 1.11
**Impact:** Medium | **Effort:** S | **Priority:** P1
**Category:** UX Improvement

#### Problem Statement

When loading the encoder in the prefill notebook, three confusing warnings appear:
1. "The fast path is not available because one of the required library is not installed."
2. "Warning: You are sending unauthenticated requests to the HF Hub."
3. "Could not cache non-existence of file. Will ignore error and continue."

These are harmless but alarm first-time users who may think something is broken. The "Evaluator" persona (AI team leads, PMs) is especially likely to be concerned by warning messages.

#### Current Behavior

Cell 7 output includes 3+ warning lines mixed with the loading progress bar.

#### Expected Behavior

Either:
- Warnings suppressed with `warnings.filterwarnings` and `transformers.logging.set_verbosity_error()`
- Or a markdown note above the cell: "You may see warnings about 'fast path' and 'HF Hub' — these are harmless and can be ignored."

#### Proposed Solution

Add warning suppression before the encoder load:

```python
import warnings
warnings.filterwarnings("ignore", message=".*fast path.*")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")

import transformers
transformers.logging.set_verbosity_error()
```

And add a markdown note in cell 6:

```markdown
> **Note:** You may see warnings about "fast path" and "HF Hub" during encoder loading — these are harmless. The encoder works correctly without the optional flash-attention library.
```

#### Files to Modify

| File | Change Type | Description |
|------|------------|-------------|
| `notebooks/quickstart-prefill.ipynb` | Edit | Add warning suppression to cell 7, add note to cell 6 |

#### Acceptance Criteria

- [ ] No confusing warnings appear during encoder loading
- [ ] Or: a clear note explains warnings before they appear
- [ ] Encoder still loads and routes correctly

#### Dependencies

None — this plan can be implemented independently.

#### Effort Breakdown

| Task | Estimate |
|------|----------|
| Add warning filters and/or documentation note | 10 min |
| Re-run notebook to verify clean output | 5 min |
| **Total** | **~15 min** |

---

## Appendix

### A. Full Command Log

```
13:00:00 $ python3 -c "[KMeans Track A execution script]"
Exit: 0 | Duration: 22.1s
  - Checkpoint load: 11.3s
  - Embed (easy): 1.3s
  - Route (easy): 38ms
  - Embed (hard): 351ms
  - API call (easy): 0.4s
  - API call (hard): 1.1s

13:00:00 $ python3 -c "[Prefill Track B execution script]"
Exit: 0 | Duration: 148.6s
  - Checkpoint load: 380ms
  - Encoder load: 17.1s
  - Route (easy): 5.8s
  - Route (hard): 5.0s
  - API call (easy): 1.6s
  - API call (hard): 2.5s

13:02:30 $ python3 -c "[Tolerance sensitivity test]"
Exit: 0 | Duration: 67.6s
  - Tolerance 0.30 → both nem-nothink
  - Tolerance 0.20 → both nem-nothink
  - Tolerance 0.15 → both nem-nothink
  - Tolerance 0.10 → both nem-think
  - Tolerance 0.05 → both nem-think
  - Tolerance 0.01 → both nem-think
```

### B. Environment Snapshot

```
Python: 3.12.9 (miniforge3)
pip: 25.0.1
OS: macOS darwin 25.1.0
GPU: CPU only (CUDA: False, MPS: False)
torch: 2.7.0
transformers: installed (Qwen3.5-0.8B compatible)
Docker: not tested (not needed for J1)
model-router-toolkit: 0.1.0 (editable install)
```

### C. Artifact Inventory

| Artifact | Produced By | Used By | Path |
|----------|-------------|---------|------|
| KMeans checkpoint | Pre-bundled | Step 1.3 | `checkpoints/kmeans_c100_db.pkl` |
| Prefill checkpoint | Pre-bundled | Step 1.11 | `checkpoints/prefill_qwen08b.pt` |
| Qwen3.5-0.8B encoder | HuggingFace (cached) | Step 1.11 | `~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/` |
| Embedding vector (easy) | Step 1.4 | Step 1.5 | In-memory |
| Embedding vector (hard) | Step 1.4 | Step 1.7 | In-memory |
| Route result (easy) | Step 1.5 / 1.13 | Step 1.6 / 1.14 | In-memory |
| Route result (hard) | Step 1.7 / 1.15 | Step 1.6b / 1.14b | In-memory |

### D. Tolerance Sensitivity Analysis

The default tolerance of 0.20 is too generous for the example questions. Here's how tolerance affects routing with the KMeans checkpoint:

```
Tolerance    Easy Question       Hard Question
-------------------------------------------------
     0.30    nem-nothink         nem-nothink
     0.20    nem-nothink         nem-nothink    ← default
     0.15    nem-nothink         nem-nothink
     0.10    nem-think           nem-think
     0.05    nem-think           nem-think
     0.01    nem-think           nem-think
```

Key observation: at no tolerance level do the easy and hard questions route to *different* models. The KMeans checkpoint's probability distributions for these two questions are structurally similar. The prefill checkpoint shows more spread (0.999 vs 0.885 for nem-nothink), which would differentiate at tolerance ~0.10, but the default 0.20 still selects the cheapest model for both.

**Implication**: To demonstrate differentiation, the demo needs either (a) different example questions with wider probability spread, or (b) a tolerance low enough that the hard question's spread crosses the threshold.
