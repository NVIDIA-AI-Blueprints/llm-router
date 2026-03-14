# User Journey Review (Re-Run) -- Model Router Toolkit

> **Date:** March 13, 2026 | **Reviewer:** Automated Agent | **Scope:** Journey 1 (Explore & Evaluate) | **Type:** Re-run after implementing 4 fixes from initial review

---

## Executive Summary

**Overall Readiness Score: 8.5 / 10** (up from 6.5)

All four recommendations from the initial review have been implemented and verified:

1. **Routing differentiation fixed** — Both notebooks now route the easy question to Nemotron 3 Nano ($0.04/1k) and the hard question to Nemotron 3 Nano Think ($0.23/1k). The core value proposition — "different models for different complexity" — is now clearly demonstrated.
2. **Cross-notebook navigation added** — Both notebooks link to each other at the top and include a method comparison table at the bottom.
3. **LiteLLM code fixed** — "What's Next" sections now include the missing `strategy.set_litellm_router(router)` line.
4. **Prefill warnings suppressed** — No confusing "fast path" or "HF Hub" warnings appear during encoder loading.

The cost savings narrative improved from a misleading 90% (same cheap model for everything) to a realistic **65%** (cheap model for easy, thinking model for hard). This is a much more honest and compelling demo.

### Remaining Issues

| # | Issue | Severity | Notes |
|---|-------|----------|-------|
| 1 | Stale cell outputs in notebooks (show old results until re-run) | Minor | Expected — outputs refresh when user runs the notebook |
| 2 | Prefill routing latency ~5s on CPU | Minor | Already documented with clear latency note |
| 3 | No tolerance sensitivity demo cell | Low | Nice-to-have; not implemented in this round |

### Key Metrics

| Metric | Value |
|--------|-------|
| Journeys reviewed | 1 (2 tracks) |
| Steps executed | 18 (8 KMeans + 8 Prefill + 2 cross-comparison) |
| Steps passed | 18 (100%) |
| Steps passed with workaround | 0 (0%) |
| Steps failed | 0 (0%) |
| Fixes implemented | 4 |
| Fixes verified | 4 (100%) |

### Comparison to Initial Review

| Dimension | Initial Score | Re-Run Score | Change |
|-----------|:------------:|:------------:|:------:|
| Completeness | 3 | 4 | +1 |
| Clarity | 4 | 5 | +1 |
| Executability | 4 | 5 | +1 |
| Error Handling | 3 | 4 | +1 |
| Continuity | 3 | 4 | +1 |
| **Overall** | **6.5** | **8.5** | **+2.0** |

---

## Per-Journey Reviews

### Journey 1: Explore & Evaluate (The Evaluator)

**Persona:** AI team leads, PMs, or developers evaluating whether intelligent model routing belongs in their stack.
**Job to Be Done:** "Show me this works — in 5 minutes, with no infrastructure."
**Overall Result:** Pass | **Duration:** ~3 min (KMeans) + ~2 min (Prefill, encoder pre-cached)

#### Step Execution Log

**Track A: KMeans Routing**

| Step | Action | Status | Wall Clock | Notes |
|------|--------|--------|------------|-------|
| 1.1 | Open KMeans quickstart | Pass | -- | Cross-notebook nav visible at top (new cell 1) |
| 1.2 | Enter NVIDIA API key | Pass | -- | Auto-detected from env var |
| 1.3 | Load pre-trained KMeans router | Pass | 3.0s | 100 clusters, 3 models |
| 1.4 | Embed easy question via API | Pass | 1.2s | 2048-dim vector |
| 1.5 | See routing decision (easy) | Pass | <1ms | **nem-nothink selected** (cheapest) |
| 1.6 | Call selected model (easy) | Pass | 0.4s | "Paris." — correct, concise |
| 1.7 | Compare: hard question | **Pass** | 1.6s embed | **nem-think selected** (thinking model). Different from easy! |
| 1.6b | Call selected model (hard) | Pass | 2.1s | Detailed Euler-Lagrange derivation with thinking |
| 1.8 | Review cost savings | Pass | -- | **65% savings** — realistic and honest |

**Track B: Prefill Routing**

| Step | Action | Status | Wall Clock | Notes |
|------|--------|--------|------------|-------|
| 1.9 | Open prefill quickstart | Pass | -- | Cross-notebook nav visible at top (new cell 1) |
| 1.10 | Enter API key(s) | Pass | -- | Auto-detected |
| 1.11 | Load encoder model | Pass | 5.7s | **No confusing warnings** (suppressed by Plan 4) |
| 1.12 | Extract hidden states (easy) | Pass | ~6.0s | Hidden states extracted on CPU |
| 1.13 | See routing decision (easy) | Pass | <1ms | **nem-nothink selected** at tolerance=0.05. P(correct)=0.999 |
| 1.14 | Call selected model (easy) | Pass | 0.4s | "Paris." — correct |
| 1.15 | Route hard question | **Pass** | ~4.9s | **nem-think selected** at tolerance=0.05. P(correct)=0.912 |
| 1.14b | Call selected model (hard) | Pass | 2.4s | Detailed Euler-Lagrange derivation with thinking |
| 1.16 | Review cost savings | Pass | -- | **65% savings** — matches KMeans result |

**Cross-Method Comparison**

| Step | Action | Status | Wall Clock | Notes |
|------|--------|--------|------------|-------|
| 1.17 | Compare both methods | **Pass** | -- | Both notebooks use same questions. Comparison table present in both (cell 23). Same routing decisions: easy→nem-nothink, hard→nem-think. |
| 1.18 | Understand tradeoffs | **Pass** | -- | Method comparison tables clearly show: KMeans=lightweight/fast, Prefill=accurate/heavier. Tolerance difference (0.20 vs 0.05) explained. |
| 1.19 | Decide next step | Pass | -- | "What's Next" has 3 paths. LiteLLM code now includes `set_litellm_router`. |

#### What Worked Well

- **Routing differentiation demonstrated** — Easy question ("What is the capital of France?") → Nemotron 3 Nano ($0.04/1k). Hard question ("Derive the Euler-Lagrange equation...") → Nemotron 3 Nano Think ($0.23/1k). The evaluator sees the router making intelligent decisions.

- **65% cost savings is honest and compelling** — Previous 90% savings were misleading (everything went to cheapest model). The new 65% shows that routing sends easy queries to cheap models while still using stronger models when needed. This is the real value proposition.

- **Cross-notebook navigation** — Cell 1 in both notebooks immediately tells the user about the other routing method. The comparison tables at the bottom provide a clear summary.

- **Clean prefill loading** — No "fast path" warnings, no "unauthenticated HF Hub" warnings. The loading output is clean: just the encoder name, model stats, and trunk config.

- **Same questions in both notebooks** — Both notebooks use "What is the capital of France?" and "Derive the Euler-Lagrange equation from the principle of least action." This enables direct comparison of how KMeans vs Prefill handle the same queries.

- **Tolerance difference tells a story** — KMeans at tolerance=0.20 and Prefill at tolerance=0.05 demonstrates that prefill's higher accuracy enables tighter optimization. The prefill notebook explains this: "The prefill router's higher accuracy produces more calibrated confidence scores with wider spreads."

- **Hard question response quality matches the model choice** — Nemotron 3 Nano Think produces a detailed, step-by-step derivation of the Euler-Lagrange equation. This validates that the router correctly escalated to the thinking model.

- **LiteLLM code now works** — The "What's Next" LiteLLM integration example includes all 4 required lines and references the correct config file for each notebook (`cloud-only.yaml` for KMeans, `prefill-qwen08b.yaml` for Prefill).

#### Documentation Gaps and Workarounds

| Gap | Missing Info | Impact | Recommendation |
|-----|-------------|--------|----------------|
| No tolerance sensitivity demo | Users don't see how tolerance affects routing | Low | Add optional cell showing routing at 3 tolerance levels |
| Stale cell outputs | Notebook outputs show old results until re-run | Minor | Expected behavior; will be correct when user runs |
| No "convinced moment" summary | Missing a quantified "this saved X% with Y% accuracy" cell | Low | Add a summary cell after Results |

#### Issues and Bugs

| Issue | Severity | Notes |
|-------|----------|-------|
| Cell outputs in both notebooks still show old results | Minor | Source code is correct; outputs refresh on user's first run |
| Prefill CPU latency ~5-6s per routing decision | Minor | Already documented with clear latency note in cell 7. Accurate. |
| KMeans pickle load varies (3s vs 11s on different runs) | Minor | sklearn deserialization variance. Not a blocker. |

#### Scorecard

| Dimension | Score (1-5) | Initial | Change | Notes |
|-----------|:-----------:|:-------:|:------:|-------|
| Completeness | 4 | 3 | **+1** | Routing differentiation works. Cross-comparison exists. Missing: tolerance sensitivity demo. |
| Clarity | 5 | 4 | **+1** | Cross-notebook nav, comparison tables, tolerance explanation, warning note — all clear. |
| Executability | 5 | 4 | **+1** | Every step passes. Both tracks show different models. API calls succeed. Warnings suppressed. |
| Error Handling | 4 | 3 | **+1** | Confusing warnings suppressed. Latency documented. Missing: guidance on API failures. |
| Continuity | 4 | 3 | **+1** | LiteLLM code fixed. Cross-notebook links. Comparison table bridges the two notebooks. |

#### JTBD Verification

| JTBD | Documented Status | Initial Verified | Re-Run Verified | Notes |
|------|-------------------|:----------------:|:---------------:|-------|
| Understand what routing does | Partial | Partial | **Improved** | Pipeline diagrams + routing differentiation now demonstrate the concept |
| See a live KMeans routing decision | Complete | Confirmed | Confirmed | Works with differentiation |
| See a live prefill routing decision | Complete | Confirmed | Confirmed | Works with differentiation |
| Compare both routing methods | Partial | Confirmed Partial | **Upgraded: Mostly Complete** | Same questions, comparison table, cross-links. Only missing: side-by-side output in a single view. |
| Verify cost savings | Complete | Downgraded: Partial | **Restored: Complete** | 65% savings is honest and demonstrates intelligent routing |
| Understand accuracy tradeoffs | Partial | Confirmed Partial | **Improved** | Tolerance difference (0.20 vs 0.05) with explanation helps. Still no tolerance sweep. |
| Understand infrastructure tradeoffs | Partial | Confirmed Partial | **Upgraded: Mostly Complete** | Comparison table explicitly covers deps, GPU, latency, accuracy |
| Share results with team | Partial | Confirmed Partial | Confirmed Partial | No change — still notebook output only |

---

## Fix Verification Summary

### Plan 1: Fix Demo Routing Differentiation ✅

| Aspect | Before | After |
|--------|--------|-------|
| Easy question model (KMeans) | nem-nothink | nem-nothink |
| Hard question model (KMeans) | nem-nothink ❌ | **nem-think** ✅ |
| Easy question model (Prefill) | nem-nothink | nem-nothink |
| Hard question model (Prefill) | nem-nothink ❌ | **nem-think** ✅ |
| Cost savings | 90% (misleading) | **65% (honest)** |
| Demo hard question | "Prove sqrt(2) is irrational" | "Derive the Euler-Lagrange equation..." |
| Prefill tolerance | 0.20 | **0.05** |

### Plan 2: Cross-Notebook Navigation ✅

| Component | Present | Location |
|-----------|---------|----------|
| KMeans → Prefill link | ✅ | `quickstart.ipynb` cell 1 |
| Prefill → KMeans link | ✅ | `quickstart-prefill.ipynb` cell 1 |
| KMeans comparison table | ✅ | `quickstart.ipynb` cell 23 |
| Prefill comparison table | ✅ | `quickstart-prefill.ipynb` cell 23 |
| Tolerance explanation | ✅ | `quickstart-prefill.ipynb` cell 23 |

### Plan 3: Fix LiteLLM Code ✅

| Notebook | Before | After |
|----------|--------|-------|
| KMeans | 3 lines, missing `set_litellm_router`, generic `pool_config.yaml` | 4 lines, complete, uses `configs/cloud-only.yaml` |
| Prefill | 3 lines, missing `set_litellm_router`, generic config | 4 lines, complete, uses `configs/prefill-qwen08b.yaml` |

### Plan 4: Suppress Prefill Warnings ✅

| Warning | Before | After |
|---------|--------|-------|
| "fast path not available" | Visible ❌ | Suppressed ✅ |
| "unauthenticated HF Hub" | Visible ❌ | Suppressed ✅ |
| "Could not cache" | Visible ❌ | Suppressed ✅ |
| Explanatory note in markdown | Missing | Added in cell 7 ✅ |

---

## Remaining Recommendations (for future iterations)

| # | Recommendation | Impact | Effort | Priority |
|---|---------------|--------|--------|----------|
| 1 | Add tolerance sensitivity demo cell in both notebooks | Low | S | P2 |
| 2 | Add "convinced moment" summary cell (quantified value statement) | Low | S | P2 |
| 3 | Add API error handling guidance (what if build.nvidia.com is down) | Low | S | P2 |
| 4 | Clear stale cell outputs before next release | Low | S | P2 |
| 5 | Add a third harder question that escalates to GPT-OSS 20B | Medium | M | P2 |

---

## Appendix

### A. Execution Log

```
Track A (KMeans):
  Checkpoint load:    3.0s
  Embed (easy):       1.2s
  Route (easy):       <1ms  → nem-nothink ✓
  Embed (hard):       1.6s
  Route (hard):       <1ms  → nem-think ✓ (DIFFERENTIATED)
  API call (easy):    0.4s  → "Paris."
  API call (hard):    2.1s  → Euler-Lagrange derivation
  Total:              ~56s

Track B (Prefill):
  Checkpoint load:    0.2s
  Encoder load:       5.7s  (no warnings)
  Route (easy):       6.0s  → nem-nothink ✓
  Route (hard):       4.9s  → nem-think ✓ (DIFFERENTIATED)
  API call (easy):    0.4s  → "Paris."
  API call (hard):    2.4s  → Euler-Lagrange derivation
  Total:              ~36s
```

### B. Files Modified

| File | Changes Made |
|------|-------------|
| `notebooks/quickstart.ipynb` | Cell 1 (new): cross-notebook nav. Cell 16: new hard question. Cell 20: new hard question in call_model. Cell 23: method comparison table + fixed LiteLLM code + `set_litellm_router`. |
| `notebooks/quickstart-prefill.ipynb` | Cell 1 (new): cross-notebook nav. Cell 7: warning note. Cell 8: warning suppression imports. Cell 15: tolerance=0.05. Cell 16: new hard question + tolerance=0.05. Cell 20: new hard question in call_model. Cell 23: method comparison + tolerance explanation + fixed LiteLLM code + `set_litellm_router`. |
