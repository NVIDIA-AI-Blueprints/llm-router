# Implementation Log -- Model Router Toolkit

## TODO — Remaining Work for Customer Shipping

### Not Implemented (Requires Custom Implementation)

These items are referenced in docs or configs but require net-new implementation work beyond what's in this project:

- [ ] **KMeans training pipeline** (`kmeans/train.py`) — Currently raises `NotImplementedError`. Planned pipeline: embed all questions, fit KMeans (n_clusters=100), compute per-cluster per-model accuracy, fit Platt calibrators, save pkl. Blocked at CLI with a clear message.
- [x] ~~**Encoder server** (`scripts/serve-encoder.py`) — Stub that exits immediately.~~ Resolved: `server/router_app.py` + `model-router serve-router` CLI. Serves routing decisions only (no LLM inference) via `POST /v1/route`. Standalone script removed in favor of CLI subcommand.
- [ ] **LLM-as-judge** (`collect.py`, `--judge llm`) — Not implemented. Would use a frontier model to evaluate answer correctness instead of majority vote. Requires prompt engineering and model selection logic.
- [ ] **vLLM encoder backend** — Using HF transformers for extraction. vLLM integration would reduce prefill latency from ~5s (CPU) to <100ms (GPU). Requires vLLM client implementation in `prefill/extract.py`.
- [ ] **Multi-encoder training** — Current training uses a single encoder. Supporting multiple encoders per model (e.g., different chat templates) would require config schema extension and changes to `prefill/train.py`.
- [ ] **Expanded 7-model default checkpoint** — Current bundled pkl covers 3-4 models. A full default pool (nem-think, nem-nothink, nem-super, gptoss-20b, gptoss-120b, qwen-122b, gpt-5.2, claude-opus) requires collecting data and training a new checkpoint.
- [ ] **Pydantic request validation** for `/v1/chat/completions` — Currently uses raw `request.json()`. Should add a Pydantic model for proper 422 responses on malformed input. Requires understanding of OpenAI request schema + litellm extensions.
- [ ] **Safetensors migration** — Multiple files use `pickle.load()` and `torch.load(weights_only=False)` which can execute arbitrary code. Migrating to safetensors requires changes to the checkpoint format and all save/load paths. Security warnings have been added in the interim.
- [ ] **Update notebooks, scripts, default checkpoints, and datasets with extended model set results** — Refresh quickstart notebooks, CLI scripts, and bundled default checkpoints/datasets to reflect results from the extended model set experiments (larger model pools, updated P(correct) calibrations, new cost tables). Ensures shipped artifacts match the latest training runs.
- [ ] **Add SOTA benchmarks and benchmarking data/scripts** — Introduce new state-of-the-art benchmark datasets and corresponding evaluation scripts using the existing collect/train/evaluate pipeline. Covers adding benchmark question sets, reference answers, and pre-computed baselines so users can reproduce published routing accuracy numbers out of the box.
- [ ] **Update collection function based on Max's code** — Refactor `collect.py` to align with Max's implementation in the `max/running_eval` branch of `dl-tme/llmrouter` (https://gitlab-master.nvidia.com/dl-tme/llmrouter/-/tree/max/running_eval). Port relevant patterns and improvements.
- [ ] **Split collection function into modular data collection + encoder output generation** — Decompose the monolithic collection function in `collect.py` so that (1) data collection (running models, judging correctness) and (2) encoder output generation (prefill extraction for training) are separate, composable modules. Both should remain accessible from the collection entrypoint but independently callable.
- [ ] **Address remaining virtual review action items** — Continue working through the open items from `working/virtual-review.md`. Key remaining work: fix question format mismatch in `collect.py` (Tier 1.4), add review result persistence in telemetry (Tier 1.6), add Docker section to README (Tier 2.2), document `/api/review` endpoint (Tier 2.3), add `serve-router` to `architecture.md` (Tier 2.4), update `schema.md` (Tier 2.6), add routing failure fallback in `strategy.py` (Tier 3.1), add `--output` flag to evaluate (Tier 3.2), create monitoring guide (Tier 3.3), and replace `print()` with structured logging (Tier 3.5). See virtual review Tiers 1–3 for full priority list.

### Polish Items (Can Be Done Incrementally)

- [ ] **Replace `print()` with `logging`** throughout `evaluate.py`, `__main__.py`, `collect.py`
- [ ] **Add `__init__.py` exports** in `kmeans/` and `prefill/` (currently just docstrings, no `__all__` or re-exports)
- [ ] **API reference docs** — Public Python API (`BaseRouter`, `RoutingResult`, `ModelRoutingStrategy`, `PoolConfig`) has no reference documentation
- [ ] **Deployment / performance guide** — No docs on recommended hardware, GPU vs CPU latency, scaling, cold start times
- [ ] **Troubleshooting / FAQ** — Common issues (encoder download hangs, OpenRouter rate limits, checkpoint compatibility)
- [ ] **GPU service in Docker** — Dockerfile has `proxy-gpu` stage but no compose service or GPU resource configuration
- [ ] **Deprecated FastAPI `on_event`** in `proxy/startup.py` — Should migrate to `lifespan` pattern

## Status
- [x] Phase 1: Repo scaffold
- [x] Phase 2: Core abstractions (BaseRouter, RoutingResult, ModelRoutingStrategy, config)
- [x] Phase 3: Quickstart notebook
- [x] Phase 4: KMeans router implementation
- [x] Phase 5: Unified train + evaluate CLIs
- [x] Phase 6: Data collection (collect.py)
- [x] Phase 7: Setup wizard
- [x] Phase 8: FastAPI server + UI
- [x] Phase 9: Agent docs (AGENTS.md, rules, skills)
- [x] Phase 10: Integration docs
- [x] Phase 11: Prefill router inference (Qwen3.5-0.8B encoder on CPU)
- [x] Phase 12: Prefill quickstart notebook
- [ ] Phase 13: Expanded model pool pkl (7-model default)
- [x] Phase 14: Server Playground UI
- [x] Phase 15: Prefill training, evaluation, and collection pipeline

## Log

### 2026-03-07 -- Phase 1: Repo Scaffold
**What was done:** Created experiments/model-router-toolkit/ with full directory structure, pyproject.toml (setuptools, Python 3.10+), src/ layout, configs/, tests/, docs/, notebooks/. Bundled kmeans_c100_db.pkl.
**Tests passing:** Package installs and imports correctly.
**Challenges:** None.

### 2026-03-07 -- Phase 2: Core Abstractions
**What was done:** Built router.py (BaseRouter ABC, RoutingResult, CostEstimate), strategy.py (ModelRoutingStrategy wrapping BaseRouter for LiteLLM integration), config.py (PoolConfig with pydantic validation). KMeansRouter implementation with pkl loading, Platt calibration, tolerance-based selection. PrefillRouter stub.
**Tests passing:** 17/17 unit tests (config, kmeans router, strategy).
**Challenges:** pkl's split_models was a parallel list, not a dict -- fixed mapping logic.
**Gaps:** PrefillRouter was a stub (requires torch + trained checkpoint).

### 2026-03-07 -- Phase 3: Quickstart Notebook
**What was done:** Created notebooks/quickstart.ipynb -- standalone notebook using only requests, numpy, scikit-learn. Calls build.nvidia.com API for embeddings and chat completions. Demonstrates routing on easy vs hard questions with cost comparison.
**Tests passing:** End-to-end verified with NVIDIA_API_KEY. All cells execute.
**Challenges:** NVIDIA API model names use single-namespace lowercase format (nvidia/nemotron-3-nano-30b-a3b), not the LiteLLM nvidia_nim format. GPT-5.2 not available on build.nvidia.com -- replaced with GPT-OSS 120B in quickstart.

### 2026-03-07 -- Phases 4-8: CLI, Server, Setup
**What was done:** Full CLI with subcommands (setup, serve, train, evaluate, collect). Setup wizard with GPU detection, checkpoint model discovery, interactive endpoint mapping. FastAPI server with /api/chat SSE, /v1/chat/completions, /health, /api/models. Telemetry (SQLite). Training dispatcher (stubs for actual training logic). Evaluation with AUC, routing accuracy, cost savings. Data collection via litellm with majority vote judging.
**Tests passing:** 17/17 unit tests.
**Challenges:** Server needed nvidia_nim/ prefix auto-detection from api_base URL. Completions endpoint was bypassing the routing strategy -- fixed to let litellm.Router dispatch via set_custom_routing_strategy. Setup wizard needed to read model names from checkpoints and map to endpoints.

### 2026-03-07 -- Phases 9-10: Docs + Agent Support
**What was done:** AGENTS.md, .cursor/rules/ (project-structure, router-abstraction, config-schema), .cursor/skills/ (setup-router, train-router, integrate-app), docs/integration.md with full examples.
**Tests passing:** 17/17 unit tests.

### 2026-03-08 -- Phase 11: Prefill Router Inference
**What was done:** Full prefill router implementation using Qwen3.5-0.8B encoder running on CPU.
- Built prefill/extract.py: PrefillExtractor loads HF model, runs forward pass with output_hidden_states=True, returns per-layer last-token and mean-pooled hidden states
- Built prefill/trunk.py: SharedTrunkNet MLP (d_in=600 -> 256 -> 128 -> 4 outputs), ensemble reconstruction from checkpoint, predict_proba with sigmoid averaging
- Built prefill/transforms.py: raw_hidden + StandardScaler + PCA pipeline
- Built prefill/scorer.py: PrefillScorer loads checkpoint, lazy-loads encoder, caches extraction results per (encoder, template) combo, runs full pipeline
- prefill/router.py already had correct wiring -- PrefillRouter delegates to scorer with tolerance-based selection
- Created configs/prefill-local.yaml with 4 models mapped to OpenRouter endpoints
- Checkpoint: checkpoints/prefill_qwen08b.pt (7.6MB, trained on 10K questions, AUC 0.70-0.75)

**Tests passing:** 17/17 unit tests. End-to-end verified:
- PrefillRouter.route() returns correct RoutingResult with per-model P(correct)
- Server on port 8001 with prefill config: /v1/chat/completions works, routes via prefill, calls OpenRouter
- First call ~36s (model load + prefill), subsequent ~5s (prefill only)

**Challenges:**
- Needed transformers upgrade for Qwen3.5 architecture support (qwen3_5 model_type)
- Needed accelerate package for device_map support
- torch_dtype deprecated in favor of dtype in newer transformers

**Performance (CPU, M-series Mac):**
- Model load: ~14s (first time, cached after)
- Prefill per question: ~5s
- Total first-call latency: ~36s (load + prefill + trunk)

### 2026-03-08 -- Phase 11 addendum: Prefill router details
**What was added to Phase 11:**
- Created configs/prefill-qwen08b.yaml: 4-model pool (nem-think, nem-nothink, gptoss-high, gpt-5.2) with build.nvidia.com-compatible names
- Checkpoint scores 4 models simultaneously from a single Qwen3.5-0.8B forward pass via SharedTrunkNet ensemble (5 nets)
- Per-model feature pipeline: extract hidden states at specific layer -> StandardScaler -> PCA -> concatenate -> MLP
- Extraction caches by (encoder, chat_template_kwargs) key to avoid redundant forward passes for models sharing the same template

### 2026-03-08 -- Phase 12: Prefill Quickstart Notebook
**What was done:** Created notebooks/quickstart-prefill.ipynb -- standalone notebook (zero model_router_toolkit imports) demonstrating the prefill-based router. Uses torch, transformers, numpy, scikit-learn, requests only.
- Loads prefill_qwen08b.pt checkpoint directly via torch.load()
- Defines SharedTrunkNet MLP class inline and reconstructs ensemble from checkpoint state dicts
- Loads Qwen3.5-0.8B encoder via AutoModelForCausalLM on CPU
- Extracts hidden states, applies per-model PCA+scaler transforms, runs MLP ensemble for P(correct)
- Routes to cheapest model above tolerance, calls build.nvidia.com API for responses
- 3-model pool on build.nvidia.com (nem-nothink, nem-think, gptoss-high); GPT-5.2 excluded (not on NVIDIA endpoint) but features still computed for MLP input
- Same cell-for-cell structure as KMeans quickstart.ipynb

**Tests passing:** All cells execute successfully via `jupyter nbconvert --execute` (~49s total).
- Prefill router correctly routes easy question to Nemotron 3 Nano (P=0.999, $0.04/1k)
- Hard question also routes to Nemotron 3 Nano (P=0.885 within tolerance of best P=0.981)
- 90% cost savings vs always using GPT-OSS 20B

**Challenges:**
- `torch_dtype` parameter deprecated in newer transformers -- changed to `dtype`
- Notebook must compute features for all 4 checkpoint models (including gpt-5.2) since MLP expects full feature vector, then filter routing to only the 3 available on build.nvidia.com

### 2026-03-07 -- Phase 14: Server Playground UI
**What was done:** Replaced the placeholder server UI with a full interactive playground.
- Rewrote server/static/index.html as HTML shell loading separate CSS/JS files
- Created server/static/playground.css (379 lines): NVIDIA dark theme, chat bubbles, routing card with probability bars, pipeline visualization, sidebar controls (tolerance slider, model toggles, session stats), review card styling
- Created server/static/playground.js (648 lines): chat SSE handler, routing card renderer with pipeline visualization and probability bars, tolerance/toggle state management, client-side session stats with cost estimation, auto-review SSE handler, prompt example chips, markdown rendering via marked CDN, copy-to-clipboard
- Created server/review.py (227 lines): POST /api/review SSE endpoint that judges answer correctness using the most expensive model in the pool as judge (via litellm direct call, bypassing routing strategy); if incorrect, tests other enabled models and streams comparison results
- Enhanced server/app.py: added GET /api/config endpoint (returns routing_method, review_available, judge_model), enhanced GET /api/models to include cost data, mounted review router
- Auto-review conditionally available when OPENROUTER_API_KEY is set
- Fixed chat.py stream error handling: late-stream exceptions from OpenRouter no longer show a red error box when content was already delivered successfully (tracks `tokens_sent` flag)
- Fixed TTFT measurement: now measures from `routing` SSE event (LLM call start) to first `token` event, not from request start (which included prefill routing time)

**Tests passing:** 17/17 existing unit tests unaffected.
**Challenges:**
- The litellm.Router's custom routing strategy intercepts all acompletion calls, so review.py calls litellm directly with extracted model params to target specific models for judging/comparison.
- OpenRouter streams occasionally throw `list index out of range` on EOF -- fixed by swallowing late-stream errors when tokens were already sent.
- TTFT was initially measuring end-to-end (including 4-5s prefill routing), fixed to start timer at the routing event instead of the request start.

### 2026-03-07 -- Phase 15: Prefill Training, Evaluation, and Collection Pipeline
**What was done:** Full end-to-end training and evaluation pipeline for the prefill router, ported from experiments/prefill-complexity-router/.

**Training pipeline** (`prefill/train.py`):
- Label loading with question normalization and output token statistics
- Batch prefill extraction with on-disk caching (`extract.py` rewrite: PrefillResult with torch tensors, save/load serialization, `run_extraction()` orchestrator)
- Layer/mode/PCA grid search with ternary layer search (`sweep.py`: SweepResult, cv_auc with logistic regression, sweep_model)
- SharedTrunkNet ensemble training (`trunk.py`: train_mlp with BCEWithLogitsLoss + early stopping, train_ensemble with seed selection)
- PCA transform fitting (`transforms.py`: fit_pca_pipeline)
- Self-contained .pt checkpoint saving (compatible with existing scorer/router for inference)
- Serve config generation (serve.yaml)

**Evaluation pipeline** (`evaluate.py`):
- Batch prefill extraction from checkpoint transforms (deduplicates by encoder)
- Rich metrics: per-model AUC/accuracy, oracle/best-single/router accuracy, lift, headroom captured
- Agreement zone analysis (all correct / disagree / all wrong)
- Deep routing analysis: near-miss (confidence gap), pairwise win rates

**Collection enhancements** (`collect.py`):
- Reference-based judging (--references CSV)
- tqdm progress bar
- Per-model accuracy summary

**CLI updates** (`__main__.py`):
- Train: --mode, --device, --batch-size, --n-seeds, --n-keep, --prefill-dir, --epochs, --patience, --pca-dims
- Evaluate: --device, --batch-size, --prefill-dir
- Collect: --references
- Clean error handling (actionable messages, exit code 1)

**Smoke test verified:** 2-model pool (nem-think, nem-nothink), 50 train / 20 test questions, Qwen3.5-0.8B encoder on CPU:
- Training: extract ~4m, sweep ~1s, trunk ~1s, total ~5m (with cache: ~10s)
- Evaluation: extract ~2m (with cache: instant), full report in 14s
- Inference: trained checkpoint loads and routes via PrefillRouter (backward compatible)

**Tests passing:** 17/17 unit tests. Full train→eval→inference cycle verified.

**Files created/modified:**
- NEW: `prefill/sweep.py` (sweep grid search)
- REWRITTEN: `prefill/extract.py` (batch extraction, caching, PrefillResult with torch tensors)
- REWRITTEN: `prefill/transforms.py` (added fit_pca_pipeline)
- REWRITTEN: `prefill/trunk.py` (added train_mlp, train_ensemble)
- REWRITTEN: `prefill/train.py` (full training pipeline)
- REWRITTEN: `evaluate.py` (prefill-specific rich evaluation)
- MODIFIED: `train.py` (pass-through kwargs)
- REWRITTEN: `__main__.py` (clean CLI with all args)
- REWRITTEN: `collect.py` (reference judging, progress)
- NEW: `configs/smoke-test.yaml` (2-model test config)
- NEW: `data/smoke-train.csv`, `data/smoke-test.csv`, `data/smoke-questions.txt`

**Doc updates** (Phase 15 addendum):
- REWRITTEN: `README.md` (lean workflow-focused: collect, train, evaluate, serve)
- REWRITTEN: `AGENTS.md` (updated project structure, full CLI reference with all options, config format, data format)
- REWRITTEN: `.cursor/skills/train-router/SKILL.md` (end-to-end workflow with smoke test recipe)
- REWRITTEN: `.cursor/skills/setup-router/SKILL.md` (quick start, manual config, app connection)
- REWRITTEN: `.cursor/skills/integrate-app/SKILL.md` (4 integration paths: OpenAI SDK, env var, LiteLLM SDK, direct Python)

## Known Gaps
1. **KMeans training**: Stub only (train_kmeans raises NotImplementedError). Pipeline defined but not coded.
2. ~~**Prefill training**: Stub only. Use experiments/prefill-complexity-router/ directly for training.~~ Resolved in Phase 15.
3. **Expanded pkl**: Current pkl covers 3-4 models; need 7-model pkl for full default pool.
4. ~~**Server UI**: Placeholder HTML; needs the full chat UI from litellm-kmeans-router.~~ Resolved in Phase 14.
5. ~~**LLM-as-judge**: collect.py only implements majority vote; llm/reference stubs.~~ Reference judging added in Phase 15. LLM-as-judge still stub.
6. ~~**Integration tests**: Not yet written (tests/integration/ is empty).~~ Resolved: 126 total tests (66 original + 60 new).
7. **vLLM encoder backend**: Planned but not implemented (using HF transformers).
8. **Prefill latency**: 5s per question on CPU is fine for evaluation but slow for production. GPU or vLLM would reduce to <100ms.
9. **Multi-encoder training**: Current training supports single encoder. The sweep modes (per_model, single, auto) are functionally equivalent with one encoder. Multi-encoder would require config schema extension.

### 2026-03-08 -- Comprehensive Test Coverage

**What was done:** Added 60 new tests across 10 new test files covering all previously untested modules.

**New unit tests (6 files, 31 tests):**
- `tests/test_checkpoint.py` (5 tests): `detect_checkpoint_type`, `load_checkpoint` with real .pkl and .pt files
- `tests/test_telemetry.py` (6 tests): `enabled()`, `create_session()`, `log_chat()`, `get_stats()`, disabled state error
- `tests/test_gpu.py` (5 tests): `GPUInfo`, `detect_gpus` (real + mocked), `has_sufficient_gpu`
- `tests/test_transforms.py` (6 tests): `raw_hidden`, `fit_pca_pipeline`, `apply_pipeline`, `build_features` with synthetic PrefillResult
- `tests/test_trunk.py` (6 tests): `SharedTrunkNet` forward, `train_mlp`, `train_ensemble`, `predict_proba`, `reconstruct_trunk` from real checkpoint
- `tests/test_sweep.py` (3 tests): `cv_auc` (perfect/random), `sweep_model` with synthetic data

**New integration tests (4 files, 29 tests):**
- `tests/integration/test_prefill_pipeline.py` (10 tests): `PrefillExtractor`, `PrefillResult` serialization, `PrefillScorer`, `PrefillRouter` with real Qwen3.5-0.8B encoder and smoke checkpoint
- `tests/integration/test_collect_and_evaluate.py` (6 tests): `_judge_vote`, `_judge_reference`, `run_collect` with real API, `run_evaluate` with real encoder
- `tests/integration/test_embed.py` (5 tests): `APIEmbedClient` with real NVIDIA API, `get_default_embed_client` factory
- `tests/integration/test_server_and_cli.py` (8 tests): `create_app`, `/health`, `/v1/chat/completions`, `/api/review` with real app; CLI subcommands via subprocess

**Infrastructure:**
- `tests/conftest.py`: Added 7 new fixtures (`prefill_ckpt_path`, `smoke_ckpt_path`, `smoke_train_csv`, `smoke_test_csv`, `smoke_questions_path`, `smoke_config_path`, `prefill_config_path`), `requires_torch` marker, `slow` marker with `--run-slow` CLI option
- `pyproject.toml`: Added `requires_torch`, `requires_encoder`, `slow` markers

**Source code fixes discovered during testing:**
- `test_checkpoint.py`: Checkpoint key is `shared_trunk` not `trunks` — test assertion corrected
- `test_trunk.py`: Checkpoint config key is `trunk_config` not `trunks` — test corrected
- `test_server_and_cli.py`: `python -m model_router_toolkit` subprocess doesn't output to stdout correctly — switched to `model-router` console script entry point

**Test results (full suite with `--run-slow`):**
- 126 collected, 122 passed, 4 failed
- All 4 failures are 401 authentication errors from expired API keys (NVIDIA_API_KEY and OPENROUTER_API_KEY) — not code bugs
- Without `--run-slow`: 97 collected, 97 passed, 0 failed

**Test execution tiers:**
```
pytest tests/ -v                    # Tier 1+2: 97 tests, no API keys needed
pytest tests/ -v --run-slow         # All tiers: 126 tests, needs API keys + encoder
```

### 2026-03-08 -- Router-Only Server (serve-encoder) + Refactor

**What was done:** Implemented the router-only deployment mode and refactored shared server code to eliminate duplication.

**Router-only server:**
- `src/model_router_toolkit/server/router_app.py`: FastAPI app factory (`create_router_app()`) serving `POST /v1/route` for routing decisions without LLM inference. Pydantic request/response models (`RouteRequest`, `RouteResponse`). Accepts both `{"question": "..."}` and `{"messages": [...]}` with per-request tolerance override.
- `scripts/serve-encoder.py`: Standalone script wrapping `create_router_app()` with argparse CLI.
- `__main__.py`: Added `serve-router` CLI subcommand: `model-router serve-router --config ... --port 8080`.

**Shared code refactor (eliminated 3 duplication clusters):**
- `router.py`: Added `extract_user_text()` as a standalone module-level function. Previously duplicated in `strategy.py._extract_user_text()` (method) and `router_app.py._extract_question()` (function). The canonical implementation handles both plain string and multipart content arrays. `strategy.py` now delegates to this function.
- `server/_shared.py`: Extracted `warmup_router()`, `health_dict()`, and `models_list()`. Previously copy-pasted across `app.py` and `router_app.py`. Both app factories now import from `_shared`.
- `app.py`: Removed `_warmup()`, inline `/health` dict, inline `/api/models` list — replaced with imports from `_shared`. Health response now includes `mode: "full"` (non-breaking addition).
- `router_app.py`: Removed `_warmup()`, inline `/health` dict, inline `/api/models` list, full `_extract_question()` body — replaced with shared imports. `_extract_question()` now delegates to `extract_user_text()` for the messages case.

**New tests:**
- `tests/test_router_app.py` (20 tests): Unit tests with FakeRouter mock — `_extract_question` (6 cases), `_result_to_response`, health/models/route endpoints, negative tests (no inference endpoints).
- `tests/integration/test_router_app.py` (9 tests, marked slow): Integration tests with real encoder and smoke checkpoint.

**Tests passing:** 68/68 unit tests (48 existing + 20 new), all green after refactor.

**Doc updates (same session):**
- `docs/architecture.md`: Full "Deployment Modes" section with serve vs proxy vs Docker and decision guide.
- `docs/integration.md`: Restructured into 4 integration paths with serve-vs-proxy decision table.
- `README.md`: Both server and proxy modes documented under "4. Serve".
- `docs/user-journeys.md`: Marked proxy-mode and deployment-mode gaps as resolved.


