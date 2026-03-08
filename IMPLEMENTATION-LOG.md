# Implementation Log -- Model Router Toolkit

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
- [ ] Phase 12: Expanded model pool pkl (7-model default)

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

## Known Gaps
1. **KMeans training**: Stub only (train_kmeans raises NotImplementedError). Pipeline defined but not coded.
2. **Prefill training**: Stub only. Use experiments/prefill-complexity-router/ directly for training.
3. **Expanded pkl**: Current pkl covers 3-4 models; need 7-model pkl for full default pool.
4. **Server UI**: Placeholder HTML; needs the full chat UI from litellm-kmeans-router.
5. **LLM-as-judge**: collect.py only implements majority vote; llm/reference stubs.
6. **Integration tests**: Not yet written (tests/integration/ is empty).
7. **vLLM encoder backend**: Planned but not implemented (using HF transformers).
8. **Prefill latency**: 5s per question on CPU is fine for evaluation but slow for production. GPU or vLLM would reduce to <100ms.
