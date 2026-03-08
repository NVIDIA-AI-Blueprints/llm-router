# Model Router Toolkit — User Journeys & Jobs to Be Done

> **Version:** 1.1 | **Last Updated:** March 8, 2026

This document maps every user journey and job-to-be-done (JTBD) for the Model Router Toolkit, from first-touch exploration through production deployment and ongoing optimization. It also covers broader journeys where the toolkit is one component in a larger workflow.

---

## Table of Contents

1. [Persona Map](#persona-map)
2. [Journey 1: Explore & Evaluate (The Evaluator)](#journey-1-explore--evaluate-the-evaluator)
3. [Journey 2: Deploy a Router (The Integrator)](#journey-2-deploy-a-router-the-integrator)
4. [Journey 3: Train & Optimize (The Optimizer)](#journey-3-train--optimize-the-optimizer)
5. [Journey 4: Production Deployment (The Platform Engineer)](#journey-4-production-deployment-the-platform-engineer)
6. [Journey 5: Plug into Existing LiteLLM (The LiteLLM User)](#journey-5-plug-into-existing-litellm-the-litellm-user)
7. [Journey 6: Quality Assurance & Review (The QA Lead)](#journey-6-quality-assurance--review-the-qa-lead)
8. [Broader Journeys (Toolkit as One Component)](#broader-journeys-toolkit-as-one-component)
9. [Jobs-to-Be-Done Matrix](#jobs-to-be-done-matrix)
10. [Journey Dependencies & Progression](#journey-dependencies--progression)
11. [Virtual Review Plan](#virtual-review-plan)

---

## Persona Map

| Persona | Role | Primary Goal | Entry Point | Key Metric |
|---------|------|-------------|-------------|------------|
| **Evaluator** | AI team lead, PM, developer | Prove routing works in 5 min | Quickstart notebook | Time to first routing decision |
| **Integrator** | Backend/ML engineer | Running router accepting requests | `configs/` + `model-router serve` | Time to first routed API call |
| **Optimizer** | ML engineer | Domain-tuned routing | `collect` + `train` + `evaluate` | AUC improvement, cost savings |
| **Platform Engineer** | DevOps/infra engineer | Production-grade deployment | Docker + `model-router proxy` | Uptime, container health |
| **LiteLLM User** | Developer with existing LiteLLM setup | Add routing to current stack | `ModelRoutingStrategy` SDK | Lines of code to integrate |
| **QA Lead** | Team lead, quality engineer | Validate routing quality | `/api/review` + evaluation CLI | Routing accuracy, review verdicts |

---

## Journey 1: Explore & Evaluate (The Evaluator)

### Persona

AI team leads, PMs, or developers evaluating whether intelligent model routing belongs in their stack. They want proof-of-concept with zero commitment.

### Job to Be Done

> "Show me this works — in 5 minutes, with no infrastructure."

### Journey Steps

The Evaluator tries **both** routing methods via two companion notebooks, understanding the tradeoffs between lightweight cloud-only routing and SOTA prefill-based routing.

#### Track A: KMeans Routing (Cloud-Only, No GPU)

| Step | Action | Asset | Success Criteria |
|------|--------|-------|-----------------|
| 1.1 | Open KMeans quickstart | `notebooks/quickstart.ipynb` | Notebook loads without errors |
| 1.2 | Enter NVIDIA API key | build.nvidia.com API key | Key validates successfully |
| 1.3 | Load pre-trained KMeans router | `checkpoints/kmeans_c100_db.pkl` | Pickle loads, 100 clusters visible |
| 1.4 | Embed a question via API | NVIDIA embeddings API | 4096-dim vector returned |
| 1.5 | See KMeans routing decision | Cluster → Platt → select | Model selected, per-model probabilities displayed |
| 1.6 | Call selected model | NVIDIA chat completions API | Streamed response returned |
| 1.7 | Compare easy vs. hard question | Side-by-side routing | Different models selected for different complexity |
| 1.8 | Review cost savings | Cost comparison output | Quantified savings percentage |

#### Track B: Prefill Routing (Encoder-Based, SOTA Accuracy)

| Step | Action | Asset | Success Criteria |
|------|--------|-------|-----------------|
| 1.9 | Open prefill quickstart | `notebooks/quickstart-prefill.ipynb` | Notebook loads without errors |
| 1.10 | Enter API key(s) | NVIDIA and/or OpenRouter API key | Key validates successfully |
| 1.11 | Load encoder model | Qwen3.5-0.8B via transformers | Encoder loads (CPU or GPU) |
| 1.12 | Extract hidden states for a question | Encoder forward pass | Hidden state tensor returned |
| 1.13 | See prefill routing decision | Hidden states → PCA → MLP → select | Model selected, per-model P(correct) displayed |
| 1.14 | Call selected model | API chat completion | Streamed response returned |
| 1.15 | Compare easy vs. hard question | Side-by-side routing | Different models, different confidence profiles |
| 1.16 | Review cost savings | Cost comparison output | Quantified savings percentage |

#### Cross-Method Comparison

| Step | Action | Asset | Success Criteria |
|------|--------|-------|-----------------|
| 1.17 | Compare both methods | Summary across both notebooks | Understand when KMeans vs. prefill is appropriate |
| 1.18 | Understand tradeoffs | Method comparison table | Clear on accuracy, latency, infrastructure requirements |
| 1.19 | Decide next step | "What's Next" sections | Clear path to Journey 2 or Journey 3, method chosen |

### Method Comparison (What the Evaluator Learns)

| Dimension | KMeans (Track A) | Prefill (Track B) |
|-----------|-------------------|-------------------|
| **Dependencies** | `requests`, `numpy`, `scikit-learn` | `torch`, `transformers` + API client |
| **Infrastructure** | Cloud API calls only, no GPU | Encoder model (CPU ok, GPU preferred) |
| **Routing latency** | ~100ms (embed API call) | ~1-5s (encoder forward pass, CPU) |
| **Accuracy** | Good (embedding similarity) | SOTA (hidden state complexity analysis) |
| **Checkpoint format** | `.pkl` (small, portable) | `.pt` (larger, requires torch) |
| **Best for** | Quick evaluation, cloud-only, low latency | Production accuracy, domain-specific tuning |

### Assets Touched

- `notebooks/quickstart.ipynb` — KMeans routing exploration (Track A)
- `notebooks/quickstart-prefill.ipynb` — Prefill routing exploration (Track B)
- `checkpoints/kmeans_c100_db.pkl` — bundled pre-trained KMeans router
- Prefill checkpoint (`.pt`) — pre-trained prefill router (if available)
- External: `build.nvidia.com` embeddings + chat completions APIs, OpenRouter API

### Jobs-to-Be-Done Breakdown

| JTBD | Current State | Completeness |
|------|--------------|-------------|
| Understand what routing does | Both notebooks explain concept | Partial — no visual diagram in notebooks |
| See a live KMeans routing decision | embed → cluster → Platt → select | Complete |
| See a live prefill routing decision | encode → hidden states → MLP → select | Complete (notebook exists) |
| Compare both routing methods | Two separate notebooks | Partial — no cross-comparison summary cell |
| Verify cost savings | Cost comparison in both notebooks | Complete |
| Understand accuracy tradeoffs | Tolerance parameter in both | Partial — tolerance impact not visualized |
| Understand infrastructure tradeoffs | Implicit from running both | Partial — no explicit comparison table in notebooks |
| Share results with team | Notebook output | Partial — no export/share mechanism |

### Gaps & Friction Points

1. **Checkpoint availability**: KMeans notebook assumes `../checkpoints/kmeans_c100_db.pkl` exists — no download instructions. Prefill checkpoint availability unclear.
2. **Model pool mismatch**: KMeans notebook uses 3 models (nem-nothink, nem-think, gptoss-high); prefill and default configs may use different pools
3. **No cross-notebook comparison**: The two notebooks are standalone — no shared summary or comparison cell linking them
4. **Prefill notebook dependencies**: Heavier install (`torch`, `transformers`) may surprise evaluators expecting the same 5-minute experience as KMeans
5. **API key guidance**: Link to build.nvidia.com but no step-by-step account creation guide
6. **No "convinced" moment**: Missing a summary cell in each notebook that quantifies "this saved X% cost with Y% accuracy retention"
7. **Navigation between notebooks**: No clear "start here, then try the other" flow linking the two notebooks
8. **Prefill on CPU timing**: If evaluator has no GPU, prefill may take 5+ seconds per question — expectations not set

---

## Journey 2: Deploy a Router (The Integrator)

### Persona

Backend or ML engineers building LLM-powered applications who need a running router endpoint they can point their apps at.

### Job to Be Done

> "Give me a running router I can point my app at today."

### Journey Steps

| Step | Action | CLI/Asset | Success Criteria |
|------|--------|-----------|-----------------|
| 2.1 | Install toolkit | `pip install -e .` or `pip install -e '.[prefill]'` | Package installs without errors |
| 2.2 | Copy and customize a config | Copy from `configs/` (see Sub-Journeys 2a/2b), edit checkpoint, tolerance, model pool | Valid config YAML with correct endpoints and API key set |
| 2.3 | Start server | `model-router serve --config configs/prefill-qwen08b.yaml` | Server starts on port 8000, `/health` returns OK |
| 2.4 | Open playground | Browser → `http://localhost:8000` | UI loads, model list appears |
| 2.5 | Send test message via UI | Type question in playground | Routing card shows selection, response streams |
| 2.6 | Adjust tolerance slider | Move slider in UI | Different model selected at different tolerances |
| 2.7 | Connect downstream app | Set `OPENAI_API_BASE=http://localhost:8000/v1` | App's LLM calls route through the toolkit |
| 2.8 | Verify routed traffic | Check server logs / telemetry | Requests flowing, models being selected |
| 2.9 | Test with cURL | `curl -X POST http://localhost:8000/v1/chat/completions` | Valid OpenAI-compatible response |

### Assets Touched

- `src/model_router_toolkit/server/` — FastAPI app, chat, completions
- `server/static/index.html` — playground UI
- `docs/integration.md` — downstream app configuration
- `configs/*.yaml` — example configs (prefill-qwen08b, cloud-only, smoke-test, etc.)
- `.env.example` — environment variable reference

### Sub-Journey 2a: KMeans Path (No GPU)

Start from `configs/cloud-only.yaml`.

| Step | Detail |
|------|--------|
| No GPU required | KMeans routing uses cloud embeddings |
| Embeddings via API | NVIDIA or OpenRouter embedding endpoint |
| Single process | Router server only, no encoder server |
| Checkpoint | `.pkl` file (pre-trained or custom) |

### Sub-Journey 2b: Prefill Path (GPU Available)

Start from `configs/prefill-qwen08b.yaml`.

| Step | Detail |
|------|--------|
| GPU >= 16GB recommended | Prefill routing offers higher accuracy |
| Encoder model loaded | Qwen3.5-0.8B via transformers (default) |
| Hidden state extraction | Local GPU inference for routing features |
| Checkpoint | `.pt` file (SharedTrunkNet ensemble) |

### Sub-Journey 2c: Router-Only Mode (No LLM Inference)

Deploy just the routing engine — returns which model to call, without actually calling it. Your application handles the LLM call itself.

Start from `configs/prefill-qwen08b.yaml` (or any pool config). No API keys needed.

| Step | Action | CLI/Asset | Success Criteria |
|------|--------|-----------|-----------------|
| 2c.1 | Install toolkit with prefill | `pip install -e '.[prefill]'` | Package installs without errors |
| 2c.2 | Start router-only server | `model-router serve-router --config configs/prefill-qwen08b.yaml --port 8080` | Server starts, `/health` returns `mode: router-only` |
| 2c.3 | Send a routing request | `curl -X POST http://localhost:8080/v1/route -d '{"question": "..."}'` | Returns `selected_model`, `confidences`, `costs` |
| 2c.4 | Use messages format | POST with `{"messages": [{"role": "user", "content": "..."}]}` | Same routing response |
| 2c.5 | Adjust tolerance per request | POST with `{"question": "...", "tolerance": 0.05}` | Tolerance affects model selection |
| 2c.6 | Integrate with your app | App calls `/v1/route`, gets model name, dispatches LLM call itself | Routing decoupled from inference |

**When to use this mode:**
- You have your own LLM dispatch layer (custom gateway, existing LiteLLM, vLLM, etc.)
- You want the router as a sidecar microservice
- You don't want to give the toolkit your API keys
- You want to log/audit routing decisions before acting on them

**What you get vs. full `serve` mode:**

| | `serve` (full) | `serve-router` (router-only) |
|---|---|---|
| Routing decisions | Yes | Yes |
| LLM inference | Yes (via LiteLLM) | No |
| Playground UI | Yes | No |
| API keys required | Yes (model provider) | No |
| Endpoint | `POST /v1/chat/completions` | `POST /v1/route` |
| Response | LLM-generated text + routing metadata | Routing decision only (model, confidences, costs) |

**Example response from `POST /v1/route`:**

```json
{
  "selected_model": "nem-nothink",
  "model_names": ["nem-think", "nem-nothink", "gptoss-high", "gpt-5.2"],
  "confidences": {"nem-think": 0.92, "nem-nothink": 0.89, "gptoss-high": 0.85, "gpt-5.2": 0.81},
  "costs": [
    {"model": "nem-think", "estimated_total_cost": 0.0003, "cost_per_m_input_tokens": 0.20},
    {"model": "nem-nothink", "estimated_total_cost": 0.0001, "cost_per_m_input_tokens": 0.04}
  ],
  "metadata": {"p_max": 0.92, "threshold": 0.72, "tolerance": 0.20, "route_ms": 4800}
}
```

### Jobs-to-Be-Done Breakdown

| JTBD | Current State | Completeness |
|------|--------------|-------------|
| Install and run in < 10 min | Copy config + serve | Complete |
| Get an OpenAI-compatible endpoint | `/v1/chat/completions` | Complete |
| Deploy routing without LLM inference | `model-router serve-router` | Complete |
| See what the router is doing | Playground UI with routing card | Complete (full mode only) |
| Connect my existing app | docs/integration.md with copy-paste configs | Complete |
| Configure model pool | YAML config with schema docs | Complete |
| Add/remove models | Edit YAML, restart server | Complete but manual |
| Use with multiple providers | LiteLLM handles NVIDIA, OpenRouter, OpenAI, Anthropic | Complete |
| Hot-reload config changes | Not supported | Missing |
| Monitor routing decisions | Telemetry (opt-in SQLite) | Partial |
| SSL/TLS termination | Not handled by toolkit | Out of scope — use reverse proxy |

### Gaps & Friction Points

1. **Setup wizard reliability**: No indication of how robust error handling is for edge cases (missing GPU drivers, partial API keys)
2. **Config validation feedback**: Unclear error messages when config has issues
3. **Encoder download time**: First-run downloads Qwen3.5-0.8B (~1.6GB) — no progress indication documented
4. **Port conflicts**: Default 8000 may conflict with other services; no auto-detection
5. **No systemd/launchd recipes**: No guidance for running as a daemon
6. **Playground limitations**: No conversation history, no multi-turn demo

---

## Journey 3: Train & Optimize (The Optimizer)

### Persona

ML engineers who want domain-specific routing tuned for their models, data, and cost targets.

### Job to Be Done

> "I need the router tuned for my models, my data, my cost targets."

### Journey Steps

| Step | Action | CLI/Asset | Success Criteria |
|------|--------|-----------|-----------------|
| 3.1 | Prepare questions | Text file or JSONL | 500+ domain-relevant questions |
| 3.2 | Collect training data | `model-router collect --config ... --questions ... --output ... --judge vote` | CSV with `question, model, isCorrect, output_tokens` |
| 3.3 | Split train/test | Manual 80/20 split | No question overlap between sets |
| 3.4 | Train router | `model-router train --config ... --data ... --output-dir ...` | Checkpoint saved, training logs |
| 3.5 | Evaluate checkpoint | `model-router evaluate --config ... --checkpoint ... --data ...` | Per-model AUC, routing accuracy, cost savings report |
| 3.6 | Interpret results | `docs/evaluation-guide.md` | Understand if router meets quality bar |
| 3.7 | Iterate (optional) | Adjust PCA dims, seeds, epochs, tolerance | Improved metrics |
| 3.8 | Deploy trained router | Update `routing.checkpoint` in config, restart server | Production traffic using custom router |
| 3.9 | Monitor quality | `/api/review` + telemetry | Ongoing accuracy tracking |

### Assets Touched

- `src/model_router_toolkit/collect.py` — data collection
- `src/model_router_toolkit/train.py` — training dispatcher
- `src/model_router_toolkit/evaluate.py` — evaluation
- `src/model_router_toolkit/prefill/` — full prefill training pipeline
- `docs/training-guide.md` — step-by-step walkthrough
- `docs/evaluation-guide.md` — metric interpretation
- `configs/smoke-test.yaml` — smoke test configuration

### Sub-Journey 3a: Smoke Test (Quick Validation)

| Step | Detail |
|------|--------|
| Use `configs/smoke-test.yaml` | 2-model pool, minimal data |
| `--n-seeds 2 --n-keep 1 --epochs 5` | Fast training |
| Validate pipeline works end-to-end | Before committing to full training run |

### Sub-Journey 3b: Full Training Run

| Step | Detail |
|------|--------|
| 500-2000+ questions | Domain-representative data |
| Default hyperparameters | `--n-seeds 5 --n-keep 3 --epochs 50` |
| Multiple PCA dimensions | `--pca-dims 32,64,128,256` |
| Cached prefill features | `--prefill-dir data/prefill_cache/` for iteration speed |

### Evaluation Metrics

| Metric | What It Tells You | Good Threshold |
|--------|-------------------|---------------|
| Per-model AUC | Can the router distinguish easy vs. hard per model | > 0.75 |
| Oracle accuracy | Theoretical best (always pick correct model) | Ceiling benchmark |
| Best single model accuracy | What you get without routing | Floor benchmark |
| Router accuracy | Actual routing performance | > best single model |
| Lift | Improvement over best single model | > 0 |
| Headroom captured | % of gap between single model and oracle | > 30% |
| Routing distribution | Traffic split across models | Matches cost expectations |
| Agreement zones | When all models agree vs. disagree | High agreement = easy questions |
| Near-miss analysis | Cases where router almost picked a better model | Identifies improvement opportunities |

### Jobs-to-Be-Done Breakdown

| JTBD | Current State | Completeness |
|------|--------------|-------------|
| Collect labeled data automatically | `model-router collect` with vote/reference judging | Mostly complete |
| Train a domain-specific router | Full prefill training pipeline | Complete (prefill only) |
| Train KMeans router | `NotImplementedError` | Not implemented |
| Evaluate and understand quality | Rich metrics report | Complete |
| Compare pre/post training | Manual comparison | No automated A/B |
| Iterate on hyperparameters | CLI flags for all key params | Complete |
| Cache intermediate artifacts | `--prefill-dir` for prefill features | Complete |
| Export training report | Printed to stdout | No file export |
| Track training experiments | Not supported | Missing (no MLflow/W&B integration) |

### Gaps & Friction Points

1. **KMeans training not implemented**: Only prefill training works; KMeans users stuck with pre-built checkpoints
2. **LLM judge not implemented**: `--judge llm` raises error; only `vote` and `reference` work
3. **No train/test split utility**: User must manually split CSV
4. **Data quality validation**: No checks for question diversity, minimum per-model coverage
5. **No experiment tracking**: Results printed to stdout, not persisted in structured format
6. **No incremental training**: Must retrain from scratch when adding new models to pool
7. **Question sourcing guidance**: Docs mention "production logs, domain benchmarks, curated sets" but no concrete examples

---

## Journey 4: Production Deployment (The Platform Engineer)

### Persona

DevOps and infrastructure engineers who need to deploy the router in a containerized, production-grade environment.

### Job to Be Done

> "Give me a Docker container I can deploy to our Kubernetes cluster with standard ops tooling."

### Journey Steps

| Step | Action | Asset | Success Criteria |
|------|--------|-------|-----------------|
| 4.1 | Review deployment options | `docker/`, `docs/integration.md` | Understand proxy vs. serve modes |
| 4.2 | Choose deployment mode | Proxy (LiteLLM) vs. Serve (standalone) | Decision based on existing stack |
| 4.3 | Configure environment | `.env.example`, `docker-compose.yaml` | API keys, config paths set |
| 4.4 | Build Docker image | `docker build -f docker/Dockerfile --target proxy .` | Image builds without errors |
| 4.5 | Start container | `docker compose up` | Container healthy, port 4000 accessible |
| 4.6 | Validate health | `curl http://localhost:4000/health` | Returns OK with model list |
| 4.7 | Send test request | cURL to `/v1/chat/completions` | Routed response returned |
| 4.8 | Configure monitoring | Health check endpoint | Kubernetes readiness/liveness probes |
| 4.9 | Scale and maintain | Docker compose or Kubernetes manifests | Horizontal scaling if needed |

### Deployment Modes

| Mode | Command | Port | Routing | Inference | Use Case |
|------|---------|------|---------|-----------|----------|
| **Standalone server** | `model-router serve` | 8000 | Yes | Yes | Quick deployment, includes playground UI |
| **Router-only** | `model-router serve-router` | 8080 | Yes | No | Routing decisions as a microservice, no API keys needed |
| **LiteLLM Proxy** | `model-router proxy` | 4000 | Yes | Yes | Drop-in replacement for existing LiteLLM proxy |
| **Docker (Proxy)** | `docker compose up` | 4000 | Yes | Yes | Containerized LiteLLM proxy with routing |
| **Docker (GPU)** | Build `proxy-gpu` target | 4000 | Yes | Yes | Prefill routing in container |

### Jobs-to-Be-Done Breakdown

| JTBD | Current State | Completeness |
|------|--------------|-------------|
| Containerized deployment | Multi-stage Dockerfile | Complete |
| Docker Compose for dev | `docker-compose.yaml` | Complete (proxy mode only) |
| GPU-enabled container | `proxy-gpu` Dockerfile target | Partial — no compose service |
| Health checks | `/health` endpoint, Dockerfile HEALTHCHECK | Complete |
| Config volume mounts | `configs/` and `checkpoints/` mounted | Complete |
| Environment variable config | `.env.example` | Complete |
| Kubernetes manifests | Not provided | Missing |
| Helm charts | Not provided | Missing |
| Horizontal scaling guide | Not documented | Missing |
| Log aggregation | Stdout logging | Basic — no structured logging |
| Secrets management | Environment variables | Basic — no vault integration |
| TLS/SSL | Not handled | Out of scope (use ingress) |

### Gaps & Friction Points

1. **No Kubernetes manifests or Helm charts**: Container exists but no K8s deployment artifacts
2. **Port inconsistency**: Docs say 8000 (serve), Docker says 4000 (proxy) — confusing
3. **No GPU compose service**: `proxy-gpu` target exists but no compose configuration
4. **Checkpoint provisioning**: Checkpoints are gitignored; no download or init-container strategy
5. **No resource limits**: No CPU/memory recommendations documented
6. **No rolling update strategy**: No guidance on zero-downtime deployments
7. **No observability stack**: No Prometheus metrics, no OpenTelemetry spans

---

## Journey 5: Plug into Existing LiteLLM (The LiteLLM User)

### Persona

Developers who already run LiteLLM (SDK or Proxy) and want to add intelligent routing without changing their deployment.

### Job to Be Done

> "I already use LiteLLM. Add smart routing to my existing setup with minimal changes."

### Journey Steps

| Step | Action | Asset | Success Criteria |
|------|--------|-------|-----------------|
| 5.1 | Install toolkit | `pip install model-router-toolkit` | Installs alongside existing LiteLLM |
| 5.2a | **SDK path**: Import strategy | `from model_router_toolkit import ModelRoutingStrategy` | Import succeeds |
| 5.3a | Create strategy from config | `strategy = ModelRoutingStrategy.from_config("config.yaml")` | Strategy loads checkpoint |
| 5.4a | Plug into existing Router | `router.set_custom_routing_strategy(strategy)` | Routing decisions change |
| 5.2b | **Proxy path**: Generate LiteLLM config | `model-router proxy-config --config pool.yaml --output litellm.yaml` | Config file generated |
| 5.3b | Start proxy with routing | `model-router proxy --litellm-config litellm.yaml --router-config pool.yaml` | Proxy starts with custom routing |
| 5.4b | Validate model alignment | Config bridge validates model names | No mismatches reported |
| 5.5 | Verify routing is active | Check response metadata / logs | Routing decisions visible |
| 5.6 | Adjust tolerance at runtime | `strategy.set_request_tolerance(0.15)` (SDK) or request body (proxy) | Cost/accuracy tradeoff changes |

### Integration Paths

```
                                ┌─────────────────────┐
                                │  Model Router        │
                                │  Toolkit (pip)       │
                                └──────┬──────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                      ▼
          ┌─────────────────┐                   ┌──────────────────┐
          │ SDK Integration │                   │ Proxy Integration │
          │ (3 lines of     │                   │ (CLI command)     │
          │  Python code)   │                   │                   │
          └────────┬────────┘                   └────────┬─────────┘
                   ▼                                      ▼
          ┌─────────────────┐                   ┌──────────────────┐
          │ Existing         │                   │ LiteLLM Proxy    │
          │ litellm.Router  │                   │ (patched at      │
          │ instance        │                   │  startup)        │
          └─────────────────┘                   └──────────────────┘
```

### Jobs-to-Be-Done Breakdown

| JTBD | Current State | Completeness |
|------|--------------|-------------|
| Add routing to LiteLLM SDK in 3 lines | `set_custom_routing_strategy()` | Complete |
| Add routing to LiteLLM Proxy | `model-router proxy` command | Complete |
| Generate proxy config from pool config | `model-router proxy-config` | Complete |
| Validate model alignment | `validate_model_alignment()` | Complete |
| Per-request tolerance control | `set_request_tolerance()` / body param | Complete |
| Access routing metadata | `strategy.last_result` | Complete |
| Fall back gracefully if routing fails | Fallback to default LiteLLM routing | Present but underdocumented |
| Version compatibility with LiteLLM | Requires `>=1.50.0` | Checked at startup |
| Use with LiteLLM callbacks | Not integrated | Missing |

### Gaps & Friction Points

1. ~~**Proxy mode not documented in integration.md**: `model-router proxy` exists but integration.md only covers server mode and SDK~~ Resolved: integration.md now covers all four paths (serve, proxy, SDK, direct library) with a decision guide
2. **Config bridge edge cases**: What happens with model names that don't match between pool and LiteLLM config
3. **No callback/logger integration**: Can't use ModelRoutingStrategy as a LiteLLM callback for scoring-only mode
4. **Thread safety**: `last_result` uses thread-local storage but async safety across concurrent requests is unclear
5. **No migration guide**: "I have this LiteLLM config, how do I add routing?" not explicitly documented

---

## Journey 6: Quality Assurance & Review (The QA Lead)

### Persona

Team leads and quality engineers who need to validate that routing decisions are actually correct and the system is performing as expected.

### Job to Be Done

> "Prove to me that the router is making good decisions and we're not sacrificing quality for cost."

### Journey Steps

| Step | Action | Asset | Success Criteria |
|------|--------|-------|-----------------|
| 6.1 | Enable auto-review | Send request to `/api/review` | Review stream starts |
| 6.2 | Judge answer quality | Most expensive model as judge evaluates response | Verdict: correct/incorrect with reasoning |
| 6.3 | Compare across models | Review compares what other models would have answered | Model comparison matrix |
| 6.4 | Run formal evaluation | `model-router evaluate --checkpoint ... --data ...` | Full metrics report |
| 6.5 | Interpret metrics | `docs/evaluation-guide.md` | Understand AUC, lift, headroom, agreement zones |
| 6.6 | Identify problem areas | Near-miss analysis, disagreement zones | Targeted improvement plan |
| 6.7 | Decide on retraining | Compare current vs. target metrics | Decision: retrain, adjust tolerance, or accept |
| 6.8 | Ongoing monitoring | Telemetry DB + periodic evaluation | Quality doesn't degrade over time |

### Jobs-to-Be-Done Breakdown

| JTBD | Current State | Completeness |
|------|--------------|-------------|
| Real-time answer quality review | `/api/review` endpoint with SSE | Complete |
| Offline batch evaluation | `model-router evaluate` CLI | Complete |
| Understand per-model performance | AUC + accuracy per model | Complete |
| See where routing fails | Near-miss analysis, disagreement zones | Complete |
| Compare models pairwise | Pairwise win rates in eval | Complete |
| Track quality over time | Telemetry SQLite (opt-in) | Partial — no dashboards |
| A/B test routing strategies | Not supported | Missing |
| Alert on quality degradation | Not supported | Missing |
| Export review results | SSE events only | No persistent storage |

### Gaps & Friction Points

1. **Review requires OPENROUTER_API_KEY**: Can't use NVIDIA API for judging
2. **No review result persistence**: Reviews stream via SSE but aren't stored
3. **No quality dashboard**: Telemetry data exists but no visualization
4. **No A/B testing framework**: Can't compare two checkpoints on live traffic
5. **No alerting**: No way to detect and alert on quality regression

---

## Broader Journeys (Toolkit as One Component)

These journeys extend beyond the toolkit itself. The Model Router Toolkit is one component in a larger workflow.

### Broader Journey A: Enterprise LLM Cost Optimization

**Context**: An enterprise spending $50K+/month on LLM API calls wants to reduce costs without sacrificing quality.

| Phase | Activity | Toolkit Role | Other Components |
|-------|----------|-------------|-----------------|
| 1. Audit | Analyze current LLM spending by model and use case | — | Billing dashboards, usage logs |
| 2. Evaluate | Prove routing can maintain quality at lower cost | **Journey 1** (quickstart) | Management buy-in, ROI analysis |
| 3. Benchmark | Collect accuracy data across model pool | **Journey 3** (collect) | Internal benchmark suite |
| 4. Deploy | Set up routing in staging | **Journey 2** (deploy) or **Journey 4** (Docker) | CI/CD pipeline, staging environment |
| 5. Validate | Run formal evaluation, compare to baseline | **Journey 6** (QA) | A/B testing infrastructure |
| 6. Production | Promote to production traffic | **Journey 4** (production) | Load balancers, monitoring, alerting |
| 7. Iterate | Retrain as models/costs change | **Journey 3** (train) | Model catalog management |
| 8. Report | Quantify savings for leadership | — | FinOps dashboards, reporting |

**Toolkit covers**: Phases 2-7 directly. Missing: Phase 1 (audit tooling), Phase 5 (A/B infrastructure), Phase 8 (reporting).

### Broader Journey B: Building an AI Product with Intelligent Routing

**Context**: A startup or team building a product (coding assistant, chatbot, support agent) that needs to serve multiple LLM quality tiers.

| Phase | Activity | Toolkit Role | Other Components |
|-------|----------|-------------|-----------------|
| 1. Design | Define quality tiers and cost targets per tier | — | Product requirements, pricing model |
| 2. Select models | Evaluate candidate models for the pool | **model-pool-reference.md** | Model benchmarks, trial accounts |
| 3. Prototype | Build prototype with manual model selection | — | Application code, frontend |
| 4. Add routing | Replace manual selection with intelligent routing | **Journey 5** (LiteLLM integration) | Application LiteLLM setup |
| 5. Customize | Collect domain data and train custom router | **Journey 3** (train) | Domain question sets |
| 6. Ship | Deploy router as part of application infrastructure | **Journey 4** (Docker/K8s) | Application deployment pipeline |
| 7. Monitor | Track routing quality alongside product metrics | **Journey 6** (QA) | Application analytics, user feedback |
| 8. Evolve | Update model pool as new models launch | `model-pool-reference.md`, retrain | Model marketplace monitoring |

**Toolkit covers**: Phases 4-7 directly. Phases 2, 8 partially (model pool reference). Missing: Phase 1 (product design), Phase 3 (application prototype).

### Broader Journey C: ML Team Model Evaluation & Selection

**Context**: An ML team evaluating multiple LLMs for a use case and wanting data-driven model selection (not just vibes).

| Phase | Activity | Toolkit Role | Other Components |
|-------|----------|-------------|-----------------|
| 1. Define criteria | Accuracy, latency, cost, safety thresholds | — | Eval framework (e.g., NeMo Evaluator) |
| 2. Create eval set | Curate representative questions for domain | — | Domain expertise, existing datasets |
| 3. Run all models | Get answers from every candidate model | **`model-router collect`** | API access to candidate models |
| 4. Judge correctness | Automated judging via majority vote or reference | **`model-router collect --judge`** | Optional: human review |
| 5. Analyze results | Per-model accuracy, cost, speed tradeoffs | **`model-router evaluate`** metrics | Visualization tools |
| 6. Decide | Pick model(s) based on data | — | Stakeholder review |
| 7. Optional: Route | If multiple models selected, deploy routing | **Journey 2** or **Journey 5** | — |

**Toolkit covers**: Phases 3-5, 7. Missing: Phase 1 (criteria definition framework), Phase 6 (decision support).

### Broader Journey D: Developer Experience & Coding Assistant Integration

**Context**: A developer or team using AI coding assistants (like OpenClaw, OpenCode, Cursor, or custom tools) wants intelligent routing behind their coding workflows.

| Phase | Activity | Toolkit Role | Other Components |
|-------|----------|-------------|-----------------|
| 1. Install router | Deploy Model Router Toolkit | **Journey 2** (deploy) | — |
| 2. Configure coding tool | Point `OPENAI_API_BASE` to router | **`docs/integration.md`** | Coding assistant configuration |
| 3. Use normally | Write code, ask questions, get completions | Router handles model selection transparently | Coding assistant IDE |
| 4. Review routing | Check which models handle which queries | **Playground UI**, telemetry | — |
| 5. Customize | Collect coding-specific data, train custom router | **Journey 3** (train) | Code benchmark datasets |
| 6. Share config | Distribute config across team | Config YAML + checkpoint | Team config management |

**Toolkit covers**: All phases for the routing layer. The coding assistant itself is external.

### Broader Journey E: Multi-Tenant LLM Platform

**Context**: A platform team serving multiple internal teams, each with different model access, budgets, and quality requirements.

| Phase | Activity | Toolkit Role | Other Components |
|-------|----------|-------------|-----------------|
| 1. Define tenants | Team-specific model pools and budgets | — | Organization management, billing |
| 2. Create per-tenant configs | Separate pool_config.yaml per team | **Config system** | Config management tool |
| 3. Deploy shared infrastructure | LiteLLM Proxy with routing | **Journey 4** (Docker), **Journey 5** (proxy) | Kubernetes, API gateway |
| 4. Route per tenant | Tenant-specific routing strategy | **Not directly supported** | Request headers, API key mapping |
| 5. Train per tenant | Custom routers for each team's domain | **Journey 3** (train) per tenant | Isolated training pipelines |
| 6. Monitor per tenant | Per-tenant quality and cost tracking | **Telemetry** (partial) | Multi-tenant dashboards |

**Toolkit covers**: Core routing infrastructure (Phases 2-3, 5). Major gap: no multi-tenant support out of the box. Per-tenant routing requires running multiple instances or extending the strategy.

---

## Jobs-to-Be-Done Matrix

Complete JTBD inventory across all personas.

### Discovery & Understanding

| # | Job | Persona | Asset | Status |
|---|-----|---------|-------|--------|
| D1 | Understand what model routing is | Evaluator | Both quickstart notebooks, README | Partial |
| D2 | See a live KMeans routing decision | Evaluator | `notebooks/quickstart.ipynb` | Complete |
| D3 | See a live prefill routing decision | Evaluator | `notebooks/quickstart-prefill.ipynb` | Complete |
| D4 | Quantify cost savings potential | Evaluator, QA Lead | Evaluation metrics, notebook outputs | Complete |
| D5 | Compare routing methods (KMeans vs. prefill) | Evaluator | Both notebooks | Partial — no cross-comparison |
| D6 | Understand accuracy/cost tradeoff | All | Tolerance parameter, eval guide | Complete |

### Setup & Configuration

| # | Job | Persona | Asset | Status |
|---|-----|---------|-------|--------|
| S1 | Choose routing method for hardware | Integrator | README, config decision table in `configs/` | Complete |
| S2 | Configure API keys and providers | Integrator | Config comments, `.env.example` | Complete |
| S3 | Create a valid config file | Integrator | Copy example config from `configs/` and customize | Complete |
| S4 | Validate config before serving | Integrator | Pydantic validation in `config.py` | Complete |
| S5 | Understand config schema | All | `configs/schema.md` | Complete |
| S6 | Choose between deployment modes | Platform Eng. | `docs/architecture.md`, `docs/integration.md` | Complete — decision guide in both docs |

### Deployment & Integration

| # | Job | Persona | Asset | Status |
|---|-----|---------|-------|--------|
| I1 | Start a router server | Integrator | `model-router serve` | Complete |
| I2 | Get an OpenAI-compatible endpoint | Integrator | `/v1/chat/completions` | Complete |
| I3 | Deploy routing without inference | Integrator | `model-router serve-router` → `POST /v1/route` | Complete |
| I4 | Connect downstream apps | Integrator | `docs/integration.md` | Complete |
| I5 | Add routing to existing LiteLLM SDK | LiteLLM User | `ModelRoutingStrategy` | Complete |
| I6 | Add routing to existing LiteLLM Proxy | LiteLLM User | `model-router proxy` | Complete |
| I7 | Deploy in Docker | Platform Eng. | `docker/`, compose | Complete |
| I8 | Deploy to Kubernetes | Platform Eng. | — | Missing |
| I9 | Run as system service/daemon | Platform Eng. | — | Missing |
| I10 | Configure health checks | Platform Eng. | `/health`, Dockerfile HEALTHCHECK | Complete |

### Training & Customization

| # | Job | Persona | Asset | Status |
|---|-----|---------|-------|--------|
| T1 | Collect labeled training data | Optimizer | `model-router collect` | Complete (vote + reference) |
| T2 | Train a prefill router | Optimizer | `model-router train` | Complete |
| T3 | Train a KMeans router | Optimizer | — | Not implemented |
| T4 | Cache intermediate artifacts | Optimizer | `--prefill-dir` | Complete |
| T5 | Run a smoke test | Optimizer | `configs/smoke-test.yaml` | Complete |
| T6 | Iterate on hyperparameters | Optimizer | CLI flags | Complete |
| T7 | Use LLM-as-judge | Optimizer | `--judge llm` | Not implemented |
| T8 | Split data into train/test | Optimizer | — | Manual, no tool |

### Evaluation & Quality

| # | Job | Persona | Asset | Status |
|---|-----|---------|-------|--------|
| Q1 | Evaluate a checkpoint | Optimizer, QA | `model-router evaluate` | Complete |
| Q2 | Understand evaluation metrics | QA Lead | `docs/evaluation-guide.md` | Complete |
| Q3 | Review individual answers in real-time | QA Lead | `/api/review` | Complete |
| Q4 | Compare routing quality to baseline | QA Lead | Lift, headroom captured | Complete |
| Q5 | Track quality over time | QA Lead | Telemetry (opt-in) | Partial |
| Q6 | A/B test checkpoints | QA Lead | — | Missing |
| Q7 | Export evaluation reports | Optimizer | — | Missing (stdout only) |

### Operations & Monitoring

| # | Job | Persona | Asset | Status |
|---|-----|---------|-------|--------|
| O1 | Monitor router health | Platform Eng. | `/health` endpoint | Complete |
| O2 | View routing telemetry | Platform Eng. | SQLite telemetry | Partial (opt-in, no dashboard) |
| O3 | Structured logging | Platform Eng. | — | Missing |
| O4 | Prometheus/OpenTelemetry metrics | Platform Eng. | — | Missing |
| O5 | Alert on quality degradation | Platform Eng., QA | — | Missing |
| O6 | Hot-reload configuration | Platform Eng. | — | Missing |

---

## Journey Dependencies & Progression

```
Journey 1 (Evaluate)
    │
    ├──► Journey 2 (Deploy)
    │        │
    │        ├──► 2a/2b: Full mode (routing + inference)
    │        │        │
    │        │        ├──► Journey 4 (Production)
    │        │        │        │
    │        │        │        └──► Broader E (Multi-Tenant Platform)
    │        │        │
    │        │        ├──► Journey 6 (QA)
    │        │        │        │
    │        │        │        └──► Broader A (Cost Optimization)
    │        │        │
    │        │        └──► Broader D (Coding Assistant)
    │        │
    │        └──► 2c: Router-only mode (routing decisions, no inference)
    │                 │
    │                 └──► Broader B (AI Product — your app dispatches LLM calls)
    │
    ├──► Journey 5 (LiteLLM Integration)
    │        │
    │        └──► Broader B (AI Product)
    │
    └──► Journey 3 (Train)
             │
             ├──► Journey 6 (QA)
             │
             └──► Broader C (Model Evaluation)
```

**Natural progression paths:**
1. **Evaluator → Integrator (full) → Platform Engineer**: Try → deploy with inference → productionize
2. **Evaluator → Integrator (router-only) → AI Product**: Try → deploy routing as microservice → build your own dispatch
3. **Evaluator → Optimizer → QA Lead**: Try → customize → validate
4. **LiteLLM User → Optimizer → Platform Engineer**: Integrate → tune → operate
5. **Any → Broader Journey**: Toolkit is the routing component in larger initiatives

---

## Virtual Review Plan

### Purpose

A structured, agent-driven review of every user journey where an automated agent walks through each journey step-by-step, examines all documentation and assets, attempts to execute the jobs to be done, and produces detailed feedback with improvement recommendations.

### Review Philosophy

Each journey review follows the principle: **"Can a user with the stated persona actually complete every step, using only the documented assets, without prior knowledge of the codebase?"**

The review evaluates five dimensions per journey:

| Dimension | Question |
|-----------|----------|
| **Completeness** | Are all steps achievable with existing assets? |
| **Clarity** | Are instructions unambiguous and self-contained? |
| **Executability** | Do CLI commands, code snippets, and configs work as documented? |
| **Error Handling** | What happens when things go wrong? Are errors actionable? |
| **Continuity** | Is the path to the next journey clear and friction-free? |

### Review Methodology

Each journey review consists of three phases:

#### Phase 1: Documentation Audit

The agent reads every asset listed in the journey's "Assets Touched" section:

1. Read each documentation file end-to-end
2. Verify internal cross-references (links to other docs, configs, commands)
3. Check that code examples match actual source code
4. Verify config examples against schema definitions
5. Flag stale information (wrong model names, deprecated parameters, broken paths)

**Output**: Per-asset audit checklist with pass/fail/warning per item.

#### Phase 2: Execution Walk-through

The agent attempts to execute each journey step:

1. **CLI commands**: Parse and validate syntax, check that referenced flags exist in `__main__.py`, verify help text
2. **Config files**: Load referenced YAML configs, validate against `PoolConfig` schema, check all referenced checkpoints exist
3. **API endpoints**: Review FastAPI route definitions, verify request/response schemas match documentation
4. **Code snippets**: Trace import paths, verify classes and methods exist, check function signatures
5. **Notebook cells**: Review cell contents, verify dependencies are installable, check API endpoint URLs

For steps requiring live API calls or GPU hardware, the agent performs a **dry-run analysis**:
- Verify API endpoint URLs are valid
- Check that model names match provider catalogs
- Validate that environment variable names are consistent across docs and code
- Confirm error handling paths exist for common failure modes (no API key, wrong model, timeout)

**Output**: Per-step execution log with status (pass/dry-run-pass/fail), notes, and screenshots or output samples where applicable.

#### Phase 3: Gap Analysis & Recommendations

Synthesize findings into actionable improvements:

1. **Blockers**: Steps that cannot be completed as documented
2. **Friction points**: Steps that work but are confusing, slow, or poorly explained
3. **Missing pieces**: Jobs-to-be-done marked as "Missing" or "Partial" in the journey matrix
4. **Quick wins**: Low-effort improvements with high UX impact
5. **Strategic recommendations**: Larger investments that would significantly improve the journey

**Output**: Prioritized recommendation list with effort estimates (S/M/L) and impact ratings (1-5).

### Per-Journey Review Specification

#### Review J1: Explore & Evaluate (The Evaluator)

| Phase | Specific Actions |
|-------|-----------------|
| Doc Audit | Read `notebooks/quickstart.ipynb` (KMeans) and `notebooks/quickstart-prefill.ipynb` (prefill) cell-by-cell, README "Quick Start" section, `checkpoints/` directory contents |
| Execution (KMeans) | Verify pickle loads with expected structure (100 clusters, 3+ models), confirm NVIDIA API endpoint URLs, validate embedding dimension, check model names match build.nvidia.com catalog |
| Execution (Prefill) | Verify encoder model loading (Qwen3.5-0.8B), check hidden state extraction, verify prefill checkpoint availability and loading, check API endpoints for model calls |
| Cross-comparison | Verify both notebooks cover the same example questions, check whether results are comparable, evaluate if tradeoffs between methods are clearly communicated |
| Gaps | Evaluate time-to-first-routing-decision for each notebook, assess "convinced moment" quality, check "What's Next" sections, verify navigation between the two notebooks |

**Key questions to answer:**
- Can a new user complete the KMeans notebook in under 5 minutes?
- Can a new user complete the prefill notebook in under 10 minutes (accounting for encoder load time)?
- Do both notebooks work without `model_router_toolkit` installed?
- Are checkpoints available for both methods (`.pkl` and `.pt`)?
- Are API endpoints currently active and returning expected formats?
- Is there a clear narrative linking the two notebooks (e.g., "try KMeans first, then prefill for higher accuracy")?
- Does the evaluator understand when to choose each method after completing both?

#### Review J2: Deploy a Router (The Integrator)

| Phase | Specific Actions |
|-------|-----------------|
| Doc Audit | Read README install section, `docs/integration.md`, `.env.example`, all example config YAMLs, `configs/schema.md` |
| Execution | Validate `pyproject.toml` install extras, verify config copy-and-customize flow, verify server startup sequence, check all endpoint routes exist, test playground HTML/JS loads |
| Gaps | Measure steps from install to first routed request, check error messages for common failures (wrong key, missing checkpoint, port conflict) |

**Key questions to answer:**
- Does `pip install -e .` succeed cleanly?
- Do the example configs cover all hardware configurations (GPU, no GPU, local-only)?
- Does the playground UI load and function with both routing methods?
- Can I connect a standard OpenAI client in under 2 minutes?

#### Review J3: Train & Optimize (The Optimizer)

| Phase | Specific Actions |
|-------|-----------------|
| Doc Audit | Read `docs/training-guide.md` end-to-end, `docs/evaluation-guide.md`, smoke test config, `data/` directory contents |
| Execution | Trace `collect` → `train` → `evaluate` code paths, validate CSV schemas, check training hyperparameter defaults, verify checkpoint output structure |
| Gaps | Evaluate data preparation guidance, check for training failure modes, assess evaluation report readability |

**Key questions to answer:**
- Can I complete a smoke test without prior ML knowledge?
- Is the training guide self-contained (no missing context)?
- Does the evaluation report provide actionable next steps?
- Are hyperparameter defaults reasonable for first-time use?

#### Review J4: Production Deployment (The Platform Engineer)

| Phase | Specific Actions |
|-------|-----------------|
| Doc Audit | Read `docker/Dockerfile`, `docker-compose.yaml`, `entrypoint.sh`, `.dockerignore` |
| Execution | Parse Dockerfile stages, verify compose config, check health check configuration, validate volume mounts, verify port mappings |
| Gaps | Evaluate K8s readiness, assess security posture, check for production hardening |

**Key questions to answer:**
- Does `docker compose up` work on first try?
- Is the container suitable for production (non-root user, health checks, resource limits)?
- Can I deploy this to Kubernetes with reasonable effort?
- Are secrets handled appropriately?

#### Review J5: Plug into Existing LiteLLM (The LiteLLM User)

| Phase | Specific Actions |
|-------|-----------------|
| Doc Audit | Read SDK integration examples in `docs/integration.md`, proxy module README, config bridge documentation |
| Execution | Verify `ModelRoutingStrategy.from_config()` code path, trace `set_custom_routing_strategy()` integration, check proxy startup injection, validate config bridge output |
| Gaps | Evaluate migration effort from vanilla LiteLLM, check compatibility matrix, assess fallback behavior |

**Key questions to answer:**
- Can I add routing to an existing `litellm.Router` in 3 lines as claimed?
- Does the proxy mode work as a drop-in replacement?
- What happens if the routing strategy fails (fallback behavior)?
- Is the `proxy-config` output compatible with standard LiteLLM proxy configs?

#### Review J6: Quality Assurance & Review (The QA Lead)

| Phase | Specific Actions |
|-------|-----------------|
| Doc Audit | Read `docs/evaluation-guide.md`, review endpoint docs, telemetry module docs |
| Execution | Trace review endpoint flow, verify judge model selection, check evaluation metric computation, validate telemetry schema |
| Gaps | Evaluate quality monitoring completeness, check for ongoing quality assurance workflows |

**Key questions to answer:**
- Does the review endpoint provide actionable quality signals?
- Can I build a quality monitoring workflow from the available tools?
- Are the evaluation metrics well-explained for a non-ML audience?
- Is there a path from "problem detected" to "problem fixed"?

### Review Output Document Structure

The virtual review produces a single structured document:

```
# Model Router Toolkit — Virtual User Journey Review

## Executive Summary
- Overall readiness score (1-10)
- Top 3 blockers across all journeys
- Top 3 quick wins
- Recommended priority order for improvements

## Per-Journey Reviews

### Journey N: [Title]

#### Documentation Audit
| Asset | Status | Issues |
|-------|--------|--------|

#### Execution Walk-through
| Step | Status | Notes |
|------|--------|-------|

#### Gap Analysis
| Gap | Severity | Effort | Impact | Recommendation |
|-----|----------|--------|--------|----------------|

#### Score Card
| Dimension | Score (1-5) | Notes |
|-----------|-------------|-------|
| Completeness | | |
| Clarity | | |
| Executability | | |
| Error Handling | | |
| Continuity | | |

### [Repeat for each journey...]

## Cross-Journey Analysis
- Common patterns across journeys
- Shared infrastructure gaps
- Documentation consistency issues
- Priority improvement matrix

## Recommendations
### Tier 1: Blockers (fix before any user testing)
### Tier 2: Quick Wins (high impact, low effort)
### Tier 3: Strategic Improvements (high impact, higher effort)
### Tier 4: Nice-to-Have (lower impact, any effort)

## Appendix
- Full asset inventory
- Command reference validation
- Config schema compliance matrix
```

### Execution Instructions for the Review Agent

The review agent should follow this protocol:

1. **Read this document first** to understand all journeys and their assets
2. **For each journey (J1-J6)**, execute the three phases in order
3. **Use the Read tool** to examine every file — do not grep as a shortcut
4. **For executability checks**, trace code paths through the actual source (not just docs)
5. **For config validation**, load YAML files and check against `PoolConfig` class definition
6. **Document every finding** with file path, line number, and specific issue
7. **Score each dimension** independently per journey (1 = non-functional, 5 = excellent)
8. **Prioritize recommendations** by impact × (1/effort) — high impact, low effort first
9. **Produce the output document** following the structure above
10. **Be specific**: "Button X doesn't work" > "UI has issues"; "Line 42 of train.py references nonexistent flag" > "training might have bugs"

### Estimated Review Duration

| Journey | Estimated Asset Count | Review Complexity |
|---------|----------------------|-------------------|
| J1: Evaluate | 5 files (2 notebooks + README + 2 checkpoints) | Medium (two notebooks, cross-comparison) |
| J2: Deploy | 12+ files | High (multi-module) |
| J3: Train | 8+ files | High (pipeline complexity) |
| J4: Production | 5 files | Medium (Docker + config) |
| J5: LiteLLM | 6+ files | Medium (integration surface) |
| J6: QA | 5+ files | Medium (metrics + endpoints) |
| **Cross-journey** | All above | High (consistency checks) |

**Total**: Full review touches 40+ unique files across the repository.
