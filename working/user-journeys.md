# Model Router Toolkit — User Journeys & Jobs to Be Done

> **Version:** 2.0 | **Last Updated:** March 12, 2026

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
8. [Journey 7: Gateway & Webhook Integration (The Gateway Admin)](#journey-7-gateway--webhook-integration-the-gateway-admin)
9. [Journey 8: Extend & Contribute (The Contributor)](#journey-8-extend--contribute-the-contributor)
10. [Broader Journeys (Toolkit as One Component)](#broader-journeys-toolkit-as-one-component)
11. [Jobs-to-Be-Done Matrix](#jobs-to-be-done-matrix)
12. [Journey Dependencies & Progression](#journey-dependencies--progression)
13. [Virtual Review Plan](#virtual-review-plan)

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
| **Gateway Admin** | Platform/infra engineer with existing API gateway | Add routing to gateway (OpenClaw, Portkey, etc.) | Router sidecar + plugin/webhook | Request overhead, fallback rate |
| **Contributor** | ML engineer, platform developer | Add custom routing method, adapter, or plugin | `docs/extending.md` | Time to integrate custom method |

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

- `src/model_router_toolkit/adapters/litellm/` — Full-mode FastAPI app (routing + inference + UI): `app.py`, `completions.py`, `chat.py`, `review.py`
- `src/model_router_toolkit/adapters/litellm/static/` — Playground UI (`index.html`, `playground.js`, `playground.css`)
- `src/model_router_toolkit/adapters/http/` — Router-only FastAPI app: `app.py`, `route.py`, `auth.py`, `_shared.py`
- `docs/integration.md` — Seven integration paths with decision matrix
- `docs/quickstart.md` — Quick start with install extras, Python routing, sidecar, server, SDK
- `configs/*.yaml` — Example configs: `prefill-qwen08b`, `cloud-only`, `smoke-test`, `nvidia-nim-smoke`, `openrouter-kmeans`, `local-prefill`, `litellm-proxy`
- `configs/schema.md` — Configuration field reference

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

### Sub-Journey 2d: Model Pinning (Router-Per-Subagent)

For multi-turn agent chains where mid-chain model switches would hurt quality. Route once, then pin the selected model for subsequent calls.

| Step | Action | CLI/Asset | Success Criteria |
|------|--------|-----------|-----------------|
| 2d.1 | Send first routing request | `POST /v1/route` with question | Returns `selected_model` via ML inference |
| 2d.2 | Capture selected model | Read `selected_model` from response | Model name stored |
| 2d.3 | Pin on subsequent calls | `POST /v1/route` with `{"model": "nem-think"}` | Instant response with `metadata.pinned: true`, no ML inference |
| 2d.4 | Use with LiteLLM SDK | `metadata={"pin_model": "nem-think"}` in request_kwargs | Strategy calls `resolve()` instead of `route()` |
| 2d.5 | Use in Direct Python | `router.resolve("nem-think")` | Returns `RoutingResult` instantly |

**When to use this mode:**
- Multi-turn agent chains (subagent should keep using the same model)
- High-throughput systems where you route once per session, not per request
- Debugging: pin to a specific model to isolate routing from inference issues

**How pinning works across adapters:**

| Adapter | Pin signal | Mechanism |
|---------|-----------|-----------|
| HTTP sidecar | `model` field in request body | Presence of `model` triggers `resolve()` instead of `route()` |
| LiteLLM strategy | `metadata.pin_model` in request_kwargs | Strategy checks metadata before running encoder |
| Direct Python | Caller calls `router.resolve(name)` | Explicit programmatic pin |

### Jobs-to-Be-Done Breakdown

| JTBD | Current State | Completeness |
|------|--------------|-------------|
| Install and run in < 10 min | Copy config + serve | Complete |
| Get an OpenAI-compatible endpoint | `/v1/chat/completions` | Complete |
| Deploy routing without LLM inference | `model-router serve-router` → `POST /v1/route` | Complete |
| See what the router is doing | Playground UI with routing cards, pipeline viz, TTFT, probability bars | Complete (full mode only) |
| Connect my existing app | `docs/integration.md` — 7 integration paths with decision matrix | Complete |
| Configure model pool | YAML config with `configs/schema.md` reference | Complete |
| Add/remove models | Edit YAML, restart server | Complete but manual |
| Use with multiple providers | LiteLLM handles NVIDIA, OpenRouter, OpenAI, Anthropic | Complete |
| Pin a model for multi-turn chains | `resolve()` / `model` field / `metadata.pin_model` | Complete |
| Secure the sidecar endpoint | `WebhookAuthMiddleware` — HMAC-SHA256 + bearer token | Complete |
| Hot-reload config changes | Not supported | Missing |
| Monitor routing decisions | Telemetry (opt-in SQLite) | Partial |
| SSL/TLS termination | Not handled by toolkit | Out of scope — use reverse proxy |

### Gaps & Friction Points

1. **Encoder download time**: First-run downloads Qwen3.5-0.8B (~1.6GB) — no progress indication documented
2. **Port conflicts**: Default 8000 may conflict with other services; no auto-detection
3. **No systemd/launchd recipes**: No guidance for running as a daemon
4. **Playground limitations**: No conversation history, no multi-turn demo
5. **No hot-reload**: Config changes require server restart
6. **Telemetry integration gap**: `telemetry.py` is implemented but not wired into adapters — chat events not logged automatically

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
| **Router-only sidecar** | `model-router serve-router` | 8079 | Yes | No | Routing decisions as a microservice, no API keys needed |
| **LiteLLM Proxy** | `model-router proxy` | 4000 | Yes | Yes | Drop-in replacement for existing LiteLLM proxy |
| **Docker (Proxy)** | `docker compose up` | 4000 | Yes | Yes | Containerized LiteLLM proxy with routing |
| **Docker (GPU)** | Build `proxy-gpu` target | 4000 | Yes | Yes | Prefill routing in container |

### Docker Details

- **Base image**: `python:3.12-slim`
- **Dockerfile targets**: `proxy` (CPU, `.[proxy]`) and `proxy-gpu` (GPU, `.[proxy,prefill]`)
- **Health check**: `curl -f http://localhost:4000/health` (built into Dockerfile)
- **Entrypoint**: `docker/entrypoint.sh` runs `model-router proxy` with `LITELLM_CONFIG` and `ROUTER_CONFIG` env vars
- **Volumes**: `configs/` and `checkpoints/` mounted read-only via compose

### Sidecar Security

When deploying the router-only sidecar (`serve-router`) in production, enable webhook authentication:

```bash
export ROUTER_WEBHOOK_SECRET=my-shared-secret
model-router serve-router --config pool.yaml --port 8079
```

Supports HMAC-SHA256 (`X-Webhook-Signature` header) and bearer token (`Authorization: Bearer <secret>`). Health endpoint is always exempt from auth. See Journey 7 for full gateway integration details.

### Jobs-to-Be-Done Breakdown

| JTBD | Current State | Completeness |
|------|--------------|-------------|
| Containerized deployment | Multi-stage Dockerfile (`proxy`, `proxy-gpu` targets) | Complete |
| Docker Compose for dev | `docker/docker-compose.yaml` | Complete (proxy-gpu mode) |
| GPU-enabled container | `proxy-gpu` Dockerfile target | Complete (target exists, compose uses it) |
| Health checks | `/health` endpoint, Dockerfile HEALTHCHECK | Complete |
| Config volume mounts | `configs/` and `checkpoints/` mounted read-only | Complete |
| Environment variable config | Env vars in compose + entrypoint | Complete |
| Secure the sidecar | HMAC-SHA256 + bearer token via `ROUTER_WEBHOOK_SECRET` | Complete |
| Kubernetes manifests | Not provided | Missing |
| Helm charts | Not provided | Missing |
| Horizontal scaling guide | Not documented | Missing |
| Log aggregation | Stdout logging | Basic — no structured logging |
| Secrets management | Environment variables | Basic — no vault integration |
| TLS/SSL | Not handled | Out of scope (use ingress) |

### Gaps & Friction Points

1. **No Kubernetes manifests or Helm charts**: Container exists but no K8s deployment artifacts
2. **Port inconsistency across modes**: Serve uses 8000, sidecar uses 8079, proxy uses 4000 — document the reasoning
3. **Checkpoint provisioning**: Checkpoints are gitignored; no download or init-container strategy
4. **No resource limits**: No CPU/memory recommendations documented
5. **No rolling update strategy**: No guidance on zero-downtime deployments
6. **No observability stack**: No Prometheus metrics, no OpenTelemetry spans
7. **No structured logging**: All output is `print()` to stdout — no JSON logging for log aggregation

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

The toolkit now provides **seven** integration paths (`docs/integration.md`), three of which are LiteLLM-specific:

```
                           ┌───────────────────────┐
                           │  Model Router Toolkit  │
                           │        (pip)           │
                           └──────────┬────────────┘
                                      │
         ┌────────────┬───────────────┼───────────────┬─────────────┐
         ▼            ▼               ▼               ▼             ▼
   ┌───────────┐ ┌──────────┐ ┌──────────────┐ ┌──────────┐ ┌───────────┐
   │ LiteLLM   │ │ LiteLLM  │ │ Standalone   │ │ Router   │ │ Direct    │
   │ SDK       │ │ Proxy    │ │ Server       │ │ Sidecar  │ │ Python    │
   │ (embed)   │ │ (inject) │ │ (full mode)  │ │ (HTTP)   │ │ (library) │
   └─────┬─────┘ └────┬─────┘ └──────┬───────┘ └────┬─────┘ └─────┬─────┘
         │            │               │               │             │
         ▼            ▼               ▼               ▼             ▼
   litellm.Router  LiteLLM Proxy  /v1/completions  /v1/route    In-process
                   (patched)       + Playground     (+ webhook)   (no server)
                                                    (+ plugin)
```

Plus: **Webhook Integration** (authenticated sidecar) and **OpenClaw Plugin** (gateway hook) — see Journey 7.

### Jobs-to-Be-Done Breakdown

| JTBD | Current State | Completeness |
|------|--------------|-------------|
| Add routing to LiteLLM SDK in 4 lines | `set_custom_routing_strategy()` | Complete |
| Add routing to LiteLLM Proxy | `model-router proxy` command | Complete |
| Generate proxy config from pool config | `model-router proxy-config` | Complete |
| Validate model alignment | `validate_model_alignment()` in config_bridge | Complete |
| Per-request tolerance control | `set_request_tolerance()` (contextvars, async-safe) | Complete |
| Access routing metadata | `strategy.last_result` | Complete |
| Pin model for multi-turn chains | `metadata.pin_model` in SDK, `model` in sidecar | Complete |
| Fall back gracefully if routing fails | Fallback to first deployment | Complete — documented in `docs/adapters.md` |
| Version compatibility with LiteLLM | Requires `>=1.50.0,<2.0` | Checked at startup |
| Choose between 7 integration paths | Decision matrix in `docs/integration.md` | Complete |
| Use with LiteLLM callbacks | Not integrated | Missing |

### Gaps & Friction Points

1. ~~**Proxy mode not documented in integration.md**~~ Resolved: `docs/integration.md` now covers all seven paths with a decision matrix
2. **Config bridge edge cases**: `validate_model_alignment()` warns on mismatches but doesn't auto-fix
3. **No callback/logger integration**: Can't use ModelRoutingStrategy as a LiteLLM callback for scoring-only mode
4. **No migration guide**: "I have this LiteLLM config, how do I add routing?" — could benefit from a step-by-step migration walkthrough

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

### Assets Touched

- `src/model_router_toolkit/adapters/litellm/review.py` — Auto-judge endpoint (SSE)
- `src/model_router_toolkit/evaluate.py` — Batch evaluation with rich metrics
- `src/model_router_toolkit/telemetry.py` — SQLite session/chat logging
- `docs/evaluation-guide.md` — Metric interpretation guide
- `adapters/litellm/static/` — Playground UI with auto-review toggle

### Jobs-to-Be-Done Breakdown

| JTBD | Current State | Completeness |
|------|--------------|-------------|
| Real-time answer quality review | `/api/review` SSE — judge verdict + model comparison | Complete |
| Offline batch evaluation | `model-router evaluate` — rich prefill report, basic kmeans report | Complete |
| Understand per-model performance | AUC + accuracy per model | Complete |
| See where routing fails | Near-miss analysis, disagreement zones, pairwise win rates | Complete |
| Compare models pairwise | Pairwise confidence win rates (≤6 models) | Complete |
| Auto-review in playground | Toggle in playground UI, streams judge verdict | Complete |
| Track quality over time | Telemetry SQLite (opt-in) — sessions and chat_events | Partial — schema exists but not wired into adapters |
| A/B test routing strategies | Not supported | Missing |
| Alert on quality degradation | Not supported | Missing |
| Export review results | SSE events only | No persistent storage |
| Export evaluation reports | Printed to stdout | Missing — no `--output` flag |

### Gaps & Friction Points

1. **Review requires API key**: Uses the most expensive model in pool as judge — requires provider API key (OPENROUTER_API_KEY or similar)
2. **No review result persistence**: Reviews stream via SSE but aren't stored in telemetry
3. **Telemetry not wired**: `telemetry.py` has `create_session()` and `log_chat()` but adapters don't call them
4. **No quality dashboard**: Telemetry data exists but no visualization
5. **No A/B testing framework**: Can't compare two checkpoints on live traffic
6. **No alerting**: No way to detect and alert on quality regression
7. **No evaluation export**: `model-router evaluate` prints to stdout only — no `--output` flag for file export

---

## Journey 7: Gateway & Webhook Integration (The Gateway Admin)

### Persona

Platform and infrastructure engineers who run an existing API gateway (OpenClaw, Portkey, TrueFoundry, Cloudflare, Kong, Envoy) and want to add intelligent routing without replacing their gateway.

### Job to Be Done

> "I already have a gateway. Add smart routing as a sidecar without touching my inference pipeline."

### Journey Steps

| Step | Action | Asset | Success Criteria |
|------|--------|-------|-----------------|
| 7.1 | Install router sidecar | `pip install 'model-router-toolkit[server,prefill]'` | Package installs without errors |
| 7.2 | Start router sidecar | `model-router serve-router --config pool.yaml --port 8079` | `/health` returns `mode: router-only` |
| 7.3 | Verify routing decisions | `curl -X POST http://localhost:8079/v1/route -d '{"question": "..."}'` | Returns `selected_model`, `confidences`, `costs` |
| 7.4 | Enable webhook auth | `export ROUTER_WEBHOOK_SECRET=my-secret` + restart | Unauthenticated requests get 401 |
| 7.5 | Choose integration method | See sub-journeys below | Decision made |
| 7.6 | Configure gateway hook | Point gateway's pre-request hook at sidecar | Gateway calls sidecar before each LLM request |
| 7.7 | Map model names | Configure pool mapping (router names → gateway provider/model) | Selected model resolves to correct gateway model |
| 7.8 | Test end-to-end | Send request through gateway | Request routed through sidecar, correct model selected |
| 7.9 | Test fallback | Stop sidecar, send request | Gateway falls back to default model selection |

### Sub-Journey 7a: OpenClaw Plugin

TypeScript plugin using the `before_model_resolve` hook. Full implementation provided.

| Step | Action | Asset | Success Criteria |
|------|--------|-------|-----------------|
| 7a.1 | Copy plugin to OpenClaw | `cp -r src/.../plugins/openclaw/ /path/to/openclaw/plugins/model-router/` | Plugin files in place |
| 7a.2 | Configure plugin | Add to OpenClaw config with `sidecarUrl`, `tolerance`, `pool` mapping | Config valid |
| 7a.3 | Test routing | Send request through OpenClaw | Plugin calls sidecar, overrides model/provider |
| 7a.4 | Verify fallback | Stop sidecar | Plugin returns `{}`, OpenClaw uses default |

**Assets**: `plugins/openclaw/index.ts`, `plugins/openclaw/openclaw.plugin.json`, `docs/plugins.md`

### Sub-Journey 7b: Webhook Integration (Portkey, TrueFoundry, Custom)

HTTP webhook pattern — gateway calls sidecar's `/v1/route` with HMAC or bearer auth.

| Step | Action | Asset | Success Criteria |
|------|--------|-------|-----------------|
| 7b.1 | Set shared secret | `export ROUTER_WEBHOOK_SECRET=my-secret` | Secret configured |
| 7b.2 | Configure HMAC auth | `X-Webhook-Signature: HMAC(secret, body, SHA256).hexdigest()` | Authenticated requests pass |
| 7b.3 | Or configure bearer auth | `Authorization: Bearer my-secret` | Authenticated requests pass |
| 7b.4 | Build name mapping | Map router model names to gateway model identifiers | Mapping table complete |
| 7b.5 | Implement webhook handler | See examples in `docs/plugins.md` (Portkey, LangChain) | Handler calls sidecar and maps result |

**Assets**: `adapters/http/auth.py`, `docs/plugins.md` Part 2 (writing plugins for other platforms), `docs/integration.md` (webhook section)

### Sub-Journey 7c: Custom Gateway Plugin

Build a new plugin for any platform with a pre-request hook.

| Step | Action | Asset | Success Criteria |
|------|--------|-------|-----------------|
| 7c.1 | Identify platform hook | Find pre-request / model-selection extension point | Hook identified |
| 7c.2 | Call `/v1/route` from hook | HTTP POST with question + tolerance | Route response returned |
| 7c.3 | Map `selected_model` to platform | Build lookup table for router → platform model IDs | Mapping works |
| 7c.4 | Handle failures gracefully | Return empty/default on sidecar timeout or error | Fallback verified |
| 7c.5 | Add health check | Call `/health` on startup | Startup check passes |

**Assets**: `docs/plugins.md` Part 2 (step-by-step guide + examples for Portkey, LangChain, Kong, Envoy)

### Architecture

```
┌──────────────────┐     POST /v1/route     ┌──────────────────────┐
│  Gateway          │ ────────────────────>  │  Router Sidecar      │
│  (OpenClaw,       │ <────────────────────  │  (model-router       │
│   Portkey, Kong,  │     RouteResponse      │   serve-router)      │
│   Envoy, etc.)    │                        │                      │
│  + Hook/Plugin    │     Auth (optional):   │  + WebhookAuthMiddleware │
└──────────────────┘     HMAC / Bearer       └──────────────────────┘
```

### Jobs-to-Be-Done Breakdown

| JTBD | Current State | Completeness |
|------|--------------|-------------|
| Deploy sidecar alongside gateway | `model-router serve-router` | Complete |
| Authenticate sidecar requests | HMAC-SHA256 + bearer via `ROUTER_WEBHOOK_SECRET` | Complete |
| OpenClaw integration | Full plugin with `before_model_resolve` hook | Complete |
| Portkey webhook integration | Example in `docs/plugins.md` | Complete (example only) |
| LangChain custom router | Example in `docs/plugins.md` | Complete (example only) |
| Map router model names to gateway models | Pool config in OpenClaw, lookup tables in examples | Complete |
| Graceful fallback on sidecar failure | All examples return default on error | Complete |
| Plugin health check on startup | OpenClaw `gateway_start` hook | Complete (OpenClaw only) |
| Other gateway plugins (Kong, Envoy, Cloudflare) | Listed in docs, no implementations | Partial — patterns described |

### Gaps & Friction Points

1. **Only OpenClaw has a packaged plugin**: Other gateways require custom implementation using the webhook/HTTP pattern
2. **No Docker Compose for sidecar + gateway**: No ready-made compose file pairing the sidecar with a gateway
3. **Sidecar warm-up latency**: First request with prefill encoder takes 5-15s — gateway timeout may need increasing
4. **No mTLS**: Auth is HMAC/bearer only — no mutual TLS for sidecar communication
5. **No distributed tracing**: No trace propagation between gateway and sidecar

---

## Journey 8: Extend & Contribute (The Contributor)

### Persona

ML engineers and platform developers who want to add custom routing methods, write new adapters for their platforms, or contribute improvements back to the toolkit.

### Job to Be Done

> "I have a better routing algorithm / a platform integration to add. How do I plug it in?"

### Journey Steps

| Step | Action | Asset | Success Criteria |
|------|--------|-------|-----------------|
| 8.1 | Understand architecture | Read `docs/architecture.md` | Clear on BaseRouter, adapters, config dispatch |
| 8.2 | Read extending guide | Read `docs/extending.md` | Understand extension points and conventions |
| 8.3 | Choose extension type | Custom routing method, adapter, or plugin | Decision made |
| 8.4 | Implement extension | Follow step-by-step guide (see sub-journeys) | Code compiles, imports work |
| 8.5 | Write tests | Unit tests with mocked router, integration tests if applicable | Tests pass |
| 8.6 | Update documentation | `docs/adapters.md`, `docs/plugins.md`, or `AGENTS.md` | Docs reflect new capability |
| 8.7 | Run CI checks | `ruff check`, `ruff format`, `mypy`, `pytest` | All checks pass |

### Sub-Journey 8a: Custom Routing Method

Add a new routing algorithm (e.g., embedding-based, LLM-as-judge, ensemble).

| Step | Action | Asset | Success Criteria |
|------|--------|-------|-----------------|
| 8a.1 | Subclass `BaseRouter` | Implement `route()`, `load()`, `unload()` | Returns valid `RoutingResult` |
| 8a.2 | Register in config dispatch | Add to `build_router_from_config()` in `config.py` | `method: my_method` in YAML works |
| 8a.3 | Add training pipeline (optional) | `my_method/train.py` + register in `train.py` dispatcher | `model-router train` dispatches correctly |
| 8a.4 | Add evaluation (optional) | `my_method/evaluate.py` | `model-router evaluate` works with new checkpoints |
| 8a.5 | Add lazy import | `__init__.py` `__getattr__` | `from model_router_toolkit import MyRouter` |

**Assets**: `docs/extending.md` Part 1 (full guide with code templates and checklist)

### Sub-Journey 8b: Custom Adapter

Add a new platform integration (e.g., gRPC service, Kafka consumer, serverless function).

| Step | Action | Asset | Success Criteria |
|------|--------|-------|-----------------|
| 8b.1 | Create adapter directory | `adapters/my_platform/__init__.py` + modules | Package structure in place |
| 8b.2 | Import only core types | `BaseRouter`, `RoutingResult`, `CostEstimate`, `extract_user_text` | No framework deps in adapter imports |
| 8b.3 | Build router from config | `load_config()` + `build_router_from_config()` | Router loads and routes |
| 8b.4 | Translate to platform interface | Map `RoutingResult` to platform's format | Platform receives correct format |
| 8b.5 | Add optional deps in `pyproject.toml` | New extras group | `pip install model-router-toolkit[my_platform]` |
| 8b.6 | Add import guard | Try/except with helpful error message | Clear install instructions on ImportError |
| 8b.7 | Reuse shared helpers | `warmup_router()`, `health_dict()`, `models_list()` from `_shared.py` | No code duplication |

**Assets**: `docs/extending.md` Part 2 (guide + checklist), `docs/adapters.md` Part 2 (writing custom adapters with FastAPI/webhook/gRPC examples)

### Sub-Journey 8c: Custom Gateway Plugin

Write a plugin for a new API gateway platform.

| Step | Action | Asset | Success Criteria |
|------|--------|-------|-----------------|
| 8c.1 | Identify platform hook | Pre-request or model-selection extension point | Hook identified |
| 8c.2 | Call `/v1/route` | HTTP POST to router sidecar | Route response received |
| 8c.3 | Map model names | `selected_model` → platform's model/provider IDs | Correct model dispatched |
| 8c.4 | Handle failures | Return default behavior on error/timeout | Graceful fallback works |
| 8c.5 | Add to `plugins/` directory | Plugin manifest + implementation | Plugin packaged |

**Assets**: `docs/plugins.md` Part 2 (step-by-step + examples for Portkey, LangChain, Kong, Envoy)

### Development Workflow

| Tool | Command | Purpose |
|------|---------|---------|
| **ruff** | `ruff check src/ tests/` | Lint |
| **ruff** | `ruff format src/ tests/` | Format |
| **mypy** | `mypy src/` | Type check |
| **pytest** | `pytest tests/ -v` | Unit tests (97 tests, no API keys) |
| **pytest** | `pytest tests/ -v --run-slow` | Full suite (126 tests, needs API keys + encoder) |

### Jobs-to-Be-Done Breakdown

| JTBD | Current State | Completeness |
|------|--------------|-------------|
| Understand extension points | `docs/architecture.md` + `docs/extending.md` | Complete |
| Add custom routing method | Step-by-step guide with code templates and checklist | Complete |
| Add custom adapter | Guide with FastAPI, webhook, gRPC examples | Complete |
| Write gateway plugins | Guide with Portkey, LangChain examples | Complete |
| Run CI checks locally | ruff + mypy + pytest | Complete |
| API reference for core types | No reference docs for `BaseRouter`, `RoutingResult`, `PoolConfig` | Missing |
| Contribution guide | `docs/extending.md` Part 3 (setup, style, testing, PR checklist) | Complete |

### Gaps & Friction Points

1. **No API reference docs**: `BaseRouter`, `RoutingResult`, `CostEstimate`, `PoolConfig` lack generated reference documentation
2. **No plugin test harness**: No mock sidecar for testing plugins without running the real router
3. **Extending guide assumes Python**: Gateway plugins may be TypeScript, Go, Lua — only TypeScript (OpenClaw) has an example
4. **No CI pipeline**: No GitHub Actions / GitLab CI for automated checks on contributions

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
| 1. Choose deployment | Standalone server, sidecar + OpenClaw plugin, or LiteLLM proxy | **Journey 2** or **Journey 7** | — |
| 2. Configure coding tool | Point `OPENAI_API_BASE` to router, or install OpenClaw plugin | **`docs/integration.md`**, **`docs/plugins.md`** | Coding assistant configuration |
| 3. Use normally | Write code, ask questions, get completions | Router handles model selection transparently | Coding assistant IDE |
| 4. Review routing | Check which models handle which queries | **Playground UI**, telemetry | — |
| 5. Customize | Collect coding-specific data, train custom router | **Journey 3** (train) | Code benchmark datasets |
| 6. Share config | Distribute config across team | Config YAML + checkpoint | Team config management |

**Toolkit covers**: All phases for the routing layer. OpenClaw integration is now a first-class path via the plugin (Journey 7a). For other coding tools, the standalone server or environment variable approach works.

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

**Toolkit covers**: Core routing infrastructure (Phases 2-3, 5). The webhook auth middleware (Journey 7) helps with Phase 3 security, and the sidecar pattern helps with Phase 4 isolation. Major gap: no multi-tenant support out of the box. Per-tenant routing requires running multiple sidecar instances or extending the strategy.

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
| S2 | Configure API keys and providers | Integrator | Config comments, env vars | Complete |
| S3 | Create a valid config file | Integrator | Copy from 7 example configs in `configs/` | Complete |
| S4 | Validate config before serving | Integrator | Pydantic validation in `config.py` | Complete |
| S5 | Understand config schema | All | `configs/schema.md`, `docs/configuration.md` | Complete |
| S6 | Choose between 7 deployment topologies | Platform Eng., Gateway Admin | `docs/architecture.md`, `docs/integration.md` | Complete — decision matrix in both docs |

### Deployment & Integration

| # | Job | Persona | Asset | Status |
|---|-----|---------|-------|--------|
| I1 | Start a router server (full mode) | Integrator | `model-router serve` | Complete |
| I2 | Get an OpenAI-compatible endpoint | Integrator | `/v1/chat/completions` | Complete |
| I3 | Deploy routing without inference | Integrator | `model-router serve-router` → `POST /v1/route` | Complete |
| I4 | Connect downstream apps | Integrator | `docs/integration.md` — 7 paths with decision matrix | Complete |
| I5 | Add routing to existing LiteLLM SDK | LiteLLM User | `ModelRoutingStrategy` | Complete |
| I6 | Add routing to existing LiteLLM Proxy | LiteLLM User | `model-router proxy` | Complete |
| I7 | Deploy in Docker | Platform Eng. | `docker/Dockerfile` (proxy + proxy-gpu), compose | Complete |
| I8 | Deploy to Kubernetes | Platform Eng. | — | Missing |
| I9 | Run as system service/daemon | Platform Eng. | — | Missing |
| I10 | Configure health checks | Platform Eng. | `/health`, Dockerfile HEALTHCHECK | Complete |
| I11 | Pin model for multi-turn chains | Integrator | `resolve()`, `model` field, `metadata.pin_model` | Complete |
| I12 | Secure sidecar with webhook auth | Gateway Admin | `WebhookAuthMiddleware` — HMAC + bearer | Complete |
| I13 | Integrate with OpenClaw gateway | Gateway Admin | `plugins/openclaw/` — full plugin | Complete |
| I14 | Integrate with Portkey/TrueFoundry | Gateway Admin | `docs/plugins.md` — webhook pattern + examples | Complete (examples) |
| I15 | Use router as Python library | Integrator | `load_config()` + `build_router_from_config()` | Complete |

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
| Q3 | Review individual answers in real-time | QA Lead | `/api/review` SSE (judge + comparison) | Complete |
| Q4 | Compare routing quality to baseline | QA Lead | Lift, headroom captured, agreement zones | Complete |
| Q5 | Track quality over time | QA Lead | Telemetry schema exists | Partial — not wired into adapters |
| Q6 | A/B test checkpoints | QA Lead | — | Missing |
| Q7 | Export evaluation reports | Optimizer | — | Missing (stdout only) |
| Q8 | Auto-review in playground UI | QA Lead | Playground toggle, streams verdict | Complete |

### Gateway & Webhook Integration

| # | Job | Persona | Asset | Status |
|---|-----|---------|-------|--------|
| G1 | Deploy router sidecar | Gateway Admin | `model-router serve-router` | Complete |
| G2 | Authenticate sidecar requests | Gateway Admin | `WebhookAuthMiddleware` | Complete |
| G3 | Integrate with OpenClaw | Gateway Admin | `plugins/openclaw/` | Complete |
| G4 | Integrate with Portkey (webhook) | Gateway Admin | `docs/plugins.md` example | Complete (example) |
| G5 | Integrate with LangChain | Gateway Admin | `docs/plugins.md` example | Complete (example) |
| G6 | Write plugin for new gateway | Gateway Admin, Contributor | `docs/plugins.md` Part 2 | Complete (guide) |
| G7 | Map router model names to gateway models | Gateway Admin | Pool config in plugin/webhook | Complete |
| G8 | Graceful fallback on sidecar failure | Gateway Admin | Documented pattern in all examples | Complete |

### Extensibility

| # | Job | Persona | Asset | Status |
|---|-----|---------|-------|--------|
| E1 | Add a custom routing method | Contributor | `docs/extending.md` Part 1 (guide + checklist) | Complete |
| E2 | Add a custom adapter | Contributor | `docs/extending.md` Part 2 + `docs/adapters.md` Part 2 | Complete |
| E3 | Write a gateway plugin | Contributor | `docs/plugins.md` Part 2 | Complete |
| E4 | Understand architecture and design | Contributor | `docs/architecture.md` | Complete |
| E5 | API reference for core types | Contributor | — | Missing |
| E6 | Run CI checks locally | Contributor | ruff + mypy + pytest | Complete |

### Operations & Monitoring

| # | Job | Persona | Asset | Status |
|---|-----|---------|-------|--------|
| O1 | Monitor router health | Platform Eng. | `/health` endpoint (all modes) | Complete |
| O2 | View routing telemetry | Platform Eng. | SQLite telemetry (schema exists) | Partial — not wired into adapters |
| O3 | Structured logging | Platform Eng. | — | Missing (`print()` throughout) |
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
    │        │        ├──► Journey 4 (Production / Docker)
    │        │        │        │
    │        │        │        └──► Broader E (Multi-Tenant Platform)
    │        │        │
    │        │        ├──► Journey 6 (QA / Review)
    │        │        │        │
    │        │        │        └──► Broader A (Cost Optimization)
    │        │        │
    │        │        └──► Broader D (Coding Assistant)
    │        │
    │        ├──► 2c: Router-only mode (no inference)
    │        │        │
    │        │        ├──► Journey 7 (Gateway / Webhook Integration)
    │        │        │        │
    │        │        │        ├──► 7a: OpenClaw Plugin
    │        │        │        ├──► 7b: Portkey / TrueFoundry Webhook
    │        │        │        └──► 7c: Custom Gateway Plugin
    │        │        │
    │        │        └──► Broader B (AI Product — your app dispatches LLM calls)
    │        │
    │        └──► 2d: Model pinning (router-per-subagent)
    │
    ├──► Journey 5 (LiteLLM Integration)
    │        │
    │        └──► Broader B (AI Product)
    │
    ├──► Journey 3 (Train)
    │        │
    │        ├──► Journey 6 (QA)
    │        │
    │        └──► Broader C (Model Evaluation)
    │
    └──► Journey 8 (Extend & Contribute)
             │
             ├──► 8a: Custom routing method
             ├──► 8b: Custom adapter
             └──► 8c: Custom gateway plugin
```

**Natural progression paths:**
1. **Evaluator → Integrator (full) → Platform Engineer**: Try → deploy with inference → productionize in Docker
2. **Evaluator → Integrator (sidecar) → Gateway Admin**: Try → deploy sidecar → plug into OpenClaw/Portkey/etc.
3. **Evaluator → Optimizer → QA Lead**: Try → customize → validate
4. **LiteLLM User → Optimizer → Platform Engineer**: Integrate → tune → operate
5. **Gateway Admin → Contributor**: Integrate gateway → build custom plugin for new platform
6. **Any → Broader Journey**: Toolkit is the routing component in larger initiatives

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

> **Note**: v2.0 adds Journey 7 (Gateway Integration) and Journey 8 (Extend & Contribute). Review specs for J7 and J8 appear below after J6.

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

#### Review J7: Gateway & Webhook Integration (The Gateway Admin)

| Phase | Specific Actions |
|-------|-----------------|
| Doc Audit | Read `docs/plugins.md` end-to-end, `docs/integration.md` (webhook and OpenClaw sections), `adapters/http/auth.py`, `plugins/openclaw/index.ts`, `plugins/openclaw/openclaw.plugin.json` |
| Execution | Verify sidecar startup with `ROUTER_WEBHOOK_SECRET`, test HMAC and bearer auth, validate OpenClaw plugin config schema, trace `before_model_resolve` hook, verify pool mapping logic |
| Gaps | Evaluate sidecar warm-up latency impact on gateway timeouts, test fallback when sidecar is down, check auth error messages |

**Key questions to answer:**
- Does the sidecar start and serve authenticated requests correctly?
- Does the OpenClaw plugin handle all edge cases (sidecar down, timeout, bad response)?
- Are the webhook examples (Portkey, LangChain) complete and runnable?
- Is the pool mapping (router name → gateway model) intuitive?
- Can a gateway admin set this up in under 30 minutes?

#### Review J8: Extend & Contribute (The Contributor)

| Phase | Specific Actions |
|-------|-----------------|
| Doc Audit | Read `docs/extending.md` end-to-end, `docs/adapters.md` Part 2, `docs/architecture.md`, `docs/plugins.md` Part 2 |
| Execution | Verify code templates compile, trace `build_router_from_config()` dispatch, check that adapter/plugin checklists are accurate, validate dev setup instructions |
| Gaps | Evaluate completeness of extending guide, check for missing patterns, assess time-to-first-contribution |

**Key questions to answer:**
- Can a developer add a new routing method following only the guide (no codebase exploration)?
- Are the code templates correct and copy-pasteable?
- Does the adapter checklist cover all requirements?
- Is the contributing guide complete (setup, style, testing, PR)?

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
| J2: Deploy | 15+ files (adapters/litellm/, adapters/http/, configs, docs) | High (multi-module, 7 integration paths) |
| J3: Train | 8+ files | High (pipeline complexity) |
| J4: Production | 6 files (Docker, compose, entrypoint, auth) | Medium (Docker + config + security) |
| J5: LiteLLM | 8+ files (strategy, proxy, config_bridge, docs) | Medium (integration surface) |
| J6: QA | 6+ files (review, evaluate, telemetry, docs) | Medium (metrics + endpoints) |
| J7: Gateway | 8+ files (auth, plugins, docs) | Medium (sidecar + plugin + webhook) |
| J8: Extend | 6+ files (extending, adapters, architecture docs) | Medium (guide completeness) |
| **Cross-journey** | All above | High (consistency checks) |

**Total**: Full review touches 55+ unique files across the repository.
