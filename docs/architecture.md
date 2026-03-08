# Architecture

The Model Router Toolkit is built around a **BaseRouter** abstraction. All routing methods inherit from BaseRouter and produce a model selection with confidence scores. **ModelRoutingStrategy** wraps any BaseRouter and implements LiteLLM's `CustomRoutingStrategyBase` for drop-in integration.

## Class Hierarchy

```
BaseRouter (abstract)
    |
    +-- KMeansRouter      # Embedding-based clustering; no GPU required
    |
    +-- PrefillRouter     # Prefill complexity scoring; CPU or GPU

ModelRoutingStrategy
    +-- wraps BaseRouter
    +-- implements CustomRoutingStrategyBase (LiteLLM)
```

## Inference Flow

### KMeans Path

```
Question --> Embed API (build.nvidia.com) --> KMeansRouter --> LiteLLM dispatch
                  |                              |
                  v                              v
          nvidia/llama-nemotron-         checkpoint.pkl
          embed-1b-v2                  (centroids, Platt calibrators)
```

1. Question is embedded via API
2. KMeansRouter assigns to nearest cluster, applies Platt calibration for P(correct) per model
3. Selects cheapest model above tolerance threshold
4. LiteLLM dispatches to selected provider

### Prefill Path

```
Question --> Encoder (Qwen3.5-0.8B) --> PrefillRouter --> LiteLLM dispatch
                  |                          |
                  v                          v
          hidden states               checkpoint.pt
          (single forward pass)       (PCA + MLP ensemble)
```

1. Question is run through the encoder model (single forward pass, `output_hidden_states=True`)
2. Hidden states at the best layer are extracted (last-token or mean-pooled)
3. Per-model StandardScaler + PCA reduces dimensions
4. Concatenated features go through SharedTrunkNet MLP ensemble
5. Sigmoid outputs give P(correct) per target model
6. Cheapest model with P(correct) within `tolerance` of the best is selected

The encoder (Qwen3.5-0.8B, 0.8B parameters) runs on CPU in ~5s per question. GPU reduces this to <100ms.

## Training Pipeline

Training bypasses the BaseRouter interface and works directly with prefill components for batch efficiency.

```
train.csv --> Load Labels --> Batch Extract Prefill
                                    |
                              Sweep (layer/mode/PCA per target)
                                    |
                              Fit Transforms (StandardScaler + PCA)
                                    |
                              Train SharedTrunkNet Ensemble
                                    |
                              Save .pt Checkpoint + serve.yaml
```

**Sweep**: For each target model, grid-searches over hidden state mode (last-token vs mean-pooled) and PCA dimension, with ternary search over encoder layers. Uses 5-fold CV AUC with logistic regression as the quality metric.

**Trunk training**: BCEWithLogitsLoss with Adam optimizer, early stopping on validation split. Trains N seeds (default 10), keeps the top K by validation loss (default 5). Final ensemble averages sigmoid outputs.

**Checkpoint**: Self-contained `.pt` file with pool config, per-model transforms (scaler, PCA, layer, mode), trunk state dicts, trunk architecture config, and cost table. The same checkpoint is used for both training evaluation and serving inference.

## Evaluation Pipeline

Evaluation also bypasses BaseRouter for batch extraction:

```
test.csv + checkpoint.pt --> Batch Extract --> Apply Transforms --> Run Trunk
                                                                       |
                                                                  Rich Metrics Report
```

Reports per-model AUC, oracle vs router accuracy, lift, headroom captured, routing distribution, agreement zone analysis, near-miss diagnostics, and pairwise confidence win rates.

## Config-Driven Dispatch

All behavior is config-driven:

```yaml
routing:
  method: prefill          # or kmeans
  checkpoint: path/to.pt   # trained checkpoint
  tolerance: 0.20          # accuracy-cost tradeoff
  encoder: Qwen/Qwen3.5-0.8B  # HF encoder for prefill

models:
  - name: nem-think
    litellm_model: openrouter/nvidia/nemotron-3-nano-30b-a3b
    cost_per_m_input_tokens: 0.20
    cost_per_m_output_tokens: 0.20
```

`routing.method` determines which BaseRouter is instantiated. The model pool, costs, and endpoints are all in the config. No code changes needed to add or remove models.

## Server

The FastAPI server (`model-router serve`) creates a `litellm.Router` internally and registers the `ModelRoutingStrategy` via `set_custom_routing_strategy()`. Endpoints:

- `POST /v1/chat/completions` -- OpenAI-compatible (streaming + non-streaming)
- `POST /api/chat` -- SSE chat endpoint for the playground UI
- `GET /api/models` -- Pool info with costs
- `GET /health` -- Health check
- `GET /` -- Interactive playground UI
