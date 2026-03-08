# Architecture

The Model Router Toolkit is built around a **BaseRouter** abstraction and the **ModelRoutingStrategy** pattern. All routing implementations (KMeans embedding-based and prefill complexity-based) inherit from BaseRouter and produce a model selection plus score. The ModelRoutingStrategy wraps a BaseRouter instance and implements LiteLLM's `CustomRoutingStrategyBase`, enabling drop-in integration with any LiteLLM proxy or application.

## Class Hierarchy

```
BaseRouter (abstract)
    |
    +-- KMeansRouter      # Embedding-based clustering; no GPU required
    |
    +-- PrefillRouter     # Prefill complexity scoring; requires GPU + encoder server

ModelRoutingStrategy
    |
    +-- wraps BaseRouter
    +-- implements CustomRoutingStrategyBase (LiteLLM)
```

## Data Flow

1. **Question** - The user query or prompt arrives at the router.
2. **Embed/Prefill** - Depending on strategy:
   - KMeans: question is embedded via API (e.g., `nvidia/llama-nemotron-embed-1b-v2`).
   - Prefill: question is sent to a local encoder server for a single forward pass; hidden states are extracted.
3. **Score** - The router computes a score per model in the pool:
   - KMeans: distance to cluster centroids or nearest-neighbor scoring.
   - Prefill: MLP head predicts P(correct) per target model from encoder hidden states.
4. **Select** - The router selects the best model (e.g., argmax over score, or cost-aware selection above a tolerance threshold).
5. **LiteLLM dispatch** - The selected model identifier is passed to LiteLLM, which dispatches the request to the appropriate provider (NVIDIA NIM, OpenRouter, etc.).

## Architecture Diagrams

### KMeans Path (no GPU)

```
+----------+     +------------------+     +----------------+     +------------------+
| Question | --> | Embed API        | --> | KMeansRouter   | --> | LiteLLM          |
|          |     | (build.nvidia.com|     | (score/select) |     | (dispatch)        |
|          |     |  or OpenRouter)  |     |                |     |                  |
+----------+     +------------------+     +----------------+     +------------------+
                        |                          |
                        v                          v
                 nvidia/llama-              checkpoint.pkl
                 nemotron-embed-1b-v2        (centroids, model map)
```

### Prefill Path (with GPU + vLLM encoder server)

```
+----------+     +----------------------+     +----------------+     +------------------+
| Question | --> | Encoder Server       | --> | PrefillRouter  | --> | LiteLLM          |
|          |     | (vLLM, GPU, local)   |     | (score/select) |     | (dispatch)        |
|          |     | Qwen3.5-35B-A3B     |     |                |     |                  |
+----------+     +----------------------+     +----------------+     +------------------+
                        |                              |
                        v                              v
                 prefill hidden states          checkpoint (MLP heads,
                 (single forward pass)           layer, pooling config)
```

## Config-Driven Dispatch

Routing behavior is fully config-driven. A YAML config specifies:

- **routing.method** - `kmeans` or `prefill`; determines which BaseRouter implementation is used.
- **routing.checkpoint** - Path to the trained checkpoint (centroids for KMeans, MLP weights for Prefill).
- **models** - The model pool: name, display name, LiteLLM model ID, cost per M tokens, and optional chat template kwargs.

At startup, the toolkit loads the config, instantiates the appropriate router with the checkpoint, and registers it with LiteLLM. All model selection logic is derived from the config and checkpoint; no code changes are required to add or remove models from the pool.
