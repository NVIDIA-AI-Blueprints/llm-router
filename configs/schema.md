# Pool Config Schema

This document describes the YAML configuration schema for the Model Router Toolkit. Config files define the routing method, checkpoint, tolerance, embedding settings, and the model pool.

## Top-Level Structure

```yaml
routing:
  # Routing section (see below)

models:
  # List of model entries (see below)
```

---

## Routing Section

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `method` | string | Yes | Routing strategy: `kmeans` (embedding-based clustering) or `prefill` (prefill complexity-based). |
| `checkpoint` | string | Yes | Path to the trained router checkpoint. For KMeans: `.pkl` file (e.g. `checkpoints/kmeans_c100_db.pkl`). For prefill: `.pt` file (e.g. `checkpoints/prefill.pt`). |
| `tolerance` | float | No | Accuracy tolerance threshold (0.0–1.0). Router selects the cheapest model that meets this accuracy target. Default varies by implementation. |
| `embed_model` | string | No | HuggingFace model ID or API model name for embeddings. Used when `method` is `kmeans`. |
| `embed_mode` | string | No | How embeddings are computed: `api` (remote API) or `local` (in-process). |
| `embed_api_base` | string | No | Base URL for embedding API when `embed_mode` is `api`. Examples: `https://integrate.api.nvidia.com/v1`, `https://openrouter.ai/api/v1`. |
| `encoder` | string | No | HuggingFace model ID for the prefill encoder. Used when `method` is `prefill`. |
| `encoder_server` | string | No | URL of the encoder inference server for prefill extraction. Used when `method` is `prefill`. |
| `training_mode` | string | No | Prefill training mode: `auto`, `per_model`, or `single`. Used when `method` is `prefill`. |

---

## Models Section

The `models` key is a list of model entries. Each entry defines one model in the routing pool.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique identifier for the model. Used in routing decisions and API responses. |
| `display_name` | string | No | Human-readable label for the model. |
| `litellm_model` | string | Yes | LiteLLM model identifier. Format depends on provider: `nvidia_nim/...` for build.nvidia.com, `openrouter/...` for OpenRouter, `openai/vllm/...` for local vLLM. |
| `cost_per_m_input_tokens` | float | Yes | Cost in USD per 1 million input tokens. |
| `cost_per_m_output_tokens` | float | Yes | Cost in USD per 1 million output tokens. |
| `system_prompt` | string | No | System prompt to prepend when calling this model. |
| `chat_template_kwargs` | object | No | Key-value pairs passed to the chat template (e.g. `enable_thinking`, `reasoning_effort`). |
| `api_base` | string | No | Override API base URL for this model. Used for local or custom endpoints (e.g. local vLLM). |

---

## Example Configs

| File | Use Case |
|------|----------|
| `cloud-only.yaml` | build.nvidia.com only. No OpenRouter, no GPU. KMeans with NVIDIA embedding API. |
| `openrouter-kmeans.yaml` | OpenRouter + KMeans. No GPU. Same model pool with `openrouter/` provider. |
| `local-prefill.yaml` | Local GPU + prefill router. Encoder served locally; model pool uses OpenRouter plus optional local vLLM model. |
