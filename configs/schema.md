# Pool Config Schema

This document describes the YAML configuration schema for the Model Router Toolkit. Config files define the routing method, checkpoint, tolerance, and the model pool.

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
| `method` | string | Yes | Routing strategy: `prefill` (prefill complexity-based routing via encoder hidden states). |
| `checkpoint` | string | Yes | Path to the trained router checkpoint (`.pt` file, e.g. `checkpoints/prefill_router.pt`). |
| `tolerance` | float | No | Accuracy tolerance threshold (0.0–1.0). Router selects the cheapest model that meets this accuracy target. Default: 0.20. |
| `encoder` | string | No | HuggingFace model ID for the prefill encoder (e.g. `Qwen/Qwen3.5-0.8B`). |
| `encoder_server` | string | No | URL of the encoder inference server for prefill extraction. |
| `training_mode` | string | No | Prefill training mode: `auto`, `per_model`, or `single`. |
| `encoder_backend` | string | No | Backend for encoder inference: `transformers` (local) or `server` (remote). |

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
| `prefill-qwen08b.yaml` | Default config. Qwen3.5-0.8B encoder, OpenRouter model pool. |
| `local-prefill.yaml` | Local GPU + prefill router. Encoder served locally; model pool uses OpenRouter plus optional local vLLM model. |
| `smoke-test.yaml` | Quick 2-model test configuration. |
