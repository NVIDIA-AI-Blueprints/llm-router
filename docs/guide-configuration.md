# Configuration Reference

This guide documents every field in the pool config YAML, how configurations are used across the system, and annotated examples for common scenarios.

---

## Table of Contents

- [Config File Structure](#config-file-structure)
- [Routing Section](#routing-section)
- [Models Section](#models-section)
- [Complete Field Reference](#complete-field-reference)
- [Annotated Examples](#annotated-examples)
- [Environment Variables](#environment-variables)
- [Config Validation](#config-validation)
- [Included Configs](#included-configs)

---

## Config File Structure

Pool configs are YAML files with two top-level sections:

```yaml
routing:
  # How routing works
  method: prefill
  checkpoint: checkpoints/my_router.pt
  tolerance: 0.20
  encoder: Qwen/Qwen3.5-0.8B

models:
  # What models are in the pool
  - name: my-cheap-model
    litellm_model: openrouter/provider/model-id
    cost_per_m_input_tokens: 0.05
    cost_per_m_output_tokens: 0.20
    extra_headers:
      x-foo: bar
  - name: my-expensive-model
    litellm_model: openrouter/provider/model-id
    cost_per_m_input_tokens: 2.50
    cost_per_m_output_tokens: 15.00
```

The config is loaded and validated by Pydantic:

```python
from model_router_toolkit.config import load_config

config = load_config("configs/my-pool.yaml")
# config.routing  → RoutingConfig
# config.models   → list[ModelSpec]
```

---

## Routing Section

Controls how the router operates.

### `method` (string, default: `"prefill"`)

The routing algorithm. Currently only `"prefill"` is supported.

```yaml
routing:
  method: prefill
```

### `checkpoint` (string, default: `""`)

Path to the trained `.pt` checkpoint file. Relative paths are resolved from the current working directory, not from the config file location.

```yaml
routing:
  checkpoint: checkpoints/prefill_router_qwen08b.pt
```

If empty, the router is created but cannot route until `router.load(path)` is called explicitly.

### `tolerance` (float, default: `0.20`)

The accuracy–cost tradeoff parameter. Range: `[0.0, 1.0]`.

```yaml
routing:
  tolerance: 0.20
```

How it works: the router finds the model with the highest P(correct) (`p_max`), then selects the cheapest model with P(correct) ≥ `p_max - tolerance`.

| Value | Behavior |
|-------|----------|
| `0.00` | Always pick the highest-confidence model (pure accuracy) |
| `0.05` | Mild efficiency gain, very conservative |
| `0.10` | Moderate efficiency gain |
| `0.20` | Balanced efficiency (default) |
| `0.50` | Very aggressive, large accuracy drops acceptable |
| `1.00` | Always pick the lowest-cost model |

This can be overridden per-request in all adapters.

### `encoder` (string, default: `""`)

HuggingFace model identifier for the feature extraction encoder. Downloaded and cached automatically on first use.

```yaml
routing:
  encoder: Qwen/Qwen3.5-0.8B
```

Supported encoders:

| Encoder | Size | Hidden Dim | Layers | Notes |
|---------|------|------------|--------|-------|
| `Qwen/Qwen3.5-0.8B` | 0.8B | 1536 | 24 | Recommended. Runs on CPU. |
| `Qwen/Qwen3.5-35B-A3B` | 35B MoE (~3B active) | 2560 | 64 | Higher quality. GPU required. |

The encoder must match what the checkpoint was trained with. Using a different encoder than what's in the checkpoint will produce incorrect results.

### `encoder_server` (string, default: `""`)

Reserved for remote encoder server support. Not currently used.

### `training_mode` (string, default: `"auto"`)

Training mode hint. Used by the training pipeline:
- `"auto"` — automatically determines the training approach
- Other values are reserved for future routing methods

### `encoder_backend` (string, default: `"transformers"`)

Which backend loads the encoder model. Currently only `"transformers"` (HuggingFace Transformers) is supported.

---

## Models Section

A list of models in the routing pool. Each model is a `ModelSpec` object.

### `name` (string, required)

Unique identifier for the model. This must match:
- The `model` column in training/test CSVs
- The name used in API requests and model pinning

```yaml
models:
  - name: nemotron-3-nano-reasoning
```

### `display_name` (string, default: `""`)

Human-readable name. Used in the playground UI and API responses. Defaults to `name` if empty.

```yaml
models:
  - name: nemotron-3-nano-reasoning
    display_name: Nemotron 3 Nano (Reasoning)
```

### `litellm_model` (string, default: `""`)

LiteLLM model identifier, used for inference in the standalone server and LiteLLM Proxy modes. Format: `provider/model-name`.

```yaml
models:
  - name: nemotron-3-nano-reasoning
    litellm_model: openrouter/nvidia/nemotron-3-nano-30b-a3b
```

Common prefixes:

| Prefix | Provider |
|--------|----------|
| `openrouter/` | OpenRouter (requires `OPENROUTER_API_KEY`) |
| `nvidia_nim/` | NVIDIA NIM (requires `NVIDIA_API_KEY`) |
| `vercel_ai_gateway/` | Vercel AI Gateway (requires `VERCEL_AI_GATEWAY_API_KEY`) |
| `openai/` | OpenAI (requires `OPENAI_API_KEY`) |
| `anthropic/` | Anthropic (requires `ANTHROPIC_API_KEY`) |

Not needed for Direct Python or Router Sidecar modes (no inference is performed).

### `cost_per_m_input_tokens` (float, default: `0.0`)

Cost per million input tokens, in dollars. Used for model ranking during selection — cheaper models are preferred when confidence is above threshold.

```yaml
models:
  - name: nemotron-3-nano-reasoning
    cost_per_m_input_tokens: 0.050
```

### `cost_per_m_output_tokens` (float, default: `0.0`)

Cost per million output tokens, in dollars. Used in `CostEstimate` calculations.

```yaml
models:
  - name: nemotron-3-nano-reasoning
    cost_per_m_output_tokens: 0.200
```

### `system_prompt` (string, default: `""`)

System message prepended to conversations when this model is called. Applied by the standalone server and collect command.

```yaml
models:
  - name: nemotron-3-nano-reasoning
    system_prompt: Think step-by-step before answering.
```

### `chat_template_kwargs` (dict, default: `{}`)

Extra keyword arguments passed to the encoder's tokenizer when formatting questions for this model. Different models may benefit from different encoding strategies.

```yaml
models:
  - name: nemotron-3-nano-reasoning
    chat_template_kwargs:
      enable_thinking: true

  - name: gpt-oss-20b-high
    chat_template_kwargs:
      reasoning_effort: high
```

During training, the sweep searches over models with different `chat_template_kwargs` to find which encoding produces the best features. The selected kwargs are stored in the checkpoint.

### `extra_headers` (dict, default: `{}`)

Extra LiteLLM request headers to apply for this model. Use this for provider-specific header options that should always travel with the model entry.

```yaml
models:
  - name: structured-output-model
    litellm_model: openai/gpt-4o-mini
    extra_headers:
      x-foo: bar
```

### `api_base` (string, default: `""`)

Custom API base URL. Used by LiteLLM for models hosted on custom endpoints.

```yaml
models:
  - name: my-custom-model
    litellm_model: openai/my-model
    api_base: https://my-custom-endpoint.example.com/v1
```

---

## Complete Field Reference

### RoutingConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `method` | string | `"prefill"` | Routing algorithm |
| `checkpoint` | string | `""` | Path to trained checkpoint |
| `tolerance` | float | `0.20` | Accuracy–cost tradeoff |
| `encoder` | string | `""` | HuggingFace encoder model |
| `encoder_server` | string | `""` | Remote encoder server (reserved) |
| `training_mode` | string | `"auto"` | Training mode hint |
| `encoder_backend` | string | `"transformers"` | Encoder loading backend |

### ModelSpec

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | (required) | Unique model identifier |
| `display_name` | string | `""` (falls back to `name`) | Human-readable name |
| `litellm_model` | string | `""` | LiteLLM model identifier for inference |
| `cost_per_m_input_tokens` | float | `0.0` | Input cost per million tokens ($) |
| `cost_per_m_output_tokens` | float | `0.0` | Output cost per million tokens ($) |
| `system_prompt` | string | `""` | System message for inference |
| `chat_template_kwargs` | dict | `{}` | Encoder template kwargs |
| `extra_headers` | dict | `{}` | Extra LiteLLM request headers |
| `api_base` | string | `""` | Custom API base URL |

---

## Annotated Examples

### Minimal config (2 models, local testing)

```yaml
routing:
  method: prefill
  checkpoint: checkpoints/my_router.pt
  encoder: Qwen/Qwen3.5-0.8B

models:
  - name: lightweight
    cost_per_m_input_tokens: 0.05
    cost_per_m_output_tokens: 0.20

  - name: capable
    cost_per_m_input_tokens: 2.50
    cost_per_m_output_tokens: 15.00
```

No `litellm_model` — this config works for Direct Python routing only.

### OpenRouter pool (serving via standalone server)

```yaml
routing:
  method: prefill
  checkpoint: checkpoints/prefill_router.pt
  tolerance: 0.15
  encoder: Qwen/Qwen3.5-0.8B

models:
  - name: nano
    litellm_model: openrouter/nvidia/nemotron-3-nano-30b-a3b
    cost_per_m_input_tokens: 0.050
    cost_per_m_output_tokens: 0.200
    system_prompt: Be concise and direct.

  - name: mid
    litellm_model: openrouter/openai/gpt-oss-120b
    cost_per_m_input_tokens: 0.113
    cost_per_m_output_tokens: 0.431

  - name: top
    litellm_model: openrouter/anthropic/claude-opus-4-6
    cost_per_m_input_tokens: 2.770
    cost_per_m_output_tokens: 25.780
```

### NVIDIA NIM pool

```yaml
routing:
  method: prefill
  checkpoint: checkpoints/nim_router.pt
  tolerance: 0.20
  encoder: Qwen/Qwen3.5-0.8B

models:
  - name: nemotron-nano
    litellm_model: nvidia_nim/nvidia/nemotron-3-nano-30b-a3b
    cost_per_m_input_tokens: 0.050
    cost_per_m_output_tokens: 0.200
    api_base: https://integrate.api.nvidia.com/v1

  - name: nemotron-super
    litellm_model: nvidia_nim/nvidia/nemotron-3-super-49b-v1
    cost_per_m_input_tokens: 0.100
    cost_per_m_output_tokens: 0.400
    api_base: https://integrate.api.nvidia.com/v1
```

### High-tolerance efficiency mode

```yaml
routing:
  method: prefill
  checkpoint: checkpoints/prefill_router.pt
  tolerance: 0.35  # aggressive — prioritize efficiency
  encoder: Qwen/Qwen3.5-0.8B

models:
  # ... model list
```

### Low-tolerance accuracy focus

```yaml
routing:
  method: prefill
  checkpoint: checkpoints/prefill_router.pt
  tolerance: 0.05  # conservative — prioritize accuracy
  encoder: Qwen/Qwen3.5-0.8B

models:
  # ... model list
```

---

## Environment Variables

Environment variables are not part of the YAML config but affect the runtime behavior of adapters and tools.

| Variable | Used By | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Standalone server, collect, LiteLLM Proxy | API key for OpenRouter |
| `NVIDIA_API_KEY` | Standalone server, collect, LiteLLM Proxy | API key for NVIDIA NIM |
| `VERCEL_AI_GATEWAY_API_KEY` | Standalone server, collect, LiteLLM Proxy | API key for Vercel AI Gateway |
| `OPENAI_API_KEY` | Standalone server, collect, LiteLLM Proxy | Fallback API key for OpenAI-compatible |
| `ROUTER_WEBHOOK_SECRET` | Router sidecar | Shared secret for webhook auth |
| `CORS_ORIGINS` | Standalone server, router sidecar | Comma-separated allowed CORS origins (default: `*`) |
| `ROUTER_TELEMETRY_DB` | Telemetry module | SQLite file path for session logging |
| `HF_HOME` / `TRANSFORMERS_CACHE` | Encoder loading | Custom HuggingFace model cache directory |

---

## Config Validation

Configs are validated by Pydantic on load. Common validation errors:

| Error | Cause | Fix |
|-------|-------|-----|
| `models: field required` | Missing `models` section | Add at least one model |
| `name: field required` | A model missing the `name` field | Add `name` to every model entry |
| `Extra inputs are not permitted` | Typo in a field name | Check spelling against the field reference |
| `value is not a valid float` | Non-numeric cost value | Use numbers, not strings, for cost fields |

Test your config:

```python
from model_router_toolkit.config import load_config

try:
    config = load_config("configs/my-pool.yaml")
    print(f"Valid: {len(config.models)} models, method={config.routing.method}")
except Exception as e:
    print(f"Invalid: {e}")
```

---

## Included Configs

| Config File | Encoder | Models | Description |
|-------------|---------|--------|-------------|
| `configs/v1-9models-qwen08b.yaml` | Qwen3.5-0.8B | 9 | Default. Fast encoder, wide cost range. |
| `configs/v1-9models-qwen35b.yaml` | Qwen3.5-35B-A3B | 9 | Same pool, larger encoder. Higher AUC. |

Both configs use OpenRouter as the inference provider and include models from $0.05/M to $25.78/M tokens.
