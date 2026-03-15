# Model Pool Reference

Reference for models and encoder used with the Model Router Toolkit.

## Default Model Pool (prefill-qwen08b config)

| Name | Display Name | LiteLLM Model | Cost (in/out per M tokens) | Notes |
|------|-------------|---------------|---------------------------|-------|
| nem-think | Nemotron 3 Nano Think | `openrouter/nvidia/nemotron-3-nano-30b-a3b` | $0.20 / $0.20 | Thinking mode enabled |
| nem-nothink | Nemotron 3 Nano | `openrouter/nvidia/nemotron-3-nano-30b-a3b` | $0.04 / $0.16 | Direct answers, cheapest |
| gptoss-high | GPT-OSS 20B | `openrouter/openai/gpt-oss-20b` | $0.30 / $0.30 | High reasoning effort |
| gpt-5.2 | GPT-5.2 | `openrouter/openai/gpt-5.2` | $1.75 / $14.00 | Premium, most expensive |

Cost range: 50x between cheapest (nem-nothink) and most expensive (gpt-5.2).

## Encoder Model (Prefill Routing)

| Model | HuggingFace Path | Parameters | VRAM | CPU Time |
|-------|------------------|------------|------|----------|
| Qwen3.5-0.8B | `Qwen/Qwen3.5-0.8B` | 0.8B | ~2GB (fp32) | ~5s/question |

The encoder runs a single forward pass per question with `output_hidden_states=True`. Hidden states at the optimal layer (found during training sweep) are extracted, PCA-reduced, and fed to the MLP trunk.

On CPU (M-series Mac): ~14s model load (first time), ~5s per question.
On GPU: <100ms per question after model load.

The encoder is specified in the config:
```yaml
routing:
  encoder: Qwen/Qwen3.5-0.8B
```

## Adding Models to the Pool

Add entries to the `models` list in your config YAML:

```yaml
models:
  - name: my-model              # unique name (used in CSV, routing, display)
    litellm_model: provider/model-id  # LiteLLM model identifier
    cost_per_m_input_tokens: 0.50     # cost per million input tokens
    cost_per_m_output_tokens: 1.00    # cost per million output tokens
    system_prompt: ""                 # optional system prompt
    chat_template_kwargs: {}          # optional template kwargs
```

After adding a model:
1. Re-collect data (`model-router collect`) to get labels for the new model
2. Re-train (`model-router train`) to include the new model in the routing decisions
3. Re-evaluate (`model-router evaluate`) to verify the router handles the expanded pool

Model names in the config must match the `model` column in your training/test CSVs.
