# Configuration Reference

All behavior in model-router-toolkit is config-driven. A single YAML file — the **pool config** — defines the routing method, model pool, and provider settings.

## Install Extras

The core package has minimal dependencies. Install extras based on your needs:

| Extra | Installs | Enables |
|-------|----------|---------|
| *(none)* | pydantic, numpy, scikit-learn, requests, PyYAML | Core routing, config |
| `[server]` | FastAPI, uvicorn | Router-only HTTP sidecar (`adapters/http/`) |
| `[litellm]` | litellm, FastAPI, uvicorn | LiteLLM strategy, standalone server (`adapters/litellm/`) |
| `[proxy]` | litellm[proxy], packaging | LiteLLM Proxy injection (`model-router proxy`) |
| `[prefill]` | torch, transformers, accelerate, tqdm | Prefill routing method |
| `[training]` | litellm | Data collection (`model-router collect`) |
| `[dev]` | pytest, pytest-asyncio, pytest-cov, httpx, ruff, mypy | Testing and linting |
| `[all]` | Everything above | Full development |

```bash
pip install -e '.[prefill,litellm]'      # Most common
pip install -e '.[all]'                  # Development
```

## Full YAML Schema

```yaml
routing:
  method: prefill                        # str — "prefill"
  checkpoint: checkpoints/router.pt      # str — path to trained checkpoint
  tolerance: 0.20                        # float [0.0–1.0] — accuracy-cost tradeoff

  encoder: Qwen/Qwen3.5-0.8B            # str — HuggingFace encoder model
  encoder_server: ""                     # str — remote encoder URL (empty = local)
  training_mode: auto                    # str — "auto", "cpu", "cuda", "mps"
  encoder_backend: transformers          # str — "transformers" or "vllm"

models:
  - name: nem-think                      # str — unique model identifier
    display_name: Nemotron Think         # str — human-readable name (defaults to name)
    litellm_model: openrouter/nvidia/nemotron-3-nano-30b-a3b  # str — litellm model string
    cost_per_m_input_tokens: 0.20        # float — USD per million input tokens
    cost_per_m_output_tokens: 0.20       # float — USD per million output tokens
    system_prompt: ""                    # str — system message prepended to requests
    chat_template_kwargs: {}             # dict — extra template args for training extraction
    api_base: ""                         # str — override API base URL for this model
```

## Field Reference

### `routing`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `method` | `str` | `"prefill"` | Routing algorithm: `"prefill"` |
| `checkpoint` | `str` | `""` | Path to trained checkpoint (`.pt`) |
| `tolerance` | `float` | `0.20` | Accuracy-cost tradeoff. `0.0` = always pick best model. `1.0` = always pick cheapest. |
| `encoder` | `str` | `""` | HuggingFace model ID for prefill encoder |
| `encoder_server` | `str` | `""` | Remote encoder URL. Empty = load locally. |
| `training_mode` | `str` | `"auto"` | Device for training: `"auto"`, `"cpu"`, `"cuda"`, `"mps"` |
| `encoder_backend` | `str` | `"transformers"` | Encoder backend: `"transformers"` or `"vllm"` |

### `models[]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | *(required)* | Unique identifier used in routing results and litellm model_list |
| `display_name` | `str` | *(same as name)* | Human-readable name for UIs |
| `litellm_model` | `str` | `""` | LiteLLM model string (e.g. `openrouter/nvidia/...`, `nvidia_nim/...`) |
| `cost_per_m_input_tokens` | `float` | `0.0` | Cost in USD per million input tokens |
| `cost_per_m_output_tokens` | `float` | `0.0` | Cost in USD per million output tokens |
| `system_prompt` | `str` | `""` | System message prepended to each request for this model |
| `chat_template_kwargs` | `dict` | `{}` | Extra kwargs passed to the tokenizer chat template during extraction |
| `api_base` | `str` | `""` | Override the API base URL for this specific model |

### Tolerance Explained

The `tolerance` parameter controls the accuracy-cost tradeoff:

```
tolerance = 0.00  →  Always select the model with highest P(correct)
tolerance = 0.10  →  Allow up to 10% lower confidence for a cheaper model
tolerance = 0.20  →  Default. Good balance of cost savings and accuracy
tolerance = 0.50  →  Aggressive cost savings, some accuracy risk
tolerance = 1.00  →  Always select the cheapest model regardless of confidence
```

The selection algorithm:
1. Compute P(correct) for each model
2. Find `p_max` = highest confidence
3. Set `threshold = p_max - tolerance`
4. Among models with confidence ≥ threshold, select the cheapest

## Environment Variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | serve, proxy, collect | API key for OpenRouter provider |
| `NVIDIA_API_KEY` | serve, proxy, collect, notebooks | API key for NVIDIA NIM / build.nvidia.com |
| `OPENAI_API_KEY` | serve, proxy | Fallback API key for OpenAI-compatible providers |
| `ROUTER_WEBHOOK_SECRET` | serve-router | Shared secret for webhook HMAC/bearer auth |
| `CORS_ORIGINS` | serve, serve-router | Comma-separated allowed CORS origins (default: `*`) |
| `LITELLM_CONFIG_FILE_PATH` | proxy | Set automatically by `model-router proxy` |

## Starter Configs

| Config | Method | Provider | When to use |
|--------|--------|----------|-------------|
| `configs/prefill-qwen08b.yaml` | Prefill | OpenRouter | Default — best accuracy |
| `configs/v1-9models-qwen08b.yaml` | Prefill | OpenRouter | Full 9-model v1 pool |
| `configs/local-prefill.yaml` | Prefill | Local | Air-gapped / local-only |

## Annotated Example: Prefill Config

```yaml
routing:
  method: prefill
  checkpoint: checkpoints/prefill_router.pt

  # How aggressively to trade accuracy for cost savings.
  # 0.20 = default, good balance. Lower = more accurate, higher = cheaper.
  tolerance: 0.20

  # HuggingFace encoder (downloaded on first run, ~1.6GB).
  # Used for prefill feature extraction at inference time.
  encoder: Qwen/Qwen3.5-0.8B

models:
  # Strong model — higher cost, better at hard questions
  - name: nem-think
    display_name: Nemotron Think (30B)
    litellm_model: openrouter/nvidia/nemotron-3-nano-30b-a3b
    cost_per_m_input_tokens: 0.20
    cost_per_m_output_tokens: 0.20
    system_prompt: Think step-by-step before answering.

  # Fast model — lower cost, good at easy questions
  - name: nem-nothink
    display_name: Nemotron Fast (30B)
    litellm_model: openrouter/nvidia/nemotron-3-nano-30b-a3b
    cost_per_m_input_tokens: 0.04
    cost_per_m_output_tokens: 0.16
```

## Annotated Example: Multi-Provider Pool

```yaml
routing:
  method: prefill
  checkpoint: checkpoints/multi_provider.pt
  tolerance: 0.20
  encoder: Qwen/Qwen3.5-0.8B

models:
  # NVIDIA NIM
  - name: nim-llama-70b
    litellm_model: nvidia_nim/meta/llama-3.1-70b-instruct
    cost_per_m_input_tokens: 0.35
    cost_per_m_output_tokens: 0.40
    api_base: https://integrate.api.nvidia.com/v1

  # OpenRouter
  - name: or-nemotron
    litellm_model: openrouter/nvidia/nemotron-3-nano-30b-a3b
    cost_per_m_input_tokens: 0.20
    cost_per_m_output_tokens: 0.20

  # Self-hosted via Ollama
  - name: local-llama
    litellm_model: ollama/llama3.1
    cost_per_m_input_tokens: 0.00
    cost_per_m_output_tokens: 0.00
    api_base: http://localhost:11434
```

## LiteLLM Proxy Config

The `proxy-config` command generates a LiteLLM-compatible `config.yaml` from your pool config:

```bash
model-router proxy-config --config configs/prefill-qwen08b.yaml --output configs/litellm-proxy.yaml
```

Generated output:

```yaml
model_list:
  - model_name: nem-think
    litellm_params:
      model: openrouter/nvidia/nemotron-3-nano-30b-a3b
      api_key: os.environ/OPENROUTER_API_KEY
  - model_name: nem-nothink
    litellm_params:
      model: openrouter/nvidia/nemotron-3-nano-30b-a3b
      api_key: os.environ/OPENROUTER_API_KEY
router_settings:
  routing_strategy: simple-shuffle
```

You can customize the generated file (add auth, caching, rate limiting) before passing it to `model-router proxy`. See [LiteLLM Proxy docs](https://docs.litellm.ai/docs/proxy/configs) for all available proxy settings.
