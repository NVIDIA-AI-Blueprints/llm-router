# Model Pool Reference

Reference for all supported models in the Model Router Toolkit.

## Model Table

| Name | Display Name | Provider(s) | Cost (input/output per M tokens) | Context Length | Notes |
|------|--------------|-------------|----------------------------------|----------------|-------|
| nem-think | Nemotron 3 Nano Think | NVIDIA NIM | $0.20 / $0.20 | 128K | Step-by-step thinking enabled |
| nem-super | Nemotron 3 Super | NVIDIA NIM | $0.30 / $0.30 | 128K | Large, capable |
| gptoss-20b | GPT-OSS 20B | NVIDIA NIM, OpenRouter | $0.30 / $0.30 | 128K | Open-source 20B |
| gptoss-120b | GPT-OSS 120B | NVIDIA NIM, OpenRouter | $1.00 / $1.00 | 128K | Open-source 120B |
| qwen-122b | Qwen 3.5 122B | NVIDIA NIM | $0.50 / $0.50 | 128K | Qwen 3.5 122B |
| gpt-5.2 | GPT-5.2 | NVIDIA NIM, OpenRouter | $1.75 / $14.00 | 128K | Premium model |
| claude-opus | Claude Opus 4.6 | NVIDIA NIM (Bedrock) | $5.00 / $25.00 | 200K | Claude Opus 4.6 |

## Provider Availability

| Provider | Models Available |
|----------|-------------------|
| **build.nvidia.com** (NVIDIA NIM) | nem-think, nem-super, gptoss-20b, gptoss-120b, qwen-122b, gpt-5.2, claude-opus |
| **OpenRouter** | gptoss-20b, gptoss-120b, gpt-5.2 |
| **Local** | Depends on deployment; typically smaller models (e.g., Nemotron variants) when self-hosted |

## Embedding Model

Used for KMeans routing (embedding-based, no GPU required).

| Model | Availability | Notes |
|-------|---------------|-------|
| nvidia/llama-nemotron-embed-1b-v2 | build.nvidia.com, OpenRouter | 1B parameter embedding model; used for question embedding before KMeans scoring |

## Encoder Model (Prefill)

Used for prefill complexity-based routing. Requires GPU and a local encoder server.

| Model | Source | VRAM (approx) | Notes |
|-------|--------|---------------|-------|
| Qwen3.5-35B-A3B | HuggingFace | ~22GB at Q4 | Single forward pass extracts hidden states; MLP heads predict P(correct) per target model |
