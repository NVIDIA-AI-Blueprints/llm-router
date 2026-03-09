# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-03-08

### Added
- Prefill complexity-based routing via Qwen3.5-0.8B encoder
- KMeans embedding-based routing (inference only; training coming soon)
- Unified CLI: `model-router serve|train|evaluate|collect|proxy|proxy-config`
- OpenAI-compatible API server with interactive playground UI
- LiteLLM custom routing strategy integration
- Full prefill training pipeline: extract, sweep, train ensemble, save checkpoint
- Rich evaluation metrics: per-model AUC, oracle accuracy, agreement zones, near-miss analysis
- Data collection with majority vote and reference-based judging
- Docker support (CPU proxy and GPU proxy stages)
- Comprehensive documentation: architecture, training guide, evaluation guide, integration guide
- Apache 2.0 license

### Security
- Removed hardcoded API key from quickstart notebook
- CORS origins now configurable via `CORS_ORIGINS` env var (was hardcoded wildcard)
- Race condition on shared tolerance fixed with per-request contextvars scoping
- Async routing now uses `asyncio.to_thread()` to avoid blocking the event loop
- Security warnings added to all `pickle.load()` and `torch.load(weights_only=False)` sites
- `trust_remote_code` parameter made configurable in `LocalEmbedClient`
- API key fallback in proxy config bridge now logs a warning for unknown providers
- Telemetry disabled by default (opt-in via `ROUTER_TELEMETRY_DB` env var)
