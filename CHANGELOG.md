# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- **Modular adapter architecture.** Moved all LiteLLM-coupled code into `adapters/litellm/` and all FastAPI-coupled code into `adapters/http/`. `litellm` and `fastapi` are now optional dependencies — bare `pip install model-router-toolkit` gives routing logic only with no framework deps.
- `pyproject.toml` new extras: `[server]` (HTTP sidecar), `[litellm]` (LiteLLM strategy + serve), `[training]` (data collection via LiteLLM), `[proxy]` (LiteLLM proxy mode), `[all]` (everything).
- `__init__.py` lazy-loads `ModelRoutingStrategy` from `adapters/litellm/strategy.py` with a clear `ImportError` if litellm is not installed.
- CLI commands unchanged; internal imports updated to `adapters.litellm` and `adapters.http`.

### Added
- `adapters/http/auth.py` — webhook authentication middleware (HMAC-SHA256 + bearer token) for enterprise gateway integrations (Portkey, TrueFoundry, Cloudflare).
- `plugins/openclaw/` — TypeScript plugin template for OpenClaw's `before_model_resolve` hook, calling the toolkit's HTTP sidecar for per-prompt routing.
- `docs/quickstart.md` — 5-minute getting started guide.
- `docs/configuration.md` — full pool config YAML reference + install extras matrix.
- `docs/adapters.md` — using bundled adapters, writing custom adapters, API reference.
- `docs/plugins.md` — OpenClaw plugin guide, writing plugins for other platforms.
- `docs/extending.md` — custom routing methods, adding adapters, contributing guide.
- `tests/adapters/test_http.py` — unit tests for HTTP adapter + webhook auth (8 new tests).
- `tests/adapters/test_litellm.py` — migrated strategy tests.
- `tests/integration/test_openclaw_sidecar.py` — OpenClaw sidecar integration test.

### Removed
- `server/` directory — split into `adapters/litellm/` (full serve) and `adapters/http/` (router-only).
- `proxy/` directory — moved to `adapters/litellm/proxy.py` and `adapters/litellm/config_bridge.py`.
- Top-level `strategy.py` — moved to `adapters/litellm/strategy.py`.

### Fixed
- Completions endpoint now ensures `content` key is always present in response choices (prevents `KeyError` when providers return responses without explicit content field).
- CLI `collect` test timeout increased from 120s to 600s for slow API providers.
- CLI `evaluate` test gracefully skips on encoder architecture version mismatches instead of failing with opaque error codes.

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
