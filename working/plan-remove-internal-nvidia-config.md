# Plan: Remove Internal NVIDIA Inference API References

**Status:** Planned (not yet executed)
**Date:** 2026-03-13

## Context

`configs/nvidia-nim-smoke.yaml` targets `REDACTED_INTERNAL_ENDPOINT` -- an internal NVIDIA endpoint not available to external users. It is the **only** config using that domain, and `NVIDIA_INTERNAL_API_KEY_REDACTED` exists solely to support it. All other configs use `integrate.api.nvidia.com` (public) or OpenRouter.

This must be removed before publishing the repo.

## Scope -- 6 files to change, 1 file to delete

### 1. Delete config file

- **Delete** `configs/nvidia-nim-smoke.yaml` -- the only config referencing `REDACTED_INTERNAL_ENDPOINT`

### 2. Source code cleanup (3 files)

**`src/model_router_toolkit/adapters/litellm/app.py`:**
- Lines 31-32: Remove the `if "inference-api.nvidia" in api_base` branch from `_resolve_api_key()`
- Line 128: Remove `NVIDIA_INTERNAL_API_KEY_REDACTED` from the `review_available` check

**`src/model_router_toolkit/adapters/litellm/config_bridge.py`:**
- Lines 117-118: Remove the `if "inference-api.nvidia" in api_base` branch from `_api_key_env_var()`

**`src/model_router_toolkit/adapters/litellm/review.py`:**
- Line 215: Remove `NVIDIA_INTERNAL_API_KEY_REDACTED` from the `has_key` check
- Line 219: Remove `NVIDIA_INTERNAL_API_KEY_REDACTED` from the error message string

### 3. Working doc update (1 file)

**`working/user-journeys.md`:**
- Line 166: Remove `nvidia-nim-smoke` from the config list

### 4. Log the change

**`working/IMPLEMENTATION-LOG.md`:**
- Append entry documenting what was removed and why

## What stays unchanged

- `NVIDIA_API_KEY` -- still used for `integrate.api.nvidia.com` / build.nvidia.com (the public endpoint)
- `.env.example` -- does not mention `NVIDIA_INTERNAL_API_KEY_REDACTED` (already clean)
- `AGENTS.md` -- does not reference the internal endpoint
- `docs/` -- no references to `inference-api.nvidia` or `NVIDIA_INTERNAL_API_KEY_REDACTED`
- `tests/` -- no references
- `skills/` -- no references
