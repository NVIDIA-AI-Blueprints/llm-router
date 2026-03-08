#!/usr/bin/env bash
set -euo pipefail

LITELLM_CONFIG="${LITELLM_CONFIG:-/app/configs/litellm-proxy.yaml}"
ROUTER_CONFIG="${ROUTER_CONFIG:-/app/configs/prefill-qwen08b.yaml}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-4000}"

exec model-router proxy \
    --litellm-config "$LITELLM_CONFIG" \
    --router-config "$ROUTER_CONFIG" \
    --host "$HOST" \
    --port "$PORT"
