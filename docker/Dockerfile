FROM python:3.12-slim AS base

LABEL org.opencontainers.image.title="model-router-toolkit" \
      org.opencontainers.image.description="LLM routing toolkit with intelligent model selection" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser && useradd -r -g appuser appuser

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY configs/ configs/

RUN pip install --no-cache-dir '.[proxy]'

# -------------------------------------------------------------------
# Default stage: proxy mode (litellm proxy + intelligent routing)
# -------------------------------------------------------------------
FROM base AS proxy

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER appuser
EXPOSE 4000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:4000/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]

# -------------------------------------------------------------------
# GPU stage: adds prefill routing support (torch + transformers)
# -------------------------------------------------------------------
FROM base AS proxy-gpu

RUN pip install --no-cache-dir '.[proxy,prefill]'

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER appuser
EXPOSE 4000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:4000/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
