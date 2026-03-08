FROM python:3.12-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir '.[proxy]'

# -------------------------------------------------------------------
# Default stage: proxy mode (litellm proxy + intelligent routing)
# -------------------------------------------------------------------
FROM base AS proxy

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 4000

ENTRYPOINT ["/entrypoint.sh"]

# -------------------------------------------------------------------
# GPU stage: adds prefill routing support (torch + transformers)
# -------------------------------------------------------------------
FROM base AS proxy-gpu

RUN pip install --no-cache-dir '.[proxy,prefill]'

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 4000

ENTRYPOINT ["/entrypoint.sh"]
