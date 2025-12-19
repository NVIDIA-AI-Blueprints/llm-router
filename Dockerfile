FROM nvcr.io/nvidia/base/ubuntu:jammy-20250619


# Install required ubuntu packages for setting up python and minimal OpenCV dependencies
RUN apt update && \
     apt install -y curl vim python3-pip \
     libgl1-mesa-glx libglib2.0-0 && \
     apt-get clean

# Install uv 
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
     echo 'export PATH="/root/.cargo/bin:$PATH"' >> /root/.bashrc && \
     export PATH="/root/.cargo/bin:$PATH"

RUN rm -rf /var/lib/apt/lists/*

WORKDIR /opt/dgxcbot/router

ENV PATH="/root/.local/bin:/root/.cargo/bin:$PATH"

## Verify uv installation
RUN uv --version

# Copy only dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

RUN uv venv --seed .venv --python 3.12.12

RUN uv sync --prerelease=allow

# Copy the entire application code including pre-trained model artifacts
COPY . .

RUN uv pip install --no-cache-dir .


# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

CMD ["uv", "run", "nat", "serve", "--config_file", "config.yml", "--host", "0.0.0.0", "--port", "8001"]
