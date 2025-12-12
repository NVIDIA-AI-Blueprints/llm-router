FROM nvcr.io/nvidia/pytorch:25.10-py3

# Install required ubuntu packages for setting up python and minimal OpenCV dependencies
# RUN apt update && \
#     apt install -y curl vim python3-pip \
#     libgl1-mesa-glx libglib2.0-0 && \
#     apt-get clean

# # Install uv 
# RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
#     echo 'export PATH="/root/.cargo/bin:$PATH"' >> /root/.bashrc && \
#     export PATH="/root/.cargo/bin:$PATH"

# RUN rm -rf /var/lib/apt/lists/*

WORKDIR /opt/dgxcbot/router

#ENV PATH="/root/.local/bin:/root/.cargo/bin:$PATH"

## Verify uv installation
#RUN uv --version

# Copy only dependency files first for better layer caching
# #COPY pyproject.toml uv.lock ./

# RUN uv venv --seed .venv --python 3.12

# RUN uv sync --prerelease=allow

RUN pip install flash-attn --no-build-isolation
RUN pip install nvidia-nat safetensors transformers==4.51.3 scikit-learn matplotlib

# Copy the entire application code including pre-trained model artifacts
COPY . .

# Install the package with all data files (embeddings, models, configs)
# Artifacts are included via MANIFEST.in and pyproject.toml [tool.setuptools.package-data]
# This ensures trained router_artifacts/nn_router.pth is packaged and available at runtime
RUN pip install clip-client
RUN pip install --no-cache-dir .


# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Create volume mount points for dynamic configuration
# Allows changing thresholds and costs without rebuilding
VOLUME ["/opt/dgxcbot/router/configs"]

CMD ["nat", "serve", "--config_file", "configs/config.yml", "--host", "0.0.0.0", "--port", "8001"]
