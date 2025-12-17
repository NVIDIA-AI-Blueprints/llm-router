# Multimodal LLM Router Demo

A simple web application that showcases the functionality of a multimodal LLM router which directs conversation chats to different LLM endpoints based on intelligent routing decisions.

## Features

- **Chat Interface**: Clean, intuitive chat UI built with Gradio
- **Image Upload**: Support for multimodal conversations with image uploads
- **Intelligent Routing**: Automatic model selection based on request analysis
- **Model Transparency**: See which model was selected for each request
- **Session History**: Message history maintained during your session

## Routing Methods

Two routing strategies are supported (only one can be active at a time due to GPU constraints):

| Method | Routing Backend | Best For |
|--------|----------------|----------|
| **Intent-Based** (default) | Qwen3-1.7B LLM | Fast, lightweight routing via text classification |
| **Neural Network** | CLIP embeddings + trained NN | Complex multimodal routing with learned patterns |

**Configuration**: Edit `objective_fn` in `src/nat_sfc_router/configs/config.yml`:
- Intent-based: `objective_fn: hf_intent_objective_fn`
- Neural network: `objective_fn: nn_objective_fn`

## Supported Models

- **GPT-5-chat** (Azure OpenAI)
- **Nemotron Nano v2** (NVIDIA Build API)
- **Nemotron Nano VLM 12B** (NVIDIA Build API) - Multimodal

## Quick Start with Docker Compose

1. **Configure environment**:
   ```bash
   cp demo/env_template.txt .env
   # Edit .env with your API keys (OPENAI_API_KEY, NVIDIA_API_KEY, AZURE_OPENAI_ENDPOINT)
   ```

2. **Choose and start your routing method**:
   
   **Intent-based router** (default):
   ```bash
   docker compose --profile intent up -d --build
   ```
   
   **Neural network router**:
   ```bash
   # First, update config.yml objective_fn to nn_objective_fn
   docker compose --profile nn up -d --build
   ```

3. **Wait for services to be ready** (first time ~2-3 minutes):
   
   Services start in order with health checks:
   - `qwen-router` or `clip-server` loads model (~2 min)
   - `router-backend` waits for routing service to be healthy
   - `demo-app` waits for router-backend to be healthy
   
   Check status: `docker compose ps`

4. **Access the UI**: http://localhost:7860

## Switching Routing Methods

To switch between routing methods:

1. **Stop current services**:
   ```bash
   docker compose down  # or: docker compose --profile <current-profile> down
   ```

2. **Update configuration** in `src/nat_sfc_router/configs/config.yml`:
   
   Change lines 63 and 68:
   ```yaml
   # For intent-based:
   objective_fn: hf_intent_objective_fn
   
   # For neural network:
   objective_fn: nn_objective_fn
   ```

3. **Start with new profile**:
   ```bash
   docker compose --profile intent up -d --build   # or --profile nn
   ```

## Local Development Setup

### Prerequisites
```bash
cd demo
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp env_template.txt .env  # Edit with your API keys
```

### Option A: Intent-Based Router

```bash
# Terminal 1: Start Qwen router
docker run -d --rm --name qwen-router --gpus all -p 8011:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/qwen3_nonthinking.jinja:/app/qwen3_nonthinking.jinja \
  vllm/vllm-openai:latest --model Qwen/Qwen3-1.7B \
  --chat-template /app/qwen3_nonthinking.jinja

# Terminal 2: Start router service
cd llm-router && ./scripts/run_local.sh

# Terminal 3: Start demo
cd demo && python app.py
```

Ensure `config.yml` has `objective_fn: hf_intent_objective_fn`

### Option B: Neural Network Router

```bash
# Terminal 1: Start CLIP server
docker run -d --rm --name clip-server --gpus all -p 51000:51000 \
  jinaai/clip-as-service:latest

# Terminal 2: Start router service (with CLIP_SERVER env var)
export CLIP_SERVER=localhost:51000
cd llm-router && ./scripts/run_local.sh

# Terminal 3: Start demo
cd demo && python app.py
```

Ensure `config.yml` has `objective_fn: nn_objective_fn`

**Access**: http://localhost:7860

## Usage

- **Text queries**: Type and click "Send"
- **Multimodal**: Upload image + optional text, then "Send"
- **Clear**: Use "Clear Chat" to reset

Example queries:
- "Explain quantum computing in simple terms"
- "Write a Python function to calculate fibonacci numbers"
- "Describe what you see in this image" (with image)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Services taking long to start | **Normal on first start** (~2-3 min). Models must load before backend starts.<br>• Check status: `docker compose ps`<br>• Check health: `docker inspect <container> --format='{{.State.Health.Status}}'`<br>• View logs: `docker logs qwen-router` or `docker logs clip-server` |
| Router connection refused | Verify router service is running on port 8001: `docker ps` |
| API key errors | Check `.env` has correct keys. Azure OpenAI needs both `AZURE_OPENAI_ENDPOINT` and `OPENAI_API_KEY` |
| Router fails or wrong routing | Ensure `objective_fn` in `config.yml` matches running service:<br>• `hf_intent_objective_fn` needs `qwen-router`<br>• `nn_objective_fn` needs `clip-server`<br>Check: `docker ps` |
| CLIP server connection failed | 1. Verify running: `docker ps \| grep clip`<br>2. Check logs: `docker logs clip-server`<br>3. Check health: `docker inspect clip-server --format='{{.State.Health.Status}}'` |
| Qwen router connection failed | 1. Verify running: `docker ps \| grep qwen`<br>2. Check logs: `docker logs qwen-router`<br>3. Check health: `curl http://localhost:8011/health` |
| Image upload errors | Ensure image format is supported (JPEG, PNG, etc.) |

## Quick Command Reference

### Docker Compose
```bash
# Start with profile
docker compose --profile intent up -d --build  # or: --profile nn

# Check service status and health
docker compose ps
docker inspect <container-name> --format='{{.State.Health.Status}}'

# Stop
docker compose down

# View logs
docker logs router-backend
docker logs qwen-router      # for intent profile
docker logs clip-server      # for nn profile
docker logs router-demo

# Follow logs in real-time
docker logs -f qwen-router   # or clip-server

# Restart specific service
docker compose restart router-backend
```


## Configuration Files

- **Router config**: `src/nat_sfc_router/configs/config.yml`
- **Environment**: `demo/.env` or `.env` (project root)
- **Docker Compose**: `docker-compose.yml`

## Project Structure

```
demo/
├── app.py              # Main Gradio application
├── requirements.txt    # Python dependencies
├── env_template.txt    # Environment template
├── Dockerfile          # Demo app container
└── README.md           # This file
```

## Customization

- **Port**: Modify `server_port` in `app.py`
- **Models**: Update `MODELS` dictionary in `app.py`
- **UI**: Adjust Gradio components in `create_demo()` function
