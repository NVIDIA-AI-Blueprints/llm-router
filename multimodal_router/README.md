# Multimodal Router v2

> **Note**: This is a standalone routing system independent from the broader [llm-router](../) repository. It represents a next-generation approach to model routing with full multimodal support.

A production-ready multimodal routing system that intelligently directs conversations to optimal Vision-Language Models (VLMs) and Large Language Models (LLMs) based on content analysis. Built on the NVIDIA NeMo Agent Toolkit, this system provides a FastAPI-compatible chat completions endpoint that returns model recommendations in real-time.

## Key Features

- **Multimodal Support**: Route based on both text and images, optimized for VLMs
- **Two Routing Strategies**: Intent-based OR auto-routing (neural network)
- **Interactive Demo**: Demo web app

## Architecture


### Router Backend (`src/`)

The router is packaged as a NVIDIA NeMo Agent Toolkit component that:
- Exposes an OpenAI chat completions-compatible FastAPI endpoint
- Accepts multimodal messages (text + images)
- Returns the optimal model name for the given context

**Endpoint**: `POST /sfc_router/chat/completions`

**Input**: Standard OpenAI chat completion request with multimodal content
**Output**: Model identifier in the response content field

## 🎯 Routing Methods

### 1. Intent-Based Routing (Arch-Router)

Uses the [Arch-Router-1.5B](https://huggingface.co/katanemo/Arch-Router-1.5B) model to match user intents to specific models.

**Advantages**:
- ✅ No training required
- ✅ Understands semantic intent
- ✅ Works out-of-the-box
- ✅ Easily configurable via intent mappings

**Use Case**: When you have clear intent categories (e.g., "visual analysis" → VLM, "code generation" → specialized LLM)

**Configuration**: See `src/nat_sfc_router/configs/config.yml` and `src/nat_sfc_router/functions/hf_intent_objective_fn.py` 

```python
route_config = [
    {
        "name": "hard_question",
        "description": "A question that requires deep reasoning, or complex problem solving, or if the user asks for careful thinking or careful consideration",
    },
    {
        "name": "chit_chat",
        "description": "Any social chit chat, small talk, or casual conversation.",
    },
    {
        "name": "try_again",
        "description": "Only if the user explicitly says the previous answer was incorrect or incomplete.",
    },
    {
        "name": "image_understanding",
        "description": "A question that requires understanding an image.",
    },
    {
        "name": "image_question",
        "description": "A question that requires the assistant to see the user eg a question about their appearance, environment, scene or surroundings.",
    },
]

MAP_INTENT_TO_PIPELINE = {
    "other": "nvidia/nvidia-nemotron-nano-9b-v2",
    "chit_chat": "nvidia/nvidia-nemotron-nano-9b-v2",
    "hard_question": "gpt-5-chat",
    "image_understanding": "nvidia/nemotron-nano-12b-v2-vl",
    "image_question": "nvidia/nemotron-nano-12b-v2-vl",
    "try_again": "gpt-5-chat",
}
```

### 2. Auto-Routing (CLIP + Neural Network)

Uses CLIP embeddings to encode text/image pairs, then a trained neural network to predict the optimal model.

**Advantages**:
- ✅ Learns from actual usage patterns
- ✅ Optimizes for quality, latency, and cost
- ✅ Adapts to your specific workload
- ✅ Handles multimodal input natively

**Use Case**: When you have historical data and want data-driven routing decisions

**Training Required**: See notebooks for training pipeline:
- `2_Embedding_NN_Training.ipynb` - Training the neural network
- `3_Embedding_NN_Usage.ipynb` - Using the trained router

## Notebooks

Three Jupyter notebooks are included for exploration and training:

1. **`1_ArchRouter_Example.ipynb`**: 
   - Introduction to intent-based routing
   - Configuration examples
   - Usage patterns

2. **`2_Embedding_NN_Training.ipynb`**: 
   - Generate CLIP embeddings for training data
   - Train neural network router
   - Evaluate routing performance

3. **`3_Embedding_NN_Usage.ipynb`**: 
   - Load and use trained router
   - Integration examples
   - Performance analysis

## Quick Start with Docker Compose

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with Docker runtime (for router backend and Arch-Router)
- OpenAI API key
- NVIDIA Build API key

### 1. Configure API Keys

Create a `.env` file in the project root:

```bash
# API Keys for demo app
OPENAI_API_KEY=sk-your-openai-key-here
NVIDIA_API_KEY=nvapi-your-nvidia-key-here
```

### 2. Launch All Services

```bash
docker-compose up -d --build
```

This starts three services:
- **router-backend** (port 8001): Main routing service
- **arch-router** (port 8011): Arch-Router-1.5B model server
- **demo-app** (port 7860): Interactive web interface

### 3. Access the Demo

Open your browser to: **http://localhost:7860**

### 4. Stop Services

```bash
docker-compose down
```

## Local Development

### Router Backend

```bash
# Setup environment
uv venv --python 3.12 --seed .venv
uv sync

# Start router service
./scripts/run_local.sh
```

The router will be available at `http://localhost:8001`

### Demo App

```bash
cd demo

# Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp env_template.txt .env
# Edit .env with your API keys

# Run the app
python app.py
```

The demo will be available at `http://localhost:7860`

## Router API Usage

### Making a Request

```bash
curl -X POST http://localhost:8001/sfc_router/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Explain quantum computing"
      }
    ],
    "stream": false
  }'
```

### Multimodal Request (with image)

```bash
curl -X POST http://localhost:8001/sfc_router/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What is in this image?"},
          {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}
          }
        ]
      }
    ]
  }'
```

### Response Format

```json
{
  "id": "chatcmpl-1765473022",
  "choices": [{
    "message": {
      "content": "nvidia/nemotron-nano-12b-v2-vl",
      "role": "assistant"
    }
  }],
  "model": "hf_intent_objective_fn"
}
```

The selected model is returned in `choices[0].message.content`.

### Router-Only Response

For use cases where you only need the routing decision without the ChatCompletion wrapper, use the `/router` endpoint. This returns a purpose-built response format with model probabilities and selection metadata.

**Endpoint**: `POST /router`

**Request** (same format as chat completions):

```bash
curl -X POST http://localhost:8001/router \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Explain quantum computing"
      }
    ]
  }'
```

**Response Format**:

```json
{
  "id": "routing-1734567890123",
  "object": "routing.decision",
  "created": 1734567890,
  "selected_model": "gpt-5-chat",
  "classifications": [
    {"label": "gpt-5-chat", "score": 0.75},
    {"label": "Qwen/Qwen3-VL-8B-Instruct", "score": 0.20},
    {"label": "nvidia/nvidia-nemotron-nano-9b-v2", "score": 0.05}
  ],
  "selection_reason": "cost_optimized"
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier for this routing decision |
| `object` | string | Always `"routing.decision"` |
| `created` | integer | Unix timestamp of when the decision was made |
| `selected_model` | string | The model selected for routing |
| `classifications` | array | All models with confidence scores (sorted descending) |
| `selection_reason` | string | Why this model was selected (`cost_optimized`, `highest_probability`, `threshold_fallback`) |

## Configuration

### Router Configuration

Edit `src/nat_sfc_router/configs/config.yml`:

```yaml
functions:
  healthcheck_fn:
    _type: healthcheck
  hf_intent_objective_fn:
    _type: hf_intent_objective_fn

  # Configuration for the CLIP + Nueral Network router
  nn_objective_fn:
    _type: nn_objective_fn
    
    model_thresholds:
      'gpt-5-chat': 0.70
      'Qwen/Qwen3-VL-8B-Instruct': 0.75
      'nvidia/nvidia-nemotron-nano-9b-v2': 0.4
    
    model_costs:
      'gpt-5-chat': 1.0
      'Qwen/Qwen3-VL-8B-Instruct': 0.5
      'nvidia/nvidia-nemotron-nano-9b-v2': 0.3

  # Select which objective function is used for routing decisions    
  sfc_router_fn:
    _type: sfc_router
    objective_fn: hf_intent_objective_fn # <--- select routing function


workflow:
  _type: sfc_router
  objective_fn: hf_intent_objective_fn # <--- select routing function

```

### Demo App Configuration

The demo app automatically maps router responses to actual API endpoints:

```python
# demo/app.py - Model configurations
MODELS = {
    "gpt-4o": {
        "name": "gpt-4o",
        "provider": "openai"
    },
    "nvidia/nemotron-nano-12b-v2-vl": {
        "name": "nvidia/nemotron-nano-12b-v2-vl",
        "provider": "nvidia"
    }
}
```

## Advanced Usage

### Training Custom Router

1. Collect conversation data with preferred model labels
2. Follow `2_Embedding_NN_Training.ipynb` to train
3. Replace router artifacts in `src/nat_sfc_router/training/router_artifacts/`
4. Update config to use `nn_objective_fn`
5. Restart router service


## Integration

### As a Library

```python
from nat_sfc_router import Router

router = Router(config_path="configs/config.yml")
model = router.route(messages=[
    {"role": "user", "content": "Hello!"}
])
print(f"Selected: {model}")
```

### As a Service

Deploy the FastAPI service and make HTTP requests:

```python
import requests

response = requests.post(
    "http://router-backend:8001/sfc_router/chat/completions",
    json={"messages": [{"role": "user", "content": "Hello!"}]}
)
model = response.json()["choices"][0]["message"]["content"]
```

## 📁 Project Structure

```
multimodal_router/
├── src/                           # Router source code
│   └── nat_sfc_router/
│       ├── configs/              # Configuration files
│       ├── functions/            # Objective functions
│       ├── schema/               # Request/response schemas
│       └── training/             # Training artifacts
├── demo/                         # Web demo application
│   ├── app.py                   # Gradio application
│   ├── requirements.txt         # Python dependencies
│   ├── Dockerfile               # Demo container
│   └── env_template.txt         # Environment template
├── scripts/                      # Utility scripts
│   └── run_local.sh            # Local development script
├── notebooks/                    # Jupyter notebooks (1-3)
├── docker-compose.yml           # Multi-service orchestration
├── Dockerfile                   # Router backend container
└── README.md                    # This file
```

## Troubleshooting

### Router Not Starting

```bash
# Check GPU availability
nvidia-smi

# Check logs
docker-compose logs router-backend

# Verify config
cat src/nat_sfc_router/configs/config.yml
```

### Demo Can't Connect to Router

```bash
# Verify router is running
curl http://localhost:8001/health

# Check network
docker network inspect multimodal_router_router-network

# Check environment variables
docker-compose exec demo-app env | grep ROUTER_ENDPOINT
```

### API Key Issues

```bash
# Verify keys are set
echo $OPENAI_API_KEY
echo $NVIDIA_API_KEY

# Test keys
curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"
```


## Acknowledgments

- Built on [NVIDIA NeMo Agent Toolkit](https://github.com/NVIDIA/NeMo)
- Uses [Arch-Router](https://github.com/katanemo/archgw) for intent-based routing
- [CLIP embeddings](https://build.nvidia.com/nvidia/nvclip)
- Demo UI powered by [Gradio](https://gradio.app)
