 <h2>NVIDIA AI Blueprint: LLM Router v2 (Experimental)</h2>

> **⚠️ EXPERIMENTAL BRANCH**: This branch contains LLM Router v2, a next-generation routing system with multimodal support. For the production-ready LLM Router v1, please visit the [main branch](https://github.com/NVIDIA-AI-Blueprints/llm-router/tree/main).

## Important Notes

**LLM Router v2 is currently experimental** and not yet backwards compatible with v1. Key differences:

| Feature | v1 (Main Branch) | v2 (Experimental) |
|---------|------------------|-------------------|
| **Server Implementation** | Rust proxy | NVIDIA NeMo Agent Toolkit (FastAPI) |
| **Inference Backend** | BERT model + NVIDIA Triton Inference Server | Qwen 1.7B LLM or CLIP + Neural Network |
| **Functionality** | Classification + Proxying to LLM | Classification only (returns model name) |
| **Input Support** | Text only | Text + Images (multimodal) |
| **Routing Methods** | Task or complexity classification | Intent-based or Auto-routing (neural network) |

**Future Plans**: The intent is to make v2 fully backwards compatible with v1's proxying capabilities, then merge to main and retire the experimental label.

## Overview

Ever struggled to decide which LLM or Vision-Language Model (VLM) to use for a specific task? In an ideal world the most accurate model would also be the cheapest and fastest, but in practice modern agentic AI systems have to make trade-offs between accuracy, speed, and cost.

This blueprint provides an experimental next-generation router that automates these tradeoffs by analyzing user prompts and identifying optimal models. Given a user prompt (text or multimodal), the router:

- applies one of two routing strategies: intent-based classification or auto-routing (based on a trained neural network)
- analyzes the prompt content, including images if present
- returns the name of the most appropriate LLM or VLM for the task

For example, using intent-based routing:

| User Prompt | Intent Classification | Recommended Model |
|---|---|---|
| "What's in this image?" (with image) | image_understanding | nvidia/nemotron-nano-12b-v2-vl |
| "Solve this complex math problem: ..." | hard_question | gpt-5-chat |
| "Hello, how are you?" | chit_chat | nvidia/nvidia-nemotron-nano-9b-v2 |

The key features of the experimental LLM Router v2 are:

- **Multimodal Support**: Route based on both text and images, optimized for VLMs
- **Two Routing Strategies**: Intent-based (using Qwen 1.7B) OR auto-routing (using CLIP embeddings + trained neural network)
- **OpenAI API compliant**: Returns model recommendations via chat completions endpoint
- **Flexible**: Use pre-configured intent mappings or train custom neural network routers on your own data

### Models

This blueprint is pre-configured to route between three complementary models:

| Model | Type | Provider | Use Case |
|-------|------|----------|----------|
| **gpt-5-chat** | Frontier LLM | Azure OpenAI or OpenAI | Complex reasoning, hard questions |
| **nvidia/nemotron-nano-12b-v2-vl** | Open VLM | NVIDIA Build API | Multimodal queries, image understanding |
| **nvidia/nvidia-nemotron-nano-9b-v2** | Small Open LLM | NVIDIA Build API | Simple text queries, chit chat |

- **gpt-5-chat**: Can be sourced from Azure OpenAI (default) or standard OpenAI API
- **Nemotron models**: Configured to use NVIDIA Build API endpoints (hosted), but can also use locally deployed NVIDIA NIMs for on-premise deployment

#### Using Different Models

The three default models are **examples only** - you can route to any models by (1) updating the intent router's configuration or (b) re-training the auto-router.

*The main goal of the LLM router is to intelligently route across frontier and open models* to optimize the cost-quality-latency tradeoff.

## Target Audience

This experimental blueprint is for:

- **AI Engineers and Developers**: Developers interested in exploring next-generation routing approaches with multimodal support.
- **MLOps Teams**: Teams interested in learning-based routing optimization and custom model selection strategies.
- **Research Teams**: Teams evaluating different routing strategies for production deployment.

## Prerequisites

### Software

- Linux operating systems (Ubuntu 22.04 or later recommended) or macOS
- [Docker](https://docs.docker.com/engine/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- For local development: Python 3.12+ and [uv](https://github.com/astral-sh/uv) package manager

### Clone repository

```bash
git clone https://github.com/NVIDIA-AI-Blueprints/llm-router
cd llm-router
git checkout experimental  # or the appropriate v2 branch name
```

### Get API Keys

1. **NVIDIA Build API key**

   - Navigate to [NVIDIA API Catalog](https://build.nvidia.com/explore/discover)
   - Click one of the models, such as nemotron-nano-12b-v2-vl
   - Select the "Python" input option
   - Click "Get API Key"
   - Click "Generate Key" and copy the resulting key (starts with `nvapi-`)

2. **Azure OpenAI API access**

   This project uses Azure OpenAI for the GPT-5-chat model. You'll need:
   
   - An Azure subscription with Azure OpenAI service enabled
   - An Azure OpenAI resource deployed with `gpt-5-chat` model
   - Your Azure OpenAI endpoint URL (format: `https://your-resource-name.openai.azure.com/`)
   - Your Azure OpenAI API key (found in the Azure portal under your resource's "Keys and Endpoint")

   Set these environment variables:
   ```bash
   export AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/"
   export OPENAI_API_KEY="your-azure-openai-api-key"
   ```

   > **Using regular OpenAI instead:** If you prefer to use OpenAI's API instead of Azure OpenAI, you'll need to update:
   > - `demo/app.py`: Change `call_model_azure_openai()` to use `OpenAI()` client with `base_url="https://api.openai.com/v1"` and update model provider from `"azure_openai"` to `"openai"`
   > - `demo/env_template.txt` and `demo/.env`: Replace `AZURE_OPENAI_ENDPOINT` with `OPENAI_API_KEY=sk-...`
   > - `src/nat_sfc_router/training/prepare_hf_data.py`: Replace `AzureOpenAI` client initialization with `OpenAI` client
   > - `2_Embedding_NN_Training.ipynb`: Update cells that reference `AZURE_OPENAI_ENDPOINT` and `AzureOpenAI` client
   > - Get your OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)

## Hardware Requirements

For the Qwen 1.7B model:

| GPU | Family | Memory | # of GPUs (min.) |
| ------ | ------ | ------ | ------ |
| T4 or newer | Any | 16GB | 1 |

For training and using the auto-router (CLIP + Neural Network):

| Component | GPU Required | Memory | Notes |
| ------ | ------ | ------ | ------ |
| CLIP Embedding Server | Yes | 8GB+ | NVIDIA NVClip NIM (required for generating embeddings) |
| Neural Network Training | Optional | 4GB+ (if GPU) | Can run on CPU, but GPU accelerates training |
| Neural Network Inference | No | N/A | Router inference runs on CPU |

**Note**: Training the auto-router requires:
1. A running CLIP server (GPU required) to generate embeddings from text and images
2. PyTorch for neural network training (GPU optional but recommended for faster training)
3. Once trained, the router artifacts can be used for inference on CPU-only systems 


## Quickstart Guide

After meeting the prerequisites, follow these steps to start a demo chat application that uses the intent based router and supports multimodal inputs:

#### 1. Configure API Keys

Create a `.env` file in the project root:

```bash
# API Keys
OPENAI_API_KEY=sk-your-openai-key-here
NVIDIA_API_KEY=nvapi-your-nvidia-key-here
```

#### 2. Launch Services with Docker Compose

**Option A: Intent-Based Router (Default, Recommended for Getting Started)**

```bash
docker compose --profile intent up -d --build
```

This starts three services:
- **router-backend** (port 8001): Main routing service using NVIDIA NeMo Agent Toolkit
- **qwen-router** (port 8011): Qwen 1.7B model server for intent-based routing
- **demo-app** (port 7860): Interactive web interface

**Option B: Neural Network Router**

```bash
docker compose --profile nn up -d --build
```

This starts three services:
- **router-backend** (port 8001): Main routing service using NVIDIA NeMo Agent Toolkit
- **clip-server** (port 51000): CLIP embedding server for neural network routing
- **demo-app** (port 7860): Interactive web interface

> **Note**: You must also update the `objective_fn` in `src/nat_sfc_router/configs/config.yml` to match your chosen profile:
> - For intent-based router: `objective_fn: hf_intent_objective_fn`
> - For neural network router: `objective_fn: nn_objective_fn`
>
> See the [demo README](demo/README.md) for detailed instructions on switching between routing methods.

#### 3. Access the Demo

Open your browser to: **http://localhost:7860**

Try sending messages with or without images to see routing decisions in real-time.

### Quickstart Alternative -  Explore the Notebooks 

Bring up Jupyter to explore the routing methods and training pipeline:

```bash
jupyter lab --no-browser --ip 0.0.0.0 --NotebookApp.token=''
```

Open the notebooks:
- `1_IntentRouter_Example.ipynb` - Intent-based routing examples
- `2_Embedding_NN_Training.ipynb` - Train custom neural network router
- `3_Embedding_NN_Usage.ipynb` - Use trained neural network router

## Software Components

The experimental LLM Router v2 has three main components:

- **Router Backend** - A service built on NVIDIA NeMo Agent Toolkit that exposes a FastAPI endpoint compatible with OpenAI's chat completions API. The router backend analyzes prompts (text and images) and returns the optimal model name. Code is available in `src/nat_sfc_router/`.

- **Routing Models** - Two routing strategies are available:
  - **Intent-Based Router**: Uses the Qwen 1.75B model to match user intents to specific models. Requires the Qwen LLM service running on port 8011.
  - **Auto-Router**: Uses CLIP embeddings and a trained neural network to predict optimal models based on quality, latency, and cost metrics. Requires the CLIP service running and a trained neural network model.

- **Demo Application** - An interactive Gradio web interface that demonstrates the router in action. After receiving a routing decision, the demo app calls the recommended model's API and displays results. Code is available in `demo/`.

**Note**: Unlike v1, v2 does not proxy requests to downstream LLMs. It only returns model recommendations. The demo app handles the actual API calls to recommended models.

![Architecture Overview](https://assets.ngc.nvidia.com/products/api-catalog/llm-router/diagram.jpg)

## Routing Methods

The experimental v2 router provides two distinct routing approaches:

### 1. Intent-Based Routing 

Uses a small LLM like Qwen 1.7B to match user intents to specific models.

**Advantages**:
- No training required
- Understands semantic intent
- Works out-of-the-box
- Easily configurable via intent mappings

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
- Learns from actual usage patterns
- Optimizes for quality, latency, and cost
- Adapts to your specific workload
- Handles multimodal input natively

**Use Case**: When you have historical data and want data-driven routing decisions

**Training Recommended**: See notebooks for training pipeline:
- `2_Embedding_NN_Training.ipynb` - Training the neural network
- `3_Embedding_NN_Usage.ipynb` - Using the trained router

> Note: The GitHub repository includes a pre-trained neural network and the weights are stored in `llm-router/src/nat_sfc_router/training/router_artifacts`.  The notebook `2_Embedding_NN_Training.ipynb` re-trains the neural network and over-writes those weights. You can run the usage notebook or demo app without running the training notebook to use the existing neural network OR you can run the training notebook and then use this notebook or demo app with your neural network.

## Deployment Options

### Docker Deployment

```bash
docker-compose up -d --build
```

This starts three services:
- **router-backend** (port 8001): Main routing service using NVIDIA NeMo Agent Toolkit
- **qwen-router** (port 8011): Qwen 1.7B model server (for intent-based routing)
- **demo-app** (port 7860): Interactive Gradio web interface

Access the demo at: **http://localhost:7860**

## Understand the Blueprint

The experimental LLM Router v2 is structured around selecting the right model for a given request:

### Router Backend

The router backend is built on the NVIDIA NeMo Agent Toolkit and exposes a FastAPI service at `http://localhost:8001/sfc_router/chat/completions`. The endpoint accepts OpenAI-compatible chat completion requests with multimodal content (text and images) and returns the name of the optimal model.

The router backend is configured via `src/nat_sfc_router/configs/config.yml`:

```yaml
functions:
  healthcheck_fn:
    _type: healthcheck
  hf_intent_objective_fn:
    _type: hf_intent_objective_fn

  nn_objective_fn:
    _type: nn_objective_fn
    
    model_thresholds:
      'gpt-5-chat': 0.70
      'nvidia/nemotron-nano-12b-v2-vl': 0.75
      'nvidia/nvidia-nemotron-nano-9b-v2': 0.4
    
    model_costs:
      'gpt-5-chat': 1.0
      'nvidia/nemotron-nano-12b-v2-vl': 0.5
      'nvidia/nvidia-nemotron-nano-9b-v2': 0.3

  sfc_router_fn:
    _type: sfc_router
    objective_fn: hf_intent_objective_fn # <--- select routing function

workflow:
  _type: sfc_router
  objective_fn: hf_intent_objective_fn # <--- select routing function
```

### Routing Strategies

The router backend can use one of two strategies, configured by setting the `objective_fn` parameter:

1. **Intent-Based Routing (`hf_intent_objective_fn`)**: Uses the Qwen 1.7B model to classify user intents and map them to models. Intent mappings are defined in `src/nat_sfc_router/functions/hf_intent_objective_fn.py`. No training required.

2. **Auto-Routing (`nn_objective_fn`)**: Uses CLIP embeddings and a trained neural network to predict optimal models. Models are stored in `src/nat_sfc_router/training/router_artifacts/` and can be retrained.

### Using the Router

The LLM Router v2 is compatible with OpenAI chat completion requests. Unlike v1, the router **does not proxy** requests to downstream models - it only returns the recommended model name. Here's an example request:

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

Response:

```json
{
  "id": "chatcmpl-1765473022",
  "choices": [{
    "message": {
      "content": "nvidia/nvidia-nemotron-nano-9b-v2",
      "role": "assistant"
    }
  }],
  "model": "hf_intent_objective_fn"
}
```

The selected model name is in `choices[0].message.content`. Your application is responsible for calling the recommended model's API.

### Multimodal Support

The router supports multimodal requests with images encoded as base64 data URLs:

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

## Next Steps

The experimental blueprint includes several resources to help you understand, evaluate, and customize the LLM Router v2:

- **Explore the notebooks**: Three Jupyter notebooks demonstrate the routing methods and training pipeline:
  - `1_IntentRouter_Example.ipynb` - Intent-based routing examples and configuration
  - `2_Embedding_NN_Training.ipynb` - Train custom neural network router on your data
  - `3_Embedding_NN_Usage.ipynb` - Use and evaluate trained routers

- **Try the demo application**: An interactive Gradio web interface in `demo/` demonstrates end-to-end routing and model calling.

- **Review the source code**: The router implementation is in `src/nat_sfc_router/` with detailed documentation.

- **Train a custom router**: Follow the notebooks to create a router optimized for your specific use case and workload.

## License 3<sup>rd</sup> Party

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

## Security Considerations

- The LLM Router v2 Blueprint doesn't generate any code that may require sandboxing.
- The LLM Router v2 Blueprint is shared as a reference and is provided "as is". The security in the production environment is the responsibility of the end users deploying it. When deploying in a production environment, please have security experts review any potential risks and threats; define the trust boundaries, implement logging and monitoring capabilities, secure the communication channels, integrate AuthN & AuthZ with appropriate access controls, keep the deployment up to date, ensure the containers/source code are secure and free of known vulnerabilities.
- A frontend that handles AuthN & AuthZ should be in place as missing AuthN & AuthZ could provide un gated access to the router if directly exposed to e.g. the internet.
- API keys for downstream models (OpenAI, NVIDIA Build) are configured in the demo application's `.env` file. The end users are responsible for safeguarding these credentials.
- The LLM Router doesn't require any privileged access to the system.
- The end users are responsible for ensuring the availability of their deployment.
- The end users are responsible for building the container images and keeping them up to date.
- The end users are responsible for ensuring that OSS packages used by the blueprint are current.
- The logs from the router backend and demo app are printed to standard out and include input prompts and routing decisions for development purposes. The end users are advised to handle logging securely and avoid information leakage for production use cases.
