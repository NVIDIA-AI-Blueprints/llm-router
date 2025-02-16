# NVIDIA AI Blueprint: LLM Router

## Overview

Ever struggled to decide which LLM to use for a specific task? In an ideal world the most accurate LLM would also be the cheapest and fastest, but in practice modern agentic AI systems have to make trade-offs between accuracy, speed, and cost.

This blueprint provides a router that automates these tradeoffs by routing user prompts between different LLMs. Given a user prompt, the router:

- applies a policy (eg task classification or intent classification)
- uses a router trained for that policy to map the prompt to an appropriate LLM
- proxies the prompt to the identified fit-for-purpose LLM

For example, using a task classification policy, the following user prompts can be classified into tasks and directed to the appropriate LLM.

| User Prompt | Task Classification | Route To |
|---|---|---|
| "Help me write a python function to load salesforce data into my warehouse." | Code Generation | deepseek |
| "Tell me about your return policy " | Open QA | llama 70B | 
| "Rewrite the user prompt to be better for an LLM agent. User prompt: what is the best coffee recipe" | Rewrite | llama 8B |

The key features of the LLM Router framework are:

- OpenAI API compliant: use the LLM Router as a drop-in replacement in code regardless of which LLM framework you use.
- Flexible: use the default policy and router, or create your own policy and fine tune a router. We expect additional trained routers to be available from the community as well.
- Configurable: easily configure which backend models are available. 
- Performant: LLM Router uses Rust and NVIDIA Triton Inference Server to add minimal latency compared to routing requests directly to a model.


## Software Components 

The LLM Router has three components: 
- <b>Router Controller</b> - is a service similar to a proxy that routes OpenAI compatible requests. The controller is implemented as a Rust proxy and the code is available in `src/router-controller`.
- <b>Router Server</b> - is a service that classifies the user's prompt using a pre-trained model. In this blueprint, the router server is implemented as a NVIDIA Triton Inference Server with pre-trained router models based off of [`Nvidia/prompt-task-and-complexity-classifier`](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier). The pre-trained router models are available on NGC.
- <b>Downstream LLMs</b> - are the LLMs the prompt will be passed to, typically foundational LLMs. In this blueprint the downstream models are NVIDIA NIMs, specifically `meta/llama-3.1-70b-instruct`, `meta/llama-3.1-8b-instruct`, `mistralai/mixtral-8x22b-instruct-v0.1`, and `deepseek-ai/deepseek-r1`. Other LLMs are supported such as locally hosted NVIDIA NIMs or third party OpenAI compatible API endpoints.

![architecture diagram](assets/llm-router-blueprint.jpg)

## Target Audience 
This blueprint is for: 

-  <b>AI Engineers and Developers</b>: Developers building or maintaining AI systems benefit from LLM routers by integrating them into applications for scalable, cost-effective solutions.
- <b>MLOps Teams</b>: People can use LLM routers to optimize resource allocation, ensuring efficient use of computational resources.

## Prerequisites 

### Software

- Linux operating systems (Ubuntu 22.04 or later recommended)
- [Docker](https://docs.docker.com/engine/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (NVIDIA Driver >= `535`, CUDA >= `12.2`)

### Clone repository and install software

1. Clone Git repository

```
git clone https://github.com/NVIDIA-AI-Blueprints/llm-router
cd llm-router
```

2. Install [Docker](https://docs.docker.com/engine/install/ubuntu/)

> Tip: Ensure the Docker Compose plugin version is 2.29.1 or higher.  Run `docker compose --version` to confirm. 

3. Install **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit)** to configure Docker for GPU-accelerated containers.

### Get API Keys 

1. NVIDIA NGC API key

The NVIDIA NGC API Key is used in this blueprint in order to pull the default router models from NGC as well as the NVIDIA Triton server docker image which is used to run those models. Refer to [Generating NGC API Keys](https://docs.nvidia.com/ngc/gpu-cloud/ngc-user-guide/index.html#generating-api-key) in the NVIDIA NGC User Guide for more information.

Once you have the key, login with: 

```
docker login nvcr.io
```

Use `$oauthtoken` as the username and the API key as the password.

2. NVIDIA API Catalog key

- Navigate to **[NVIDIA API Catalog](https://build.nvidia.com/explore/discover). 
- Click one of the models, such as llama3-8b-instruct. 
- Select the "Docker" input option. 
- Click "Get API Key".
- Click "Generate Key" and copy the resulting key, save it somewhere safe for later use

> Tip: The NVIDIA API Catalog key will start with nvapi- whereas the NVIDIA NGC key will not.

## Hardware Requirements

Using LLM router models used in blueprint:
| GPU | Family | Memory | # of GPUs (min.) |
| ------ | ------ | ------ | ------ |
| V100 or newer | SXM or PCIe | 80GB | 1 |

Using a custom LLM-router model not included in blueprint (requirements may vary based on customizations):
| GPU | Family | Memory | # of GPUs (min.) |
| ------ | ------ | ------ | ------ |
| A100 or newer | SXM or PCIe | 80GB | 1 |

## Quickstart Guide

After meeting the pre-requisites follow these steps.

#### 1. Install necessary python libraries

Create and activate a Python virtual environment, then run: 

```
pip install -r requirements.txt
```

#### 3. Access Jupyter Notebook

Bring up Jupyter and open the notebook in the `launchable` direcotry called `1_Deploy_LLM_Router.ipynb`.

```
jupyter --no-browser --ip 0.0.0.0 --NotebookApp.token=’’
```

## Understand the blueprint

The LLM Router is composed of three components: 

- <b>Router Controller</b> - is a service similar to a proxy that routes OpenAI compatible requests.
- <b>Router Server</b> - is a service that classifies the user's prompt according to a routing strategy and policy. The classification is made using a pre-trained model. 
- <b>Downstream LLMs</b> - are the LLMs the prompt will be routed to, typically foundational LLMs. 

These three components are all managed in the LLM Router configuration file which is located at `src/router-controller/config.yml`. 

```yaml 
policies:
  - name: "task_router"
    url: http://router-server:8000/v2/models/task_router_ensemble/infer
    llms:
      - name: Brainstorming
        api_base: https://integrate.api.nvidia.com
        api_key: 
        model: meta/llama-3.1-70b-instruct
      - name: Chatbot
        api_base: https://integrate.api.nvidia.com
        api_key: 
        model: mistralai/mixtral-8x22b-instruct-v0.1
    ...    
  - name: "complexity_router"
    url: http://router-server:8000/v2/models/complexity_router_ensemble/infer
    llms:
      - name: Creativity
        api_base: https://integrate.api.nvidia.com
        api_key: 
        model: meta/llama-3.1-70b-instruct
      - name: Reasoning
        api_base: https://integrate.api.nvidia.com
        api_key: 
        model: deepseek-ai/deepseek-r1
    ...
```

### Policies

The configuration file specifies the routing policies. In the default configuration, a prompt can either be classified using a `task_router` policy or a `complexity_router` policy. 

The `task_router` uses a pre-trained model that will be deployed at `http://router-server:8000/v2/models/task_router_ensemble/infer`. The model classifies prompts into categories based on the task of the prompt:
  - Brainstorming
  - Chatbot
  - Classification
  - Closed QA
  - Code Generation
  - Extraction
  - Open QA
  - Other
  - Rewrite
  - Summarization
  - Text Generation
  - Unknown

For example, the prompt `Help me write a python function to load salesforce data into my warehouse` would be classified as a `Code Generation` task.

The `complexity_router` uses a different pre-trained model that will be deployed at `http://router-server:8000/v2/models/complexity_router_ensemble/infer`. This model classifies prompts into categories based on the complexity of the prompt:
  - Creativity: Prompts that require create knowledge, eg "write me a science fiction story".
  - Reasoning: Prompts that require reasoning, eg solving a riddle.
  - Contextual-Knowledge: Prompts that require background information, eg asking for technical help with a specific product.
  - Few-Shot: Prompts that include example questions and answers.
  - Domain-Knowledge: Prompts that require broad domain knowledge, such as asking for an explanation of a historical event.
  - No-Label-Reason: Prompts that are not classified into one of the other categories.
  - Constraint: Prompts that include specific constraints, eg requesting an answer in a haiku format.

The `customize/READEME.md` describes how to create your own policy and classification model, providing an example showing a policy for classifying user interactions with a bank support chatbot. 

### LLMs 

The `llms` portion of the configuration file specifies where the classified prompts should be routed. For example, in the default configuration file: 

```yaml
policies:
  - name: "task_router"
    url: http://router-server:8000/v2/models/task_router_ensemble/infer
    llms:
      - name: Brainstorming
        api_base: https://integrate.api.nvidia.com
        api_key: 
        model: meta/llama-3.1-70b-instruct
      - name: Chatbot
        api_base: https://integrate.api.nvidia.com
        api_key: 
        model: mistralai/mixtral-8x22b-instruct-v0.1
```

A prompt sent to the `task_router` policy classified as a Brainstorming task would be proxied to the NVIDIA NIM `meta/llama-3.1-70b-instruct` whereas a prompt classified as a Chatbot task would be sent to `mistralai/mixtral-8x22b-instruct-v0.1`. 

### Using the router

The LLM Router is compatible with OpenAI API requests. This means that any applications or code that normally use an OpenAI API client (such as LangChain) can use LLM Router with minimal modification. For example, this RESTful API request to LLM Router follows the OpenAI specification with a few modifications:

```console
curl -X 'POST' \
  'http://0.0.0.0:8084/v1/chat/completions' \   # the URL to the deployed LLM router
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "",                                # the model field is left blank, as the LLM router will add this based on the prompt classification
    "messages": [
      {
        "role":"user",
        "content":"Hello! How are you?"
      },
      {
        "role":"assistant",
        "content":"Hi! I am quite well, how can I help you today?"
      },
      {
        "role":"user",
        "content":"Can you write me a song? Use as many emojis as possible."
      }
    ],
    "max_tokens": 64,
    "stream": true,
    "nim-llm-router": {"policy": "task_router",
                       "routing_strategy": "triton",
                       "model": ""}
  }'
```

The primary modification is the inclusion of the `nim-llm-router` metadata in the body of the request. In most python clients this metadata would be added as `extra_body`, see `src/test_router.py` for an example. The required metadata is:

- policy: the policy to use for classification, by default either `task_router`  or `complexity_router`.
- routing_strategy: either `triton` which means the prompt is sent to a model for classification or `manual` which means that the classification is skipped - use this if the client needs to make a manual over-ride
- model: if the `routing_strategy` is `triton` leave this blank, if the routing strategy is `manual` specify the model over-ride

## Next Steps

The blueprint includes a variety of tools to help understand, evaluate, customize, and monitor the LLM Router.

- Read more about the router controller implementation and capabilities in the source README located at `src/router-controller/readme.md`.
- A sample client application is available in the `demo/app` folder.
- Metrics are automatically collected and can be exported via Prometheus to Grafana. Details are available in the source README and an example is provided in the quickstart notebook.
- A sample loadtest is available in the `demo/loadtest` folder with instructions in the associated README.
- The blueprint includes two default routing policies available for download from NGC. The `customize` directory includes two notebooks showing how each policy model was created. There is also an example notebook showing how to create a third policy. The `intent_router` is created by fine-tuning a model to classify prompts based on a user's intent, assuming they are interacting with a support chatbot at a bank.

## Security Considerations

- The LLM Router Blueprint doesn't generate any code that may require sandboxing.
- The LLM Router Development Blueprint is shared as a reference and is provided "as is". The security in the production environment is the responsibility of the end users deploying it. When deploying in a production environment, please have security experts review any potential risks and threats; define the trust boundaries, implement logging and monitoring capabilities, secure the communication channels, integrate AuthN & AuthZ with appropriate access controls, keep the deployment up to date, ensure the containers/source code are secure and free of known vulnerabilities.
- The users need to be aware that the api_key for the end LLM for the router-controller is obtained by populating the config.yaml file and this might lead to the leakage of api_key and unauthorized access for the end LLM and the end users are responsible for safeguarding the config.yaml file and the api_key in it.
- The LLM Router doesn't require any privileged access to the system.
- The end users are responsible for ensuring the availability of their deployment.
- The end users are responsible for building the container images and keeping them up to date.
- The end users are responsible for ensuring that OSS packages used by the developer blueprint are current.
- The logs from router-controller, router-server, and the demo app are printed to standard out, they include input prompts and output completions for development purposes. The end users are advised to handle logging securely and avoid information leakage for production use cases.
