# Router Service

This service is responsible for determining the most optimal model for a given multimodal LLM context. Optimizing the model selection can deliver an improved chat experience in terms of:

1. Generation quality
2. Latency
3. Cost efficiency

Built on the NeMo Agent Toolkit, this system provides streamlined configuration, evaluation, profiling, observability, and type safety.

## Local Development

### Setup Virtual Environment

```bash
uv venv --python 3.12 --seed .venv
```

### Install Dependencies

```bash
uv sync
```

### Start the Service

```bash
./scripts/run_local.sh
```

## Docker Deployment

### Build Container

```bash
docker build -t superfast-chatbot-router .
```

### Run Container

```bash
docker run --env-file .env -p 8000:8000 superfast-chatbot-router
```

###  Objective Functions

The router uses an objective function responsible for taking in a configured list of candidate routes and an OpenAI chat completions request (including messages and context). The objective function then returns the optimal candidate model. 

There are currently two objective functions:
- cycle_objective_fn: round robins requests between candidate models
- linear_mf_cost_latency

### Linear MF Cost Latency 

The linear_mf_cost_latency objective function picks the candidate model that maximizes:

```
c*CLn(prompt) - a*Cn - b*Ln
```

This is a linear function that takes:

CLn(prompt) - This represents the normalized Correctness Likelihood, which is calculated using the RouteLLM Matrix Factorization model (MF). MF takes in a prompt and returns a score that is roughly correlated with how likely the given model is to answer the prompt more correctly than a known "golden" model, trained on Chat Arena preference data. Taken liberally, this means the CLn number represents how smart the model's answer to the prompt will be, higher being better.

Cn - This is a fixed cost value based on the model's total cost to run the Artifical Analysis Intelligence exam --- essentially a proxy for how expensive this model is relative to the other candidate models. The value is normalized so the most expensive candidate model has a Cn of 1.

Ln - This is a fixed latency value based on the model's time to generate 500 tokens as reported by the Artifical Analysis Intelligence exam --- essentially a proxy for the model's latency relative to the other candidate models. The value is normalized so the most expensive candidate model has a Cn of 1.


Configurable hyper parameters:

c - correctness likelihood weight aka mf_weight, a higher value indicates a stronger preference for correct answers

a - cost weight, a higher value indicates a stronger preference for cheaper models

b - latency weight, a higher value indicates a stronger preference for fast models

