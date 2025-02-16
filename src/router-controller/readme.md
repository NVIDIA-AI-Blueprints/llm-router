# Router Controller API and Metrics

## Overview

The `router-controller` is a service that routes OpenAI compatible requests to appropriate LLMs based on a specified policy. It acts as a proxy, classifying user prompts and directing them to the best-fit model.

## API Endpoints

### `/config`
- **Description**: Returns the current configuration of the router.
- **Method**: `GET`
- **Response**: JSON object containing the sanitized router configuration.

### `/health`
- **Description**: Health check endpoint.
- **Method**: `GET`
- **Response**: JSON object with status `OK`.

### `/metrics`
- **Description**: Provides Prometheus metrics for monitoring the router's performance.
- **Method**: `GET`
- **Response**: Prometheus formatted metrics.

### `/v1/chat/completions` or `/completions`
- **Description**: Main endpoint for processing chat completions.
- **Method**: `POST`
- **Request Body**: JSON object containing the user prompt and additional parameters.
- **Response**: JSON object with the completion result from the selected LLM.

#### Request Payload

The payload for the POST call to `/v1/chat/completions` should be a JSON object with the following structure:

```json
{
  "model": "",
  "messages": [
    {
      "role": "user",
      "content": "Your input prompt here"
    }
  ],
  "nim-llm-router": {
    "policy": "task_router",
    "routing_strategy": "triton",
  },
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 1.0,
  "n": 1,
  "stream": false,
  "stop": ["\n"]
}
```

* model: (string) The name of the model to use for the completion.
* messages: (array) A list of messages comprising the conversation so far.
  * role: (string) The role of the message author, either "user" or "system".
  * content: (string) The content of the message.
* nim-llm-router: (object) Routing information for the LLM router.
  * policy: (string) The policy to use for routing. The policy is a mandatory argument.
  * routing_strategy: (string) The routing strategy to use, either "triton", "manual".
  * model: (string) If routing strategy is manual, model name should be specified.
* max_tokens: (integer) The maximum number of tokens to generate in the completion.
* temperature: (float) Sampling temperature to use, between 0 and 1.
* top_p: (float) Nucleus sampling probability, between 0 and 1.
* n: (integer) Number of completions to generate for each prompt.
* stream: (boolean) Whether to stream back partial progress.
* stop: (array of strings) Up to 4 sequences where the API will stop generating further tokens.

## Configuration

The `router-controller` communicates with the `router-server`, which is a Triton
Inference Server running the router models for classification. 

The router-controller configuration is defined in a YAML file and includes policies,
LLMs, and routing strategies.

We can specify multiple policies in the same `config.yaml`

### Routing Strategies
Router Controller Support two different routing strategies

- **Triton**: Uses the routing model hosted in the router server to classify prompts and route them to the appropriate LLM.
- **Manual**: Routes user prompts based on selected LLM name from the policy.


### Example Configuration

**Note**: The order of the LLMs under policies in the `config.yaml` is very important, as the router server returns a one-hot encoded vector for each classification.

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
      - name: Classification
        api_base: https://integrate.api.nvidia.com
        api_key: 
        model: meta/llama-3.1-8b-instruct
      - name: Closed QA
        api_base: https://integrate.api.nvidia.com
        api_key: 
        model: meta/llama-3.1-70b-instruct
      - name: Code Generation
        api_base: https://integrate.api.nvidia.com
        api_key: 
        model: mistralai/mixtral-8x22b-instruct-v0.1
      - name: Extraction
        api_base: https://integrate.api.nvidia.com
        api_key: 
        model: meta/llama-3.1-8b-instruct
      - name: Open QA
        api_base: https://integrate.api.nvidia.com
        api_key: 
        model: meta/llama-3.1-70b-instruct
      - name: Other
        api_base: https://integrate.api.nvidia.com
        api_key: 
        model: mistralai/mixtral-8x22b-instruct-v0.1
      - name: Rewrite
        api_base: https://integrate.api.nvidia.com
        api_key: 
        model: meta/llama-3.1-8b-instruct
      - name: Summarization
        api_base: https://integrate.api.nvidia.com
        api_key: 
        model: meta/llama-3.1-70b-instruct
      - name: Text Generation
        api_base: https://integrate.api.nvidia.com
        api_key: 
        model: mistralai/mixtral-8x22b-instruct-v0.1
      - name: Unknown
        api_base: https://integrate.api.nvidia.com
        api_key: 
        model: meta/llama-3.1-8b-instruct
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
        model: mistralai/mixtral-8x22b-instruct-v0.1
      - name: Contextual-Knowledge
        api_base: https://integrate.api.nvidia.com
        api_key: 
        model: meta/llama-3.1-8b-instruct
      - name: Few-Shot
        api_base: https://integrate.api.nvidia.com
        api_key: 
        model: meta/llama-3.1-70b-instruct
      - name: Domain-Knowledge
        api_base: https://integrate.api.nvidia.com
        api_key: 
        model: mistralai/mixtral-8x22b-instruct-v0.1
      - name: No-Label-Reason
        api_base: https://integrate.api.nvidia.com
        api_key: 
        model: meta/llama-3.1-8b-instruct
      - name: Constraint
        api_base: https://integrate.api.nvidia.com
        api_key: 
        model: meta/llama-3.1-8b-instruct
```

### `config.yaml` Parameters
  * policies: A list of routing policies. Each policy defines how to route user prompts to the appropriate LLMs.
  * name: The name of the policy.
  * url: The URL of the routing model hosted in the router server.
  * llms: A list of LLMs (Large Language Models) associated with the policy.
    * name: User defined name of the LLM that you want to associate with the classification.
    * api_base: The base URL of the LLM API.
    * api_key: The API key to access the LLM.
    * model: The specific model to use for the LLM.

### Example of Order Mapping 

In the above example, the order of the LLMs under the `task_router` policy is crucial. The router server returns a one-hot encoded vector for each classification, which corresponds to the order of the LLMs listed. For example:

* If the router server classifies a prompt as
 `Brainstorming`, it will return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].
* If the router server classifies a prompt as `Chatbot`, it will return [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].
* If the router server classifies a prompt as `Classifiction`, it will return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0].
* .
* .

The router-controller uses this one-hot encoded vector to route the prompt to the appropriate LLM based on the order specified in the config.yaml. Therefore, maintaining the correct order is essential for accurate routing.

### Error Types by Routing Strategy

#### Triton Routing Strategy Errors
- `5xx`: Server-side errors
  - Triton server unavailable (503)
  - Router model loading failures (503)
  - Triton service error with detailed message and code.

#### Manual Routing Strategy Errors
- `4xx`: Client-side errors
  - Model not found in policy (404)
  - Missing model parameter (400)
  - Invalid routing parameters (400)
  - Client error with detailed message and type.

#### LLM Service Errors
- Original status codes from LLM services are passed through
  - Rate limiting (429)
  - Service unavailable (503)
  - Quota Unavailable (402)
  - Other LLM-specific errors

### Error Response Format
When an error occurs, the response will contain:
```json
{
  "error": {
    "message": "Detailed error message",
    "type": "error_type",
    "status": status_code
  }
}
```

## Metrics

The `router-controller` exposes various metrics to help monitor its performance and behavior. These metrics can be accessed via the `/metrics` endpoint and are formatted for Prometheus.

### Stream Options

When making a request to the `/v1/chat/completions` endpoint with the `stream` parameter set to `true`, you can track token usage by including the `stream_options` object in the request payload with the `include_usage` field set to `true`. This ensures that token usage information is included in the response when streaming is enabled.

Example Request Payload with Streaming and Token Usage Tracking

```json
{
  "model": "",
  "messages": [
    {
      "role": "user",
      "content": "Your input prompt here"
    }
  ],
  "nim-llm-router": {
    "policy": "task_router",
    "routing_strategy": "triton"
  },
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 1.0,
  "n": 1,
  "stream": true,
  "stop": ["\n"],
  "stream_options": {
    "include_usage": true
  }
}
```
In this example, the `stream_options` object includes the `include_usage` field set to `true`, indicating that token usage information should be included in the response. This behavior allows the router controller to track token usage for streaming requests. Make sure to add the `stream_options` object with `include_usage: true` from the client when making streaming requests to enable this feature. Without this, the router controller will not report token usage for streaming requests.

### Available Metrics

- **Total Requests**: 
  - **Name**: `num_requests`
  - **Description**: Total number of requests received.

- **Requests Per Policy**: 
  - **Name**: `requests_per_policy`
  - **Description**: Total number of requests per policy.
  - **Labels**: `policy`

- **Requests Per Model**: 
  - **Name**: `requests_per_model`
  - **Description**: Total number of requests per model.
  - **Labels**: `model`

- **Request Latency**: 
  - **Name**: `request_latency_seconds`
  - **Description**: Latency of processing requests in seconds.

- **Successful Requests**: 
  - **Name**: `request_success_total`
  - **Description**: Total successful requests.

- **Failed Requests**: 
  - **Name**: `request_failure_total`
  - **Description**: Total failed requests, broken down by error type.
  - **Labels**: `error_type`
    - `4xx`: Client errors (e.g., invalid input, bad request)
    - `5xx`: Server errors (e.g., internal server errors, gateway timeouts)
    - `system`: System-level errors (e.g., network failures, connection timeouts)
    - `other`: Unclassified errors

- **Routing Policy Usage**: 
  - **Name**: `routing_policy_usage`
  - **Description**: Number of times each routing policy was used.
  - **Labels**: `routing_policy`

- **Model Selection Time**: 
  - **Name**: `model_selection_time_seconds`
  - **Description**: Time taken for model selection in seconds.

- **LLM Response Time**: 
  - **Name**: `llm_response_time_seconds`
  - **Description**: Response time for each LLM in seconds.
  - **Labels**: `llm`

- **Token Usage**: 
  - **Name**: `llm_token_usage`
  - **Description**: Token usage per LLM.
  - **Labels**: `llm`, `category`

- **Proxy Overhead Latency**: 
  - **Name**: `proxy_overhead_latency_seconds`
  - **Description**: Overhead latency of the proxy, calculated as overall latency minus model selection and LLM response time.
