# Routing to Local Endpoints on OpenShift AI

This guide explains how to configure the NVIDIA LLM Router to route requests to local LLM inference endpoints running on the same OpenShift cluster, instead of (or alongside) NVIDIA's hosted Build API.

This works with any OpenAI-compatible inference server, including:
- **vLLM** (via KServe InferenceService or standalone)
- **NVIDIA NIM** (self-hosted)
- **Models deployed through Red Hat OpenShift AI (RHOAI)** model serving

Each LLM entry in the router configuration has its own `api_base` URL, so different models can live at different endpoints — including a mix of local and remote (NVIDIA Build API) models in the same deployment.

## Prerequisites

- **LLM Router deployed on OpenShift** — see the [OpenShift Deployment Guide](openshift.md)
- **One or more running LLM endpoints** exposing an OpenAI-compatible API (`/v1/chat/completions`)
- **For each endpoint, know:**
  - The Route URL (e.g., `https://my-model-my-namespace.apps.cluster.example.com`)
  - The model ID as returned by `/v1/models`
  - The authentication method (e.g., OpenShift OAuth bearer token, API key, or none)

### Verifying Your Endpoints

Before configuring the router, confirm each endpoint works independently:

```bash
# Check the model ID
curl -sk https://<model-route>/v1/models \
  -H "Authorization: Bearer $(oc whoami -t)"

# Test inference
curl -sk https://<model-route>/v1/chat/completions \
  -H "Authorization: Bearer $(oc whoami -t)" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model-id>",
    "messages": [{"role": "user", "content": "Say hello in one word"}]
  }'
```

## Step 1: Create the Bearer Token Secret

The router-controller sends requests to LLM endpoints using Bearer token authentication. Store your token in the `llm-api-keys` secret:

```bash
oc create secret generic llm-api-keys \
  --from-literal=nvidia_api_key="unused" \
  --from-literal=ocp_bearer_token="$(oc whoami -t)" \
  --dry-run=client -o yaml | oc apply -f -
```

> **Note**: The `nvidia_api_key` key is required by the Helm template even when not using NVIDIA's API. Set it to `"unused"` or your actual NVIDIA key if you also route to Build API.

### Token Expiry

OpenShift OAuth tokens expire (typically 24 hours). For production, create a long-lived ServiceAccount token:

```bash
# Create a service account
oc create sa llm-router-client

# Create a long-lived token (no expiry)
oc create token llm-router-client --duration=0 > /tmp/sa-token

# Update the secret with the SA token
oc create secret generic llm-api-keys \
  --from-literal=nvidia_api_key="unused" \
  --from-literal=ocp_bearer_token="$(cat /tmp/sa-token)" \
  --dry-run=client -o yaml | oc apply -f -

rm /tmp/sa-token
```

> **Important**: The service account may need RBAC permissions to access endpoints in other namespaces. Consult your cluster administrator if you get 403 errors.

## Step 2: Manual Routing

Manual routing lets the client choose which model to use by name. No GPU or Triton server required.

### Create the Custom ConfigMap

```bash
cat <<'EOF' | oc apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: llm-router-local-config
data:
  config.yaml: |
    policies:
      - name: "task_router"
        url: ""
        llms:
          - name: <display-name-1>
            api_base: https://<model-1-route>
            api_key: ${OCP_BEARER_TOKEN}
            model: <model-1-id>
          - name: <display-name-2>
            api_base: https://<model-2-route>
            api_key: ${OCP_BEARER_TOKEN}
            model: <model-2-id>
          - name: <display-name-3>
            api_base: https://<model-3-route>
            api_key: ${OCP_BEARER_TOKEN}
            model: <model-3-id>
EOF
```

**Field reference:**

| Field | Description | Example |
|-------|-------------|---------|
| `name` | Display name used by clients to select this model | `Llama-8B` |
| `api_base` | The endpoint's base URL (without `/v1/chat/completions`) | `https://my-model-ns.apps.cluster.com` |
| `api_key` | Bearer token — use `${OCP_BEARER_TOKEN}` to reference the env var | `${OCP_BEARER_TOKEN}` |
| `model` | The model ID as the endpoint knows it (from `/v1/models`) | `my-llama-8b` |

> **How it works**: The controller builds the full URL as `{api_base}/v1/chat/completions` and sends the request with `Authorization: Bearer {api_key}`. The `model` field is injected into the forwarded request body.

### Deploy

```bash
helm upgrade llm-router ./deploy/helm/llm-router \
  --reuse-values \
  --set routerServer.enabled=false \
  --set routerController.config.existingConfigMap=llm-router-local-config \
  --set 'routerController.env[0].name=OCP_BEARER_TOKEN' \
  --set 'routerController.env[0].valueFrom.secretKeyRef.name=llm-api-keys' \
  --set 'routerController.env[0].valueFrom.secretKeyRef.key=ocp_bearer_token'
```

### Test

```bash
CONTROLLER_ROUTE=$(oc get route llm-router-controller -o jsonpath='{.spec.host}')

curl -sk -X POST "https://$CONTROLLER_ROUTE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "nim-llm-router": {
      "policy": "task_router",
      "routing_strategy": "manual",
      "model": "<display-name-1>"
    }
  }'
```

The `model` field in `nim-llm-router` must match the `name` field of an LLM entry in the config (e.g., `Llama-8B`), not the model ID.

## Step 3: Triton Routing (Automatic Classification)

Triton routing uses ML classification models to automatically route prompts to the most appropriate LLM based on task type or complexity. This requires:
- The router-server (Triton) running with loaded classification models — see [Triton Model Setup Guide](openshift-triton-model-setup.md)
- A GPU node for the router-server

### How Triton Routing Works

1. Client sends a prompt to the router-controller
2. The controller forwards the prompt text to Triton's classification model
3. Triton returns a probability distribution over categories (e.g., "Code Generation", "Chatbot", "Summarization")
4. The controller selects the category with the highest probability
5. The controller routes the request to the LLM mapped to that category

### Category-to-Model Mapping

The NGC classification models define a **fixed order** of categories. Each category maps to an LLM entry **by position** in the config — the first category maps to the first LLM, the second to the second, and so on.

**Task Router** — 12 categories (fixed order):

| Index | Category | Description | Suggested Tier |
|-------|----------|-------------|---------------|
| 0 | Brainstorming | Creative ideation | Strongest |
| 1 | Chatbot | General conversation | Mid-tier |
| 2 | Classification | Categorization tasks | Fastest |
| 3 | Closed QA | Factual question answering | Strongest |
| 4 | Code Generation | Writing code | Strongest |
| 5 | Extraction | Information extraction | Fastest |
| 6 | Open QA | Open-ended questions | Mid-tier |
| 7 | Other | Uncategorized | Mid-tier |
| 8 | Rewrite | Text transformation | Fastest |
| 9 | Summarization | Content summarization | Mid-tier |
| 10 | Text Generation | General text writing | Mid-tier |
| 11 | Unknown | Unclassified prompts | Fastest |

**Complexity Router** — 7 categories (fixed order):

| Index | Category | Suggested Tier |
|-------|----------|---------------|
| 0 | Creativity | Strongest |
| 1 | Reasoning | Strongest |
| 2 | Contextual-Knowledge | Mid-tier |
| 3 | Few-Shot | Mid-tier |
| 4 | Domain-Knowledge | Strongest |
| 5 | No-Label-Reason | Fastest |
| 6 | Constraint | Fastest |

> **Suggested Tier** is a guideline — route complex/creative tasks to your strongest model and simple tasks to your fastest. The actual mapping depends on your available models and priorities.

### Create the Triton ConfigMap

You must provide **exactly 12 LLM entries** for `task_router` and **exactly 7** for `complexity_router`, in the order shown above. Multiple categories can point to the same model.

```bash
cat <<'EOF' | oc apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: llm-router-local-triton-config
data:
  config.yaml: |
    policies:
      - name: "task_router"
        url: http://llm-router-router-server:8000/v2/models/task_router_ensemble/infer
        llms:
          # Index 0: Brainstorming → strongest model
          - name: Brainstorming
            api_base: https://<strongest-model-route>
            api_key: ${OCP_BEARER_TOKEN}
            model: <strongest-model-id>
          # Index 1: Chatbot → mid-tier model
          - name: Chatbot
            api_base: https://<midtier-model-route>
            api_key: ${OCP_BEARER_TOKEN}
            model: <midtier-model-id>
          # Index 2: Classification → fastest model
          - name: Classification
            api_base: https://<fastest-model-route>
            api_key: ${OCP_BEARER_TOKEN}
            model: <fastest-model-id>
          # Index 3: Closed QA → strongest model
          - name: Closed QA
            api_base: https://<strongest-model-route>
            api_key: ${OCP_BEARER_TOKEN}
            model: <strongest-model-id>
          # Index 4: Code Generation → strongest model
          - name: Code Generation
            api_base: https://<strongest-model-route>
            api_key: ${OCP_BEARER_TOKEN}
            model: <strongest-model-id>
          # Index 5: Extraction → fastest model
          - name: Extraction
            api_base: https://<fastest-model-route>
            api_key: ${OCP_BEARER_TOKEN}
            model: <fastest-model-id>
          # Index 6: Open QA → mid-tier model
          - name: Open QA
            api_base: https://<midtier-model-route>
            api_key: ${OCP_BEARER_TOKEN}
            model: <midtier-model-id>
          # Index 7: Other → mid-tier model
          - name: Other
            api_base: https://<midtier-model-route>
            api_key: ${OCP_BEARER_TOKEN}
            model: <midtier-model-id>
          # Index 8: Rewrite → fastest model
          - name: Rewrite
            api_base: https://<fastest-model-route>
            api_key: ${OCP_BEARER_TOKEN}
            model: <fastest-model-id>
          # Index 9: Summarization → mid-tier model
          - name: Summarization
            api_base: https://<midtier-model-route>
            api_key: ${OCP_BEARER_TOKEN}
            model: <midtier-model-id>
          # Index 10: Text Generation → mid-tier model
          - name: Text Generation
            api_base: https://<midtier-model-route>
            api_key: ${OCP_BEARER_TOKEN}
            model: <midtier-model-id>
          # Index 11: Unknown → fastest model
          - name: Unknown
            api_base: https://<fastest-model-route>
            api_key: ${OCP_BEARER_TOKEN}
            model: <fastest-model-id>
      - name: "complexity_router"
        url: http://llm-router-router-server:8000/v2/models/complexity_router_ensemble/infer
        llms:
          # Index 0: Creativity → strongest
          - name: Creativity
            api_base: https://<strongest-model-route>
            api_key: ${OCP_BEARER_TOKEN}
            model: <strongest-model-id>
          # Index 1: Reasoning → strongest
          - name: Reasoning
            api_base: https://<strongest-model-route>
            api_key: ${OCP_BEARER_TOKEN}
            model: <strongest-model-id>
          # Index 2: Contextual-Knowledge → mid-tier
          - name: Contextual-Knowledge
            api_base: https://<midtier-model-route>
            api_key: ${OCP_BEARER_TOKEN}
            model: <midtier-model-id>
          # Index 3: Few-Shot → mid-tier
          - name: Few-Shot
            api_base: https://<midtier-model-route>
            api_key: ${OCP_BEARER_TOKEN}
            model: <midtier-model-id>
          # Index 4: Domain-Knowledge → strongest
          - name: Domain-Knowledge
            api_base: https://<strongest-model-route>
            api_key: ${OCP_BEARER_TOKEN}
            model: <strongest-model-id>
          # Index 5: No-Label-Reason → fastest
          - name: No-Label-Reason
            api_base: https://<fastest-model-route>
            api_key: ${OCP_BEARER_TOKEN}
            model: <fastest-model-id>
          # Index 6: Constraint → fastest
          - name: Constraint
            api_base: https://<fastest-model-route>
            api_key: ${OCP_BEARER_TOKEN}
            model: <fastest-model-id>
EOF
```

### Deploy

```bash
helm upgrade llm-router ./deploy/helm/llm-router \
  --reuse-values \
  --set routerServer.enabled=true \
  --set routerController.config.existingConfigMap=llm-router-local-triton-config \
  --set 'routerController.env[0].name=OCP_BEARER_TOKEN' \
  --set 'routerController.env[0].valueFrom.secretKeyRef.name=llm-api-keys' \
  --set 'routerController.env[0].valueFrom.secretKeyRef.key=ocp_bearer_token'
```

> **Note**: If the router-server pod gets stuck in `ContainerCreating` due to the RWO PVC being held by the old pod during rolling update, scale to 0 and back to 1:
> ```bash
> oc scale deployment/llm-router-router-server --replicas=0
> # Wait a few seconds
> oc scale deployment/llm-router-router-server --replicas=1
> ```

### Test

```bash
CONTROLLER_ROUTE=$(oc get route llm-router-controller -o jsonpath='{.spec.host}')

# Task-based routing — classifier picks the model automatically
curl -sk -X POST "https://$CONTROLLER_ROUTE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a Python function to sort a list"}],
    "nim-llm-router": {
      "policy": "task_router",
      "routing_strategy": "triton"
    }
  }'

# Complexity-based routing
curl -sk -X POST "https://$CONTROLLER_ROUTE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain quantum entanglement step by step"}],
    "nim-llm-router": {
      "policy": "complexity_router",
      "routing_strategy": "triton"
    }
  }'
```

Check which model was selected by looking at the `model` field in the response JSON.

## Notes

### In-Cluster Service URLs

If the LLM endpoints are in the same OpenShift cluster, you can use internal service URLs instead of external routes:

```
http://<service-name>.<namespace>.svc.cluster.local:8000
```

This bypasses the OpenShift Router, avoids OAuth, and reduces latency. Use this when:
- The model endpoints don't require external route authentication
- NetworkPolicies allow cross-namespace traffic

### Mixed Local and Remote Routing

You can mix local endpoints and NVIDIA's Build API in the same config. For example, route code generation to a local model and everything else to NVIDIA:

```yaml
- name: Code Generation
  api_base: https://<local-model-route>
  api_key: ${OCP_BEARER_TOKEN}
  model: local-code-model
- name: Chatbot
  api_base: https://integrate.api.nvidia.com
  api_key: ${NVIDIA_API_KEY}
  model: meta/llama-3.1-70b-instruct
```

When mixing, inject both env vars via `routerController.env`.

### Route Timeout

OpenShift Routes have a default 30-second timeout. Larger models or complex prompts may exceed this. To increase:

```bash
oc annotate route llm-router-controller \
  haproxy.router.openshift.io/timeout=120s --overwrite
```

### Using Models Deployed via RHOAI

Models deployed through **Red Hat OpenShift AI (RHOAI)** model serving (single-model or multi-model) are exposed as KServe InferenceServices with OpenAI-compatible endpoints. They work with this guide as-is:

1. Find the InferenceService route:
   ```bash
   oc get inferenceservice -n <rhoai-project> -o jsonpath='{.items[*].status.url}'
   ```
2. Get the model ID:
   ```bash
   curl -sk <inference-service-url>/v1/models \
     -H "Authorization: Bearer $(oc whoami -t)"
   ```
3. Use the route URL as `api_base` and the model ID as `model` in the ConfigMap above.

RHOAI model serving endpoints follow the same OpenAI-compatible API contract (`/v1/chat/completions`, `/v1/models`), so no special configuration is needed beyond what this guide describes.

## Troubleshooting

### 401 Unauthorized

The bearer token is invalid or expired.

```bash
# Refresh the token
oc create secret generic llm-api-keys \
  --from-literal=nvidia_api_key="unused" \
  --from-literal=ocp_bearer_token="$(oc whoami -t)" \
  --dry-run=client -o yaml | oc apply -f -

# Restart the controller to pick up the new token
oc rollout restart deployment/llm-router-router-controller
```

### 504 Gateway Timeout

The LLM endpoint took too long to respond. See [Route Timeout](#route-timeout) above.

### Model Not Found

The `model` field in `nim-llm-router` must match the `name` field of an LLM entry (for manual routing), not the model ID. Check your ConfigMap.

### Triton Classification Returns Wrong Category

The classifier is a pre-trained model — its categorization is a best-effort prediction. You can:
- Use manual routing to override the selection
- Adjust the category-to-model mapping so that "wrong" categories still route to a capable model
- Set a `threshold` in the request to fall back when confidence is low

Last updated: May 2026
