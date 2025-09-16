# LLM Router Helm Chart

Deploys the LLM Router with all required components on Kubernetes: Router Server (Triton), Router Controller (API), and optional Demo App.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.2.0+
- GPU nodes with NVIDIA drivers for Router Server
- NVIDIA API key from [NVIDIA API Catalog](https://build.nvidia.com/explore/discover)

## Router Controller Configuration

The router controller can be configured to work with different LLM backends. **All configurations require API key secrets** - choose your approach:

### Step 1: Create API Key Secret (Required for All)

**Default NVIDIA Cloud API:**
```bash
kubectl create secret generic llm-api-keys \
  --from-literal=nvidia_api_key=nvapi-your-key-here
```

**Custom secret name:**
```bash
kubectl create secret generic my-custom-secret \
  --from-literal=my_api_key=your-key-here
```

**Multiple providers:**
```bash
kubectl create secret generic multi-provider-keys \
  --from-literal=nvidia_key=nvapi-your-key \
  --from-literal=openai_key=sk-your-key \
  --from-literal=anthropic_key=ant-your-key
```

### Step 2: Choose Configuration Approach

#### Option A: Built-in Template (Recommended)

**Default NVIDIA Cloud API** - No additional configuration needed:
- Uses `llm-api-keys` secret with `nvidia_api_key`
- API Base: `https://integrate.api.nvidia.com`
- Pre-configured models for different tasks

**Customize models** (optional):
```yaml
# values.override.yaml
routerController:
  config:
    models:
      brainstorming: "meta/llama-3.1-405b-instruct"  # Upgrade to larger model
      creativity: "meta/llama-3.1-405b-instruct"     # Upgrade to larger model
```

**Custom secret configuration:**
```yaml
# values.override.yaml
routerController:
  config:
    apiKeySecret: "my-custom-secret"
    apiKeySecretKey: "my_api_key"
```

#### Option B: Custom Configuration (Advanced)

**Inline configuration:**
```yaml
# values.override.yaml
routerController:
  config:
    customConfig: |
      policies:
        - name: "task_router"
          url: http://llm-router-router-server:8000/v2/models/task_router_ensemble/infer
          llms:
            - name: Brainstorming
              api_base: http://your-local-llm-service:8000/v1
              api_key: ${NVIDIA_API_KEY}
              model: meta/llama-3.1-405b-instruct
```

**External ConfigMap:**
```yaml
# values.override.yaml
routerController:
  config:
    existingConfigMap: "my-custom-router-config"
```

First create your ConfigMap:
```bash
kubectl create configmap my-custom-router-config --from-file=config.yaml=my-config.yaml
```

### Step 3: Add Custom Environment Variables (Optional)

**Local LLM deployments (Dynamo, vLLM, etc.):**
```yaml
# values.override.yaml
routerController:
  env:
    - name: DYNAMO_API_KEY
      valueFrom:
        secretKeyRef:
          name: dynamo-api-secret
          key: DYNAMO_API_KEY
    - name: DYNAMO_API_BASE
      value: "http://dynamo-service.namespace.svc.cluster.local:8000"
```

**Multiple API providers:**
```yaml
# values.override.yaml
routerController:
  env:
    - name: OPENAI_API_KEY
      valueFrom:
        secretKeyRef:
          name: multi-provider-keys
          key: openai_key
    - name: ANTHROPIC_API_KEY
      valueFrom:
        secretKeyRef:
          name: multi-provider-keys
          key: anthropic_key
```

**Custom configuration:**
```yaml
# values.override.yaml
routerController:
  env:
    - name: ROUTER_TIMEOUT
      value: "30s"
    - name: MAX_CONCURRENT_REQUESTS
      value: "100"
    - name: CUSTOM_API_TOKEN
      valueFrom:
        secretKeyRef:
          name: custom-tokens
          key: api_token
```

## Deployment Steps

### Step 4: Build and Push Images

```bash
# Build all required images
docker build -t <your-registry>/router-server:latest -f src/router-server/router-server.dockerfile .
docker build -t <your-registry>/router-controller:latest -f src/router-controller/router-controller.dockerfile .
docker build -t <your-registry>/llm-router-client:app -f demo/app/app.dockerfile .

# Push to your registry
docker push <your-registry>/router-server:latest
docker push <your-registry>/router-controller:latest
docker push <your-registry>/llm-router-client:app
```

### Step 5: Create Registry Secret

```bash
# Create secret for NVIDIA Container Registry authentication
kubectl create secret docker-registry nvcr-secret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password=<your-ngc-api-key>

# Verify
kubectl get secret nvcr-secret
```

### Step 6: Prepare Models

```bash
# Download models first (follow main project README)
# Ensure you have models in local 'routers/' directory

# Step 1: Check available storage classes and create PVC
# First, check what storage classes are available in your cluster
kubectl get storageclass

# Create PVC using appropriate storage class for your environment:
# - AWS EKS: use "gp3" or "gp2" 
# - Google GKE: use "standard" or "ssd"
# - Azure AKS: use "default" or "managed-premium"
# - MicroK8s: use "microk8s-hostpath"
# - Or omit storageClassName to use cluster default

kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: router-models-pvc
spec:
  accessModes: ["ReadWriteOnce"]
  resources:
    requests:
      storage: 100Gi
EOF

# Step 2: Create temporary pod to upload models
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: model-uploader
spec:
  containers:
  - name: uploader
    image: alpine
    command: ["sleep", "3600"]
    volumeMounts:
    - name: models
      mountPath: /models
  volumes:
  - name: models
    persistentVolumeClaim:
      claimName: router-models-pvc
EOF

# Step 3: Wait for pod and copy models
kubectl wait --for=condition=ready pod/model-uploader --timeout=60s
kubectl cp routers/ model-uploader:/models/

# Step 4: Verify models were copied
kubectl exec model-uploader -- ls -la /models/

# Step 5: Clean up temporary pod
kubectl delete pod model-uploader
```

### Step 7: Install Chart

```bash
# Step 1: Copy and edit configuration
cp values.override.yaml.sample values.override.yaml

# Step 2: Configure to use existing PVC with models
# Edit values.override.yaml and uncomment/modify:
# routerServer:
#   volumes:
#     modelRepository:
#       storage:
#         persistentVolumeClaim:
#           enabled: true
#           existingClaim: "router-models-pvc"

# Step 3: Install chart
helm install llm-router ./deploy/helm/llm-router -f values.override.yaml

# Step 4: Verify deployment
echo "=== Checking all pod status ==="
kubectl get pods -l app.kubernetes.io/name=llm-router

echo "=== Verifying router server models loaded ==="
kubectl logs -l app.kubernetes.io/component=router-server | grep "Successfully loaded"

echo "=== Checking router controller health ==="
kubectl logs -l app.kubernetes.io/component=router-controller | tail -5

echo "=== Checking app status (if enabled) ==="
kubectl get pods -l app.kubernetes.io/component=app

echo "=== Testing component connectivity ==="
kubectl wait --for=condition=ready pod -l app.kubernetes.io/component=router-server --timeout=300s
kubectl wait --for=condition=ready pod -l app.kubernetes.io/component=router-controller --timeout=60s
```

### Step 8: Access Services

```bash
# Router Controller API
kubectl port-forward svc/llm-router-router-controller 8084:8084

# Demo App (if enabled)
kubectl port-forward svc/llm-router-app 8008:8008

# Test the API
curl http://localhost:8084/health
```

## Chart Parameters & Advanced Options

### Essential Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `global.imageRegistry` | Docker image registry prefix | `""` |
| `global.storageClass` | Storage class for PVCs | `""` |
| `routerServer.modelRepository.path` | Model repository path/URL | `"/model_repository"` |
| `routerController.enabled` | Enable Router Controller | `true` |
| `app.enabled` | Enable Demo App | `false` |
| `ingress.enabled` | Enable ingress | `false` |

**For complete configuration options, see `values.yaml` - it contains all available parameters with comments.**

### Model Repository Options

The chart supports flexible model storage:

**Filesystem Storage (Default):**
```yaml
routerServer:
  modelRepository:
    path: "/model_repository"
  volumes:
    modelRepository:
      enabled: true
      storage:
        persistentVolumeClaim:
          enabled: true
          size: "100Gi"
```

**Cloud Storage:**
```yaml
routerServer:
  modelRepository:
    path: "s3://my-bucket/models/"  # or gs://, as://
  volumes:
    modelRepository:
      enabled: false  # No volume needed
```

**See `values.yaml` for all storage options: NFS, existing PVCs, different storage classes, etc.**


### Security & Production Features

The chart includes production-ready defaults:
- Non-root containers with minimal capabilities
- Health checks for all components  
- Multi-replica deployments with anti-affinity
- Resource limits and requests
- Secure volume management

**All security settings are configurable in `values.yaml` under `securityContext`, `healthChecks`, `resources`, etc.**

## Ingress Setup

```yaml
# values.override.yaml
ingress:
  enabled: true
  hosts:
    - host: llm-router.local
```

Add to `/etc/hosts`:
```bash
echo "127.0.0.1 llm-router.local" | sudo tee -a /etc/hosts
``` 

Access: http://llm-router.local/router-controller/, http://llm-router.local/app/

## Troubleshooting

**Router Server issues (models not loading):**
```bash
# Check for errors in router server logs
kubectl logs -l app.kubernetes.io/component=router-server | grep -i error

# Verify models are present in the mounted volume
kubectl exec -l app.kubernetes.io/component=router-server -- ls -la /model_repository/

# Check GPU resources
kubectl describe pod -l app.kubernetes.io/component=router-server | grep -A 5 "nvidia.com/gpu"
```

**Router Controller issues (API errors):**
```bash
# Check controller logs
kubectl logs -l app.kubernetes.io/component=router-controller

# Test controller health
kubectl exec -l app.kubernetes.io/component=router-controller -- curl -s http://localhost:8084/health

# Check API keys secret
kubectl get secret llm-api-keys -o yaml
```

**App issues (web interface not accessible):**
```bash
# Check app logs
kubectl logs -l app.kubernetes.io/component=app

# Test app health (should return 200)
kubectl exec -l app.kubernetes.io/component=app -- curl -s -o /dev/null -w "%{http_code}" http://localhost:8008/

# Verify app configuration
kubectl exec -l app.kubernetes.io/component=app -- cat /app/config.yaml
```

**General debugging:**
```bash
# Check all pod events
kubectl get events --sort-by=.metadata.creationTimestamp

# Check resource usage
kubectl top pods -l app.kubernetes.io/name=llm-router

# Test connectivity between components
kubectl exec -l app.kubernetes.io/component=router-controller -- curl -s http://llm-router-router-server:8000/v2/health/ready
```

**Storage issues:**
```bash
kubectl get pvc
kubectl describe pvc <pvc-name>
```

## Components

- **Router Server**: Triton Server hosting router models (requires GPU)
- **Router Controller**: API server for routing requests  
- **Demo App**: Sample web interface (optional)

For detailed configuration, advanced scenarios, and all available options, see the heavily commented `values.yaml` file. 