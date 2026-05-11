# NVIDIA LLM Router - OpenShift Deployment Guide

This guide explains how to deploy the NVIDIA LLM Router on OpenShift using the provided Helm chart with OpenShift-specific configurations.

## Prerequisites

### Common Requirements (All Deployments)
- **OpenShift Cluster**: 4.10+ with minimum 3 worker nodes (4 vCPU, 16GB RAM each)
- **Helm**: 3.2.0+ installed and configured
- **NVIDIA API Key**: From [NVIDIA API Catalog](https://build.nvidia.com/settings/api-keys)
  - Format: `nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` (47 characters)
  - Required scopes: Model access for routing
- **NGC API Key**: From [NGC Portal](https://org.ngc.nvidia.com/setup/api-keys) 
  - Format: 40+ alphanumeric characters
  - Required for: Container registry access and model downloads
- **Container Images**: Pre-built images for router-controller and app components
  - Must be pushed to accessible container registry
  - Architecture: linux/amd64 (OpenShift standard)
- **Network Access**: Outbound HTTPS to `api.nvidia.com`, `build.nvidia.com`, `api.ngc.nvidia.com`

### Additional Requirements (Triton Routing Only)
- **GPU Hardware**: Nodes with 16GB+ VRAM (Tesla T4, V100, A10, A100, etc.)
- **NVIDIA GPU Operator**: Installed and validated on OpenShift cluster
- **Storage**: Persistent storage for model repository. ReadWriteMany only if using a separate download pod (see [Triton Model Setup Guide](openshift-triton-model-setup.md))
- **Network**: Egress access to api.ngc.nvidia.com and build.nvidia.com
- **Container Images**: router-server component built for your architecture
- **GPU Node Configuration**: Tolerations for tainted GPU nodes (if applicable)


## Routing Strategy Overview

The NVIDIA LLM Router supports two routing strategies. For detailed architecture and component information, see the [main project documentation](../README.md#software-components).

### Manual Routing
- **Direct policy-based routing** to NVIDIA's hosted models via Build API
- **Requirements**: Standard OpenShift nodes, NVIDIA API key
- **Use case**: Production deployments, immediate functionality

### Triton Routing  
- **ML-based routing decisions** using local classification models
- **Requirements**: GPU nodes (16GB+ VRAM), NGC model downloads
- **Use case**: Custom routing logic, research environments

> **Recommendation**: Start with manual routing for immediate functionality. See [NVIDIA's routing strategy documentation](https://docs.nvidia.com/nim/llm-router/latest/routing.html) for detailed comparison.

## Deployment Steps

### 1. Create Required Secrets

```bash
# Create namespace (replace 'your-namespace' with your preferred name)
oc new-project your-namespace

# Set and validate environment variables
export NVIDIA_API_KEY="nvapi-YOUR-NVIDIA-API-KEY"
export NGC_API_KEY="YOUR-NGC-API-KEY"

# Validate NVIDIA API key format (should start with 'nvapi-' and be ~47 characters)
if [[ ! "$NVIDIA_API_KEY" =~ ^nvapi-.{40,}$ ]]; then
  echo "❌ Error: Invalid NVIDIA API key format. Should start with 'nvapi-' followed by 40+ characters"
  echo "Get your key from: https://build.nvidia.com/settings/api-keys"
  exit 1
fi

# Validate NGC API key format (should be alphanumeric, 40+ characters)
if [[ ! "$NGC_API_KEY" =~ ^[A-Za-z0-9]{40,}$ ]]; then
  echo "❌ Error: Invalid NGC API key format. Should be 40+ alphanumeric characters"
  echo "Get your key from: https://org.ngc.nvidia.com/setup/api-keys"
  exit 1
fi

echo "✅ API keys validated successfully"

# Create NVIDIA API key secret (required for all deployments)
oc create secret generic llm-api-keys \
  --from-literal=nvidia_api_key="$NVIDIA_API_KEY"

# Create NGC registry secret for pulling container images from nvcr.io
oc create secret docker-registry nvcr-secret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password="$NGC_API_KEY"

# Create NGC CLI secret for model downloads (required for Triton routing only)
oc create secret generic ngc-cli-key \
  --from-literal=ngc_cli_api_key="$NGC_API_KEY"
```

### 2. Choose Your Deployment Path

#### **Option A: Manual Routing (Recommended)**
**Cost-effective deployment with immediate functionality**

```bash
# Manual routing deployment (no GPU required)
helm install llm-router ./deploy/helm/llm-router \
  -f ./deploy/helm/llm-router/openshift-values.yaml \
  --set app.enabled=true \
  --set routerServer.enabled=false \
  --set global.imageRegistry=your-registry.com/namespace/ \
  --set routerController.image.tag=your-tag \
  --set app.image.tag=your-tag
```

> **OpenShift values file**: The `openshift-values.yaml` file configures all OpenShift-specific settings:
> - **Storage class** (`gp3-csi` by default)
> - **OpenShift Routes** instead of Kubernetes Ingress  
> - **Security contexts** compliant with `restricted-v2` SCC
> - **HuggingFace cache** redirection for proper permissions
> - **GPU tolerations** for tainted GPU nodes
>
> Review and adjust `openshift-values.yaml` for your cluster before deploying.

#### **Option B: Triton Routing (Advanced)**
**GPU deployment with ML-based routing**

```bash
# Triton routing deployment (GPU required)
helm install llm-router ./deploy/helm/llm-router \
  -f ./deploy/helm/llm-router/openshift-values.yaml \
  --set app.enabled=true \
  --set routerServer.enabled=true \
  --set global.imageRegistry=your-registry.com/namespace/ \
  --set routerServer.image.tag=your-tag \
  --set routerController.image.tag=your-tag \
  --set app.image.tag=your-tag
```

> **Important Notes**:
> - **Manual routing** provides complete LLM routing functionality without GPU infrastructure
> - **Triton routing** requires GPU nodes and additional model setup (see section 4)
> - **Cache and tolerations** are pre-configured in `openshift-values.yaml`
> - **Cost consideration**: Manual routing uses standard nodes, Triton routing requires expensive GPU nodes

### 3. Access the Application

```bash
# Get the route URLs
oc get routes

# Access the demo UI
https://llm-router-app-<your-namespace>.<cluster-domain>/
```

**Using the Application**:
- **Manual routing**: Works immediately - select "manual" routing strategy in the UI
- **Triton routing**: Requires additional model setup (see section 4) before using "triton" routing strategy

### 4. Triton Model Setup (Required for Triton Routing)

By default, the Triton routing deployment (`routerServer.enabled=true`) creates a **ReadWriteOnce** PVC and starts the router-server with an empty model repository at `/model_repository`. The router-server (Triton Inference Server) will start successfully but will have **no models loaded** until you populate the model repository.

**You must load models before Triton routing will function.** For detailed instructions on downloading and loading NGC models into the router-server, see:

> **[Triton Model Setup Guide](openshift-triton-model-setup.md)**

That guide covers three approaches:
- **Option A: Download Pod with RWX PVC** -- Uses a separate Kubernetes pod to download models from NGC into a shared ReadWriteMany PVC
- **Option B: Custom Docker Image with Runtime Download** -- Builds a custom router-server image that downloads models at container startup using a standard ReadWriteOnce PVC
- **Option C: MinIO + Init Container (Recommended)** -- Stores models in MinIO and syncs them to the PVC via an init container at pod startup

Until models are loaded, the Triton routing strategy will not work.

### 5. Routing to Self-Hosted Models (Optional)

By default, the LLM Router routes to NVIDIA's hosted Build API. If you have your own LLM inference endpoints running on the cluster — for example, models deployed via **Red Hat OpenShift AI (RHOAI)** model serving, KServe InferenceServices, or standalone vLLM instances — you can configure the router to use those instead.

> **[Routing to Local OpenAI-Compatible Endpoints](routing-openshift-ai-models.md)**

That guide covers:
- Configuring the router-controller with custom model endpoints
- Manual routing (client selects the model) and Triton routing (automatic classification)
- Authentication with OpenShift OAuth bearer tokens
- Mixing local and remote (NVIDIA Build API) endpoints in the same deployment


## Architecture and Routing Strategies

The NVIDIA LLM Router supports two routing strategies. For detailed information about the architecture and components, see the main [README.md](../README.md#software-components).

### Key Differences for OpenShift Deployment

| Aspect | Manual Routing | Triton Routing |
|--------|----------------|----------------|
| **Components** | router-controller only | router-controller + router-server |
| **Infrastructure** | Standard nodes | GPU nodes (16GB+ VRAM) |
| **Setup Complexity** | Simple | Complex (model downloads) |
| **Cost** | Low | High (GPU instances) |
| **OpenShift Compatibility** | Excellent | Requires model setup (see [Model Setup Guide](openshift-triton-model-setup.md)) |

> **For complete architecture details and configuration options**, refer to:
> - [Main Documentation](../README.md) - Overall architecture and components
> - [Router Controller README](../src/router-controller/readme.md) - API and configuration details

## OpenShift-Specific Features

When `openshift.enabled: true`, the chart automatically configures:

### Security Context
- **Pod Level**: `runAsNonRoot: true`, `seccompProfile: RuntimeDefault`
- **Container Level**: `allowPrivilegeEscalation: false`, dropped capabilities
- **No hardcoded UIDs**: Lets OpenShift assign UIDs dynamically
- **SCC Compliance**: Works with `restricted-v2` SCC without cluster-admin privileges

### Networking
- **OpenShift Routes**: Created instead of standard Kubernetes Ingress
- **Edge TLS**: Automatic HTTPS termination
- **Path-based routing**: Multiple services accessible via different paths

### Storage
- **Default Storage Class**: Uses `gp3-csi` for OpenShift environments
- **PVC Creation**: Automatic persistent volume provisioning for model storage
- **Default Access Mode**: ReadWriteOnce (RWO). Override to ReadWriteMany if using download pods (see [Triton Model Setup Guide](openshift-triton-model-setup.md))
- **RHOAI Integration**: Compatible with RHOAI workbench shared storage

### RBAC
- **ServiceAccount**: Dedicated service account with minimal permissions
- **Role/RoleBinding**: Basic permissions for ConfigMap and Secret access

## Configuration Options

### OpenShift Settings

All OpenShift-specific settings are in `openshift-values.yaml`. Review and adjust before deploying:

```yaml
# openshift-values.yaml (key sections)
openshift:
  enabled: true
  storageClass: "gp3-csi"         # Adjust for your cluster
  routes:
    enabled: true
    router:
      host: ""                    # Auto-generated if empty
    controller:
      host: ""
    app:
      host: ""

# GPU tolerations (adjust key/value to match your cluster)
tolerations:
  - key: "g5-gpu"
    operator: Equal
    value: "true"
    effect: NoSchedule
```

For Router Server storage (Triton Routing), override via `--set` or a separate values file:
```yaml
routerServer:
  volumes:
    modelRepository:
      storage:
        persistentVolumeClaim:
          storageClass: ""        # Uses openshift.storageClass via helper when empty
          accessMode: ReadWriteOnce  # Override to ReadWriteMany for download pod approach
          size: "100Gi"           # Sufficient for NGC models (~1.5GB total)
```

### Service Configuration

```yaml
# Enable/disable components
routerServer:
  enabled: true                    # Triton-based routing server (requires GPU)
  
routerController:
  enabled: true                    # API orchestration service
  
app:
  enabled: true                    # Optional Gradio demo UI
```

### Cache Configuration (Required for Triton Routing Only)

**Note**: This configuration is only needed when `routerServer.enabled=true` (Triton routing).

OpenShift security contexts prevent writing to the root filesystem, which causes HuggingFace cache permission errors in the router-server component. The `openshift-values.yaml` file already includes the required cache configuration:

- HuggingFace cache env vars redirected to `/tmp/hf_cache`
- An `emptyDir` volume mounted at `/tmp/hf_cache`

No additional `--set` flags are needed when deploying with `-f openshift-values.yaml`.

## Resource Requirements

### Manual Routing Deployment
| Component | CPU | Memory | GPU | Storage |
|-----------|-----|--------|-----|---------|
| router-controller | 0.5-1 core | 2Gi | - | - |
| app (demo) | 0.1-0.5 core | 1Gi | - | - |
| **Total** | **0.6-1.5 cores** | **3Gi** | **None** | **Standard PVC** |

### Triton Routing Deployment  
| Component | CPU | Memory | GPU | Storage |
|-----------|-----|--------|-----|---------|
| router-server | 2-4 cores | 8Gi | 1x NVIDIA GPU (16GB+ VRAM) | - |
| router-controller | 0.5-1 core | 2Gi | - | - |
| app (demo) | 0.1-0.5 core | 1Gi | - | - |
| model-storage | - | - | - | 100Gi PVC |
| **Total** | **2.6-5.5 cores** | **11Gi** | **1x GPU** | **100Gi+ PVC** |

> **Cost Impact**: GPU instances typically cost 10-50x more than standard instances in cloud environments.

## Troubleshooting

### Pod Security Issues
If pods fail with security context errors:

```bash
# Check SCC assignments
oc describe pod <pod-name>

# Verify service account
oc get serviceaccount llm-router
```

### Route Access Issues
If external access doesn't work:

```bash
# Check route status
oc get routes
oc describe route llm-router-app

# Test internal connectivity
oc port-forward service/llm-router-app 8008:8008
curl http://localhost:8008
```

### GPU Allocation Issues
If router-server pods are pending:

```bash
# Check GPU availability
oc describe node <gpu-node>

# Verify GPU operator
oc get pods -n nvidia-gpu-operator

# Check resource requests
oc describe pod <router-server-pod>
```

#### GPU Node Taints
Many OpenShift GPU nodes have taints that prevent non-GPU workloads from scheduling. If you see errors like:

```
6 node(s) had untolerated taint {g5-gpu: true}
```

The `openshift-values.yaml` file includes a `tolerations` section. Edit it to match your cluster's GPU node taints:

```yaml
# In openshift-values.yaml
tolerations:
  - key: "g5-gpu"
    operator: Equal
    value: "true"
    effect: NoSchedule
```

> **Note**: Tolerations are applied at the top level (`.Values.tolerations`) and affect all components. GPU tolerations are only needed for Triton routing deployments.

Check node taints:
```bash
# List GPU nodes and their taints
oc get nodes -l nvidia.com/gpu.present=true -o custom-columns=NAME:.metadata.name,TAINTS:.spec.taints

# Check specific node
oc describe node <gpu-node-name> | grep Taints
```

### Model Repository Issues
If Triton routing fails to load models:

```bash
# Verify repository structure
oc exec deployment/llm-router-router-server -- find /model_repository -type f | head -20

# Check for actual model files (.pt files) - these are large (700MB+)
oc exec deployment/llm-router-router-server -- find /model_repository -name "*.pt" -exec ls -lh {} \;

# Check Triton server logs for detailed loading errors  
oc logs deployment/llm-router-router-server | grep -E "model|error|fail"
```

**Common Issues:**
1. **Missing model files** - Repository contains only template structure
2. **Corrupted transfers** - Large model files corrupted during pod-to-pod copy
3. **Cache permission errors** - HuggingFace models fail due to OpenShift security contexts
4. **Insufficient storage** - PVC too small for 700MB+ model files

**Solutions:**
- Ensure cache configuration is applied (see Cache Configuration section)
- See the [Triton Model Setup Guide](openshift-triton-model-setup.md) for model download options
- Verify PVC has sufficient space (recommend 100Gi+)
- Consider manual routing as alternative

### Routing Strategy Issues

If routing doesn't work as expected:

#### **Manual Routing Issues**
```bash
# Check NVIDIA API key is set
oc get secret llm-api-keys -o yaml
oc logs deployment/llm-router-router-controller | grep -i "api"

# Test API connectivity from inside cluster
oc exec <router-controller-pod> -- curl -s https://api.nvidia.com/v1/models \
  -H "Authorization: Bearer ${NVIDIA_API_KEY}"

# Verify router controller is accessible
curl https://<router-controller-route>/v1/models
```

**Common Solutions:**
- Ensure NVIDIA API key is valid and has model access permissions
- Check network connectivity to api.nvidia.com
- Verify router-controller service is accessible via OpenShift route

#### **Triton Routing Issues**  
If using Triton routing and seeing "Request for unknown model" errors:

```bash
# Check if models are loaded in Triton
curl https://<router-server-route>/v2/models

# Check Triton server logs
oc logs deployment/llm-router-router-server | grep -E "model|loaded|failed"
```

**Common Solutions:**
- Verify NGC model files were transferred successfully (see model download section)
- Ensure cache configuration is applied (see Cache Configuration section)
- Check PVC has sufficient space for model files
- Use manual routing as alternative - provides same functionality

> **Recommendation**: For production deployments, manual routing provides reliable functionality without the complexity of model file management.

### Image Pull Issues
If images fail to pull:

```bash
# Verify NGC secret for base images
oc get secret nvcr-secret
oc describe secret nvcr-secret

# Check custom registry access
oc describe pod <pod-name>

# Verify image tags match your built images
helm template . --set global.imageRegistry=your-registry.com/
```

## GPU Configuration

### Node Selection and Tolerations

For clusters with GPU node taints, configure tolerations in your values:

```yaml
# values.yaml or separate tolerations file
tolerations:
  - key: g5-gpu                    # Common AWS g5 instance taint
    operator: Equal
    value: "true"
    effect: NoSchedule
  - key: nvidia.com/gpu           # Alternative GPU taint
    operator: Exists
    effect: NoSchedule

# Optional: Force scheduling only on GPU nodes
routerServer:
  nodeSelector:
    nvidia.com/gpu.present: "true"
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: nvidia.com/gpu.present
            operator: In
            values: ["true"]
```

### Resource Limits

Adjust GPU and memory requirements based on your models:

```yaml
# values.yaml
routerServer:
  resources:
    limits:
      nvidia.com/gpu: 1           # Number of GPUs required
      memory: "16Gi"              # Memory for large models
      cpu: "4"
    requests:
      nvidia.com/gpu: 1
      memory: "8Gi"
      cpu: "2"
```

## Image Building

For OpenShift deployment, you need to build container images:

### Images Required by Deployment Type

**Manual Routing**:
```bash
# Build router-controller image  
podman build --platform=linux/amd64 \
  -f src/router-controller/router-controller.dockerfile \
  -t your-registry.com/router-controller:your-tag .
podman push your-registry.com/router-controller:your-tag

# Build demo app image
podman build --platform=linux/amd64 \
  -f demo/app/app.dockerfile \
  -t your-registry.com/llm-router-client:your-tag .
podman push your-registry.com/llm-router-client:your-tag
```

**Triton Routing** (all manual routing images plus):
```bash
# Build router-server image (GPU support required)
podman build --platform=linux/amd64 \
  -f src/router-server/router-server.dockerfile \
  -t your-registry.com/router-server:your-tag .
podman push your-registry.com/router-server:your-tag
```

> **Architecture Note**: OpenShift typically runs on x86_64, so ensure images are built with `--platform=linux/amd64`.

## Advanced Configuration

### Custom Storage Classes

```yaml
# values.yaml
openshift:
  storageClass: "fast-ssd"        # Use custom storage class

# Or disable OpenShift auto-selection
routerServer:
  volumes:
    modelRepository:
      storage:
        persistentVolumeClaim:
          storageClass: "my-custom-class"
```

### Custom Hostnames

```yaml
# values.yaml
openshift:
  routes:
    app:
      host: "llm-demo.apps.your-cluster.com"
    controller:
      host: "llm-api.apps.your-cluster.com"
```

### Resource Limits

```yaml
# values.yaml
routerServer:
  resources:
    limits:
      nvidia.com/gpu: 1
      memory: "16Gi"
      cpu: "4"
    requests:
      nvidia.com/gpu: 1
      memory: "8Gi" 
      cpu: "2"
```

## Monitoring

### Health Checks
All components include health checks:

```bash
# Check component health
oc get pods
oc logs <pod-name>

# Test endpoints
curl https://<route-url>/health
```

### Resource Usage

```bash
# Monitor resource usage
oc top pods
oc top nodes

# GPU utilization (if available)
oc exec <router-server-pod> -- nvidia-smi
```

## Uninstall

```bash
# Remove Helm release
helm uninstall llm-router

# Clean up secrets (optional)
oc delete secret llm-api-keys nvcr-secret

# Clean up PVCs (optional - will delete model data)
oc delete pvc --selector=app.kubernetes.io/name=llm-router
```

## Support

For issues specific to:
- **OpenShift deployment**: Check this guide's troubleshooting section
- **NVIDIA GPU support**: Consult [NVIDIA GPU Operator documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/)
- **LLM Router functionality**: See main project [README](../README.md)

## Version Compatibility

| LLM Router | OpenShift | NVIDIA GPU Operator | Notes |
|------------|-----------|-------------------|-------|
| 0.1.0+ | 4.10+ | 23.6.1+ | Initial OpenShift support |

Last updated: May 2026