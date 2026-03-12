# Adapters Guide

Adapters bridge the routing engine (`BaseRouter` → `RoutingResult`) with platform-specific interfaces. The core routing engine has no framework dependencies — adapters bring their own.

## Part 1: Using Bundled Adapters

### LiteLLM Adapter (`adapters/litellm/`)

**Install:** `pip install 'model-router-toolkit[litellm]'`

The LiteLLM adapter provides three integration patterns.

#### Strategy — Embedded SDK

Wraps any `BaseRouter` as a `CustomRoutingStrategyBase` for `litellm.Router`. Use this when routing should happen in your application process.

```python
from litellm import Router
from model_router_toolkit import ModelRoutingStrategy

router = Router(model_list=model_list)
strategy = ModelRoutingStrategy.from_config("configs/prefill-qwen08b.yaml")
strategy.set_litellm_router(router)
router.set_custom_routing_strategy(strategy)

response = await router.acompletion(
    model="nem-think",
    messages=[{"role": "user", "content": "Hello"}],
)
```

Key features:
- Per-request tolerance via `strategy.set_request_tolerance(0.10)` (async-safe, uses contextvars)
- Access routing metadata via `strategy.last_result`
- Automatic fallback to first deployment if routing fails

**Source:** `adapters/litellm/strategy.py`

#### Standalone Server — Full Mode

FastAPI app with routing, inference, playground UI, and auto-review.

```bash
model-router serve --config configs/prefill-qwen08b.yaml --port 8000
```

Endpoints: `/v1/chat/completions`, `/api/chat` (SSE), `/api/models`, `/api/config`, `/api/review`, `/health`, `/` (playground).

Internally: creates a `litellm.Router`, registers `ModelRoutingStrategy`, serves requests.

**Source:** `adapters/litellm/app.py`

#### LiteLLM Proxy — Injection

Starts the full LiteLLM Proxy and injects `ModelRoutingStrategy` at startup.

```bash
model-router proxy \
    --litellm-config configs/litellm-proxy.yaml \
    --router-config configs/prefill-qwen08b.yaml \
    --port 4000
```

The proxy injection hook:
1. Verifies `litellm[proxy]` is installed with compatible version
2. Registers a FastAPI `startup` event
3. At startup, builds `ModelRoutingStrategy` from pool config
4. Patches the proxy's internal `Router.set_custom_routing_strategy()`
5. Runs warmup route

**Source:** `adapters/litellm/proxy.py`

#### Config Bridge

Generates LiteLLM proxy config YAML from pool config:

```python
from model_router_toolkit.adapters.litellm.config_bridge import (
    generate_litellm_config,
    validate_model_alignment,
)

litellm_config = generate_litellm_config("pool_config.yaml", output="litellm.yaml")

warnings = validate_model_alignment("litellm.yaml", "pool_config.yaml")
for w in warnings:
    print(f"Warning: {w}")
```

**Source:** `adapters/litellm/config_bridge.py`

---

### HTTP Adapter (`adapters/http/`)

**Install:** `pip install 'model-router-toolkit[server]'`

Lightweight router-only sidecar. Returns routing decisions without LLM inference. No litellm dependency.

#### Router-Only App

```bash
model-router serve-router --config configs/prefill-qwen08b.yaml --port 8079
```

```bash
curl -X POST http://localhost:8079/v1/route \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain recursion", "tolerance": 0.20}'
```

Endpoints: `/v1/route` (POST), `/api/models` (GET), `/health` (GET).

**Source:** `adapters/http/app.py`, `adapters/http/route.py`

#### Route Request/Response

**Request** (`POST /v1/route`):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `messages` | `list[dict]` | `null` | OpenAI-format messages (uses last user message) |
| `question` | `str` | `null` | Plain text question (alternative to messages) |
| `tolerance` | `float` | `0.20` | Accuracy-cost tradeoff [0.0–1.0] |

**Response**:

| Field | Type | Description |
|-------|------|-------------|
| `selected_model` | `str` | Model chosen by the router |
| `model_names` | `list[str]` | All models in the pool |
| `confidences` | `dict[str, float]` | P(correct) per model |
| `costs` | `list[dict]` | Cost estimates per model |
| `metadata` | `dict` | Routing metadata (`p_max`, `threshold`, `route_ms`) |

#### Webhook Auth Middleware

HMAC-SHA256 and bearer token authentication for enterprise integrations.

```bash
export ROUTER_WEBHOOK_SECRET=my-secret
model-router serve-router --config pool.yaml --port 8079
```

Authentication methods (checked in order):

1. **HMAC-SHA256**: Header `X-Webhook-Signature` with `HMAC(secret, body, SHA256).hexdigest()`
2. **Bearer token**: Header `Authorization: Bearer <secret>`
3. If neither present → 401

Health endpoint (`/health`) is always exempt from auth.

When no secret is configured, all requests pass through.

**Source:** `adapters/http/auth.py`

---

## Part 2: Writing a Custom Adapter

A custom adapter translates between your platform's interface and `BaseRouter.route()`. Here's a step-by-step guide.

### Step 1: Create the adapter directory

```
src/model_router_toolkit/adapters/my_platform/
├── __init__.py
├── app.py          # or strategy.py, hook.py — whatever fits your platform
└── ...
```

### Step 2: Import only the core

Your adapter should depend only on core types:

```python
from model_router_toolkit.config import load_config, build_router_from_config, PoolConfig
from model_router_toolkit.router import BaseRouter, RoutingResult, CostEstimate, extract_user_text
```

### Step 3: Build the router from config

```python
config = load_config("pool_config.yaml")
router = build_router_from_config(config)
```

### Step 4: Call `router.route()`

```python
result: RoutingResult = router.route(question_text, tolerance=0.20)

selected = result.selected_model       # str — model name
confs = result.confidences             # list[float] — P(correct) per model
names = result.model_names             # list[str] — model names (same order)
cost = result.selected_cost            # CostEstimate for selected model
meta = result.metadata                 # dict — p_max, threshold, etc.
```

### Step 5: Map to your platform

Translate `RoutingResult` into whatever your platform expects. Examples:

**FastAPI endpoint:**

```python
from fastapi import FastAPI
from model_router_toolkit.config import load_config, build_router_from_config

app = FastAPI()
config = load_config("pool.yaml")
router = build_router_from_config(config)

@app.post("/route")
async def route(question: str, tolerance: float = 0.20):
    result = router.route(question, tolerance=tolerance)
    return {
        "model": result.selected_model,
        "confidence": result.selected_confidence,
    }
```

**Webhook handler:**

```python
def handle_webhook(payload: dict) -> dict:
    question = payload.get("prompt", "")
    result = router.route(question, tolerance=0.20)
    return {"model_override": result.selected_model}
```

**gRPC service:**

```python
class RouterService(router_pb2_grpc.RouterServiceServicer):
    def Route(self, request, context):
        result = self.router.route(request.question, tolerance=request.tolerance)
        return router_pb2.RouteResponse(
            selected_model=result.selected_model,
            confidences={n: c for n, c in zip(result.model_names, result.confidences)},
        )
```

### Step 6: Add optional dependency in pyproject.toml

```toml
[project.optional-dependencies]
my_platform = [
    "my-platform-sdk>=1.0",
]
```

### Step 7: Add a lazy import guard

In `__init__.py`:

```python
"""My Platform adapter for model-router-toolkit.

Requires: pip install model-router-toolkit[my_platform]
"""

try:
    from my_platform_sdk import SomeClass
except ImportError:
    raise ImportError(
        "my_platform adapter requires my-platform-sdk. "
        "Install with: pip install 'model-router-toolkit[my_platform]'"
    ) from None
```

### Step 8: Test your adapter

```python
import pytest
from unittest.mock import MagicMock
from model_router_toolkit.router import RoutingResult, CostEstimate

def make_mock_result(selected: str = "model-a") -> RoutingResult:
    return RoutingResult(
        model_names=["model-a", "model-b"],
        confidences=[0.9, 0.7],
        costs=[
            CostEstimate(median_output_tokens=100, cost_per_m_input_tokens=0.2,
                        cost_per_m_output_tokens=0.2),
            CostEstimate(median_output_tokens=100, cost_per_m_input_tokens=0.04,
                        cost_per_m_output_tokens=0.16),
        ],
        selected_model=selected,
        metadata={"p_max": 0.9, "threshold": 0.7},
    )

def test_my_adapter():
    mock_router = MagicMock()
    mock_router.route.return_value = make_mock_result("model-b")

    # Test your adapter logic using mock_router
    # ...
```

---

## Part 3: Reference

### RoutingResult

```python
@dataclass
class RoutingResult:
    model_names: list[str]       # All models in the pool
    confidences: list[float]     # P(correct) per model (same order as model_names)
    costs: list[CostEstimate]    # Cost estimates per model (same order)
    selected_model: str          # Model chosen by the router
    metadata: dict[str, Any]     # Routing metadata

    selected_confidence: float   # (property) confidence of the selected model
    selected_cost: CostEstimate  # (property) cost estimate of the selected model
    as_dict() -> dict            # Serializable dict representation
```

### CostEstimate

```python
@dataclass
class CostEstimate:
    median_output_tokens: int          # Median output tokens (from training data)
    cost_per_m_input_tokens: float     # USD per million input tokens
    cost_per_m_output_tokens: float    # USD per million output tokens
    estimated_input_tokens: int        # Estimated input tokens for this request
    estimated_output_cost: float       # Estimated output cost
    estimated_input_cost: float        # Estimated input cost
    estimated_total_cost: float        # estimated_input_cost + estimated_output_cost
```

### extract_user_text

```python
def extract_user_text(messages: list[dict[str, str]] | None) -> str:
```

Extracts the last user message from an OpenAI-format message list. Handles both plain string `content` and multipart content arrays (filters for `type: "text"` parts). Returns empty string if no user message found.

### BaseRouter

```python
class BaseRouter(ABC):
    @abstractmethod
    def route(self, question: str, *, tolerance: float = 0.20) -> RoutingResult:
        """Score all models and select the cheapest above threshold."""

    @abstractmethod
    def load(self, checkpoint_path: str | Path) -> None:
        """Load a trained checkpoint."""

    def unload(self) -> None:
        """Release resources (GPU memory, model weights). Optional."""
```

### Error Handling

The router may raise exceptions during:

| Scenario | Exception | Recommended handling |
|----------|-----------|---------------------|
| Missing checkpoint file | `FileNotFoundError` | Fail loudly at startup |
| Corrupt checkpoint | `RuntimeError` | Fail loudly at startup |
| Encoder model not found | `OSError` (HuggingFace) | Fail loudly at startup (first download) |
| Encoder OOM | `torch.cuda.OutOfMemoryError` | Fall back to CPU or return default model |
| Empty/missing question text | Returns valid result | Router uses first model as fallback |

Adapters should catch exceptions from `router.route()` and return a sensible fallback (e.g., first model in pool, or the platform's default). The OpenClaw plugin demonstrates this pattern — returning `{}` on any error lets the gateway fall back to its default.
