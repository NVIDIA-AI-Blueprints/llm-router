# Extending Guide

How to add custom routing methods, write new bundled adapters, and contribute to the project.

## Part 1: Custom Routing Methods

All routing methods implement the `BaseRouter` abstract class. The toolkit includes `KMeansRouter` and `PrefillRouter`; you can add your own.

### Step 1: Subclass BaseRouter

```python
# src/model_router_toolkit/my_method/router.py

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_router_toolkit.config import PoolConfig
from model_router_toolkit.router import BaseRouter, CostEstimate, RoutingResult


class MyRouter(BaseRouter):
    """Custom routing method using <your approach>."""

    def __init__(self, config: PoolConfig):
        self._config = config
        self._checkpoint_data: dict[str, Any] | None = None

    def route(self, question: str, *, tolerance: float = 0.20) -> RoutingResult:
        models = self._config.models
        model_names = [m.name for m in models]

        # --- Your scoring logic ---
        # Compute P(correct) for each model given the question
        confidences = self._score(question)

        # --- Cost estimation ---
        costs = [
            CostEstimate(
                median_output_tokens=150,
                cost_per_m_input_tokens=m.cost_per_m_input_tokens,
                cost_per_m_output_tokens=m.cost_per_m_output_tokens,
            )
            for m in models
        ]

        # --- Selection (standard tolerance-based) ---
        p_max = max(confidences)
        threshold = p_max - tolerance
        cheapest_above = None
        cheapest_cost = float("inf")

        for i, (name, conf, cost) in enumerate(zip(model_names, confidences, costs)):
            if conf >= threshold and cost.estimated_total_cost < cheapest_cost:
                cheapest_above = name
                cheapest_cost = cost.estimated_total_cost

        selected = cheapest_above or model_names[0]

        return RoutingResult(
            model_names=model_names,
            confidences=confidences,
            costs=costs,
            selected_model=selected,
            metadata={"p_max": p_max, "threshold": threshold},
        )

    def _score(self, question: str) -> list[float]:
        """Compute P(correct) for each model. Override with your logic."""
        raise NotImplementedError

    def load(self, checkpoint_path: str | Path) -> None:
        # Load your trained model/checkpoint
        import pickle
        with open(checkpoint_path, "rb") as f:
            self._checkpoint_data = pickle.load(f)

    def unload(self) -> None:
        self._checkpoint_data = None
```

### Step 2: Register in config dispatch

Add your method to `build_router_from_config()` in `config.py`:

```python
def build_router_from_config(config: PoolConfig):
    method = config.routing.method.lower()

    if method == "kmeans":
        from model_router_toolkit.kmeans.router import KMeansRouter
        router = KMeansRouter(config=config)
    elif method == "prefill":
        from model_router_toolkit.prefill.router import PrefillRouter
        router = PrefillRouter(config=config)
    elif method == "my_method":
        from model_router_toolkit.my_method.router import MyRouter
        router = MyRouter(config=config)
    else:
        raise ValueError(f"Unknown routing method: {method!r}")

    if config.routing.checkpoint:
        router.load(config.routing.checkpoint)
    return router
```

Now `method: my_method` in the YAML config will use your router.

### Step 3: Add a training pipeline (optional)

If your method needs training, add a training module:

```python
# src/model_router_toolkit/my_method/train.py

from pathlib import Path
from model_router_toolkit.config import PoolConfig


def train_my_method(
    config: PoolConfig,
    data_path: str | Path,
    output_dir: str | Path,
    **kwargs,
) -> Path:
    """Train a custom routing checkpoint.

    Args:
        config: Pool configuration
        data_path: Path to training CSV (question, model, isCorrect, output_tokens)
        output_dir: Directory to save checkpoint

    Returns:
        Path to saved checkpoint
    """
    # Load data
    import pandas as pd
    df = pd.read_csv(data_path)

    # Your training logic ...

    # Save checkpoint
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "my_method.pkl"

    import pickle
    with open(checkpoint_path, "wb") as f:
        pickle.dump({"weights": ..., "config": config.model_dump()}, f)

    return checkpoint_path
```

Register in `train.py` (the unified training dispatcher):

```python
def train(config, data_path, output_dir, **kwargs):
    method = config.routing.method.lower()
    if method == "prefill":
        from model_router_toolkit.prefill.train import train_prefill
        return train_prefill(config, data_path, output_dir, **kwargs)
    elif method == "my_method":
        from model_router_toolkit.my_method.train import train_my_method
        return train_my_method(config, data_path, output_dir, **kwargs)
    ...
```

### Step 4: Add evaluation (optional)

```python
# src/model_router_toolkit/my_method/evaluate.py

def evaluate_my_method(config, checkpoint_path, data_path, **kwargs):
    """Evaluate a custom routing checkpoint against labeled data."""
    ...
```

### Step 5: Add a lazy import in `__init__.py`

```python
# In __init__.py __getattr__
if name == "MyRouter":
    from model_router_toolkit.my_method.router import MyRouter
    return MyRouter
```

### Routing Method Checklist

- [ ] Subclass `BaseRouter` with `route()`, `load()`, `unload()`
- [ ] `route()` returns `RoutingResult` with all fields populated
- [ ] `load()` loads from a checkpoint file
- [ ] Registered in `build_router_from_config()` dispatch
- [ ] Config-driven — no hardcoded model names or paths
- [ ] Tolerance parameter respected (p_max - tolerance threshold)
- [ ] Cost estimates populated from config `ModelSpec`
- [ ] (Optional) Training pipeline in `my_method/train.py`
- [ ] (Optional) Evaluation in `my_method/evaluate.py`
- [ ] (Optional) Lazy import in `__init__.py`
- [ ] Tests in `tests/test_my_method.py`

---

## Part 2: Adding a Bundled Adapter

Bundled adapters live in `src/model_router_toolkit/adapters/` and are included in the package distribution.

### Directory Structure

```
src/model_router_toolkit/adapters/my_platform/
├── __init__.py          # Docstring, optional public imports
├── app.py               # Main entry point (server, hook, or strategy)
├── ...                  # Additional modules as needed
```

### Step 1: Create the adapter package

```python
# adapters/my_platform/__init__.py
"""My Platform adapter — brief description.

Requires: pip install model-router-toolkit[my_platform]
"""
```

### Step 2: Add optional dependencies

In `pyproject.toml`:

```toml
[project.optional-dependencies]
my_platform = [
    "my-platform-sdk>=1.0",
]
all = [
    "model-router-toolkit[prefill,litellm,server,training,proxy,my_platform,dev]",
]
```

### Step 3: Use shared utilities

Reuse helpers from `adapters/http/_shared.py` where applicable:

```python
from model_router_toolkit.adapters.http._shared import warmup_router, health_dict, models_list
```

### Step 4: Follow the import guard pattern

Check that platform-specific deps are installed:

```python
try:
    import my_platform_sdk
except ImportError:
    raise ImportError(
        "my_platform adapter requires my-platform-sdk. "
        "Install with: pip install 'model-router-toolkit[my_platform]'"
    ) from None
```

### Step 5: Add CLI integration (if applicable)

If your adapter needs a CLI command, add it to `__main__.py`:

```python
def _cmd_my_platform(args):
    from model_router_toolkit.adapters.my_platform.app import start
    start(config=args.config, port=args.port)
```

### Step 6: Write tests

```
tests/
├── test_my_platform_adapter.py      # Unit tests (mock the platform SDK)
└── integration/
    └── test_my_platform_live.py     # Integration tests (needs running platform)
```

Unit test example:

```python
import pytest
from unittest.mock import MagicMock, patch

from model_router_toolkit.router import RoutingResult, CostEstimate


def _mock_result() -> RoutingResult:
    return RoutingResult(
        model_names=["a", "b"],
        confidences=[0.9, 0.7],
        costs=[
            CostEstimate(median_output_tokens=100,
                        cost_per_m_input_tokens=0.2, cost_per_m_output_tokens=0.2),
            CostEstimate(median_output_tokens=100,
                        cost_per_m_input_tokens=0.04, cost_per_m_output_tokens=0.16),
        ],
        selected_model="b",
    )


def test_adapter_maps_result():
    from model_router_toolkit.adapters.my_platform.app import map_result
    result = _mock_result()
    output = map_result(result)
    assert output["model"] == "b"
```

### Step 7: Add documentation

Add a section to `docs/adapters.md` under Part 1 and update the topology table in `docs/integration.md`.

### Bundled Adapter Checklist

- [ ] Directory under `adapters/`
- [ ] `__init__.py` with docstring and install instructions
- [ ] Optional deps in `pyproject.toml`
- [ ] Import guards for optional deps
- [ ] Uses `BaseRouter.route()` for routing (no direct scorer access)
- [ ] Reuses shared helpers where applicable
- [ ] Handles errors gracefully (returns sensible default on failure)
- [ ] Unit tests with mocked router
- [ ] Integration test (if applicable)
- [ ] Documentation added to `docs/adapters.md`
- [ ] Topology table updated in `docs/integration.md`

---

## Part 3: Contributing Guide

### Development Setup

```bash
git clone https://github.com/NVIDIA/model-router-toolkit.git
cd model-router-toolkit
pip install -e '.[all]'
```

### Code Style

| Tool | Config | Purpose |
|------|--------|---------|
| **ruff** | `pyproject.toml` [tool.ruff] | Linting and import sorting |
| **mypy** | `pyproject.toml` [tool.mypy] | Type checking |
| **Line length** | 100 | Configured in ruff |
| **Target** | Python 3.10 | Use `X \| Y` union syntax |

Run checks:

```bash
ruff check src/ tests/
ruff format src/ tests/
mypy src/
```

### Testing

```bash
# Unit tests (no API keys, no checkpoints)
pytest tests/ --ignore=tests/integration/ -v

# Integration tests (mocked, no API keys needed)
pytest tests/integration/ -v

# All tests with coverage
pytest tests/ -v --cov=model_router_toolkit --cov-report=term-missing
```

Test markers:

| Marker | Meaning |
|--------|---------|
| `@pytest.mark.requires_torch` | Needs PyTorch installed |
| `@pytest.mark.requires_encoder` | Needs Qwen encoder cached |
| `@pytest.mark.requires_nvidia_api_key` | Needs `NVIDIA_API_KEY` |
| `@pytest.mark.requires_openrouter_api_key` | Needs `OPENROUTER_API_KEY` |
| `@pytest.mark.slow` | Slow test (encoder or API) |

### Project Conventions

1. **Config-driven dispatch**: Routing method is determined by `routing.method` in YAML, never by code branching on model names
2. **BaseRouter abstraction**: All inference paths go through `BaseRouter.route()`. Training and evaluation may bypass it for batch efficiency.
3. **Lazy imports**: Heavy dependencies (torch, litellm, transformers) are imported inside functions, not at module level
4. **Optional extras**: Platform-specific deps go in `[project.optional-dependencies]`, never in core `dependencies`
5. **Graceful degradation**: Adapters and plugins handle errors and fall back to sensible defaults

### PR Checklist

- [ ] Code passes `ruff check` and `ruff format --check`
- [ ] Code passes `mypy src/`
- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] New functionality has tests
- [ ] Documentation updated (relevant docs in `docs/`)
- [ ] `AGENTS.md` updated if project structure changed
- [ ] No API keys or secrets in code
- [ ] No hardcoded model names or paths
- [ ] Lazy imports for optional dependencies
- [ ] Import guards with helpful error messages

### Commit Message Format

```
<type>: <short description>

<optional body with details>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

Examples:

```
feat: add gRPC adapter for model routing

Adds adapters/grpc/ with a unary Route RPC that wraps BaseRouter.
Requires new [grpc] optional extra (grpcio, grpcio-tools).
```

```
fix: handle empty messages in extract_user_text

Previously returned the system message content when no user
messages were present. Now correctly returns empty string.
```
