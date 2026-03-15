import os
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

requires_nvidia_api_key = pytest.mark.skipif(
    not os.environ.get("NVIDIA_API_KEY"),
    reason="NVIDIA_API_KEY not set",
)

requires_openrouter_api_key = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


requires_torch = pytest.mark.skipif(not _has_torch(), reason="torch not installed")


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_pool_config_dict():
    return {
        "routing": {
            "method": "prefill",
            "checkpoint": "checkpoints/prefill_router.pt",
            "tolerance": 0.20,
        },
        "models": [
            {
                "name": "nem-think",
                "display_name": "Nemotron 3 Nano Think",
                "litellm_model": "nvidia_nim/nvidia/nvidia/Nemotron-3-Nano-30B-A3B",
                "cost_per_m_input_tokens": 0.20,
                "cost_per_m_output_tokens": 0.20,
            },
            {
                "name": "gpt-5.2",
                "display_name": "GPT-5.2",
                "litellm_model": "nvidia_nim/openai/openai/gpt-5.2",
                "cost_per_m_input_tokens": 1.75,
                "cost_per_m_output_tokens": 14.00,
            },
        ],
    }


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def project_root():
    return Path(__file__).parent.parent


@pytest.fixture
def prefill_ckpt_path(project_root):
    p = project_root / "checkpoints" / "prefill_qwen08b.pt"
    if not p.exists():
        pytest.skip("prefill_qwen08b.pt not found")
    return p


@pytest.fixture
def smoke_ckpt_path(project_root):
    p = project_root / "checkpoints" / "smoke" / "prefill_router.pt"
    if not p.exists():
        pytest.skip("smoke/prefill_router.pt not found")
    return p


@pytest.fixture
def smoke_train_csv(project_root):
    p = project_root / "data" / "smoke-train.csv"
    if not p.exists():
        pytest.skip("data/smoke-train.csv not found")
    return p


@pytest.fixture
def smoke_test_csv(project_root):
    p = project_root / "data" / "smoke-test.csv"
    if not p.exists():
        pytest.skip("data/smoke-test.csv not found")
    return p


@pytest.fixture
def smoke_questions_path(project_root):
    p = project_root / "data" / "smoke-questions.txt"
    if not p.exists():
        pytest.skip("data/smoke-questions.txt not found")
    return p


@pytest.fixture
def smoke_config_path(project_root):
    p = project_root / "configs" / "smoke-test.yaml"
    if not p.exists():
        pytest.skip("configs/smoke-test.yaml not found")
    return p


@pytest.fixture
def prefill_config_path(project_root):
    p = project_root / "configs" / "prefill-qwen08b.yaml"
    if not p.exists():
        pytest.skip("configs/prefill-qwen08b.yaml not found")
    return p
