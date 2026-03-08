import os
import tempfile
from pathlib import Path

import pytest


requires_nvidia_api_key = pytest.mark.skipif(
    not os.environ.get("NVIDIA_API_KEY"),
    reason="NVIDIA_API_KEY not set",
)

requires_openrouter_api_key = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)


@pytest.fixture
def sample_pool_config_dict():
    return {
        "routing": {
            "method": "kmeans",
            "checkpoint": "checkpoints/kmeans_c100_db.pkl",
            "tolerance": 0.20,
            "embed_model": "nvidia/llama-nemotron-embed-1b-v2",
            "embed_mode": "api",
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
def pkl_path(project_root):
    p = project_root / "checkpoints" / "kmeans_c100_db.pkl"
    if not p.exists():
        pytest.skip("kmeans_c100_db.pkl not found")
    return p
