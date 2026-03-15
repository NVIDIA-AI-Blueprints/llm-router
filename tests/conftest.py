import csv
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
def v1_config_path(project_root):
    p = project_root / "configs" / "v1-9models-qwen08b.yaml"
    if not p.exists():
        pytest.skip("configs/v1-9models-qwen08b.yaml not found")
    return p


@pytest.fixture
def v1_ckpt_path(project_root):
    p = project_root / "checkpoints" / "prefill_router_qwen08b.pt"
    if not p.exists():
        pytest.skip("checkpoints/prefill_router_qwen08b.pt not found")
    return p


_V1_CSV_ROWS_PER_MODEL = 5


@pytest.fixture
def v1_test_csv_subset(project_root, tmp_path):
    """Read test_v1.csv and write a small subset (first N rows per model)."""
    src = project_root / "data" / "test_v1.csv"
    if not src.exists():
        pytest.skip("data/test_v1.csv not found")

    counts: dict[str, int] = {}
    dest = tmp_path / "test_v1_subset.csv"
    with open(src, newline="") as fin, open(dest, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(
            fout, fieldnames=["question", "model", "isCorrect", "output_tokens"],
        )
        writer.writeheader()
        for row in reader:
            model = row["model"]
            if counts.get(model, 0) >= _V1_CSV_ROWS_PER_MODEL:
                if all(v >= _V1_CSV_ROWS_PER_MODEL for v in counts.values()):
                    break
                continue
            counts[model] = counts.get(model, 0) + 1
            writer.writerow({
                "question": row["question"],
                "model": row["model"],
                "isCorrect": row["isCorrect"],
                "output_tokens": row["output_tokens"],
            })
    return dest


@pytest.fixture
def v1_questions(v1_test_csv_subset, tmp_path):
    """Extract unique questions from the v1 test subset."""
    seen: set[str] = set()
    questions: list[str] = []
    with open(v1_test_csv_subset, newline="") as f:
        for row in csv.DictReader(f):
            q = row["question"].strip()
            if q not in seen:
                seen.add(q)
                questions.append(q)
    dest = tmp_path / "v1_questions.txt"
    dest.write_text("\n".join(questions) + "\n")
    return dest


@pytest.fixture
def prefill_config_path(project_root):
    p = project_root / "configs" / "v1-9models-qwen08b.yaml"
    if not p.exists():
        pytest.skip("configs/v1-9models-qwen08b.yaml not found")
    return p
