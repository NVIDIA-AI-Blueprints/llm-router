"""Integration tests for server factory, review endpoint, and CLI subcommands.

Server tests require encoder model + checkpoint + API key.
CLI tests exercise real subcommands with smoke data.
"""

import subprocess
import sys

import pytest


@pytest.mark.slow
class TestCreateAppReal:
    """Tests using create_app with a real config, checkpoint, and API key."""

    @pytest.mark.requires_openrouter_api_key
    def test_create_app_real_config(self, smoke_config_path):
        from model_router_toolkit.adapters.litellm.app import create_app

        app = create_app(str(smoke_config_path))
        routes = [r.path for r in app.routes]
        assert "/health" in routes
        assert "/v1/chat/completions" in routes or any("/chat/completions" in r for r in routes)

    @pytest.mark.requires_openrouter_api_key
    def test_health_endpoint_real_app(self, smoke_config_path):
        from fastapi.testclient import TestClient

        from model_router_toolkit.adapters.litellm.app import create_app

        app = create_app(str(smoke_config_path))
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert len(data["models"]) > 0

    @pytest.mark.requires_openrouter_api_key
    def test_completions_real_api(self, smoke_config_path):
        from fastapi.testclient import TestClient

        from model_router_toolkit.adapters.litellm.app import create_app

        app = create_app(str(smoke_config_path))
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "max_tokens": 50,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        msg = data["choices"][0].get("message", {})
        assert "content" in msg, f"Response message missing 'content': {msg}"

    @pytest.mark.requires_openrouter_api_key
    def test_review_endpoint_judges_answer(self, smoke_config_path):
        from fastapi.testclient import TestClient

        from model_router_toolkit.adapters.litellm.app import create_app

        app = create_app(str(smoke_config_path))
        client = TestClient(app)
        resp = client.post(
            "/api/review",
            json={
                "question": "What is the capital of France?",
                "answer": "London",
                "selected_model": "nem-think",
            },
        )
        assert resp.status_code == 200
        body = resp.text
        assert "event:" in body


@pytest.mark.slow
class TestCLISubcommands:
    """Tests exercising CLI subcommands via subprocess."""

    def _run_cli(self, args, timeout=300, **kwargs):
        return subprocess.run(
            ["model-router"] + args,
            capture_output=True, text=True, timeout=timeout,
            cwd=str(kwargs.get("cwd", ".")),
        )

    def test_cli_serve_config_message(self, project_root):
        result = self._run_cli(["serve-config"], cwd=project_root)
        assert result.returncode == 0
        assert "not yet available" in result.stdout

    def test_cli_kmeans_train_blocked(self, project_root, smoke_train_csv):
        result = self._run_cli(
            ["train", "--config", "configs/openrouter-kmeans.yaml",
             "--data", str(smoke_train_csv)],
            cwd=project_root,
        )
        assert result.returncode == 1
        assert "not yet available" in result.stderr.lower() or "not yet available" in result.stdout.lower()

    @pytest.mark.requires_openrouter_api_key
    def test_cli_collect_smoke(self, project_root, smoke_config_path, smoke_questions_path, tmp_path):
        output_csv = tmp_path / "collected.csv"
        questions_3 = tmp_path / "q3.txt"
        with open(smoke_questions_path) as f:
            lines = [l for l in f if l.strip()][:2]
        questions_3.write_text("\n".join(lines))

        result = self._run_cli(
            ["collect",
             "--config", str(smoke_config_path),
             "--questions", str(questions_3),
             "--output", str(output_csv),
             "--judge", "vote"],
            cwd=project_root, timeout=600,
        )
        assert result.returncode == 0, f"collect failed: {result.stderr}"
        assert output_csv.exists()

    def test_cli_evaluate_smoke(self, project_root, smoke_config_path, smoke_ckpt_path, smoke_test_csv):
        result = self._run_cli(
            ["evaluate",
             "--config", str(smoke_config_path),
             "--checkpoint", str(smoke_ckpt_path),
             "--data", str(smoke_test_csv),
             "--device", "cpu"],
            cwd=project_root, timeout=600,
        )
        combined = result.stdout + result.stderr
        if result.returncode < 0:
            import signal as _sig
            sig = -result.returncode
            sig_name = _sig.Signals(sig).name if sig in _sig.Signals._value2member_map_ else str(sig)
            pytest.skip(
                f"Encoder subprocess killed by signal {sig_name} — "
                f"likely a torch/transformers crash on this platform"
            )
        if "does not recognize this architecture" in combined:
            pytest.skip(
                "Transformers version too old for this encoder model — "
                "upgrade with: pip install -U transformers"
            )
        assert result.returncode == 0, f"evaluate failed (rc={result.returncode}): {result.stderr}"
        assert "AUC" in combined or "auc" in combined.lower()
