"""Integration tests for data collection and evaluation.

Collection tests require OPENROUTER_API_KEY.
Evaluation tests require the encoder model + checkpoint.
"""

import csv

import pytest

from model_router_toolkit.collect import _judge_reference, _judge_vote, _normalize


class TestJudgeFunctions:
    """Unit tests for judging helpers (no API keys needed)."""

    def test_judge_vote_majority(self):
        outputs = ["Paris", "paris", "London"]
        majority = _judge_vote(outputs)
        assert majority == "paris"

    def test_judge_reference_match(self):
        refs = {_normalize("What is 2+2?"): "4"}
        assert _judge_reference("4", "What is 2+2?", refs) is True

    def test_judge_reference_no_match(self):
        refs = {_normalize("What is 2+2?"): "4"}
        assert _judge_reference("5", "What is 2+2?", refs) is False


@pytest.mark.slow
class TestRunCollectRealAPI:
    """Integration tests calling real model APIs via LiteLLM."""

    @pytest.mark.requires_openrouter_api_key
    def test_run_collect_vote_real_api(self, v1_config_path, tmp_path):
        from model_router_toolkit.collect import run_collect

        questions_file = tmp_path / "questions.txt"
        questions_file.write_text("What is the capital of France?\nWhat is 2+2?\n")
        output_file = tmp_path / "collected.csv"

        run_collect(
            str(v1_config_path),
            str(questions_file),
            str(output_file),
            judge_method="vote",
        )

        assert output_file.exists()
        with open(output_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) > 0
        for row in rows:
            assert "question" in row
            assert "model" in row
            assert row["isCorrect"] in ("0", "1")
            assert int(row["output_tokens"]) >= 0

    @pytest.mark.requires_openrouter_api_key
    def test_run_collect_reference_real_api(self, v1_config_path, tmp_path):
        from model_router_toolkit.collect import run_collect

        questions_file = tmp_path / "questions.txt"
        questions_file.write_text("What is the capital of France?\n")

        refs_file = tmp_path / "refs.csv"
        refs_file.write_text("question,answer\nWhat is the capital of France?,Paris\n")

        output_file = tmp_path / "collected.csv"
        run_collect(
            str(v1_config_path),
            str(questions_file),
            str(output_file),
            judge_method="reference",
            references_path=str(refs_file),
        )

        assert output_file.exists()
        with open(output_file) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0


@pytest.mark.slow
class TestRunEvaluateReal:
    """Integration tests for evaluation with real encoder + checkpoint."""

    def test_run_evaluate_v1(self, v1_config_path, v1_ckpt_path, v1_test_csv_subset, capsys):
        from model_router_toolkit.evaluate import run_evaluate

        run_evaluate(
            str(v1_config_path),
            str(v1_ckpt_path),
            str(v1_test_csv_subset),
            device="cpu",
        )
        captured = capsys.readouterr()
        assert "AUC" in captured.out or "auc" in captured.out.lower()
