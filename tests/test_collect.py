"""Unit tests for the LLM-as-judge collect pipeline (no API keys needed)."""

from __future__ import annotations

import csv
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from model_router_toolkit.collect import (
    DEFAULT_JUDGE_MODEL,
    _JUDGE_SYSTEM_PROMPT,
    _judge_llm,
    _parse_judge_response,
)


# ---------------------------------------------------------------------------
# _parse_judge_response
# ---------------------------------------------------------------------------


class TestParseJudgeResponse:
    def test_json_correct_true(self):
        assert _parse_judge_response('{"correct": true}') is True

    def test_json_correct_false(self):
        assert _parse_judge_response('{"correct": false}') is False

    def test_json_with_whitespace(self):
        assert _parse_judge_response('  \n {"correct": true}  \n') is True

    def test_json_in_markdown_fence(self):
        assert _parse_judge_response('```json\n{"correct": true}\n```') is True

    def test_json_in_markdown_fence_false(self):
        assert _parse_judge_response('```json\n{"correct": false}\n```') is False

    def test_thinking_tags_stripped(self):
        text = '<think>The answer looks right...</think>\n{"correct": true}'
        assert _parse_judge_response(text) is True

    def test_thinking_tags_multiline(self):
        text = (
            "<think>\nLet me reason about this.\n"
            "The capital of France is Paris.\n</think>\n"
            '{"correct": false}'
        )
        assert _parse_judge_response(text) is False

    def test_fallback_regex_true(self):
        assert _parse_judge_response("The answer is correct: true") is True

    def test_fallback_regex_false(self):
        assert _parse_judge_response('correct": false and that is it') is False

    def test_empty_string_defaults_false(self):
        assert _parse_judge_response("") is False

    def test_garbage_defaults_false(self):
        assert _parse_judge_response("I cannot determine the answer.") is False

    def test_json_truthy_integer(self):
        assert _parse_judge_response('{"correct": 1}') is True

    def test_json_falsy_zero(self):
        assert _parse_judge_response('{"correct": 0}') is False

    def test_json_extra_fields_ignored(self):
        text = '{"correct": true, "reason": "The answer is factually accurate."}'
        assert _parse_judge_response(text) is True


# ---------------------------------------------------------------------------
# _judge_llm
# ---------------------------------------------------------------------------


def _make_litellm_response(content: str) -> MagicMock:
    """Build a mock litellm completion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestJudgeLlm:
    @patch("litellm.completion")
    def test_correct_answer(self, mock_completion):
        mock_completion.return_value = _make_litellm_response(
            '{"correct": true}',
        )
        result = _judge_llm("What is 2+2?", "4", "some-judge-model")
        assert result is True

        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["model"] == "some-judge-model"
        assert call_kwargs.kwargs["temperature"] == 0.0
        messages = call_kwargs.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == _JUDGE_SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        assert "What is 2+2?" in messages[1]["content"]
        assert "4" in messages[1]["content"]

    @patch("litellm.completion")
    def test_incorrect_answer(self, mock_completion):
        mock_completion.return_value = _make_litellm_response(
            '{"correct": false}',
        )
        result = _judge_llm("What is 2+2?", "5", "some-judge-model")
        assert result is False

    @patch("litellm.completion")
    def test_uses_default_judge_model(self, mock_completion):
        mock_completion.return_value = _make_litellm_response(
            '{"correct": true}',
        )
        _judge_llm("q", "a", DEFAULT_JUDGE_MODEL)
        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["model"] == DEFAULT_JUDGE_MODEL

    @patch("litellm.completion")
    def test_empty_response_defaults_false(self, mock_completion):
        mock_completion.return_value = _make_litellm_response("")
        result = _judge_llm("q", "a", "judge")
        assert result is False


# ---------------------------------------------------------------------------
# run_collect with judge_method="llm"
# ---------------------------------------------------------------------------


def _minimal_config_yaml(tmp_path):
    """Write a minimal 2-model pool config and return the path."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "routing:\n"
        "  method: prefill\n"
        "models:\n"
        "  - name: model-a\n"
        "    litellm_model: provider/model-a\n"
        "  - name: model-b\n"
        "    litellm_model: provider/model-b\n"
    )
    return str(cfg)


def _minimal_questions(tmp_path):
    qf = tmp_path / "questions.txt"
    qf.write_text("What is 2+2?\nWhat is the capital of France?\n")
    return str(qf)


class TestRunCollectLlmJudge:
    """End-to-end test of run_collect with judge_method='llm', all LLM calls mocked."""

    @patch("model_router_toolkit.collect._judge_llm")
    @patch("model_router_toolkit.collect._call_model")
    def test_basic_flow(self, mock_call_model, mock_judge_llm, tmp_path):
        from model_router_toolkit.collect import run_collect

        config_path = _minimal_config_yaml(tmp_path)
        questions_path = _minimal_questions(tmp_path)
        output_path = str(tmp_path / "out.csv")

        mock_call_model.return_value = ("Paris", 10)
        # model-a correct, model-b incorrect for both questions
        mock_judge_llm.side_effect = [True, False, True, False]

        run_collect(
            config_path, questions_path, output_path,
            judge_method="llm", judge_model="test-judge",
        )

        with open(output_path) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 4  # 2 questions * 2 models
        assert all(r["isCorrect"] in ("0", "1") for r in rows)

        correct_rows = [r for r in rows if r["isCorrect"] == "1"]
        incorrect_rows = [r for r in rows if r["isCorrect"] == "0"]
        assert len(correct_rows) == 2
        assert len(incorrect_rows) == 2

    @patch("model_router_toolkit.collect._judge_llm")
    @patch("model_router_toolkit.collect._call_model")
    def test_judge_model_passed_through(self, mock_call_model, mock_judge_llm, tmp_path):
        from model_router_toolkit.collect import run_collect

        config_path = _minimal_config_yaml(tmp_path)
        questions_path = _minimal_questions(tmp_path)
        output_path = str(tmp_path / "out.csv")

        mock_call_model.return_value = ("answer", 5)
        mock_judge_llm.return_value = True

        run_collect(
            config_path, questions_path, output_path,
            judge_method="llm", judge_model="my-custom-judge",
        )

        for call in mock_judge_llm.call_args_list:
            assert call.args[2] == "my-custom-judge"

    @patch("model_router_toolkit.collect._judge_llm")
    @patch("model_router_toolkit.collect._call_model")
    def test_judge_failure_defaults_incorrect(
        self, mock_call_model, mock_judge_llm, tmp_path,
    ):
        from model_router_toolkit.collect import run_collect

        config_path = _minimal_config_yaml(tmp_path)
        qf = tmp_path / "q.txt"
        qf.write_text("What is 1+1?\n")
        output_path = str(tmp_path / "out.csv")

        mock_call_model.return_value = ("2", 3)
        mock_judge_llm.side_effect = RuntimeError("API down")

        run_collect(
            str(config_path), str(qf), output_path,
            judge_method="llm", judge_model="broken-judge",
        )

        with open(output_path) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 2  # 1 question * 2 models
        assert all(r["isCorrect"] == "0" for r in rows)

    @patch("model_router_toolkit.collect._judge_llm")
    @patch("model_router_toolkit.collect._call_model")
    def test_csv_columns(self, mock_call_model, mock_judge_llm, tmp_path):
        from model_router_toolkit.collect import run_collect

        config_path = _minimal_config_yaml(tmp_path)
        questions_path = _minimal_questions(tmp_path)
        output_path = str(tmp_path / "out.csv")

        mock_call_model.return_value = ("ans", 7)
        mock_judge_llm.return_value = True

        run_collect(
            config_path, questions_path, output_path,
            judge_method="llm",
        )

        with open(output_path) as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["question", "model", "isCorrect", "output_tokens"]
            rows = list(reader)

        for row in rows:
            assert int(row["output_tokens"]) == 7
