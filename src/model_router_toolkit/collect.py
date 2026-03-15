"""Data collection for model routing: run models on questions and judge correctness."""

from __future__ import annotations

import csv
import json
import logging
import re
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "openrouter/nvidia/nemotron-3-super-120b-a12b:free"

_JUDGE_SYSTEM_PROMPT = (
    "You are an expert answer evaluator. Given a question and a candidate answer, "
    "determine whether the answer is substantively correct.\n\n"
    'Respond with ONLY a JSON object: {"correct": true} or {"correct": false}'
)

_JUDGE_USER_TEMPLATE = "Question: {question}\n\nAnswer: {answer}"


def _normalize(text: str) -> str:
    return " ".join(re.split(r"\s+", text.strip().lower()))


def _judge_vote(outputs: list[str]) -> str:
    if not outputs:
        return ""
    normalized = [_normalize(o) for o in outputs]
    counts = Counter(normalized)
    return counts.most_common(1)[0][0]


def _judge_reference(
    content: str,
    question: str,
    references: dict[str, str],
) -> bool:
    """Compare model output against a reference answer."""
    q_norm = _normalize(question)
    ref = references.get(q_norm, "")
    if not ref:
        return False
    return _normalize(content) == _normalize(ref)


def _load_references(path: str | Path) -> dict[str, str]:
    """Load reference answers from CSV with columns: question, answer."""
    refs: dict[str, str] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            q = _normalize(row.get("question", ""))
            a = row.get("answer", "")
            if q and a:
                refs[q] = a
    return refs


def _parse_judge_response(text: str) -> bool:
    """Extract a correctness verdict from the judge model's response.

    Tries JSON parsing first, then falls back to regex matching on the raw text.
    Defaults to False if parsing fails entirely.
    """
    cleaned = text.strip()
    # Strip common markdown fences and thinking tags
    cleaned = re.sub(r"```json\s*", "", cleaned)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "correct" in obj:
            return bool(obj["correct"])
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: scan for true/false keywords (first match wins)
    lower = cleaned.lower()
    true_match = re.search(r'\bcorrect["\':\s]*true\b', lower)
    false_match = re.search(r'\bcorrect["\':\s]*false\b', lower)
    if true_match and false_match:
        return true_match.start() < false_match.start()
    if true_match:
        return True
    if false_match:
        return False

    logger.warning("Could not parse judge response, defaulting to incorrect: %.200s", text)
    return False


def _judge_llm(question: str, content: str, judge_model: str) -> bool:
    """Ask a judge LLM whether `content` correctly answers `question`."""
    import litellm

    user_msg = _JUDGE_USER_TEMPLATE.format(question=question, answer=content)
    response = litellm.completion(
        model=judge_model,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
    )
    reply = response.choices[0].message.content or ""
    return _parse_judge_response(reply)


def _call_model(
    litellm_model: str,
    question: str,
    system_prompt: str = "",
    **kwargs,
) -> tuple[str, int]:
    import litellm

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    response = litellm.completion(model=litellm_model, messages=messages, **kwargs)
    content = response.choices[0].message.content or ""
    usage = response.usage
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    return content, output_tokens


def run_collect(
    config_path: str | Path,
    questions_path: str | Path,
    output_path: str | Path,
    judge_method: str = "llm",
    *,
    references_path: str | Path | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    **kwargs,
) -> None:
    from model_router_toolkit.config import load_config

    config = load_config(config_path)
    questions_path = Path(questions_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(questions_path) as f:
        questions = [line.strip() for line in f if line.strip()]

    if not questions:
        print("  No questions found.")
        return

    if judge_method == "llm":
        print(f"  Judge model: {judge_model}")

    references: dict[str, str] = {}
    if judge_method == "reference":
        if not references_path:
            raise ValueError(
                "Reference judging requires --references CSV with columns: question, answer",
            )
        references = _load_references(references_path)
        print(f"  Loaded {len(references)} reference answers")

    try:
        from tqdm import tqdm

        iterator = tqdm(questions, desc="  Collecting")
    except ImportError:
        iterator = questions

    rows = []
    model_correct: dict[str, int] = {}
    model_total: dict[str, int] = {}

    for q in iterator:
        outputs_by_model: dict[str, tuple[str, int]] = {}
        for model_spec in config.models:
            try:
                content, out_tokens = _call_model(
                    model_spec.litellm_model,
                    q,
                    system_prompt=model_spec.system_prompt or "",
                    **model_spec.chat_template_kwargs,
                )
                outputs_by_model[model_spec.name] = (content, out_tokens)
            except Exception:
                logger.warning(
                    "Model %s failed on question: %.80s...",
                    model_spec.name,
                    q,
                    exc_info=True,
                )
                outputs_by_model[model_spec.name] = ("", 0)

        if judge_method == "vote":
            all_outputs = [o for o, _ in outputs_by_model.values()]
            majority = _judge_vote(all_outputs)
            for model_name, (content, out_tokens) in outputs_by_model.items():
                is_correct = _normalize(content) == majority
                rows.append(
                    {
                        "question": q,
                        "model": model_name,
                        "isCorrect": int(is_correct),
                        "output_tokens": out_tokens,
                    }
                )
                model_total[model_name] = model_total.get(model_name, 0) + 1
                if is_correct:
                    model_correct[model_name] = model_correct.get(model_name, 0) + 1

        elif judge_method == "reference":
            for model_name, (content, out_tokens) in outputs_by_model.items():
                is_correct = _judge_reference(content, q, references)
                rows.append(
                    {
                        "question": q,
                        "model": model_name,
                        "isCorrect": int(is_correct),
                        "output_tokens": out_tokens,
                    }
                )
                model_total[model_name] = model_total.get(model_name, 0) + 1
                if is_correct:
                    model_correct[model_name] = model_correct.get(model_name, 0) + 1

        elif judge_method == "llm":
            for model_name, (content, out_tokens) in outputs_by_model.items():
                try:
                    is_correct = _judge_llm(q, content, judge_model)
                except Exception:
                    logger.warning(
                        "Judge failed for model %s on question: %.80s...",
                        model_name,
                        q,
                        exc_info=True,
                    )
                    is_correct = False
                rows.append(
                    {
                        "question": q,
                        "model": model_name,
                        "isCorrect": int(is_correct),
                        "output_tokens": out_tokens,
                    }
                )
                model_total[model_name] = model_total.get(model_name, 0) + 1
                if is_correct:
                    model_correct[model_name] = model_correct.get(model_name, 0) + 1

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "model", "isCorrect", "output_tokens"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Collected {len(rows)} rows from {len(questions)} questions")
    print(f"  Output: {output_path}")

    if model_total:
        print("\n  Per-model accuracy:")
        for m in sorted(model_total):
            total = model_total[m]
            correct = model_correct.get(m, 0)
            print(f"    {m}: {correct}/{total} ({correct / total:.1%})")
