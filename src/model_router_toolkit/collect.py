"""Data collection for model routing: run models on questions and judge correctness."""

from __future__ import annotations

import csv
import logging
import re
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


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
    judge_method: str = "vote",
    *,
    references_path: str | Path | None = None,
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
        raise NotImplementedError(
            "LLM-as-judge not yet implemented. Use 'vote' or 'reference'.",
        )

    references: dict[str, str] = {}
    if judge_method == "reference":
        if not references_path:
            raise ValueError(
                "Reference judging requires --references CSV "
                "with columns: question, answer",
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
                    model_spec.name, q, exc_info=True,
                )
                outputs_by_model[model_spec.name] = ("", 0)

        if judge_method == "vote":
            all_outputs = [o for o, _ in outputs_by_model.values()]
            majority = _judge_vote(all_outputs)
            for model_name, (content, out_tokens) in outputs_by_model.items():
                is_correct = _normalize(content) == majority
                rows.append({
                    "question": q,
                    "model": model_name,
                    "isCorrect": int(is_correct),
                    "output_tokens": out_tokens,
                })
                model_total[model_name] = model_total.get(model_name, 0) + 1
                if is_correct:
                    model_correct[model_name] = model_correct.get(model_name, 0) + 1

        elif judge_method == "reference":
            for model_name, (content, out_tokens) in outputs_by_model.items():
                is_correct = _judge_reference(content, q, references)
                rows.append({
                    "question": q,
                    "model": model_name,
                    "isCorrect": int(is_correct),
                    "output_tokens": out_tokens,
                })
                model_total[model_name] = model_total.get(model_name, 0) + 1
                if is_correct:
                    model_correct[model_name] = model_correct.get(model_name, 0) + 1

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["question", "model", "isCorrect", "output_tokens"],
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
