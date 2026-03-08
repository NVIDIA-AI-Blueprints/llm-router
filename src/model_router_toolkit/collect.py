"""Data collection for model routing: run models on questions and judge correctness."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path


def _normalize(text: str) -> str:
    return " ".join(re.split(r"\s+", text.strip().lower()))


def _judge_vote(outputs: list[str]) -> str:
    if not outputs:
        return ""
    normalized = [_normalize(o) for o in outputs]
    counts = Counter(normalized)
    return counts.most_common(1)[0][0]


def _judge_llm(outputs: list[str], question: str) -> str:
    raise NotImplementedError(
        "LLM-as-judge not yet implemented. Use judge_method='vote' or 'reference'."
    )


def _judge_reference(outputs: list[str], references: list[str]) -> list[bool]:
    raise NotImplementedError(
        "Reference-based judging not yet implemented. Use judge_method='vote'."
    )


def _call_model(litellm_model: str, question: str, system_prompt: str = "", **kwargs) -> tuple[str, int]:
    import litellm

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    response = litellm.completion(
        model=litellm_model,
        messages=messages,
        **kwargs,
    )
    content = response.choices[0].message.content or ""
    usage = response.usage
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    return content, output_tokens


def run_collect(
    config_path: str | Path,
    questions_path: str | Path,
    output_path: str | Path,
    judge_method: str = "vote",
    **kwargs,
) -> None:
    from model_router_toolkit.config import load_config

    config = load_config(config_path)
    questions_path = Path(questions_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(questions_path) as f:
        questions = [line.strip() for line in f if line.strip()]

    if judge_method == "llm":
        raise NotImplementedError("LLM-as-judge not yet implemented. Use judge_method='vote'.")

    if judge_method == "reference":
        raise NotImplementedError("Reference-based judging not yet implemented. Use judge_method='vote'.")

    rows = []
    for q in questions:
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
            except Exception as e:
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

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "model", "isCorrect", "output_tokens"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Collected {len(rows)} rows from {len(questions)} questions")
    print(f"Output: {output_path}")
