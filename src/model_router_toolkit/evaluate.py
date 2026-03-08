"""Unified evaluation for routing checkpoints."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import roc_auc_score

from model_router_toolkit.config import load_config
from model_router_toolkit.router import BaseRouter


def _load_data(data_path: str | Path) -> tuple[list[str], dict[str, dict[str, tuple[bool, int]]]]:
    questions = []
    seen = set()
    by_question: dict[str, dict[str, tuple[bool, int]]] = defaultdict(dict)

    with open(data_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("question", "").strip()
            model = row.get("model", "").strip()
            is_correct = str(row.get("isCorrect", "0")).lower() in ("1", "true", "yes")
            try:
                out_tokens = int(row.get("output_tokens", 0))
            except ValueError:
                out_tokens = 0

            if q and model:
                by_question[q][model] = (is_correct, out_tokens)
                if q not in seen:
                    seen.add(q)
                    questions.append(q)

    return questions, dict(by_question)


def _build_cost_table(config, by_question: dict) -> dict[str, dict]:
    model_output_tokens: dict[str, list[int]] = defaultdict(list)
    for qdata in by_question.values():
        for model, (_, out_tokens) in qdata.items():
            model_output_tokens[model].append(out_tokens)

    table = {}
    for m in config.models:
        tokens = model_output_tokens.get(m.name, [500])
        median = sorted(tokens)[len(tokens) // 2] if tokens else 500
        input_est = 100
        cost = (input_est / 1e6) * m.cost_per_m_input_tokens + (median / 1e6) * m.cost_per_m_output_tokens
        table[m.name] = {
            "median_output_tokens": median,
            "cost_per_m_input_tokens": m.cost_per_m_input_tokens,
            "cost_per_m_output_tokens": m.cost_per_m_output_tokens,
            "cost": cost,
        }
    return table


def _compute_cost(question: str, model: str, qdata: dict, config) -> float:
    if model not in qdata:
        return 0.0
    _, out_tokens = qdata[model]
    spec = config.get_model(model)
    if not spec:
        return 0.0
    input_est = max(100, len(question.split()) * 2)
    return (input_est / 1e6) * spec.cost_per_m_input_tokens + (out_tokens / 1e6) * spec.cost_per_m_output_tokens


def run_evaluate(
    config_path: str | Path,
    checkpoint_path: str | Path,
    data_path: str | Path,
    **kwargs,
) -> None:
    config = load_config(config_path)
    config.routing.checkpoint = str(checkpoint_path)

    from model_router_toolkit.config import build_router_from_config

    router: BaseRouter = build_router_from_config(config)
    questions, by_question = _load_data(data_path)
    if not questions:
        print("No test data found.")
        return

    cost_table = _build_cost_table(config, by_question)
    if hasattr(router, "set_cost_table"):
        router.set_cost_table(cost_table)

    model_names = config.model_names
    confidences_by_model: dict[str, list[float]] = {m: [] for m in model_names}
    labels_by_model: dict[str, list[int]] = {m: [] for m in model_names}
    router_correct = 0
    best_single_correct = 0
    routing_counts: dict[str, int] = defaultdict(int)
    cost_by_tolerance: dict[float, float] = {t: 0.0 for t in [0.0, 0.05, 0.10, 0.15, 0.20]}

    model_acc = {m: 0 for m in model_names}
    for qdata in by_question.values():
        for m, (correct, _) in qdata.items():
            if m in model_acc:
                model_acc[m] += int(correct)
    best_single_model = max(model_names, key=lambda m: model_acc.get(m, 0))

    for q in questions:
        qdata = by_question[q]
        result = router.route(q, tolerance=config.routing.tolerance)
        selected = result.selected_model
        routing_counts[selected] += 1

        sel_correct = qdata.get(selected, (False, 0))[0]
        if sel_correct:
            router_correct += 1
        if qdata.get(best_single_model, (False, 0))[0]:
            best_single_correct += 1

        for m in model_names:
            conf = result.confidences[result.model_names.index(m)] if m in result.model_names else 0.0
            confidences_by_model[m].append(conf)
            labels_by_model[m].append(int(qdata.get(m, (False, 0))[0]))

        for tol in cost_by_tolerance:
            res_t = router.route(q, tolerance=tol)
            sel_t = res_t.selected_model
            cost_by_tolerance[tol] += _compute_cost(q, sel_t, qdata, config)

    oracle_total_cost = 0.0
    best_single_total_cost = 0.0
    for q in questions:
        qdata = by_question[q]
        correct_models = [m for m, (c, _) in qdata.items() if c]
        if correct_models:
            costs = [
                (_compute_cost(q, m, qdata, config), m) for m in correct_models
            ]
            oracle_model = min(costs, key=lambda x: x[0])[1]
            oracle_total_cost += _compute_cost(q, oracle_model, qdata, config)
        best_single_total_cost += _compute_cost(q, best_single_model, qdata, config)

    n = len(questions)
    acc_router = router_correct / n if n else 0
    acc_best_single = best_single_correct / n if n else 0

    print("=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test samples: {n}")
    print()

    print("Per-model AUC:")
    for m in model_names:
        y_true = labels_by_model[m]
        y_score = confidences_by_model[m]
        if len(set(y_true)) < 2:
            auc = float("nan")
        else:
            try:
                auc = roc_auc_score(y_true, y_score)
            except ValueError:
                auc = float("nan")
        print(f"  {m}: {auc:.4f}")

    print()
    print("Router accuracy (vs oracle):", f"{acc_router:.2%}")
    print("Best-single model accuracy:", f"{acc_best_single:.2%}")
    print()

    print("Cost savings at tolerance levels:")
    for tol in [0.0, 0.05, 0.10, 0.15, 0.20]:
        router_cost = cost_by_tolerance[tol]
        if oracle_total_cost > 0:
            savings = (oracle_total_cost - router_cost) / oracle_total_cost
        else:
            savings = 0.0
        print(f"  tolerance={tol:.2f}: cost={router_cost:.4f}, savings vs oracle={savings:.2%}")

    print()
    print("Routing distribution:")
    for m in model_names:
        pct = routing_counts[m] / n * 100 if n else 0
        print(f"  {m}: {pct:.1f}%")
