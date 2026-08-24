"""Helpers for assembling exact evidence cost curves from public artifacts."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

EVIDENCE_BUDGETS: tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096, 8192)


def _rate(values: list[bool]) -> dict[str, Any]:
    return {
        "n": len(values),
        "rate": sum(values) / len(values) if values else None,
    }


def _token_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "min": None, "mean": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, int((0.95 * len(ordered)) - 1e-12)))
    return {
        "n": len(ordered),
        "min": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "median": ordered[(len(ordered) - 1) // 2]
        if len(ordered) % 2
        else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2,
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def _quality_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if not bool(row.get("is_adversarial", False))]
    adversarial = [row for row in rows if bool(row.get("is_adversarial", False))]
    correct = [bool(row["correct"]) for row in answerable if isinstance(row.get("correct"), bool)]
    refusals = [bool(row.get("abstained", False)) for row in rows]
    false_refusals = [bool(row.get("abstained", False)) for row in answerable]
    citation_rows = [row.get("citation_metrics") for row in rows]
    available = [item for item in citation_rows if isinstance(item, Mapping) and item.get("available")]
    citation: dict[str, Any] = {
        "available": bool(available),
        "n": len(available),
        "reason_code": None if available else "benchmark_answer_has_no_citation_channel",
    }
    if available:
        for key in ("precision", "recall", "coverage"):
            values = [float(item[key]) for item in available if isinstance(item.get(key), (int, float))]
            citation[key] = sum(values) / len(values) if values else None
    return {
        "accuracy": _rate(correct),
        "refusal": _rate(refusals),
        "false_refusal": _rate(false_refusals),
        "adversarial_refusal": _rate([bool(row.get("abstained", False)) for row in adversarial]),
        "citation_metrics": citation,
    }


def _group_metrics(rows: list[Mapping[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        value = str(row.get(key, "unknown"))
        groups.setdefault(value, []).append(row)
    result: dict[str, Any] = {}
    for value, group in sorted(groups.items()):
        exact = [int(row["evidence_tokens_exact"]) for row in group]
        total = [int(row["input_tokens_exact"]) for row in group if isinstance(row.get("input_tokens_exact"), int)]
        result[value] = {
            "n": len(group),
            **_quality_metrics(group),
            "evidence_tokens_exact": _token_stats(exact),
            "input_tokens_exact": _token_stats(total),
        }
    return result


def evidence_cost_curve(
    outcomes: Iterable[Mapping[str, Any]],
    budgets: Iterable[int] = EVIDENCE_BUDGETS,
) -> list[dict[str, Any]]:
    """Return deterministic cumulative cost points for the fixed budget ladder.

    This is a reporting helper. A normal run reports quality for questions observed below each
    budget. It does not claim that generation was actually constrained by that budget unless the
    per question artifact records an actual ``evidence_budget`` field.
    """
    rows = list(outcomes)
    normalized_budgets = tuple(int(budget) for budget in budgets)
    if not normalized_budgets or tuple(sorted(set(normalized_budgets))) != normalized_budgets:
        raise ValueError("evidence budgets must be a nonempty sorted unique sequence")
    has_budgeted_records = any("evidence_budget" in row for row in rows)
    points: list[dict[str, Any]] = []
    for budget in normalized_budgets:
        if budget < 1:
            raise ValueError("evidence budgets must be positive")
        eligible = [
            row
            for row in rows
            if isinstance(row.get("evidence_tokens_exact"), int)
            and int(row["evidence_tokens_exact"]) <= budget
            and (not has_budgeted_records or row.get("evidence_budget") == budget)
        ]
        values = [int(row["evidence_tokens_exact"]) for row in eligible]
        input_values = [
            int(row["input_tokens_exact"])
            for row in eligible
            if isinstance(row.get("input_tokens_exact"), int)
        ]
        measured = has_budgeted_records and bool(eligible)
        quality = _quality_metrics(eligible)
        points.append(
            {
                "budget_tokens": budget,
                "n": len(eligible),
                "coverage": len(eligible) / len(rows) if rows else None,
                "evidence_tokens_exact": _token_stats(values),
                "input_tokens_exact": _token_stats(input_values),
                "by_query_class": _group_metrics(eligible, "query_class"),
                "by_routing_profile": _group_metrics(eligible, "routing_profile"),
                "records": [dict(row) for row in eligible],
                **quality,
                "measured_budget": measured,
                "quality_measurement": "budgeted" if measured else "observed_within_budget",
            }
        )
    return points


def evidence_cost_curve_from_artifacts(
    artifacts: Iterable[Mapping[str, Any]],
    budgets: Iterable[int] = EVIDENCE_BUDGETS,
) -> list[dict[str, Any]]:
    """Combine one budgeted artifact per ladder point with paired question identity checks."""
    expected = tuple(int(value) for value in budgets)
    rows: list[dict[str, Any]] = []
    ids_by_budget: dict[int, set[str]] = {}
    for artifact in artifacts:
        outcomes = artifact.get("outcomes")
        if not isinstance(outcomes, list):
            raise ValueError("each budget artifact must contain an outcomes array")
        artifact_rows = [row for row in outcomes if isinstance(row, Mapping)]
        if len(artifact_rows) != len(outcomes):
            raise ValueError("budget artifact outcomes must be objects")
        declared = artifact.get("config", {})
        config_budget = declared.get("evidence_budget") if isinstance(declared, Mapping) else None
        row_budgets = {row.get("evidence_budget") for row in artifact_rows}
        row_budgets.discard(None)
        if config_budget is not None:
            row_budgets.add(config_budget)
        if len(row_budgets) != 1:
            raise ValueError("each artifact must identify exactly one evidence budget")
        budget_value = next(iter(row_budgets))
        if not isinstance(budget_value, int):
            raise ValueError("evidence budget must be an integer")
        budget = budget_value
        if budget not in expected:
            raise ValueError(f"unsupported evidence budget: {budget}")
        normalized = []
        for row in artifact_rows:
            item = dict(row)
            item["evidence_budget"] = budget
            normalized.append(item)
        ids = {str(row.get("question_id")) for row in normalized}
        if budget in ids_by_budget:
            raise ValueError(f"duplicate artifact for evidence budget {budget}")
        ids_by_budget[budget] = ids
        rows.extend(normalized)
    missing = [budget for budget in expected if budget not in ids_by_budget]
    if missing:
        raise ValueError(f"missing budget artifacts: {missing}")
    paired = set.intersection(*(ids_by_budget[budget] for budget in expected))
    if any(ids_by_budget[budget] != paired for budget in expected):
        raise ValueError("budget artifacts do not share paired question identities")
    points = evidence_cost_curve(rows, expected)
    for point in points:
        point["paired_question_ids"] = sorted(paired)
        point["pairing_complete"] = True
    return points


__all__ = [
    "EVIDENCE_BUDGETS",
    "evidence_cost_curve",
    "evidence_cost_curve_from_artifacts",
]
