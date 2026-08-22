"""Dependency free summaries for paired benchmark records."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
from statistics import mean, median
from typing import Any, Callable

from .schema import RECALL_OFF, RECALL_ON, SessionRecord


@dataclass(frozen=True)
class MetricSummary:
    metric: str
    higher_is_better: bool
    n: int
    missing: int
    on_mean: float | None
    off_mean: float | None
    delta_mean: float | None
    on_median: float | None
    off_median: float | None
    delta_median: float | None
    on_better: int
    off_better: int
    ties: int


def _summary(
    metric: str,
    values: list[tuple[float | None, float | None]],
    *,
    higher_is_better: bool,
) -> MetricSummary:
    complete = [(on, off) for on, off in values if on is not None and off is not None]
    deltas = [on - off for on, off in complete]
    on_values = [on for on, _ in complete]
    off_values = [off for _, off in complete]
    on_better = sum(
        (on > off if higher_is_better else on < off) for on, off in complete
    )
    off_better = sum(
        (off > on if higher_is_better else off < on) for on, off in complete
    )
    return MetricSummary(
        metric=metric,
        higher_is_better=higher_is_better,
        n=len(complete),
        missing=len(values) - len(complete),
        on_mean=mean(on_values) if on_values else None,
        off_mean=mean(off_values) if off_values else None,
        delta_mean=mean(deltas) if deltas else None,
        on_median=median(on_values) if on_values else None,
        off_median=median(off_values) if off_values else None,
        delta_median=median(deltas) if deltas else None,
        on_better=on_better,
        off_better=off_better,
        ties=len(complete) - on_better - off_better,
    )


def _metric_values(
    pairs: list[tuple[SessionRecord, SessionRecord]],
    getter: Callable[[SessionRecord], float | None],
) -> list[tuple[float | None, float | None]]:
    return [(getter(on), getter(off)) for on, off in pairs]


def summarize_recall_overhead(records: list[SessionRecord]) -> dict[str, Any]:
    """Describe what the memory layer cost and how often it was used, on the on arm alone.

    `recall_latency_ms` and `recall_call_count` cannot be paired deltas: the off arm has no RE-call
    by construction, so every pair is incomplete and the paired row is permanently null. Reported
    as a paired metric it looks like a measurement that failed rather than a question that does
    not apply. These are one-armed statistics and are presented as such.

    `used_rate` is the one that matters and is easy to forget: a session that had the tool and
    never reached for it is a real result about discoverability, and it is included here rather
    than excluded as a non-event.
    """

    on_arm = [record for record in records if record.variant == RECALL_ON]
    if not on_arm:
        return {"sessions": 0}
    used = [record for record in on_arm if record.recall_call_count > 0]
    latencies = [
        record.recall_latency_ms for record in on_arm if record.recall_latency_ms is not None
    ]
    calls = [float(record.recall_call_count) for record in on_arm]
    return {
        "sessions": len(on_arm),
        "sessions_that_searched": len(used),
        "used_rate": len(used) / len(on_arm),
        "calls_mean": mean(calls),
        "calls_median": median(calls),
        "latency_ms_mean": mean(latencies) if latencies else None,
        "latency_ms_median": median(latencies) if latencies else None,
        "latency_ms_total": sum(latencies) if latencies else None,
        "latency_missing": len(on_arm) - len(latencies),
    }


def summarize_pairs(records: list[SessionRecord]) -> dict[str, Any]:
    """Summarize complete paired records and preserve incomplete task IDs."""

    by_task: dict[str, dict[str, SessionRecord]] = defaultdict(dict)
    for record in records:
        by_task[record.task_id][record.variant] = record

    pairs: list[tuple[SessionRecord, SessionRecord]] = []
    incomplete: list[str] = []
    for task_id in sorted(by_task):
        arms = by_task[task_id]
        on = arms.get(RECALL_ON)
        off = arms.get(RECALL_OFF)
        if on is None or off is None or not on.is_complete or not off.is_complete:
            incomplete.append(task_id)
            continue
        pairs.append((on, off))

    metrics = {
        "success": (lambda record: float(record.success), True),
        "total_tokens": (lambda record: record.total_tokens, False),
        "input_tokens": (lambda record: record.input_tokens, False),
        "output_tokens": (lambda record: record.output_tokens, False),
        "model_turns": (lambda record: record.model_turns, False),
        "tool_calls": (lambda record: float(len(record.tool_calls)), False),
        "recall_call_count": (lambda record: float(record.recall_call_count), False),
        "recall_latency_ms": (lambda record: record.recall_latency_ms, False),
        "wall_time_ms": (lambda record: record.wall_time_ms, False),
        "system_cost_usd": (lambda record: record.system_cost_usd, False),
        "evaluator_cost_usd": (lambda record: record.evaluator_cost_usd, False),
    }
    summaries = {
        name: asdict(
            _summary(
                name,
                _metric_values(pairs, getter),
                higher_is_better=higher_is_better,
            )
        )
        for name, (getter, higher_is_better) in metrics.items()
    }

    return {
        "record_count": len(records),
        "complete_pairs": len(pairs),
        "incomplete_task_ids": incomplete,
        "metrics": summaries,
    }
