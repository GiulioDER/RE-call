"""Aggregate preregistered graph precision observations with paired statistics."""

from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


def _values(rows: list[dict[str, object]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def _mean(rows: list[dict[str, object]], key: str) -> float | None:
    values = _values(rows, key)
    return mean(values) if values else None


def _bootstrap_interval(values: list[float], seed: int = 20260825) -> list[float | None]:
    if not values:
        return [None, None]
    rng = random.Random(seed)
    samples = []
    for _ in range(10000):
        samples.append(mean(rng.choice(values) for _ in values))
    samples.sort()
    return [samples[250], samples[9749]]


def _binomial_cdf(k: int, n: int) -> float:
    return sum(math.comb(n, index) for index in range(k + 1)) / (2**n)


def _sign_test(values: list[float]) -> float | None:
    signs = [value for value in values if value != 0]
    if not signs:
        return None
    positive = sum(value > 0 for value in signs)
    n = len(signs)
    lower = _binomial_cdf(min(positive, n - positive), n)
    return min(1.0, 2.0 * lower)


def _paired_delta(
    left: dict[str, dict[str, object]],
    right: dict[str, dict[str, object]],
    metric: str,
) -> list[float]:
    deltas = []
    for query in sorted(set(left) & set(right)):
        left_value = left[query].get(metric)
        right_value = right[query].get(metric)
        if left_value is not None and right_value is not None:
            deltas.append(float(right_value) - float(left_value))
    return deltas


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: summarize_graph_precision_eval.py OUTPUT_JSON BATCH_JSON ...")
    rows: list[dict[str, object]] = []
    for path in sys.argv[2:]:
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8")))

    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["variant"]), str(row["relation_control"]))].append(row)

    summaries: dict[str, object] = {}
    for (variant, control), variant_rows in sorted(groups.items()):
        arms = {}
        for arm in ("off", "one_hop"):
            arm_rows = [row for row in variant_rows if row["arm"] == arm]
            valid = sum(bool(row.get("answer_valid")) for row in arm_rows)
            arms[arm] = {
                "rows": len(arm_rows),
                "valid_structural_answers": valid,
                "mean_evidence_recall": _mean(arm_rows, "evidence_recall"),
                "mean_evidence_precision": _mean(arm_rows, "evidence_precision"),
                "mean_trusted_items": _mean(arm_rows, "trusted_items"),
                "mean_retrieval_latency_ms": _mean(arm_rows, "retrieval_latency_ms"),
                "mean_graph_latency_ms": _mean(arm_rows, "graph_latency_ms"),
                "mean_graph_candidates": _mean(arm_rows, "graph_candidates"),
                "mean_graph_rejected": _mean(arm_rows, "graph_rejected"),
            }
        summaries[f"{variant}:{control}"] = {"variant": variant, "control": control, "arms": arms}

    keyed: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for row in rows:
        keyed[(str(row["variant"]), str(row["relation_control"]), str(row["query"]), str(row["arm"]))] = row
    baseline = {
        query: row
        for (variant, control, query, arm), row in keyed.items()
        if variant == "baseline" and control == "none" and arm == "one_hop"
    }
    comparisons = {}
    for key in sorted(groups):
        variant, control = key
        candidate = {
            query: row
            for (row_variant, row_control, query, arm), row in keyed.items()
            if row_variant == variant and row_control == control and arm == "one_hop"
        }
        if variant == "baseline" and control == "none":
            continue
        metrics = {}
        for metric in (
            "evidence_precision",
            "evidence_recall",
            "trusted_items",
            "retrieval_latency_ms",
            "graph_latency_ms",
        ):
            deltas = _paired_delta(baseline, candidate, metric)
            metrics[metric] = {
                "n": len(deltas),
                "mean_delta": mean(deltas) if deltas else None,
                "bootstrap_95": _bootstrap_interval(deltas),
                "exact_paired_sign_p": _sign_test(deltas),
            }
        comparisons[f"{variant}:{control}"] = metrics

    report = {
        "artifact": "RE-call graph precision tuning evaluation",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "preregistration": "benchmarks/PREREGISTRATION-evidence-graph-precision-tuning-v1.md",
        "judge": "human review required, no model judge used",
        "rows": len(rows),
        "groups": summaries,
        "comparisons_to_current_graph_baseline": comparisons,
        "raw_batches": sys.argv[2:],
        "observations": rows,
    }
    Path(sys.argv[1]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "groups": len(groups), "output": sys.argv[1]}))


if __name__ == "__main__":
    main()
