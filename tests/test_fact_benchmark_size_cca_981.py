"""Regression proof observed 2026-10-26 on the isolated testbench.
Baseline: 51c400cc96fef80c5354c632edb6fd0ae44a2c97. The unchanged cases listed
below reached their intended assertion failures; setup errors and skips were not RED.
All 17 cases passed after repair and again after hash-verified mutation restoration.
No timing result is asserted.
test_fact_benchmark_size_cca_981.py::test_benchmark_reports_actual_churn_event_count[100]: baseline control GREEN, production mutation RED, restored GREEN.
test_fact_benchmark_size_cca_981.py::test_benchmark_reports_actual_churn_event_count[101]: baseline assertion RED, repaired GREEN.
test_fact_benchmark_size_cca_981.py::test_benchmark_reports_actual_churn_event_count[102]: baseline control GREEN, production mutation RED, restored GREEN.
test_fact_benchmark_size_cca_981.py::test_benchmark_reports_actual_churn_event_count[103]: baseline assertion RED, repaired GREEN.
Control mutation: fact_current_read_cost.lite: actual event count reduced by one.
Both controls failed their intended assertion and passed after restoration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks import fact_current_read_cost as benchmark


@pytest.mark.parametrize("requested", [100, 101, 102, 103])
def test_benchmark_reports_actual_churn_event_count(
    requested: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the real CLI and builder; independently count the resulting ledger events.

    Intended RED: reported events differ from the real built ledger count. The build
    observer delegates the real implementation and never changes its input or result.
    """
    counts: list[int] = []
    original = benchmark.build

    def build(ledger: Any, tenant: str, shape: str, events: int) -> Any:
        probe = original(ledger, tenant, shape, events)
        counts.append(len(ledger.events))
        return probe

    monkeypatch.setattr(benchmark, "build", build)
    output = tmp_path / "cardinality.json"
    assert benchmark.main([
        "--backend", "lite", "--shapes", "churn", "--events", str(requested),
        "--samples", "2", "--out", str(output),
    ]) == 0
    (record,) = json.loads(output.read_text(encoding="utf-8"))
    assert len(counts) == 1 and counts[0] >= requested
    assert record["events"] == counts[0], (
        f"benchmark labeled {counts[0]} real ledger events as {record['events']}"
    )
