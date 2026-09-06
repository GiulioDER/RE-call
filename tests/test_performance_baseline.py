from __future__ import annotations

import importlib
import json
from pathlib import Path


def test_performance_fixture_is_immutable_and_nonempty() -> None:
    payload = json.loads(
        Path("benchmarks/fixtures/performance_baseline.json").read_text()
    )
    assert len(payload["corpus"]) >= 20
    assert len(payload["queries"]) >= 15
    assert len({item["id"] for item in payload["corpus"]}) == len(payload["corpus"])


def test_rss_measurement_is_available() -> None:
    module = importlib.import_module("benchmarks.performance_baseline")
    assert module._rss_bytes() > 0


def test_performance_summary_reports_errors_and_stage_percentiles() -> None:
    module = importlib.import_module("benchmarks.performance_baseline")
    rows = [
        {
            "ok": True,
            "elapsed_ms": 10.0,
            "rss_bytes": 100,
            "stage_ms": {name: 1.0 for name in module.STAGES},
        },
        {
            "ok": True,
            "elapsed_ms": 20.0,
            "rss_bytes": 120,
            "stage_ms": {name: 2.0 for name in module.STAGES},
        },
        {
            "ok": False,
            "elapsed_ms": 30.0,
            "rss_bytes": 130,
            "error_reason": "queue_full",
        },
    ]
    summary = module._summary(rows)
    assert summary["attempted"] == 3
    assert summary["failures"] == 1
    assert summary["errors"] == {"queue_full": 1}
    assert summary["stages"]["trust_evaluation"]["n"] == 2
    assert summary["rss_bytes"]["peak"] == 130
