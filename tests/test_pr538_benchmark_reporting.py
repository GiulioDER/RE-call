"""Exercise the artifact/report boundary with offline synthetic timings.

SEC-003 and NUM-001 nodes fail their assertions on baseline 76a0b36ea.
No database, API or latency measurement is performed by these tests.
"""

from __future__ import annotations

import argparse
import json

import pytest

from benchmarks import write_time_connection_reuse as bench


def run_benchmark(tmp_path, monkeypatch, capsys, dsn):
    counter = iter(range(30))
    monkeypatch.setattr(bench, "OUT", tmp_path)
    monkeypatch.setattr(bench, "_one_request", lambda *a: {
        "status": "ok", "hits": [], "elapsed_ms": float(next(counter)),
    })
    monkeypatch.setattr(bench, "_relay_requests", lambda _dsn, _tenant, queries: [
        {"status": "ok", "hits": [], "elapsed_ms": float(i + 100)}
        for i in range(len(queries))
    ])
    assert bench._run(argparse.Namespace(dsn=dsn, tenant="test")) == 0
    return json.loads(capsys.readouterr().out), next(tmp_path.glob("*.json")).read_text()


@pytest.mark.parametrize("dsn", [
    "host=localhost dbname=test user=test password='credential-sentinel'",
    "postgresql://localhost/test?user=test&password=credential-sentinel",
    "postgresql://localhost/test?%70assword=credential-sentinel",
])
def test_artifact_omits_connection_credentials(tmp_path, monkeypatch, capsys, dsn):
    """SEC-003 red: baseline persists the synthetic password in artifact.dsn."""
    _, artifact = run_benchmark(tmp_path, monkeypatch, capsys, dsn)
    assert "credential-sentinel" not in artifact


def test_report_averages_the_central_even_samples(tmp_path, monkeypatch, capsys):
    """NUM-001 red: baseline reports 15/115 instead of 14.5/114.5."""
    summary, _ = run_benchmark(tmp_path, monkeypatch, capsys, "postgresql://localhost/test")
    assert summary["cold_median_ms"] == 14.5
    assert summary["relay_median_ms"] == 114.5
