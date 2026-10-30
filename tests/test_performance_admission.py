"""The performance worker must measure contention on one shared admission queue."""

from __future__ import annotations

import concurrent.futures
import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from benchmarks import performance_baseline as baseline
from recall.profiles import QUALITY_PROFILE


@pytest.mark.parametrize("warmup", [0, 2])
def test_worker_enforces_shared_admission(monkeypatch, tmp_path, warmup):
    """BUG-001: concurrent requests must respect the worker's execution bound.

    Red-proof target: baseline 9807af5f, _one_request's per-request limiter.
    The unchanged node must fail the peak-concurrency assertion before repair.
    Events observe requests entering retrieval or being refused, rather than
    inferring concurrency from elapsed time. Cold and warmup requests remain controls.
    """
    profile = replace(
        QUALITY_PROFILE, max_concurrency=1, queue_capacity=1, latency_budget_ms=100
    )
    monkeypatch.setitem(baseline.PROFILES, "quality", profile)
    monkeypatch.setattr(baseline, "_build_embedder", lambda _: SimpleNamespace(dim=64))
    monkeypatch.setattr(baseline, "_build_reranker", lambda *_: None)
    monkeypatch.setattr(baseline, "embedding_profile_id", lambda _: "test-embedder")
    monkeypatch.setattr(baseline, "_rss_bytes", lambda: 1)
    monkeypatch.setattr(baseline, "_assemble", lambda _: None)
    closed = []
    monkeypatch.setattr(
        baseline, "PgVectorStore", lambda *_, **__: SimpleNamespace(close=lambda: closed.append(True))
    )
    lock = threading.Lock()
    local = threading.local()
    start = threading.Barrier(4)
    observed = threading.Event()
    release = threading.Event()
    state = {"entered": 0, "refused": 0, "active": 0, "peak": 0, "controls": 0}
    original = baseline._one_request

    def observe():
        if state["entered"] + state["refused"] == 4:
            observed.set()

    def request(*args, **kwargs):
        local.request_id = args[8]
        if local.request_id > 0:
            start.wait(timeout=10)
        row = original(*args, **kwargs)
        if local.request_id > 0 and not row["ok"]:
            with lock:
                state["refused"] += 1
                observe()
        return row

    def search(*args, **kwargs):
        if local.request_id <= 0:
            with lock:
                state["controls"] += 1
        else:
            with lock:
                state["entered"] += 1
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
                observe()
            try:
                assert release.wait(timeout=10), "test did not release retrieval"
            finally:
                with lock:
                    state["active"] -= 1
        return SimpleNamespace(hits=[], diagnostics=SimpleNamespace(stage_ms={}))

    monkeypatch.setattr(baseline, "_one_request", request)
    monkeypatch.setattr(baseline, "bench_search", search)
    output = tmp_path / "run.json"
    config = {"profile": "quality", "k": 1, "candidate_k": 1, "concurrency": 4}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            baseline._worker, config, "unused", "unused", "unused", None, None,
            ["query"], warmup, 4, output,
        )
        try:
            assert observed.wait(timeout=10), "requests neither entered nor reported refusal"
        finally:
            release.set()
        future.result(timeout=10)

    assert state["peak"] == 1, "benchmark requests bypassed the shared execution bound"
    assert state["controls"] == warmup + 1
    rows = json.loads(output.read_text())["rows"]
    measured = [row for row in rows if row["phase"] == "warm"]
    assert sum(row["ok"] for row in measured) == 1
    assert {row["error_reason"] for row in measured if not row["ok"]} <= {
        "queue_full", "budget_exhausted"
    }
    assert sum(not row["ok"] for row in measured) == 3
    assert rows[0]["ok"] is True
    assert closed == [True]
