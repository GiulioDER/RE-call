"""Regression invariants for the CCA review of PRs 760 through 769.

Red-proof baseline: acdd27b05, with this test file unchanged. The behavior
assertions cover retry count (STAKES-001), strict JSON fields (BUG-002),
unsafe resume refusal (BUG-001), log redaction (SEC-001), and medians (NUM-001).
All transports are offline doubles; no provider request is made.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest

from benchmarks.beam import f1_storyline_replay as replay
from scripts import aml_c9_compile_citation_diagnosis as diagnosis
from tests.test_aml_compile_diagnostics import _compile_event


def test_builder_stops_retrying_after_observed_spend_cap(monkeypatch) -> None:
    """FIX-004: a charged malformed answer must not permit another paid attempt.

    Baseline production target: replay._json_call's check before its retry loop.
    Red proof requires failure of the calls == 1 assertion, not a provider error.
    """
    calls = []

    def transport(*args):
        calls.append(args)
        return 200, {"choices": [{"message": {"content": "not JSON"}}],
                     "usage": {"prompt_tokens": 10_000}}, {}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-placeholder")
    monkeypatch.setattr(replay, "http_json", transport)
    monkeypatch.setattr(replay.time, "sleep", lambda _: None)
    error = None
    try:
        replay.builder_call(replay.Spend(0.001), "old story",
                            {"date": "d", "messages": []})
    except (SystemExit, RuntimeError) as exc:
        error = exc

    assert len(calls) == 1, "a charged malformed reply crossed the cap before retry"
    assert isinstance(error, SystemExit)


@pytest.mark.parametrize("field", ["digest", "storyline"])
@pytest.mark.parametrize("bad", [None, [], {}, False, 12])
def test_builder_retries_nonstring_json_fields(monkeypatch, field, bad) -> None:
    """FIX-002: str() at replay._json_call must not invent valid memory text.

    Baseline returns the malformed first response and fails result == valid.
    """
    valid = {"digest": "new digest", "storyline": "new story"}
    answers = iter([{**valid, field: bad}, valid])

    def transport(*args):
        return 200, {"choices": [{"message": {"content": json.dumps(next(answers))}}]}, {}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-placeholder")
    monkeypatch.setattr(replay, "http_json", transport)
    monkeypatch.setattr(replay.time, "sleep", lambda _: None)
    result, meta = replay.builder_call(replay.Spend(1), "old story",
                                       {"date": "d", "messages": []})
    assert result == valid
    assert meta["attempts"] == 2


def test_diagnosis_refuses_incomplete_resume_state(tmp_path, monkeypatch) -> None:
    """FIX-001: legacy calls.jsonl cannot restore full compiler prior context.

    Baseline diagnosis.run silently skips saved calls and fails the refusal
    assertion. The log must be preserved and no client constructed on refusal.
    """
    rows = tmp_path / "calls.jsonl"
    original = json.dumps({"call": 0}) + "\n"
    rows.write_text(original, encoding="utf-8")
    clients = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-placeholder")
    monkeypatch.setattr(diagnosis, "planned_calls", lambda *args: [])

    def client(key):
        clients.append(key)
        return SimpleNamespace()

    monkeypatch.setattr(diagnosis, "build_openrouter_client", client)
    args = argparse.Namespace(out=tmp_path, locomo=tmp_path / "locomo",
                              beam=tmp_path / "beam", set="diagnosis", prior_mode="with-ids")
    refused = False
    try:
        diagnosis.run(args)
    except ValueError as exc:
        refused = "cannot resume" in str(exc)
    assert refused, "saved calls lack the full prior records required for safe resume"
    assert clients == []
    assert rows.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("cited", ["anchor_ConfidentialCustomerName", "a012_PrivateProjectName"])
def test_nonhex_anchor_suffix_never_reaches_logs(caplog, cited) -> None:
    """FIX-003: compiler._ID_SHAPED must reject free text in anchor-shaped strings.

    Baseline fails the sample equality because the raw citation is logged.
    """
    event = _compile_event(caplog, [cited])
    assert event["unknown_anchor_id_samples"] == [f"<non-id:{len(cited)} chars>"]
    assert all(cited not in item.message for item in caplog.records)
    assert all(cited not in getattr(item, "unknown_anchor_id_samples", [])
               for item in caplog.records)


@pytest.mark.parametrize("values,expected", [
    ([], None), ([4], 4.0), ([9, 2, 5], 5.0), ([2, 10], 6.0), ([20, 2, 8, 5], 6.5),
])
def test_diagnosis_reports_statistical_median(values, expected) -> None:
    """FIX-006: diagnosis._median must average the central pair for even counts.

    Baseline fails both even-count assertions. Empty/odd/singleton are controls.
    """
    assert diagnosis._median(values) == expected


def test_diagnosis_plan_uses_the_same_median(monkeypatch, capsys) -> None:
    """FIX-006: the plan consumer must not retain its own upper-middle expression.

    Baseline diagnosis.plan prints 10 and fails the expected six-word median.
    """
    monkeypatch.setattr(diagnosis, "planned_calls", lambda *args: [
        {"source": "locomo", "messages": [{"content": "word " * size}]} for size in (2, 10)
    ])
    diagnosis.plan(argparse.Namespace(locomo=None, beam=None, set="diagnosis"))
    assert "median words 6.0" in capsys.readouterr().out
