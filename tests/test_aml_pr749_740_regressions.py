"""Offline regression proofs for the PR 749 through 740 audit.

The baseline is fd0ffa22c. Every defect test was run on the test host against
that source and failed its stated assertion before the repair. No paid call is
made: only the HTTP transport, AML dataset/pipeline and hosted app are doubled.
FIX-001 covers request bounds, concurrent allowance, retries and invalid usage;
FIX-002 covers the same allowance across independently constructed clients;
FIX-003 covers cleanup after settled workers; FIX-004 covers a perfect report;
FIX-005 covers admitted ISO timestamps at both UTC year boundaries.
"""

from __future__ import annotations

from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import aml_locomo_loss_diagnosis as diagnosis  # noqa: E402


@pytest.fixture
def budget(tmp_path, monkeypatch):
    """Independent ledger fixture, also usable by the unchanged baseline client."""
    path = tmp_path / "budget.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE budget (id INTEGER PRIMARY KEY, used_nanos INTEGER, blocked INTEGER)")
        db.execute("INSERT INTO budget VALUES (1, 0, 0)")
    monkeypatch.setenv("AML_DIAG_BUDGET_LEDGER", str(path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-key")
    monkeypatch.setattr(diagnosis.time, "sleep", lambda _: None)
    return path


def _seed(path, spent):
    with sqlite3.connect(path) as db:
        db.execute("UPDATE budget SET used_nanos=?", (round(spent * 1_000_000_000),))


def _transport(monkeypatch, handler):
    original = httpx.Client
    monkeypatch.setattr(diagnosis.httpx, "Client", lambda **kw: original(
        transport=httpx.MockTransport(handler), **kw
    ))


def _reply(request, cost=0.01):
    return httpx.Response(200, request=request, json={
        "choices": [{"message": {"content": "answer"}}], "usage": {"cost": cost},
    })


def test_competing_calls_cannot_spend_one_allowance_twice(budget, monkeypatch):
    """FIX-001: baseline ends at 15.07, failing the cumulative cap assertion."""
    _seed(budget, 14.99)
    barrier = threading.Barrier(8)

    def send(request):
        barrier.wait(timeout=3)
        return _reply(request)

    _transport(monkeypatch, send)
    router = diagnosis.OpenRouter(14.99)

    def complete(_):
        try:
            router.complete(diagnosis.ANSWER_MODEL, "question")
        except RuntimeError:
            pass

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(complete, range(8)))
    assert router.spent <= diagnosis.COST_CAP_USD


def test_payload_bounds_the_reserved_charge(budget, monkeypatch):
    """FIX-001: baseline submits a payload without the asserted output bound."""
    sent = []
    _transport(monkeypatch, lambda r: (sent.append(json.loads(r.content)), _reply(r))[1])
    diagnosis.OpenRouter(0).complete(diagnosis.ANSWER_MODEL, "question")
    assert sent[0].get("max_tokens", 0) > 0
    assert sent[0]["provider"]["max_price"]["request"] == 0
    assert sent[0]["provider"]["require_parameters"] is True


def test_unbounded_model_cannot_submit(budget, monkeypatch):
    """FIX-001: baseline sends a model whose token accounting is not bounded."""
    sent = []
    _transport(monkeypatch, lambda r: (sent.append(r), _reply(r))[1])
    try:
        diagnosis.OpenRouter(0).complete("unreviewed/model:online", "question")
    except ValueError:
        pass
    assert sent == []


@pytest.mark.parametrize("spent", [-1, "NaN", "invalid", True])
def test_invalid_resumed_cost_is_rejected(budget, monkeypatch, spent):
    """FIX-001: baseline accepts these seeds without validating the allowance."""
    _transport(monkeypatch, _reply)
    error = None
    try:
        diagnosis.OpenRouter(spent)
    except ValueError as exc:
        error = exc
    assert error is not None


@pytest.mark.parametrize("cost", [None, -1, "NaN", "invalid", True, "1e100"])
def test_unknown_usage_stops_further_paid_work(budget, monkeypatch, cost):
    """FIX-001: baseline makes a second submission after invalid cost accounting."""
    sent = []
    _transport(monkeypatch, lambda r: (sent.append(r), _reply(r, cost))[1])
    for _ in range(2):
        try:
            diagnosis.OpenRouter(0).complete(diagnosis.ANSWER_MODEL, "question")
        except (RuntimeError, ValueError, OverflowError):
            pass
    assert len(sent) == 1, "unknown expenditure must halt new submissions"


def test_ordinary_recorded_costs_resume_without_roundoff(budget, monkeypatch, tmp_path):
    """FIX-002: 0.1+0.2 must neither invent a nanodollar nor block the next stage.

    The first repair rounds the float seed up and fails the no-error assertion;
    the original implementation fails the exact accumulated-cost assertion.
    """
    _seed(budget, 0.3)
    _transport(monkeypatch, _reply)
    records = tmp_path / "answers.jsonl"
    records.write_text("\n".join(json.dumps({"id": str(i), "usage": {"cost": cost}})
                                 for i, cost in enumerate((0.1, 0.2))))
    observed = diagnosis.spent_so_far(records)
    error = None
    try:
        diagnosis.OpenRouter(observed)
    except RuntimeError as exc:
        error = exc
    assert error is None, "ordinary recorded charges must remain resumable"
    assert observed == 0.3


def test_unknown_retry_expenditure_keeps_its_allowance(budget, monkeypatch):
    """FIX-001: baseline retries six times against the same remaining allowance."""
    _seed(budget, 14.75)
    sent = []

    def send(request):
        sent.append(request)
        raise httpx.ReadError("unknown provider outcome", request=request)

    _transport(monkeypatch, send)
    router = diagnosis.OpenRouter(14.75)
    try:
        router.complete(diagnosis.ANSWER_MODEL, "question")
    except RuntimeError:
        pass
    assert len(sent) == 1
    assert router.spent <= diagnosis.COST_CAP_USD


def test_independent_stages_share_the_budget(budget, monkeypatch):
    """FIX-002: baseline resets the second client's allowance and submits twice."""
    _seed(budget, 14.79)
    sent = []
    _transport(monkeypatch, lambda r: (sent.append(r), _reply(r))[1])
    diagnosis.OpenRouter(0).complete(diagnosis.ANSWER_MODEL, "question")
    try:
        diagnosis.OpenRouter(0).complete(diagnosis.JUDGE_MODEL, "question")
    except RuntimeError:
        pass
    assert len(sent) == 1


def test_budget_survives_a_new_process(budget, monkeypatch):
    """FIX-002: baseline child submits despite the parent's exhausted allowance."""
    _seed(budget, 14.79)
    _transport(monkeypatch, _reply)
    diagnosis.OpenRouter(0).complete(diagnosis.ANSWER_MODEL, "question")
    program = """
import json, sys, httpx
sys.path.insert(0, 'scripts')
import aml_locomo_loss_diagnosis as d
original = httpx.Client
sent = []
def send(r):
    sent.append(r)
    return httpx.Response(200, request=r, json={'choices':[{'message':{'content':'ok'}}],
                                             'usage':{'cost':0.01}})
d.httpx.Client = lambda **kw: original(transport=httpx.MockTransport(send), **kw)
try:
    d.OpenRouter(0).complete(d.JUDGE_MODEL, 'question')
except RuntimeError:
    pass
print(json.dumps({'calls':len(sent)}))
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True,
                            text=True, check=True, timeout=20)
    assert json.loads(result.stdout)["calls"] == 0


@pytest.mark.parametrize("failure", ["add", "canary", "search"])
def test_failed_collect_cleans_every_user_after_workers_settle(tmp_path, monkeypatch, failure):
    """FIX-003: baseline records no delete after each injected collection failure."""
    import recall_aml.__main__ as hosted_main
    import starlette.testclient

    requests = [{"user_id": u, "session_id": "s", "request_id": u, "messages": []}
                for u in ("u1", "u2")]
    questions = [{"user_id": u, "query": u, "gold_turns": ["turn"]} for u in ("u1", "u2")]
    events = []
    searches = 0

    class Client:
        def __init__(self, app):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, *args, **kwargs):
            return httpx.Response(200, json={"variant": "v", "window_renderer_profile":
                                           "message-content-only-v1"})

        def post(self, path, json, **kwargs):
            nonlocal searches
            if path == "/v1/delete":
                events.append(("delete", json["user_id"]))
                return httpx.Response(500 if json["user_id"] == "u1" else 200)
            if path == "/v1/add":
                events.append(("add-finished", json["user_id"]))
                if failure == "add" and json["user_id"] == "u1":
                    raise RuntimeError("add failed")
                return httpx.Response(200, json={})
            searches += 1
            if failure == "canary":
                return httpx.Response(503, json={"error": "unavailable"})
            if searches > 1:
                raise RuntimeError("search failed")
            return httpx.Response(200, json={"data": []})

    monkeypatch.setattr(starlette.testclient, "TestClient", Client)
    monkeypatch.setattr(hosted_main, "build_app", lambda: object())
    monkeypatch.setattr(diagnosis, "build_corpus", lambda *a: (requests, questions))
    monkeypatch.setattr(diagnosis.time, "sleep", lambda _: None)
    monkeypatch.setenv("RECALL_AML_API_KEY", "offline-test-key")
    data = tmp_path / "data.json"
    data.write_text("[]")
    try:
        diagnosis.collect(Namespace(data=data, run_id="r", limit_conversations=None,
                                   timestamped_windows=False, expected_variant="v", workers=2,
                                   preregistration="unchanged", out=tmp_path / "out.gz"))
    except (RuntimeError, KeyError):
        pass
    assert {u for event, u in events if event == "delete"} == {"u1", "u2"}
    first_delete = next(i for i, event in enumerate(events) if event[0] == "delete")
    assert all(event != "add-finished" for event, _ in events[first_delete:])


def test_perfect_report(tmp_path):
    """FIX-004: baseline captures ZeroDivisionError and fails the no-error assertion."""
    collected = tmp_path / "collected.gz"
    collected.write_bytes(gzip.compress(json.dumps({"rows": [
        {"id": "q", "category": 1, "route": "code", "first_gold_rank": 1}
    ]}).encode()))
    answers, judged, classified = [tmp_path / name for name in ("answers", "judged", "classified")]
    answers.write_text(json.dumps({"id": "q", "prompt_chars": 1}) + "\n")
    judged.write_text(json.dumps({"id": "q", "label": "CORRECT"}) + "\n")
    classified.write_text("")
    output = tmp_path / "report.json"
    error = None
    try:
        diagnosis.report(Namespace(collected=collected, answers=answers, judged=judged,
                                   classified=classified, out=output))
    except ZeroDivisionError as exc:
        error = exc
    assert error is None, "a valid perfect report must not divide by zero"
    result = json.loads(output.read_text())
    assert result["accuracy"]["accuracy"] == 1
    assert all(bucket["share_of_wrong"] is None for bucket in result["wrong_buckets"].values())


@pytest.mark.parametrize("stamp", ["9999-12-31T23:59:59-01:00", "0001-01-01T00:00:00+01:00"])
def test_boundary_timestamp(stamp):
    """FIX-005: baseline captures OverflowError and fails the no-error assertion."""
    from recall_aml.models import AddRequest
    from recall_aml.service import build_chunks

    request = AddRequest.model_validate_json(json.dumps({
        "request_id": "r", "user_id": "u", "session_id": "s",
        "messages": [{"role": "user", "content": "hello", "timestamp": stamp}],
    }))
    error = None
    try:
        chunks = build_chunks(request, [], word_window_size=160, word_window_stride=120,
                              content_only_windows=True)
    except OverflowError as exc:
        error = exc
    assert error is None, "an admitted optional timestamp must remain processable"
    assert chunks


def test_valid_boundary_offsets_keep_their_timestamps():
    """FIX-005 control: red mutation makes _iso_timestamp return None for valid dates."""
    from recall_aml.models import Message

    for stamp in ("9999-12-31T22:59:59-01:00", "0001-01-01T01:00:00+01:00"):
        message = Message.model_validate({"role": "user", "content": "hello", "timestamp": stamp})
        assert message.timestamp is not None
        assert message.timestamp.isoformat() == stamp
