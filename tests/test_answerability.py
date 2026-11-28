"""The opt-in answerability judge fills `query_support` and nothing else. No network: the HTTP
opener is replaced, so the request body the judge would send is read here instead of sent.

Invariants and the failure each one catches:
- A1 off by default: no setting means no judge, and a typo or a missing key refuses at startup.
- A2 the request is the measured one (model, reasoning on, temperature 0, 8000 tokens, usage
  included, no provider pin), with the measured system prompt and only the citable hits as evidence.
- A3 the VERDICT decides: `stated: false, confidence: 99` is `not_stated`.
- A4 a call that raises is `failed`; replies the strict parser refuses three times are `unparsed`,
  and a refused reply is retried before a verdict is taken.
- A5 nothing citable means no call and `not_checked`.
- A6 through `recall_search` and `recall_evidence` the verdict reaches `query_support` while hits,
  abstention and citable evidence are unchanged.
- A7 the reasoning response decoder keeps a judged support and refuses an inconsistent one.
- A8 every attempt shares one deadline: past it, no further call is made and the support is
  `failed` (CCA PERF-001, run 20261128T104334Z).
- A9 a search that finds `MAX_CONCURRENT_CALLS` judge calls in flight is `failed` without a call
  (CCA PERF-002).
- A10 the judge's call-rate ceiling refuses a call over it as `failed`, and a bad ceiling refuses
  at startup (CCA PERF-003).
- A11 `recall_search` and `recall_evidence` send the judge the same bytes for the same hits, with
  the source label the bundle shows, and report the judge as its own stage, outside the retrieval
  budget (CCA BUG-001, CODE-001, PERF-004).
- A12 the CLI judges exactly the evidence it prints (CCA BUG-002).
- A13 a reply that trickles in through a real HTTP response is abandoned at the first piece read
  after the deadline (CCA FA-1-1, FA-2-1).
- A14 the search waits no longer than the deadline for a call stuck anywhere, and the call keeps
  its in-flight slot until it ends (CCA FA-2-2).

Red proof, 2026-11-28, on the remote test host, each mutation applied alone to `recall/answerability.py` or the named
file, then reverted; every test below failed in its own assertion:
- A1 `test_a1_off_by_default_and_refuses_a_bad_setting`: `resolve_answerability_judge` returning
  None without reading the key (the `raise ValueError` for an empty key deleted).
- A2 `test_a2_the_request_is_the_measured_one`: `"reasoning": {"enabled": True}` removed from the
  body; and `citable_evidence` dropping its `if hit.verdict == "ok"` filter (since replaced by
  `bundle_evidence` over the evidence bundle, which admits `ok` hits only).
- A3 `test_a3_the_verdict_decides_not_the_confidence`: status computed as
  `"stated" if _confidence >= 50 else "not_stated"`.
- A4 `test_a4_failures_are_distinct_from_verdicts`: the `except Exception` branch returning
  `QuerySupport(status="stated", ...)`; and `ATTEMPTS = 1`.
- A5 `test_a5_nothing_citable_sends_nothing`: the `if not evidence` early return deleted.
- A6 `test_a6_search_and_evidence_carry_the_verdict_only`: the `judged(...)` line in
  `recall_mcp/retrieval.py::search_memory` deleted, and separately the `judge_query_support` line
  in `evidence_memory`.
- A7 `test_a7_the_decoder_keeps_and_checks_judged_support`: `_query_support_from_dict` restored to
  its pre-change body, which refuses any non-null `method`.
"""

from __future__ import annotations

import http.client
import io
import json
import threading
import time
from dataclasses import asdict, replace

import pytest

from recall import answerability
from recall.answer_provider import CallRateGate
from recall.cli_commands.index_search import _print_evidence
from recall.evidence import EvidenceValidationError
from recall.reasoning_response import reasoning_response_from_dict
from recall.support import QuerySupport
from recall_mcp.settings import _validate_runtime_options
from tests.test_query_support_reasoning import _response
from tests import test_search_evidence
from tests.test_search_evidence import _FakeSearch, _both

ON = {"RECALL_ANSWERABILITY_JUDGE": "1", "OPENROUTER_API_KEY": "sk-or-offline-fixture"}


class _Opener:
    """Records each request and answers with the next canned assistant text."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.bodies: list[dict] = []
        self.headers: list[dict] = []
        self.timeouts: list[float] = []

    def open(self, req, timeout):
        self.timeouts.append(timeout)
        self.bodies.append(json.loads(req.data))
        self.headers.append(dict(req.header_items()))
        reply = self.replies.pop(0)
        if callable(reply):
            reply = reply()
        if isinstance(reply, Exception):
            raise reply
        if hasattr(reply, "read"):
            return reply
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": reply}}]}).encode())


@pytest.fixture
def opener(monkeypatch):
    def install(*replies):
        fake = _Opener(*replies)
        monkeypatch.setattr(answerability, "_OPENER", fake)
        return fake
    return install


@pytest.fixture(autouse=True)
def _fresh_rate_gate(monkeypatch):
    """Each test starts with an empty window: the gate is process wide on purpose."""
    monkeypatch.setattr(answerability, "_RATE_GATE", CallRateGate(
        answerability.RATE_VARIABLE, "answerability judge", answerability.AnswerabilityRateLimited))


def _result(verdicts):
    return _FakeSearch(verdicts)(None, None, "q?", k=5)


def _judge(verdicts, env=ON):
    transport = answerability.resolve_answerability_judge(env)
    assert transport is not None
    return answerability.judged(_result(verdicts), 5, transport).query_support


def test_a1_off_by_default_and_refuses_a_bad_setting() -> None:
    assert answerability.resolve_answerability_judge({}) is None
    assert answerability.resolve_answerability_judge({"RECALL_ANSWERABILITY_JUDGE": "0"}) is None
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        answerability.resolve_answerability_judge({"RECALL_ANSWERABILITY_JUDGE": "1"})
    with pytest.raises(ValueError, match="not a boolean"):
        answerability.resolve_answerability_judge({"RECALL_ANSWERABILITY_JUDGE": "ture"})
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        _validate_runtime_options({"RECALL_ANSWERABILITY_JUDGE": "1"})


def test_a2_the_request_is_the_measured_one(opener) -> None:
    fake = opener('{"stated": true, "confidence": 97}')
    support = _judge(["ok", "superseded", "ok"])
    assert support.status == "stated"
    (body,) = fake.bodies
    assert {k: v for k, v in body.items() if k != "messages"} == {
        "model": "deepseek/deepseek-v4-pro-0813", "temperature": 0, "max_tokens": 8000,
        "usage": {"include": True}, "reasoning": {"enabled": True},
    }, "the request is not the measured configuration"
    system, user = body["messages"]
    assert system == {"role": "system", "content": answerability.JUDGE_SYSTEM}
    assert json.loads(user["content"]) == {"question": "q?", "evidence": [
        {"chunk_id": "c0", "source": "note0.md", "text": "text 0"},
        {"chunk_id": "c2", "source": "note2.md", "text": "text 2"},
    ]}, "the judge saw evidence that is not citable"
    assert fake.headers[0]["Authorization"] == "Bearer sk-or-offline-fixture"


def test_a3_the_verdict_decides_not_the_confidence(opener) -> None:
    opener('{"stated": false, "confidence": 99}')
    assert _judge(["ok"]) == QuerySupport(
        status="not_stated", method="llm_judge", model=answerability.MODEL)


def test_a4_failures_are_distinct_from_verdicts(opener) -> None:
    opener(OSError("connection reset"))
    assert _judge(["ok"]).status == "failed"
    fake = opener("no json", '{"stated": "yes", "confidence": 90}', '{"stated": true, "confidence": 101}')
    assert _judge(["ok"]).status == "unparsed"
    assert len(fake.bodies) == 3
    opener("no json", '{"stated": true, "confidence": 80}')
    assert _judge(["ok"]).status == "stated", "a refused reply was not retried"


def test_a5_nothing_citable_sends_nothing(opener) -> None:
    fake = opener()
    assert _judge(["superseded"]) == QuerySupport()
    assert fake.bodies == []


def test_a6_search_and_evidence_carry_the_verdict_only(opener) -> None:
    plain_search, plain_evidence = _both(["ok", "superseded", "ok"])
    fake = opener('{"stated": false, "confidence": 3}', '{"stated": false, "confidence": 3}')
    search, evidence = _both(["ok", "superseded", "ok"], env=ON)
    assert len(fake.bodies) == 2
    assert search.query_support.status == "not_stated", "recall_search did not carry the verdict"
    assert evidence.query_support.status == "not_stated", "recall_evidence did not carry the verdict"
    assert plain_search.query_support.status == "not_checked"

    def rest(model):
        data = model.model_dump()
        return {k: v for k, v in data.items() if k not in {"query_support"} and not k.endswith("_ms")}
    assert rest(search) == rest(plain_search), "the judge changed something besides query_support"
    assert rest(evidence) == rest(plain_evidence), "the judge changed something besides query_support"


def test_a7_the_decoder_keeps_and_checks_judged_support() -> None:
    payload = _response().to_dict()
    judged = {"status": "stated", "scope": "returned_evidence", "method": "llm_judge", "model": "m"}
    payload["trusted_evidence"]["query_support"] = judged
    try:
        decoded = reasoning_response_from_dict(payload)
    except EvidenceValidationError as exc:
        pytest.fail(f"the decoder refused a judged support: {exc}")
    assert asdict(decoded.trusted_evidence.query_support) == judged
    payload["trusted_evidence"]["query_support"] = {**judged, "method": None}
    with pytest.raises(EvidenceValidationError, match="exactly when a judge ran"):
        reasoning_response_from_dict(payload)


def _reply(text: str) -> bytes:
    return json.dumps({"choices": [{"message": {"content": text}}]}).encode()


class _Clock:
    """`answerability.time`, advanced only by the test."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def at(self, when: float, reply):
        def answer():
            self.now = when
            return reply
        return answer


def test_a8_every_attempt_shares_one_deadline(opener, monkeypatch) -> None:
    clock = _Clock()
    monkeypatch.setattr(answerability, "time", clock)
    fake = opener(clock.at(50.0, "no json"), clock.at(61.0, '{"stated": true, "confidence": 90}'))
    assert _judge(["ok"]).status == "failed", "a reply was taken past the shared deadline"
    assert fake.timeouts == [answerability.TIMEOUT_SECONDS, answerability.TIMEOUT_SECONDS - 50.0], (
        "an attempt got a fresh timeout instead of the time remaining"
    )


class _TrickleSocket:
    """A socket whose reply headers arrive at once and whose body arrives one byte per receive,
    each 25 s after the last: inside any socket timeout, so urllib's timeout never fires.

    Read through a real `http.client.HTTPResponse` over an `io.BufferedReader`, the stack urllib
    returns, so `read(n)` waits for n bytes or the end of the body exactly as it does in production.
    """

    def __init__(self, clock: _Clock, body: bytes) -> None:
        head = b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % len(body)
        self.raw = _TrickleRaw(clock, head, body)

    def makefile(self, mode: str) -> io.BufferedReader:
        return io.BufferedReader(self.raw)


class _TrickleRaw(io.RawIOBase):
    def __init__(self, clock: _Clock, head: bytes, body: bytes) -> None:
        self.clock, self.head, self.body = clock, head, body

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        if self.head:
            size = min(len(buffer), len(self.head))
            buffer[:size], self.head = self.head[:size], self.head[size:]
            return size
        if not self.body:
            return 0
        self.clock.now += 25.0
        buffer[0], self.body = self.body[0], self.body[1:]
        return 1


def _trickling_response(clock: _Clock, text: str) -> http.client.HTTPResponse:
    response = http.client.HTTPResponse(_TrickleSocket(clock, _reply(text)))  # type: ignore[arg-type]  # a fake socket: HTTPResponse only calls makefile
    response.begin()
    return response


def test_a13_a_trickling_reply_is_abandoned_at_the_deadline(opener, monkeypatch) -> None:
    """FA-1-1 and FA-2-1 (CCA run 20261128T104334Z, fix audit passes 1 and 2): a reply body that
    keeps arriving inside every socket timeout is abandoned at the first piece read after the
    shared deadline, so the judgement's own thread ends there instead of reading on.
    """
    clock = _Clock()
    monkeypatch.setattr(answerability, "time", clock)
    opener(_trickling_response(clock, '{"stated": true, "confidence": 90}'))
    assert _judge(["ok"]).status == "failed", "a trickling reply was read past the deadline"
    assert clock.now < answerability.TIMEOUT_SECONDS + 25.0, "the read went on after the deadline"


def test_a14_the_search_waits_no_longer_than_the_deadline(opener, monkeypatch) -> None:
    """FA-2-2 (CCA run 20261128T104334Z, fix audit pass 2): a call stuck where no socket read
    returns (resolving a name, connecting, waiting for headers) does not hold the search past
    `TIMEOUT_SECONDS`; it keeps its in-flight slot until it ends.
    """
    stuck = threading.Event()
    busy = threading.BoundedSemaphore(1)
    monkeypatch.setattr(answerability, "_IN_FLIGHT", busy)
    monkeypatch.setattr(answerability, "TIMEOUT_SECONDS", 0.2)

    def hang():
        stuck.wait(10.0)
        return '{"stated": true, "confidence": 90}'

    opener(hang)
    started = time.monotonic()
    try:
        assert _judge(["ok"]).status == "failed", "a stuck call produced a verdict"
        assert time.monotonic() - started < 5.0, "the search waited for a stuck call"
        assert not busy.acquire(blocking=False), "a call still running gave up its slot"
    finally:
        stuck.set()
    assert busy.acquire(timeout=10.0), "the call never gave its slot back once it ended"


def test_a9_a_busy_judge_is_failed_without_a_call(opener, monkeypatch) -> None:
    busy = threading.BoundedSemaphore(1)
    busy.acquire()
    monkeypatch.setattr(answerability, "_IN_FLIGHT", busy)
    fake = opener('{"stated": true, "confidence": 90}')
    assert _judge(["ok"]).status == "failed", "a search waited for a busy judge"
    assert fake.bodies == []
    busy.release()
    assert _judge(["ok"]).status == "stated"
    assert busy.acquire(blocking=False), "the judge did not give its slot back"


def test_a10_the_call_rate_ceiling_refuses_as_failed(opener) -> None:
    env = {**ON, answerability.RATE_VARIABLE: "1"}
    fake = opener('{"stated": true, "confidence": 90}', '{"stated": true, "confidence": 90}')
    assert _judge(["ok"], env).status == "stated"
    assert _judge(["ok"], env).status == "failed", "a call over the ceiling was made"
    assert len(fake.bodies) == 1
    with pytest.raises(ValueError, match=answerability.RATE_VARIABLE):
        _validate_runtime_options({**ON, answerability.RATE_VARIABLE: "0"})


def test_a11_both_tools_judge_the_same_bytes_and_time_the_judge(opener, monkeypatch) -> None:
    original = test_search_evidence._hit

    def walked(i, verdict, demoted_confidence=0.2):
        # An indexed chunk's source is the path as walked; the bundle shows the relative file.
        hit = original(i, verdict, demoted_confidence)
        return replace(hit, chunk=replace(hit.chunk, source=f"/corpus/root/note{i}.md"))

    monkeypatch.setattr(test_search_evidence, "_hit", walked)
    fake = opener('{"stated": true, "confidence": 90}', '{"stated": true, "confidence": 90}')
    search, evidence = _both(["ok", "ok"], env=ON)
    search_body, evidence_body = fake.bodies
    assert search_body["messages"] == evidence_body["messages"], "the two tools judged different bytes"
    evidence_sent = json.loads(search_body["messages"][1]["content"])["evidence"]
    assert [item["source"] for item in evidence_sent] == ["note0.md", "note1.md"]
    for model in (search, evidence):
        assert "answerability_judge" in model.stage_ms, "the judge's time is not its own stage"


def test_a12_the_cli_judges_what_it_prints(opener, capsys) -> None:
    fake = opener('{"stated": true, "confidence": 90}')
    transport = answerability.resolve_answerability_judge(ON)
    _print_evidence(_result(["ok", "ok", "ok"]), max_items=1, judge=transport)
    printed = json.loads(capsys.readouterr().out)
    (body,) = fake.bodies
    judged_ids = [item["chunk_id"] for item in json.loads(body["messages"][1]["content"])["evidence"]]
    assert judged_ids == [item["chunk_id"] for item in printed["bundle"]["items"]], (
        "the CLI judged passages it did not print"
    )
    assert printed["bundle"]["query_support"]["status"] == "stated"
