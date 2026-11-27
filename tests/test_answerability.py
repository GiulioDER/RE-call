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

Red proof, 2026-11-28, VPS2, each mutation applied alone to `recall/answerability.py` or the named
file, then reverted; every test below failed in its own assertion:
- A1 `test_a1_off_by_default_and_refuses_a_bad_setting`: `resolve_answerability_judge` returning
  None without reading the key (the `raise ValueError` for an empty key deleted).
- A2 `test_a2_the_request_is_the_measured_one`: `"reasoning": {"enabled": True}` removed from the
  body; and `citable_evidence` dropping its `if hit.verdict == "ok"` filter.
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

import io
import json
from dataclasses import asdict

import pytest

from recall import answerability
from recall.evidence import EvidenceValidationError
from recall.reasoning_response import reasoning_response_from_dict
from recall.support import QuerySupport
from recall_mcp.settings import _validate_runtime_options
from tests.test_query_support_reasoning import _response
from tests.test_search_evidence import _both, _hit

ON = {"RECALL_ANSWERABILITY_JUDGE": "1", "OPENROUTER_API_KEY": "sk-or-offline-fixture"}


class _Opener:
    """Records each request and answers with the next canned assistant text."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.bodies: list[dict] = []
        self.headers: list[dict] = []

    def open(self, req, timeout):
        self.bodies.append(json.loads(req.data))
        self.headers.append(dict(req.header_items()))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": reply}}]}).encode())


@pytest.fixture
def opener(monkeypatch):
    def install(*replies):
        fake = _Opener(*replies)
        monkeypatch.setattr(answerability, "_OPENER", fake)
        return fake
    return install


def _judge(hits):
    transport = answerability.resolve_answerability_judge(ON)
    assert transport is not None
    return answerability.judge_query_support("q?", answerability.citable_evidence(hits), transport)


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
    support = _judge([_hit(0, "ok"), _hit(1, "superseded"), _hit(2, "ok")])
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
    assert _judge([_hit(0, "ok")]) == QuerySupport(
        status="not_stated", method="llm_judge", model=answerability.MODEL)


def test_a4_failures_are_distinct_from_verdicts(opener) -> None:
    opener(OSError("connection reset"))
    assert _judge([_hit(0, "ok")]).status == "failed"
    fake = opener("no json", '{"stated": "yes", "confidence": 90}', '{"stated": true, "confidence": 101}')
    assert _judge([_hit(0, "ok")]).status == "unparsed"
    assert len(fake.bodies) == 3
    opener("no json", '{"stated": true, "confidence": 80}')
    assert _judge([_hit(0, "ok")]).status == "stated", "a refused reply was not retried"


def test_a5_nothing_citable_sends_nothing(opener) -> None:
    fake = opener()
    assert _judge([_hit(0, "superseded")]) == QuerySupport()
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
