"""The compile journal lines name their Add and show which unknown anchor ids were cited.

Round 2 step 0 of the compiler-fallback plan (2026-09-25): on the Textual smoke, 25 of 134 C9
Adds fell back, mostly because every proposed record cited anchor ids that were never sent, but
``compiler_anchor_compile_complete`` carried neither the Add's ``request_digest`` nor the ids, so
neither which Add nor what was cited could be read back. Logging only; no behaviour changes.

Red proof, 2026-09-25, each by mutating the named production line with this file unchanged and
watching the named test fail on its assertion, then restoring it:

* ``test_a_compile_line_names_its_add_and_samples_unknown_ids``: replacing the
  ``fields["request_digest"] = request_digest`` line in ``_log_diagnostics``
  (``recall_aml/compiler.py``) with ``pass`` failed ``assert event["request_digest"] == DIGEST``
  with ``KeyError: 'request_digest'``.
* ``test_a_quoted_phrase_is_never_logged_as_an_id``: making ``unknown_anchor_id_sample`` return
  ``cited`` unconditionally failed the ``<non-id:`` assertion.
* ``test_the_service_names_the_add_inside_the_compile_thread``: setting
  ``COMPILE_REQUEST_DIGEST`` to ``None`` instead of the request digest in
  ``HostedService._add_once`` (``recall_aml/service.py``) failed ``assert seen == [expected]``
  with ``[None]``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import recall_aml.compiler as compiler_module
from recall_aml.compiler import COMPILE_REQUEST_DIGEST, OpenAICompiler, unknown_anchor_id_sample
from recall_aml.identity import canonical_digest
from recall_aml.models import Message
from tests.test_aml_c9_coding_hardening import RecordingRepository, add, c9_client

DIGEST = "0123456789abcdef"
PHRASE = "Caroline went to the support group"


def _messages() -> list[Message]:
    words = " ".join(f"topic{i} detail{i} value{i}" for i in range(360))
    return [Message(role="user", content=words)]


def _compiler(answer: dict[str, Any]) -> OpenAICompiler:
    def complete(**_: Any) -> Any:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(answer)))]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    return OpenAICompiler(client, sleep=lambda _: None)


def _proposal(anchor_ids: list[str]) -> dict[str, Any]:
    # Fields quote nothing from the text, so exact-text recovery cannot rescue the record.
    return {
        "kind": "architectural decision",
        "action": "the user settled on a plan",
        "evidence_anchor_ids": anchor_ids,
        "source_session_id": "session",
    }


def _compile_event(caplog: Any, cited: list[str]) -> dict[str, Any]:
    compiler = _compiler({"records": [_proposal(cited)]})
    token = COMPILE_REQUEST_DIGEST.set(DIGEST)
    try:
        with caplog.at_level("INFO", logger="recall_aml"):
            try:
                compiler.compile_anchored_v3(_messages(), "session", [])
            except Exception:  # BROAD-CATCH: a fully rejected answer may raise; the log is the subject
                pass
    finally:
        COMPILE_REQUEST_DIGEST.reset(token)
    line = next(
        item.message
        for item in caplog.records
        if item.message.startswith("compiler_anchor_compile_complete ")
    )
    return json.loads(line.removeprefix("compiler_anchor_compile_complete "))


def test_a_compile_line_names_its_add_and_samples_unknown_ids(caplog) -> None:
    anchors = compiler_module.build_evidence_anchors(_messages(), "session", identifier_version=3)
    cited = ["a999_deadbeefdeadbeef", "anchor_0badc0de", "a001_0000000000000000", "a002_ffff"]

    event = _compile_event(caplog, cited)

    assert event["request_digest"] == DIGEST
    assert event["sent_anchor_count"] == len(anchors)
    assert event["invalid_anchor_references"] == 4
    assert event["unknown_anchor_id_samples"] == cited[:3]


def test_a_quoted_phrase_is_never_logged_as_an_id(caplog) -> None:
    event = _compile_event(caplog, [PHRASE])

    assert event["unknown_anchor_id_samples"] == [f"<non-id:{len(PHRASE)} chars>"]
    assert all(PHRASE not in item.message for item in caplog.records)
    assert unknown_anchor_id_sample("a012") == "a012"


def test_the_service_names_the_add_inside_the_compile_thread(caplog) -> None:
    seen: list[str | None] = []

    class CapturingCompiler:
        def compile_anchored_v3(self, *args: Any) -> list[Any]:
            seen.append(COMPILE_REQUEST_DIGEST.get())
            raise TimeoutError("provider timeout")

    client = c9_client(RecordingRepository(), CapturingCompiler())
    with caplog.at_level("INFO", logger="recall_aml"):
        response = add(client, [{"role": "user", "content": "we chose postgres for the queue"}])

    expected = canonical_digest("r1")[:16]
    assert response.status_code == 200
    assert seen == [expected]
    assert COMPILE_REQUEST_DIGEST.get() is None
    fallback = [item for item in caplog.records if item.message == "hosted_compiler_fallback"]
    assert [(item.request_digest, item.error_class) for item in fallback] == [
        (expected, "TimeoutError")
    ]


def test_the_fallback_line_names_the_http_status(caplog, monkeypatch) -> None:
    """A 402 (credit exhausted) is told apart from other ``APIStatusError`` answers.

    Red proof, 2026-09-26: replacing ``fallback_fields["http_status"] = http_status`` in
    ``HostedService._add_once`` (``recall_aml/service.py``) with ``pass`` failed the last
    assertion with ``assert [None] == [402]``.

    Since 2026-09-28 C9 stops an Add on a 402 instead (see the tests below), so this runs with that
    switched off to keep covering the fallback line.
    """
    monkeypatch.setenv("RECALL_AML_STOP_ON_CREDIT_EXHAUSTED", "0")

    class CreditExhausted(Exception):
        status_code = 402

    class RefusedCompiler:
        def compile_anchored_v3(self, *args: Any) -> list[Any]:
            raise CreditExhausted("insufficient credits")

    client = c9_client(RecordingRepository(), RefusedCompiler())
    with caplog.at_level("INFO", logger="recall_aml"):
        response = add(client, [{"role": "user", "content": "we chose postgres for the queue"}])

    assert response.status_code == 200
    fallback = [item for item in caplog.records if item.message == "hosted_compiler_fallback"]
    assert [getattr(item, "http_status", None) for item in fallback] == [402]


def test_a_prior_record_id_is_logged_verbatim() -> None:
    """A prior compiled record's id is an id, and the class the 2026-09-25 replay found.

    Red proof: removing the ``mem_[0-9a-fA-F]{1,64}`` alternative from ``_ID_SHAPED`` in
    ``recall_aml/compiler.py`` logged it as ``<non-id:68 chars>`` and failed this assertion.
    """
    prior_id = "mem_" + "4ea0ca6794ffd978" * 4

    assert unknown_anchor_id_sample(prior_id) == prior_id


# ------------------------------------------------------------------ stop on exhausted credit
#
# 2026-09-28, owner request: when the compile provider is out of credit (HTTP 402), C9 fails the
# Add 503 with Retry-After instead of storing it raw-only, so AML retries it and, failing that,
# the run stops and is resumed. Red proofs, each by mutating the named production line of
# ``recall_aml/service.py`` or ``recall_aml/app.py`` with this file unchanged, then restoring:
#
# * ``test_c9_stops_an_add_when_the_compile_credit_is_out``: deleting the ``if http_status == 402
#   and self.stop_on_credit_exhausted:`` branch in ``HostedService._compile_and_persist`` failed
#   ``response.status_code == 503`` with 200 (the Add was stored raw-only).
# * ``test_c9_stops_an_add_when_the_compile_credit_is_out`` again: deleting the
#   ``except CompilerCreditExhausted`` clause in ``protected`` (``recall_aml/app.py``) failed
#   ``response.headers["retry-after"] == "60"`` (the generic 503 carries no Retry-After).
# * ``test_the_retried_add_goes_through_once_the_credit_is_back``: building the chunks before the
#   compile would store the first attempt; this test pins that nothing is stored or receipted, so
#   the retry is not a 409 conflict. Mutation used: recording a receipt in the 402 branch before
#   raising failed ``repository.receipts == []``.
# * ``test_a_bad_stop_on_credit_value_stops_startup``: dropping ``self.stop_on_credit_exhausted``
#   from the startup validation tuple failed with DID NOT RAISE.


class _OutOfCredit(Exception):
    status_code = 402


class _RefusingCompiler:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def compile_anchored_v3(self, *args: Any) -> list[Any]:
        self.calls += 1
        raise self.error


MESSAGE = [{"role": "user", "content": "we chose postgres for the queue"}]


def test_c9_stops_an_add_when_the_compile_credit_is_out(caplog) -> None:
    repository = RecordingRepository()
    client = c9_client(repository, _RefusingCompiler(_OutOfCredit("insufficient credits")))
    with caplog.at_level("INFO", logger="recall_aml"):
        response = add(client, MESSAGE)

    assert response.status_code == 503
    assert response.headers["retry-after"] == "60"
    assert response.json() == {"error": "compiler_credit_exhausted"}
    assert repository.persisted == {}
    assert repository.receipts == []
    assert any(r.message == "hosted_compiler_credit_exhausted" for r in caplog.records)
    assert not any(r.message == "hosted_compiler_fallback" for r in caplog.records)


def test_other_compile_failures_still_fall_back() -> None:
    """Only a 402 stops the Add; any other provider status keeps the raw-only fallback."""

    class BadRequest(Exception):
        status_code = 400

    repository = RecordingRepository()
    response = add(c9_client(repository, _RefusingCompiler(BadRequest("bad"))), MESSAGE)

    assert response.status_code == 200
    assert response.json()["compiler_fallback"] is True
    assert repository.receipts == ["r1"]


def test_the_retried_add_goes_through_once_the_credit_is_back() -> None:
    """AML resends the same request_id and payload; with nothing stored by the refused attempt,
    the resend is processed afresh rather than answered 409."""
    repository = RecordingRepository()
    refused = add(c9_client(repository, _RefusingCompiler(_OutOfCredit("insufficient"))), MESSAGE)
    assert refused.status_code == 503
    assert repository.receipts == []

    retried = add(c9_client(repository, _RefusingCompiler(TimeoutError("slow"))), MESSAGE)

    assert retried.status_code == 200, retried.text
    assert repository.receipts == ["r1"]


def test_a_bad_stop_on_credit_value_stops_startup(monkeypatch) -> None:
    import pytest

    monkeypatch.setenv("RECALL_AML_STOP_ON_CREDIT_EXHAUSTED", "yes")
    with pytest.raises(ValueError, match="RECALL_AML_STOP_ON_CREDIT_EXHAUSTED must be 1 or 0"):
        c9_client(RecordingRepository(), _RefusingCompiler(TimeoutError("x")))


def test_c9_reports_the_stop_on_credit_setting() -> None:
    from recall_aml.variants import variant

    assert variant("C9_routed_specialists_grounded_graph_atomic").stop_on_credit_exhausted is True
    assert variant("C7_routed_specialists").stop_on_credit_exhausted is False
    client = c9_client(RecordingRepository(), _RefusingCompiler(TimeoutError("x")))
    assert client.get("/version", headers={"X-Api-Key": "k"}).json()["stop_on_credit_exhausted"] is True

