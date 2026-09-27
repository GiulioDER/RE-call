"""The prior records an anchored compile sends are bounded by encoded size, not only by count.

On the official Textual Full of 2026-09-26 a user with very large messages made the compile prompt
grow about 35,000 tokens per Add within a session (7k, 34k, 77k, 113k) until it passed
gpt-4o-mini's 128k window, and every later Add of that session was refused with HTTP 400 and kept
no compiled record. ``fit_prior_records`` keeps the newest records that fit the budget.

Red proof, 2026-09-26, each by mutating the named production line with this file unchanged,
watching the named assertion fail, then restoring it:

* ``test_an_oversized_set_keeps_the_newest_records_that_fit``: dropping from the END instead of
  the start in ``fit_prior_records`` (``entries[:len(entries) - start]``) failed
  ``sent == everything[len(everything) - expected:]`` (the oldest records were sent).
* ``test_an_oversized_set_keeps_the_newest_records_that_fit`` again: removing the ``while`` loop's
  ``total > budget_chars`` condition (dropping every record) failed ``len(sent) == expected``.
* ``test_a_set_within_the_budget_is_sent_unchanged``: making ``fit_prior_records`` always drop the
  oldest record failed the equality with the unbounded payload.
* ``test_served_c9_bounds_its_prior_records``: removing ``anchor_prior_records_max_chars``
  from the C9 variant (``recall_aml/variants.py``, then 40,000, now 145,000) failed the equality;
  dropping
  ``max_prior_record_chars=`` from ``build_compiler`` (``recall_aml/__main__.py``) failed the
  builder assertion.
* ``test_the_budget_counts_cjk_as_the_escaped_text_the_model_is_sent``, 2026-09-27: measuring
  each entry in ``fit_prior_records`` as raw text
  (``json.dumps(entry, ensure_ascii=False, separators=(",", ":"))`` in place of
  ``_encode_stored_data(entry)``) failed ``0 < len(sent) < len(everything)`` with ``24 < 24``,
  every record sent. The six tests above all passed under that mutation, since their text is
  ASCII.
* ``test_served_c9_prior_budget_keeps_the_payload_under_the_character_budget`` (first named
  ``..._inside_the_model_window``, renamed by audit cca789b, which showed it is not one), 2026-09-27: setting
  C9's ``anchor_prior_records_max_chars`` to 160,000 failed ``150000 + 17 + 160000 <= 300000``.
  ``test_served_c9_bounds_its_prior_records`` failed ``40000 == 145000`` before the raise.
"""

from __future__ import annotations

import json
import logging
import re
from types import SimpleNamespace
from typing import Any

import pytest

from recall_aml.compiler import (
    OpenAICompiler,
    StoredCodingRecord,
    _encode_stored_data,
    fit_prior_records,
)
from recall_aml.models import CodingMemoryRecord, Message

_STORED = re.compile(r"<stored_data>(.*?)</stored_data>", re.DOTALL)
MESSAGES = [Message(role="user", content="We moved the queue to postgres because redis lost jobs.")]


def _record(index: int, quote_chars: int, fill: str = "x") -> StoredCodingRecord:
    quote = (f"record {index} " + fill * quote_chars)[:quote_chars]
    record = CodingMemoryRecord.model_validate(
        {
            "kind": "procedure",
            "action": quote[:40],
            "evidence_spans": [
                {"message_ordinal": 0, "start": 0, "end": len(quote), "quote": quote}
            ],
            "evidence_quotes": [quote],
            "source_session_id": "s",
        }
    )
    return StoredCodingRecord(f"mem_{index:064x}", record)


def _sent(prior: list[StoredCodingRecord], **kwargs: Any) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []

    def create(**request: Any) -> Any:
        calls.append(request)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"records": []}'))]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    compiler = OpenAICompiler(client, sleep=lambda _: None, prior_record_mode="without-ids", **kwargs)
    compiler.compile_anchored_v3(MESSAGES, "s", prior)
    return json.loads(_STORED.search(calls[0]["messages"][1]["content"]).group(1))


def test_a_set_within_the_budget_is_sent_unchanged() -> None:
    prior = [_record(i, 300) for i in range(5)]

    assert _sent(prior, max_prior_record_chars=40_000) == _sent(prior)


def test_an_oversized_set_keeps_the_newest_records_that_fit(caplog: pytest.LogCaptureFixture) -> None:
    prior = [_record(i, 4_000) for i in range(24)]
    budget = 40_000
    with caplog.at_level(logging.INFO, logger="recall_aml"):
        sent = _sent(prior, max_prior_record_chars=budget)["prior_records"]

    everything = _sent(prior)["prior_records"]
    # The most that fits: the newest k records whose encoded list is within the budget.
    expected = max(
        k for k in range(len(everything) + 1) if len(_encode_stored_data(everything[len(everything) - k:])) <= budget  # type: ignore[arg-type]
    )
    assert len(sent) == expected
    assert 0 < expected < len(everything)
    assert sent == everything[len(everything) - expected:]
    assert sent[-1]["record"]["evidence_quotes"][0].startswith("record 23 ")
    assert len(_encode_stored_data(sent)) <= budget  # type: ignore[arg-type]
    assert any("compiler_prior_records_fitted" in r.getMessage() for r in caplog.records)


def test_the_budget_counts_cjk_as_the_escaped_text_the_model_is_sent() -> None:
    # The prompt carries ``ensure_ascii`` JSON, so one CJK character is six characters there
    # (``\uXXXX``) and about 0.6 gpt-4o-mini tokens per character, against about 0.19 for English.
    # Measured on raw text, this set fits 40,000 and all 24 records would go, about six times the
    # budget once escaped; that is the overflow the budget exists to stop.
    prior = [_record(i, 600, fill="缓") for i in range(24)]
    budget = 40_000
    everything = _sent(prior)["prior_records"]
    assert len(json.dumps(everything, ensure_ascii=False, separators=(",", ":"))) <= budget
    assert len(_encode_stored_data(everything)) > 4 * budget  # type: ignore[arg-type]

    sent = _sent(prior, max_prior_record_chars=budget)["prior_records"]

    assert 0 < len(sent) < len(everything)
    assert sent == everything[len(everything) - len(sent):]
    assert len(_encode_stored_data(sent)) <= budget  # type: ignore[arg-type]


def test_a_newest_record_over_the_budget_sends_none() -> None:
    prior = [_record(i, 4_000) for i in range(3)]

    assert _sent(prior, max_prior_record_chars=1_000)["prior_records"] == []


def test_the_helper_is_the_identity_without_a_budget() -> None:
    entries = [{"record": {"n": i}} for i in range(30)]

    assert fit_prior_records(entries, None) is entries


def test_a_non_positive_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="max_prior_record_chars"):
        OpenAICompiler(object(), max_prior_record_chars=0)


def test_served_c9_bounds_its_prior_records() -> None:
    from recall_aml.__main__ import build_compiler
    from recall_aml.config import HostedSettings
    from recall_aml.variants import variant

    served = variant("C9_routed_specialists_grounded_graph_atomic")
    settings = HostedSettings(
        database_url="postgresql://unused", api_key="k", git_commit="c", openrouter_api_key="or"
    )
    compiler = build_compiler(settings, served, client_factory=lambda **_: object())

    assert served.anchor_prior_records_max_chars == 145_000
    assert compiler is not None
    assert compiler._max_prior_record_chars == 145_000


def test_served_c9_prior_budget_keeps_the_payload_under_the_character_budget() -> None:
    """C9's two limits together stay under ``ANCHOR_PAYLOAD_BUDGET_CHARS``, so the anchor
    fitting never runs for it.

    The anchors-only limit is measured on ``{"session_id", "anchors"}``; the prior records add the
    key ``,"prior_records":`` and a list the budget bounds including its brackets. Raised from
    40,000 to 145,000 on 2026-09-27 (owner decision, after the audit measured 40,000 trimming
    ordinary Coding sessions).

    This is a CHARACTER invariant and not a window guarantee. It was first named as one, and audit
    cca789b (NUM-001) measured an accepted C9 payload of escaped CJK at 174,997 o200k tokens
    against a 125,600-token room; the window is kept by the resend without prior records, tested
    by ``test_a_400_with_prior_records_is_resent_once_without_them``.
    """
    from recall_aml.compiler import ANCHOR_PAYLOAD_BUDGET_CHARS
    from recall_aml.variants import variant

    served = variant("C9_routed_specialists_grounded_graph_atomic")
    anchors = served.anchor_compile_max_payload_chars
    prior = served.anchor_prior_records_max_chars
    assert anchors is not None and prior is not None
    assert anchors + len(',"prior_records":') + prior <= ANCHOR_PAYLOAD_BUDGET_CHARS


class _ContextLengthExceeded(Exception):
    """What the provider answers for a prompt past gpt-4o-mini's window: an HTTP 400."""

    status_code = 400


def _refusing_while_prior_records_are_sent() -> tuple[OpenAICompiler, list[dict[str, Any]]]:
    """A client that refuses any prompt still carrying prior records, and compiles otherwise."""
    calls: list[dict[str, Any]] = []

    def create(**request: Any) -> Any:
        stored = json.loads(_STORED.search(request["messages"][1]["content"]).group(1))
        calls.append(stored)
        if stored.get("prior_records"):
            raise _ContextLengthExceeded("maximum context length is 128000 tokens")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"records": []}'))]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    compiler = OpenAICompiler(
        client, sleep=lambda _: None, prior_record_mode="without-ids", max_prior_record_chars=145_000
    )
    return compiler, calls


def test_a_400_with_prior_records_is_resent_once_without_them(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A payload inside every character limit can still pass the model's window, because
    characters do not bound tokens (escaped CJK and random base64 run at 1.4 to 1.7 characters
    per o200k token). The Add's own anchors are what it must compile, so a 400 on a payload that
    carries prior records is resent once without them.

    Red proof, 2026-09-27 (audit cca789b, NUM-001): with ``recall_aml/compiler.py`` at
    ``dcd6ea44``, before the ``compiler_prior_records_dropped`` branch of
    ``OpenAICompiler._compile_anchored`` existed, ``compile_anchored_v3`` below raised
    ``_ContextLengthExceeded: maximum context length is 128000 tokens`` from the first attempt
    instead of resending. Restored, green.
    """
    compiler, calls = _refusing_while_prior_records_are_sent()
    prior = [_record(i, 600, fill="缓") for i in range(24)]

    with caplog.at_level(logging.INFO, logger="recall_aml"):
        compiler.compile_anchored_v3(MESSAGES, "s", prior)

    assert len(calls) == 2
    assert calls[0]["prior_records"] and calls[1]["prior_records"] == []
    assert calls[1]["anchors"] == calls[0]["anchors"]
    assert any("compiler_prior_records_dropped" in r.getMessage() for r in caplog.records)


def test_a_400_without_prior_records_still_raises_at_once() -> None:
    """Nothing is left to drop, and the identical request is refused identically."""
    compiler, calls = _refusing_while_prior_records_are_sent()

    def always_refuse(**request: Any) -> Any:
        calls.append(request)
        raise _ContextLengthExceeded("bad request")

    compiler._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=always_refuse))
    )
    with pytest.raises(_ContextLengthExceeded):
        compiler.compile_anchored_v3(MESSAGES, "s", [])

    assert len(calls) == 1


# ---------------------------------------------------------------- L6 advisories, audit cca789b
#
# Red proofs, 2026-09-27, each against ``recall_aml/compiler.py`` at ``d9b27633`` (the approved fix,
# before these advisories) or a named mutation, failing at the assertion named, then restored green:
#
# * ``test_a_400_on_the_last_attempt_still_gets_its_resend``: at d9b27633 the drop ran on the third
#   and last attempt and nothing followed, so ``compile_anchored_v3`` raised
#   ``_ContextLengthExceeded`` after 3 calls instead of compiling on a 4th.
# * ``test_the_drop_logs_the_providers_error_code``: at d9b27633 the log row had no ``error_code``,
#   failing ``getattr(dropped, "error_code", None) == "context_length_exceeded"``.
# * ``test_a_400_that_survives_the_drop_raises_after_one_resend``: passes at d9b27633 (it pins
#   behaviour the advisory found untested), so by mutation: deleting
#   ``sent = {**sent, "prior_records": []}`` from the drop branch resends the same prior records,
#   failing ``counts == [counts[0], 0]``. (A first mutation, forcing the drop guard to ``True``,
#   looped until the backoff overflowed: termination then rested on that one guard, so the drop
#   is now also bounded by ``prior_dropped``.)


class _RateLimited(Exception):
    status_code = 429


def _scripted(answers: list[Exception | None]) -> tuple[OpenAICompiler, list[dict[str, Any]]]:
    """Each call takes the next answer: an exception to raise, or None for an empty success."""
    calls: list[dict[str, Any]] = []

    def create(**request: Any) -> Any:
        calls.append(json.loads(_STORED.search(request["messages"][1]["content"]).group(1)))
        answer = answers[min(len(calls), len(answers)) - 1]
        if answer is not None:
            raise answer
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"records": []}'))]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    compiler = OpenAICompiler(
        client, sleep=lambda _: None, prior_record_mode="without-ids", max_prior_record_chars=145_000
    )
    return compiler, calls


def test_a_400_on_the_last_attempt_still_gets_its_resend() -> None:
    """Two 429s use up the ordinary attempts; the over-window 400 on the third must still be
    resent without prior records, or the drop never reaches the provider and the record is lost."""
    too_long = _ContextLengthExceeded("maximum context length is 128000 tokens")
    compiler, calls = _scripted([_RateLimited("slow down"), _RateLimited("slow down"), too_long, None])
    prior = [_record(i, 600, fill="缓") for i in range(24)]

    compiler.compile_anchored_v3(MESSAGES, "s", prior)

    counts = [len(call["prior_records"]) for call in calls]
    assert counts[0] > 0 and counts == [counts[0]] * 3 + [0]


def test_a_400_that_survives_the_drop_raises_after_one_resend() -> None:
    """Prior records present, and the anchors alone still refused: one resend, then the 400."""
    compiler, calls = _scripted([_ContextLengthExceeded("bad request")])
    prior = [_record(i, 600, fill="缓") for i in range(24)]

    with pytest.raises(_ContextLengthExceeded):
        compiler.compile_anchored_v3(MESSAGES, "s", prior)

    counts = [len(call["prior_records"]) for call in calls]
    assert counts[0] > 0 and counts == [counts[0], 0]


def test_the_drop_logs_the_providers_error_code(caplog: pytest.LogCaptureFixture) -> None:
    """A later cost count must be able to tell length refusals from other 400s that also cost the
    one extra call."""
    too_long = _ContextLengthExceeded("maximum context length is 128000 tokens")
    too_long.code = "context_length_exceeded"  # type: ignore[attr-defined]
    compiler, _ = _scripted([too_long, None])

    with caplog.at_level(logging.INFO, logger="recall_aml"):
        compiler.compile_anchored_v3(MESSAGES, "s", [_record(i, 600) for i in range(3)])

    dropped = next(r for r in caplog.records if "compiler_prior_records_dropped" in r.getMessage())
    assert getattr(dropped, "error_code", None) == "context_length_exceeded"
    assert getattr(dropped, "attempt", None) == 1
