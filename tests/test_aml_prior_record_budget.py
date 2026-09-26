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
* ``test_served_c9_bounds_its_prior_records``: removing ``anchor_prior_records_max_chars=40_000``
  from the C9 variant (``recall_aml/variants.py``) failed ``== 40_000``; dropping
  ``max_prior_record_chars=`` from ``build_compiler`` (``recall_aml/__main__.py``) failed the
  builder assertion.
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


def _record(index: int, quote_chars: int) -> StoredCodingRecord:
    quote = (f"record {index} " + "x" * quote_chars)[:quote_chars]
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

    assert served.anchor_prior_records_max_chars == 40_000
    assert compiler is not None
    assert compiler._max_prior_record_chars == 40_000
