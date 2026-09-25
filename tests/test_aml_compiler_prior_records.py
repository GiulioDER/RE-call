"""What an anchored compile sends of the session's earlier compiled records.

docs/preregistrations/2026-09-25-c9-prior-record-ids.md: gpt-4o-mini cited the ids of the prompt's
``prior_records`` as evidence anchors (41 of 41 unknown ids in the fallback calls of the
2026-09-25 replay), which rejects the record, while those ids' one use, ``supersedes``, was never
set. ``prior_record_mode`` chooses what is sent.

Red proof, 2026-09-25, each by mutating ``OpenAICompiler._compile_anchored`` or
``OpenAICompiler.__init__`` in ``recall_aml/compiler.py`` with this file unchanged, then restoring:

* ``test_each_mode_sends_the_prior_records_it_names``: sending ``{"id": item.id}`` in every mode
  failed the ``without-ids`` assertion ``"id" not in entry``.
* ``test_the_default_payload_is_the_one_v3_always_sent``: a default of ``"without-ids"`` failed
  the equality with the payload as v3 built it before the mode existed.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any

import pytest

from recall_aml.compiler import OpenAICompiler, StoredCodingRecord
from recall_aml.models import CodingMemoryRecord, Message

_STORED = re.compile(r"<stored_data>(.*?)</stored_data>", re.DOTALL)
MESSAGES = [Message(role="user", content="We moved the queue to postgres because redis lost jobs.")]


def _prior() -> list[StoredCodingRecord]:
    record = CodingMemoryRecord.model_validate(
        {
            "kind": "architectural decision",
            "action": "moved the queue to postgres",
            "evidence_spans": [
                {"message_ordinal": 0, "start": 0, "end": 8, "quote": "We moved"}
            ],
            "source_session_id": "s",
        }
    )
    return [StoredCodingRecord("mem_" + "ab" * 32, record)]


def _sent_payload(**kwargs: Any) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []

    def create(**request: Any) -> Any:
        calls.append(request)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"records": []}'))]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    compiler = OpenAICompiler(client, sleep=lambda _: None, **kwargs)
    compiler.compile_anchored_v3(MESSAGES, "s", _prior())
    return json.loads(_STORED.search(calls[0]["messages"][1]["content"]).group(1))


def test_each_mode_sends_the_prior_records_it_names() -> None:
    with_ids = _sent_payload(prior_record_mode="with-ids")["prior_records"]
    without_ids = _sent_payload(prior_record_mode="without-ids")["prior_records"]
    none = _sent_payload(prior_record_mode="none")

    assert [entry["id"] for entry in with_ids] == ["mem_" + "ab" * 32]
    assert all("id" not in entry for entry in without_ids)
    assert [entry["record"] for entry in without_ids] == [entry["record"] for entry in with_ids]
    assert "prior_records" not in none
    assert none["anchors"]


def test_the_default_payload_is_the_one_v3_always_sent() -> None:
    sent = _sent_payload()
    before = [
        {"id": item.id, "record": item.record.model_dump(mode="json")} for item in _prior()
    ]

    assert list(sent) == ["session_id", "anchors", "prior_records"]
    assert sent["prior_records"] == json.loads(json.dumps(before))


def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="prior_record_mode"):
        OpenAICompiler(object(), prior_record_mode="some")
