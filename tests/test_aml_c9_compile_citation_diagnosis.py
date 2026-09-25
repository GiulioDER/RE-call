"""The citation classifier of ``scripts/aml_c9_compile_citation_diagnosis.py`` sorts ids correctly.

The diagnosis is only as good as this sort: a wrong rule would report a class that did not occur
and steer the round 2 fix. Red proof, 2026-09-25, each by mutating the named line of
``classify`` with this file unchanged, then restoring it:

* ``test_each_unknown_citation_lands_in_its_own_class``: deleting the
  ``if digest in hashes: return "hash_of_other_anchor"`` branch failed the
  ``a001_5555666677778888`` case, reported as ``right_index_wrong_hash``.
* ``test_the_sent_ids_are_read_from_the_request_the_model_saw``: making ``sent_anchor_ids`` search
  ``request_messages[:1]`` only failed on the assertion line, where ``sent_anchor_ids`` raised
  ``ValueError: request carried no <stored_data>``, because the first message is the system
  prompt.
"""

from __future__ import annotations

import json

import pytest

from recall_aml.compiler import _encode_stored_data, build_evidence_anchors
from recall_aml.models import Message
from scripts.aml_c9_compile_citation_diagnosis import classify, sent_anchor_ids

SENT = ["a000_1111222233334444", "a001_aaaabbbbccccdddd", "a002_5555666677778888"]


@pytest.mark.parametrize(
    ("cited", "expected"),
    [
        ("a001_aaaabbbbccccdddd", "known"),
        ("a001", "bare_index_known"),
        ("a009", "bare_index_out_of_range"),
        ("a001_aaaabbbb", "truncated_id"),
        ("a001_aaaabbbbccccdd00", "right_index_wrong_hash"),
        ("a001_5555666677778888", "hash_of_other_anchor"),
        ("a007_0123456789abcdef", "index_out_of_range"),
        ("a007_1111222233334444", "hash_of_sent_anchor"),
        ("anchor_9f3c", "v2_anchor_form"),
        ("Caroline went to the support group", "quoted_text"),
        ("evidence1", "other"),
    ],
)
def test_each_unknown_citation_lands_in_its_own_class(cited: str, expected: str) -> None:
    assert classify(cited, SENT) == expected


def test_the_sent_ids_are_read_from_the_request_the_model_saw() -> None:
    messages = [Message(role="user", content="topic one " * 200)]
    anchors = build_evidence_anchors(messages, "s", identifier_version=3)
    payload = {"session_id": "s", "anchors": [{"id": a.id, "quote": a.quote} for a in anchors]}
    request = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": f"<stored_data>{_encode_stored_data(payload)}</stored_data>"},
    ]

    assert sent_anchor_ids(request) == [a.id for a in anchors]
    assert json.loads(json.dumps(sent_anchor_ids(request)))
