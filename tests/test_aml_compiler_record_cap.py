"""An anchored compiler answer longer than our caps keeps its first entries instead of failing.

Found on the live C9 control Add of 2026-09-24: gpt-4o-mini proposed nine records, the payload
schema capped the list at eight and rejected the whole answer, the three retries resent the
identical prompt, and the Add stored no compiled record. Each behaviour test records the mutation
it was proved red against, with this file unchanged.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

import recall_aml.compiler as compiler_module
from recall_aml.compiler import OpenAICompiler
from recall_aml.models import (
    AnchoredCodingMemoryProposal,
    AnchoredCompilerPayload,
    Message,
)


def proposal(action: str, anchor_id: str = "a000_0000000000000000", **fields: Any) -> dict:
    return {
        "kind": "successful repair",
        "action": action,
        "evidence_anchor_ids": [anchor_id],
        "source_session_id": "session",
        **fields,
    }


def test_a_ninth_proposal_is_dropped_not_fatal():
    """Red proof: `AnchoredCompilerPayload.keep_first_records` deleted. This test then failed on
    `assert payload is not None` (pydantic: List should have at most 8 items, not 9)."""
    raw = {"records": [proposal(f"step{i}") for i in range(9)]}

    try:
        payload = AnchoredCompilerPayload.model_validate_json(json.dumps(raw))
    except ValidationError:
        payload = None

    assert payload is not None
    assert [record.action for record in payload.records] == [f"step{i}" for i in range(8)]


@pytest.mark.parametrize(
    ("field", "size", "kept"),
    [("entities", 33, 32), ("evidence_anchor_ids", 9, 8), ("supersedes", 9, 8)],
)
def test_an_over_long_proposal_list_keeps_its_prefix(field, size, kept):
    """Red proof, two mutations, each run against all three parameters:
    `keep_first_entities` deleted failed the `entities` case, and `keep_first_references`
    deleted failed the `evidence_anchor_ids` and `supersedes` cases, each on
    `assert parsed is not None` (pydantic `too_long`)."""
    values = [f"a{i:03d}_{i:016d}" for i in range(size)]
    raw = {**proposal("step0"), field: values}

    try:
        parsed = AnchoredCodingMemoryProposal.model_validate_json(json.dumps(raw))
    except ValidationError:
        parsed = None

    assert parsed is not None
    assert getattr(parsed, field) == values[:kept]


def test_a_nine_record_answer_compiles_eight_records_in_one_call():
    """The live failure end to end: one provider call, the first eight records kept, in order.

    Red proof: `AnchoredCompilerPayload.keep_first_records` deleted. This test then failed on
    `assert records is not None`: every attempt was rejected and the compile raised.
    """
    content = " ".join(f"step{i} fixed module{i}." for i in range(9))
    messages = [Message(role="assistant", content=content)]
    anchor = compiler_module.build_evidence_anchors(messages, "session", identifier_version=3)[0]
    answer = {
        "records": [proposal(f"step{i} fixed module{i}.", anchor.id) for i in range(9)]
    }
    calls: list[dict] = []

    def complete(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(answer)))]
        )

    compiler = OpenAICompiler(
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete))),
        sleep=lambda _: None,
    )

    try:
        records = compiler.compile_anchored_v3(messages, "session", [])
    except ValidationError:
        records = None

    assert records is not None
    assert len(calls) == 1
    assert [record.action for record in records] == [
        f"step{i} fixed module{i}." for i in range(8)
    ]
