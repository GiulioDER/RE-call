"""Regression tests for Lane 1 proof obligations.

Red proof receipt: after the green implementation, replace the equality check in
``validate_proof_assessment`` that compares ``item.text[span.start : span.end]`` to
``span.quote`` with ``True``.  ``test_rejects_forged_span_quote`` fails because the forged
quote would otherwise become accepted trusted support.  Restore the equality check before the
green run.
"""

from __future__ import annotations

import json

import pytest

from recall.evidence import EvidenceBundle, EvidenceItem
from recall.proof_obligations import (
    ProofValidationError,
    run_proof_obligations,
    validate_proof_assessment,
    parse_proof_assessment,
)


def _bundle(*items: EvidenceItem, generation: str = "generation-1") -> EvidenceBundle:
    return EvidenceBundle(
        query="Which release date applies to project alpha?",
        decision="answer",
        reason_code=None,
        decision_state="supported",
        calibrated=True,
        stale=False,
        embedding_profile="test-profile",
        retrieval_profile="test-retrieval",
        index_generation=generation,
        items=items,
        trust_state="trusted",
    )


def _item(chunk_id: str, text: str) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=chunk_id,
        text=text,
        source="alpha.md",
        ordinal=1,
        indexed_at=None,
        valid_from=None,
        valid_until=None,
        cosine=0.9,
        confidence=0.9,
    )


def _assessment(*, date_supported: bool, repair: object | None) -> dict[str, object]:
    support: list[dict[str, object]] = [
        {"slot_id": "project", "chunk_id": "one", "start": 0, "end": 13, "quote": "Project alpha"}
    ]
    if date_supported:
        support.append(
            {"slot_id": "release_date", "chunk_id": "two", "start": 15, "end": 25, "quote": "2026-10-15"}
        )
    return {
        "schema_version": 1,
        "slots": [
            {"slot_id": "project", "kind": "entity", "requirement": "project identity", "required": True},
            {"slot_id": "release_date", "kind": "time", "requirement": "release date", "required": True},
        ],
        "support": support,
        "repair": repair,
    }


def test_rejects_forged_span_quote() -> None:
    bundle = _bundle(_item("one", "Project alpha ships."))
    payload = _assessment(date_supported=False, repair=None)
    payload["support"] = [
        {"slot_id": "project", "chunk_id": "one", "start": 0, "end": 13, "quote": "Project beta"}
    ]
    with pytest.raises(ProofValidationError, match="quote does not match"):
        validate_proof_assessment(parse_proof_assessment(payload), bundle)


def test_missing_required_slot_without_repair_abstains() -> None:
    initial = _bundle(_item("one", "Project alpha ships."))
    calls: list[str] = []

    def provider(_system: str, _user: str) -> str:
        calls.append("proof")
        return json.dumps(_assessment(date_supported=False, repair=None))

    result = run_proof_obligations(initial, provider)

    assert result.decision == "abstain"
    assert result.reason_code == "proof_slot_gap"
    assert result.repair_attempted is False
    assert result.model_calls == 1
    assert calls == ["proof"]


def test_one_scoped_repair_can_supply_only_missing_required_slot() -> None:
    initial = _bundle(_item("one", "Project alpha ships."))
    repaired = _bundle(
        _item("two", "The release is 2026-10-15."),
    )
    outputs = iter(
        [
            _assessment(
                date_supported=False,
                repair={"query": "Project alpha release date", "missing_slot_ids": ["release_date"]},
            ),
            _assessment(date_supported=True, repair=None),
        ]
    )
    repair_queries: list[tuple[str, tuple[str, ...]]] = []

    def provider(_system: str, _user: str) -> dict[str, object]:
        return next(outputs)

    def retrieve(repair: object) -> EvidenceBundle:
        repair_queries.append((repair.query, repair.missing_slot_ids))  # type: ignore[union-attr]
        return repaired

    result = run_proof_obligations(initial, provider, retrieve)

    assert result.decision == "sufficient"
    assert result.repair_attempted is True
    assert result.model_calls == 2
    assert repair_queries == [("Project alpha release date", ("release_date",))]
    assert [item.chunk_id for item in result.evidence.items] == ["one", "two"]


def test_repair_cannot_cross_generation_boundary() -> None:
    initial = _bundle(_item("one", "Project alpha ships."))
    wrong_generation = _bundle(_item("two", "The release is 2026-10-15."), generation="generation-2")
    output = _assessment(
        date_supported=False,
        repair={"query": "Project alpha release date", "missing_slot_ids": ["release_date"]},
    )

    result = run_proof_obligations(initial, lambda _system, _user: output, lambda _repair: wrong_generation)

    assert result.decision == "abstain"
    assert result.reason_code == "proof_repair_failure"
    assert result.model_calls == 1


def test_post_repair_slots_are_immutable() -> None:
    initial = _bundle(_item("one", "Project alpha ships."))
    repaired = _bundle(_item("two", "The release is 2026-10-15."))
    outputs = iter(
        [
            _assessment(
                date_supported=False,
                repair={"query": "Project alpha release date", "missing_slot_ids": ["release_date"]},
            ),
            {
                "schema_version": 1,
                "slots": [{"slot_id": "project", "kind": "entity", "requirement": "project identity", "required": True}],
                "support": [{"slot_id": "project", "chunk_id": "one", "start": 0, "end": 13, "quote": "Project alpha"}],
                "repair": None,
            },
        ]
    )

    result = run_proof_obligations(initial, lambda _system, _user: next(outputs), lambda _repair: repaired)

    assert result.decision == "abstain"
    assert result.reason_code == "proof_provider_invalid"
    assert result.model_calls == 2


def test_the_evidence_payload_carries_no_output_template() -> None:
    """A schema template inside the data payload is echoed back, not followed.

    Version 1 of the prompt put a `proof_schema` object with placeholder values beside the
    evidence. A small instruction model returned that container and copied placeholders such as
    `lowercase_identifier` as slot ids, so no response satisfied the proof contract.

    Red proof: restoring a `"proof_schema": {...}` entry to the payload in `render_proof_prompt`
    fails on the exact key set.
    """
    from recall.proof_obligations import render_proof_prompt

    _, user = render_proof_prompt(_bundle(_item("one", "Project alpha ships.")))
    assert user.startswith("<proof_evidence>") and user.endswith("</proof_evidence>")
    payload = json.loads(user[len("<proof_evidence>") : -len("</proof_evidence>")])
    assert set(payload) == {"query", "evidence"}
    assert "lowercase_identifier" not in user


def test_the_prompt_digest_changes_with_the_prompt() -> None:
    """A receipt must name the prompt that ran; version 1 hashed a fixed label instead.

    Red proof: replacing the derived `PROOF_PROMPT_DIGEST` with a hash of a constant string fails
    on the equality with the recomputed digest.
    """
    import hashlib

    from recall import proof_obligations as po

    expected = hashlib.sha256(f"{po.PROOF_SCHEMA_VERSION}:{po.SYSTEM_PROMPT}".encode()).hexdigest()
    assert po.PROOF_PROMPT_DIGEST == expected
