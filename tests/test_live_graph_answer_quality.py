"""Tests for bounded answer replay context selection."""

from __future__ import annotations

import json

from recall.evidence import (
    EVIDENCE_CLOSE,
    EVIDENCE_OPEN,
    EvidenceBundle,
    EvidenceItem,
    render_compact_evidence_prompt,
)
from scripts.run_live_graph_answer_quality import _cap_evidence


def _bundle(count: int = 5) -> EvidenceBundle:
    items = tuple(
        EvidenceItem(
            chunk_id=f"chunk-{index}",
            text=f"evidence {index}",
            source="source.md",
            ordinal=index,
            indexed_at=None,
            valid_from=None,
            valid_until=None,
            cosine=0.9 - index / 100,
            confidence=0.8,
        )
        for index in range(count)
    )
    return EvidenceBundle(
        query="question",
        decision="answer",
        reason_code=None,
        decision_state="supported",
        calibrated=True,
        stale=False,
        embedding_profile="test",
        retrieval_profile="fast",
        index_generation="generation",
        items=items,
    )


def test_answer_replay_cap_preserves_order_and_bundle_metadata() -> None:
    """Node replay-cap-01: the old unbounded prompt would fail the cap assertion."""
    bundle = _bundle()

    capped = _cap_evidence(bundle, 3)

    assert [item.chunk_id for item in capped.items] == ["chunk-0", "chunk-1", "chunk-2"]
    assert capped.query == bundle.query
    assert capped.trust_state == bundle.trust_state


def test_answer_replay_cap_none_is_identity() -> None:
    bundle = _bundle()

    assert _cap_evidence(bundle, None) is bundle


def test_answer_replay_cap_rejects_nonpositive_values() -> None:
    try:
        _cap_evidence(_bundle(), 0)
    except ValueError as exc:
        assert str(exc) == "max_evidence_items must be positive"
    else:
        raise AssertionError("Node replay-cap-02: nonpositive cap was accepted")


def test_compact_prompt_keeps_ids_and_text_but_omits_retrieval_metadata() -> None:
    """Node compact-prompt-01: a metadata restoring mutation fails this payload contract."""
    _system, user = render_compact_evidence_prompt(_bundle(1))

    payload = json.loads(user[len(EVIDENCE_OPEN) : -len(EVIDENCE_CLOSE)])
    assert payload["evidence"] == [{"chunk_id": "chunk-0", "text": "evidence 0"}]
    assert "confidence" not in payload["evidence"][0]
    assert "indexed_at" not in payload["evidence"][0]
