"""What the shipped contradiction detector can and cannot reach, pinned.

The reasoning layer is the thing Track B was going to build on, so what it actually keys off is
load bearing. It is not semantic: `_contradictory_validity_window_proposals` is three conjuncts,
and this module pins each one, so widening the vocabulary or changing subject derivation shows up
as a test change rather than as a quiet shift in what the layer reaches.

⚠️ The property that matters most is the last one: when the detector DOES fire, the planner
`_fail_closed`s with `ambiguous_evidence`. The layer's answer to a contradiction is to ABSTAIN,
not to resolve it. It is a safety mechanism, and no amount of it can raise a correctness score.

Properties, one test each:
  1. An EnterpriseRAG shaped conflict (numeric, cross document, no status words) yields NOTHING.
  2. A memo shaped conflict (shared subject, opposing status words) yields a contradiction.
  3. The subject conjunct: same status words, different subjects, no contradiction.
  4. The vocabulary conjunct: same subject, disagreement outside the nine words, no contradiction.
  5. The vocabulary is a closed nine-word pair, enumerated here so widening it is visible.
"""
from __future__ import annotations

import pytest

from benchmarks.probe_reasoning_reach import ENTERPRISE_SHAPED, MEMO_SHAPED
from recall.reasoning_graph import build_reasoning_graph
from recall.reasoning_proposals._deterministic import (
    _opposing_validity_text,
    deterministic_inference_proposals,
)
from recall.types import Chunk


def _contradictions(chunks: list[Chunk]) -> list[object]:
    graph = build_reasoning_graph(
        chunks, tenant_id="t", generation_id="g", include_text=True
    )
    proposals = deterministic_inference_proposals(graph, pipeline_id="p")
    return [p for p in proposals if p.proposed_relation == "contradicts"]


def _pair(subject_a: str, subject_b: str, text_a: str, text_b: str) -> list[Chunk]:
    return [
        Chunk(id="a", source=subject_a, text=text_a, metadata={"file": subject_a}),
        Chunk(id="b", source=subject_b, text=text_b, metadata={"file": subject_b}),
    ]


def test_an_enterprise_shaped_conflict_is_not_reached():
    """Zero proposals of ANY kind, not merely zero contradictions.

    These are the `conflicting_info` rows that `FINDING-where-the-deficit-actually-is.md`
    identifies as clean answer-control failures. The layer does not engage them.
    """
    graph = build_reasoning_graph(
        ENTERPRISE_SHAPED, tenant_id="t", generation_id="g", include_text=True
    )
    assert deterministic_inference_proposals(graph, pipeline_id="p") == ()


def test_a_memo_shaped_conflict_is_reached():
    """The positive control. Without it, test 1 passes for a broken detector."""
    assert len(_contradictions(MEMO_SHAPED)) == 1


def test_the_subject_conjunct_gates_it():
    """Opposing status words are not enough; the claims must share a subject."""
    same = _pair(
        "policy_v1.md", "policy_v2.md",
        "The policy is approved and active.", "The policy is rejected and disabled.",
    )
    assert len(_contradictions(same)) == 1

    different = _pair(
        "alpha_policy.md", "beta_runbook.md",
        "The policy is approved and active.", "The runbook is rejected and disabled.",
    )
    assert _contradictions(different) == []


def test_the_vocabulary_conjunct_gates_it():
    """A shared subject is not enough; the disagreement must be spelled in the nine words."""
    numeric = _pair(
        "limits_v1.md", "limits_v2.md",
        "The upload limit is 10 MiB per file.", "The upload limit is 25 MiB per file.",
    )
    assert _contradictions(numeric) == [], (
        "a numeric contradiction on one subject is invisible to the detector, which is why "
        "EnterpriseRAG's conflicts are out of reach"
    )


@pytest.mark.parametrize(
    ("positive", "negative"),
    [("enabled", "disabled"), ("active", "inactive"), ("valid", "invalid"),
     ("approved", "rejected"), ("ship", "blocked"), ("on", "off")],
)
def test_the_status_vocabulary_is_the_closed_pair_it_looks_like(positive: str, negative: str):
    """Enumerated so widening the vocabulary is a visible test change, not a silent reach change."""
    assert _opposing_validity_text(f"it is {positive}", f"it is {negative}")
    assert _opposing_validity_text(f"it is {negative}", f"it is {positive}")


def test_a_word_outside_the_vocabulary_is_not_opposition():
    """`deprecated` and `superseded` are the words a reader expects here, and neither is in it."""
    assert not _opposing_validity_text("it is deprecated", "it is current")
    assert not _opposing_validity_text("it is superseded", "it is authoritative")
    assert not _opposing_validity_text("the limit is 10 MiB", "the limit is 25 MiB")
