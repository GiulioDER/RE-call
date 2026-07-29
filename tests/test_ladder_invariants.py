"""The artefact assertions. Exit code 0 is not a measurement.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

Each of these catches a failure that would otherwise produce a PLAUSIBLE NUMBER rather than an
error — which is the only kind of failure worth writing an assertion for.
"""
from __future__ import annotations

import pytest

from benchmarks.ladder.invariants import (
    InvariantViolation,
    assert_excised_absent,
    assert_manifest_digest,
    assert_originals_were_answered,
    assert_ring_zero_has_survivors,
    assert_survivors_present,
)
from benchmarks.ladder.manifest import (
    LABEL_ANSWERABLE,
    LABEL_UNANSWERABLE,
    RING_MAX,
    RING_ORIGINAL,
    Instance,
    manifest_digest,
)


def _inst(**kw) -> Instance:
    base = dict(
        instance_id="i1",
        corpus="locomo",
        source_question_id="q1",
        question="when?",
        label=LABEL_UNANSWERABLE,
        ring=0,
        excised_doc_ids=("c/D1:3",),
        gold_doc_ids=("c/D1:3",),
        pair_id="p1",
    )
    base.update(kw)
    return Instance(**base)


def _original(instance_id: str = "o1") -> Instance:
    return _inst(
        instance_id=instance_id,
        label=LABEL_ANSWERABLE,
        ring=RING_ORIGINAL,
        excised_doc_ids=(),
    )


def test_excised_absent_passes_when_the_system_really_dropped_them():
    assert_excised_absent(_inst(), frozenset({"c/D1:1", "c/D1:2"}))


def test_excised_absent_raises_when_a_system_cached_across_rings():
    with pytest.raises(InvariantViolation, match="still indexed"):
        assert_excised_absent(_inst(), frozenset({"c/D1:3"}))


def test_ring_zero_needs_surviving_neighbours():
    cluster = ["c/D1:1", "c/D1:2", "c/D1:3"]
    assert_ring_zero_has_survivors(_inst(), frozenset({"c/D1:1", "c/D1:2"}), cluster)


def test_ring_zero_with_no_survivors_is_secretly_ring_max():
    with pytest.raises(InvariantViolation, match="d=max"):
        assert_ring_zero_has_survivors(_inst(), frozenset(), ["c/D1:3"])


def test_non_zero_rings_are_not_subject_to_the_survivor_rule():
    assert_ring_zero_has_survivors(_inst(ring=16), frozenset(), ["c/D1:3"])


def test_survivors_present_passes_when_ingest_kept_everything_it_should():
    cluster = ["c/D1:1", "c/D1:2", "c/D1:3"]
    assert_survivors_present(_inst(), frozenset({"c/D1:1", "c/D1:2"}), cluster)


def test_an_empty_index_no_longer_looks_like_a_perfect_excision():
    """The hole this invariant exists to close: absence alone cannot tell a correct excision from
    an ingest that did nothing."""
    cluster = ["c/D1:1", "c/D1:2", "c/D1:3"]
    with pytest.raises(InvariantViolation, match="are missing from the index"):
        assert_survivors_present(_inst(), frozenset(), cluster)


def test_a_partial_ingest_is_caught_although_every_negative_check_passes():
    """10-of-646 ingest: excised really is absent, so assert_excised_absent is happy, and the
    resulting flat curve would be read as a genuine H1 FAIL."""
    cluster = ["c/D1:1", "c/D1:2", "c/D1:3", "c/D1:4"]
    partial = frozenset({"c/D1:1"})
    assert_excised_absent(_inst(), partial)  # the negative check passes
    with pytest.raises(InvariantViolation, match="are missing from the index"):
        assert_survivors_present(_inst(), partial, cluster)


def test_mid_rungs_are_checked_too_not_only_d_zero():
    cluster = ["c/D1:1", "c/D1:2", "c/D1:3"]
    mid = _inst(ring=16, excised_doc_ids=("c/D1:3",))
    with pytest.raises(InvariantViolation, match="are missing from the index"):
        assert_survivors_present(mid, frozenset(), cluster)


def test_d_max_passes_trivially_because_the_whole_cluster_is_meant_to_be_gone():
    cluster = ["c/D1:1", "c/D1:2", "c/D1:3"]
    at_max = _inst(ring=RING_MAX, excised_doc_ids=tuple(cluster))
    assert_survivors_present(at_max, frozenset(), cluster)


def test_originals_answered_passes_when_at_least_one_was_answered():
    assert_originals_were_answered({"o1": True}, [_original()])


def test_originals_all_abstained_means_the_questions_are_broken_not_hard():
    with pytest.raises(InvariantViolation, match="broken"):
        assert_originals_were_answered({"o1": False}, [_original()])


def test_the_all_abstained_message_now_names_the_offending_ids():
    with pytest.raises(InvariantViolation, match="o1"):
        assert_originals_were_answered({"o1": False}, [_original()])


def _header(instances) -> dict:
    return {
        "digest": manifest_digest(instances, ring_widths=[0], corpus_hashes={"locomo": "x"}),
        "ring_widths": [0],
        "corpus_hashes": {"locomo": "x"},
    }


def test_manifest_digest_mismatch_is_refused():
    instances = [_inst()]
    assert_manifest_digest(instances, _header(instances))
    with pytest.raises(InvariantViolation, match="digest"):
        assert_manifest_digest(instances, {**_header(instances), "digest": "deadbeef"})


def test_a_forged_corpus_hash_is_refused_even_with_the_right_bodies():
    """The digest covers provenance, so editing which corpus these came from must not pass."""
    instances = [_inst()]
    forged = {**_header(instances), "corpus_hashes": {"locomo": "forged"}}
    with pytest.raises(InvariantViolation, match="digest"):
        assert_manifest_digest(instances, forged)
