"""The runner's v2 behaviour: ingest scope widens to `scope_cluster_ids`, invariants get the
right slice — and getting that argument backwards silently disables the check it exists to run.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

`test_ladder_run.py`'s existing tests exercise v1 behaviour (empty `scope_cluster_ids`) and must
keep passing unchanged — this file is additive, not a replacement. Same fake `MemorySystem`
pattern as that file, for the same reason stated there: what needs pinning is the runner's own
logic, not a real Postgres-backed adapter.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pytest

from benchmarks.ladder.adapter import Document, Response
from benchmarks.ladder.invariants import InvariantViolation
from benchmarks.ladder.manifest import (
    LABEL_ANSWERABLE,
    LABEL_UNANSWERABLE,
    MANIFEST_VERSION_V2,
    RING_ORIGINAL,
    Instance,
    write_manifest,
)
from benchmarks.ladder.rings import fraction_to_ring
from benchmarks.ladder.run import run

OWN = {f"c/D1:{i}": f"turn {i} about the support group" for i in range(1, 5)}  # c/D1:1..4
DISTRACTOR = {f"d/D1:{i}": f"unrelated turn {i}" for i in range(1, 3)}  # d/D1:1..2
OUT_OF_SCOPE = {f"e/D1:{i}": f"never in scope turn {i}" for i in range(1, 2)}  # e/D1:1
DOCS = {**OWN, **DISTRACTOR, **OUT_OF_SCOPE}
CLUSTER_MEMBERS = {
    "c": tuple(sorted(OWN)),
    "d": tuple(sorted(DISTRACTOR)),
    "e": tuple(sorted(OUT_OF_SCOPE)),
}


class _Fake:
    name = "fake"

    def __init__(self) -> None:
        self._docs: dict[str, str] = {}
        self.ingest_calls = 0

    def ingest(self, docs: Iterable[Document]) -> None:
        self.ingest_calls += 1
        self._docs = {d.doc_id: d.text for d in docs}  # replaces, never merges

    def indexed_doc_ids(self) -> frozenset[str]:
        return frozenset(self._docs)

    def query(self, question: str) -> Response:
        if not self._docs:
            return Response(answer=None)
        first = sorted(self._docs)[0]
        return Response(answer=self._docs[first], cited_ids=(first,), tokens=10)


def _v2_inst(
    instance_id: str,
    *,
    source_question_id: str,
    ring: int,
    excised_doc_ids: tuple[str, ...],
    gold_doc_ids: tuple[str, ...] = ("c/D1:1",),
    scope_cluster_ids: tuple[str, ...] = ("c", "d"),
    label: str = LABEL_UNANSWERABLE,
) -> Instance:
    return Instance(
        instance_id=instance_id,
        corpus="locomo",
        source_question_id=source_question_id,
        question="when?",
        label=label,
        ring=ring,
        excised_doc_ids=excised_doc_ids,
        gold_doc_ids=gold_doc_ids,
        pair_id=source_question_id,
        scope_cluster_ids=scope_cluster_ids,
    )


def _write(tmp_path: Path, instances: list[Instance]) -> Path:
    path = tmp_path / "manifest.jsonl"
    write_manifest(
        path, instances, ring_widths=[0], corpus_hashes={"locomo": "x"},
        manifest_version=MANIFEST_VERSION_V2,
    )
    return path


# --- ingest scope widens to the union of scope_cluster_ids ------------------------------------


def test_ingest_includes_distractor_conversations(tmp_path: Path):
    """The whole point of v2: the ingested slice is own cluster UNION distractors."""
    inst = _v2_inst(
        "p1#r0.00", source_question_id="q1", ring=0, excised_doc_ids=("c/D1:1",)
    )
    system = _Fake()
    run(_write(tmp_path, [inst]), system, tmp_path / "r.jsonl", documents=DOCS,
        cluster_members=CLUSTER_MEMBERS)
    indexed = system.indexed_doc_ids()
    assert set(DISTRACTOR) <= indexed  # distractors present
    assert not (set(OUT_OF_SCOPE) & indexed)  # cluster "e" was never in scope


def test_r1_00_excises_the_whole_own_cluster_but_distractors_survive(tmp_path: Path):
    """This is the exact check v1 could not make: at the top rung, distractors remain indexed."""
    inst = _v2_inst(
        "p1#r1.00",
        source_question_id="q1",
        ring=fraction_to_ring(1.0),
        excised_doc_ids=tuple(sorted(OWN)),
    )
    system = _Fake()
    run(_write(tmp_path, [inst]), system, tmp_path / "r.jsonl", documents=DOCS,
        cluster_members=CLUSTER_MEMBERS)
    indexed = system.indexed_doc_ids()
    assert not (set(OWN) & indexed)  # own cluster fully gone
    assert set(DISTRACTOR) == indexed  # distractors are ALL that's left


def test_two_questions_sharing_a_scope_and_excision_set_share_one_ingest(tmp_path: Path):
    a = _v2_inst("p1#r0.00", source_question_id="q1", ring=0, excised_doc_ids=("c/D1:1",))
    b = _v2_inst(
        "p2#r0.00",
        source_question_id="q2",
        ring=0,
        excised_doc_ids=("c/D1:1",),
        gold_doc_ids=("c/D1:1",),
    )
    system = _Fake()
    run(_write(tmp_path, [a, b]), system, tmp_path / "r.jsonl", documents=DOCS,
        cluster_members=CLUSTER_MEMBERS)
    assert system.ingest_calls == 1


def test_a_different_excised_set_in_the_same_scope_forces_a_second_ingest(tmp_path: Path):
    a = _v2_inst("p1#r0.00", source_question_id="q1", ring=0, excised_doc_ids=("c/D1:1",))
    b = _v2_inst(
        "p1#r1.00",
        source_question_id="q1",
        ring=fraction_to_ring(1.0),
        excised_doc_ids=tuple(sorted(OWN)),
    )
    system = _Fake()
    run(_write(tmp_path, [a, b]), system, tmp_path / "r.jsonl", documents=DOCS,
        cluster_members=CLUSTER_MEMBERS)
    assert system.ingest_calls == 2


# --- invariant arguments: the part that is easy to get subtly wrong ---------------------------


def test_survivors_present_checks_the_full_scope_including_distractors(tmp_path: Path):
    """A system that drops a distractor document must be caught — assert_survivors_present must
    see the FULL scope (own + distractors), not just the question's own cluster."""

    class _DropsADistractor(_Fake):
        def ingest(self, docs: Iterable[Document]) -> None:
            self.ingest_calls += 1
            self._docs = {d.doc_id: d.text for d in docs if d.doc_id != "d/D1:1"}

    inst = _v2_inst(
        "p1#r1.00",
        source_question_id="q1",
        ring=fraction_to_ring(1.0),
        excised_doc_ids=tuple(sorted(OWN)),
    )
    with pytest.raises(InvariantViolation, match="missing from the index"):
        run(_write(tmp_path, [inst]), _DropsADistractor(), tmp_path / "r.jsonl",
            documents=DOCS, cluster_members=CLUSTER_MEMBERS)


def test_ring_zero_check_uses_the_own_cluster_only_not_the_full_scope(tmp_path: Path):
    """The bug this test exists to catch: if `assert_ring_zero_has_survivors` were handed the
    FULL scope instead of the question's own cluster, this would pass trivially forever, because
    distractors always survive (they are never excised) — silently disabling invariant 2.

    Here `ring=0` but `excised_doc_ids` covers the WHOLE own cluster (a malformed/broken instance
    that should never come out of `build_v2`, but the runner must still catch it): nothing from
    the topic itself survives, even though the distractor cluster is fully indexed. Passing the
    own cluster only must raise; passing the full scope would not.
    """
    inst = _v2_inst(
        "p1#r0.00-broken",
        source_question_id="q1",
        ring=0,  # ring-zero rung...
        excised_doc_ids=tuple(sorted(OWN)),  # ...but excises the WHOLE own cluster
    )
    with pytest.raises(InvariantViolation, match="d=max"):
        run(_write(tmp_path, [inst]), _Fake(), tmp_path / "r.jsonl",
            documents=DOCS, cluster_members=CLUSTER_MEMBERS)


def test_ring_zero_check_passes_when_the_own_cluster_genuinely_has_survivors(tmp_path: Path):
    """The positive case for the same invariant: a well-formed r=0.00 instance (gold only
    excised) has plenty of own-cluster survivors, so the check must NOT raise."""
    inst = _v2_inst(
        "p1#r0.00", source_question_id="q1", ring=0, excised_doc_ids=("c/D1:1",)
    )
    run(_write(tmp_path, [inst]), _Fake(), tmp_path / "r.jsonl",
        documents=DOCS, cluster_members=CLUSTER_MEMBERS)  # must not raise


# --- v1 fallback: scope_cluster_ids empty ------------------------------------------------------


def test_v1_fallback_ingests_only_the_gold_derived_cluster_when_scope_is_empty(tmp_path: Path):
    """The v1 behaviour `_scope_of` must reproduce exactly: no `scope_cluster_ids` means the
    ingested slice is inferred from the gold id's own cluster, and nothing else — not even a
    conversation that happens to be passed in `cluster_members`."""
    inst = _v2_inst(
        "p1#d0",
        source_question_id="q1",
        ring=0,
        excised_doc_ids=("c/D1:1",),
        scope_cluster_ids=(),  # v1 shape
    )
    system = _Fake()
    run(_write(tmp_path, [inst]), system, tmp_path / "r.jsonl", documents=DOCS,
        cluster_members=CLUSTER_MEMBERS)
    indexed = system.indexed_doc_ids()
    assert not (set(DISTRACTOR) & indexed)
    assert not (set(OUT_OF_SCOPE) & indexed)
    assert indexed <= set(OWN)
