"""The runner: ingest once per distinct corpus state, and let invariants stop a bad run early.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

These tests use a fake MemorySystem on purpose. The Postgres-backed adapter is exercised by the
real run; what needs pinning here is the runner's own logic, which is where a silent defect would
cost a whole overnight job.
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
    RING_ORIGINAL,
    Instance,
    write_manifest,
)
from benchmarks.ladder.run import AdapterSmokeCheckFailed, run

DOCS = {f"c/D1:{i}": f"turn {i} about the support group" for i in range(1, 5)}


class _Fake:
    name = "fake"

    def __init__(self, *, leak: bool = False) -> None:
        self._docs: dict[str, str] = {}
        self.ingest_calls = 0
        self._leak = leak

    def ingest(self, docs: Iterable[Document]) -> None:
        self.ingest_calls += 1
        incoming = {d.doc_id: d.text for d in docs}
        # A leaking system MERGES instead of replacing — exactly what invariant 1 exists to catch.
        self._docs = {**self._docs, **incoming} if self._leak else incoming

    def indexed_doc_ids(self) -> frozenset[str]:
        return frozenset(self._docs)

    def query(self, question: str) -> Response:
        if not self._docs:
            return Response(answer=None)
        first = sorted(self._docs)[0]
        return Response(answer=self._docs[first], cited_ids=(first,), tokens=10)


def _manifest(tmp_path: Path) -> Path:
    instances = [
        Instance(
            instance_id="p1#original", corpus="locomo", source_question_id="q1",
            question="when?", label=LABEL_ANSWERABLE, ring=RING_ORIGINAL,
            excised_doc_ids=(), gold_doc_ids=("c/D1:1",), pair_id="p1",
        ),
        Instance(
            instance_id="p1#d0", corpus="locomo", source_question_id="q1",
            question="when?", label=LABEL_UNANSWERABLE, ring=0,
            excised_doc_ids=("c/D1:1",), gold_doc_ids=("c/D1:1",), pair_id="p1",
        ),
        Instance(
            instance_id="p2#d0", corpus="locomo", source_question_id="q2",
            question="who?", label=LABEL_UNANSWERABLE, ring=0,
            excised_doc_ids=("c/D1:1",), gold_doc_ids=("c/D1:1",), pair_id="p2",
        ),
    ]
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, instances, ring_widths=[0], corpus_hashes={"locomo": "x"})
    return path


def test_writes_one_row_per_instance(tmp_path: Path):
    out = tmp_path / "responses.jsonl"
    run(_manifest(tmp_path), _Fake(), out, documents=DOCS, cluster_members={"c": tuple(DOCS)})
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {r["instance_id"] for r in rows} == {"p1#original", "p1#d0", "p2#d0"}


def test_ingests_once_per_distinct_excision_set_not_once_per_instance(tmp_path: Path):
    system = _Fake()
    run(_manifest(tmp_path), system, tmp_path / "r.jsonl", documents=DOCS,
        cluster_members={"c": tuple(DOCS)})
    # Two distinct states: nothing excised, and {c/D1:1} excised.
    assert system.ingest_calls == 2


def test_abstention_is_recorded_as_a_boolean(tmp_path: Path):
    out = tmp_path / "r.jsonl"
    run(_manifest(tmp_path), _Fake(), out, documents=DOCS, cluster_members={"c": tuple(DOCS)})
    rows = {
        json.loads(line)["instance_id"]: json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
    }
    assert isinstance(rows["p1#d0"]["abstained"], bool)


def test_a_system_that_merges_instead_of_replacing_is_stopped(tmp_path: Path):
    with pytest.raises(InvariantViolation, match="still indexed"):
        run(_manifest(tmp_path), _Fake(leak=True), tmp_path / "r.jsonl", documents=DOCS,
            cluster_members={"c": tuple(DOCS)})


def test_resume_skips_instances_already_written(tmp_path: Path):
    out = tmp_path / "r.jsonl"
    manifest = _manifest(tmp_path)
    run(manifest, _Fake(), out, documents=DOCS, cluster_members={"c": tuple(DOCS)})
    system = _Fake()
    run(manifest, system, out, documents=DOCS, cluster_members={"c": tuple(DOCS)}, resume=True)
    assert system.ingest_calls == 0


def test_ingest_is_scoped_to_the_questions_own_conversation(tmp_path: Path):
    """A question is scored against its own conversation, not the whole corpus.

    This is the difference between indexing 646 turns per state and 5 882 — at ~1 500 states, the
    difference between a run an adopter can finish and one nobody will.
    """
    two_clusters = dict(DOCS)
    two_clusters.update({f"other/D1:{i}": f"unrelated turn {i}" for i in range(1, 4)})
    system = _Fake()
    run(
        _manifest(tmp_path),
        system,
        tmp_path / "r.jsonl",
        documents=two_clusters,
        cluster_members={"c": tuple(DOCS), "other": ("other/D1:1", "other/D1:2", "other/D1:3")},
    )
    assert all(not d.startswith("other/") for d in system.indexed_doc_ids())


def test_invariant_three_still_fires_on_a_fully_resumed_run(tmp_path: Path):
    """A resume where every original was already scored must not skip the check silently."""
    out = tmp_path / "r.jsonl"
    manifest = _manifest(tmp_path)
    # Hand-write a completed run in which the answerable original was ABSTAINED on.
    rows = [
        {"instance_id": "p1#original", "system": "fake", "abstained": True, "cited_ids": [], "tokens": 0},
        {"instance_id": "p1#d0", "system": "fake", "abstained": True, "cited_ids": [], "tokens": 0},
        {"instance_id": "p2#d0", "system": "fake", "abstained": True, "cited_ids": [], "tokens": 0},
    ]
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    with pytest.raises(InvariantViolation, match="broken"):
        run(manifest, _Fake(), out, documents=DOCS, cluster_members={"c": tuple(DOCS)}, resume=True)


class _BadSignature:
    """An adapter whose method NAMES match `MemorySystem` but whose signatures do not.

    `MemorySystem` is `runtime_checkable`, which checks names only (Task 7 review) — an adapter
    shaped like this passes `isinstance` and must instead be caught by a real call, not by typing.
    """

    name = "bad"

    def ingest(self, docs, extra_required_arg) -> None:  # missing default -> TypeError on call
        raise AssertionError("should never be reached")

    def indexed_doc_ids(self) -> frozenset[str]:
        return frozenset()

    def query(self, question: str) -> Response:
        return Response(answer=None)


def test_smoke_check_calls_ingest_and_query_and_fails_fast_on_a_bad_signature():
    from benchmarks.ladder.run import AdapterSmokeCheckFailed, smoke_check

    with pytest.raises(AdapterSmokeCheckFailed):
        smoke_check(_BadSignature())


def test_smoke_check_passes_for_a_well_shaped_system():
    from benchmarks.ladder.run import smoke_check

    smoke_check(_Fake())  # must not raise


def test_run_rejects_an_adapter_whose_query_signature_is_wrong(tmp_path: Path):
    """runtime_checkable passes this class — it has all three method NAMES. Only a signature
    check catches it, and catching it here beats catching it forty minutes in."""

    class _WrongArity(_Fake):
        def query(self):  # missing `question`
            return Response(answer=None)

    with pytest.raises(AdapterSmokeCheckFailed, match="query"):
        run(_manifest(tmp_path), _WrongArity(), tmp_path / "r.jsonl",
            documents=DOCS, cluster_members={"c": tuple(DOCS)})


def test_run_rejects_an_adapter_missing_a_method_entirely(tmp_path: Path):
    class _NoIndexedIds:
        name = "broken"

        def ingest(self, docs):
            return None

        def query(self, question):
            return Response(answer=None)

    with pytest.raises(AdapterSmokeCheckFailed, match="indexed_doc_ids"):
        run(_manifest(tmp_path), _NoIndexedIds(), tmp_path / "r.jsonl",
            documents=DOCS, cluster_members={"c": tuple(DOCS)})


def test_the_signature_check_does_not_disturb_the_ingest_counters(tmp_path: Path):
    """The reason this is a signature check and not a functional smoke call."""
    system = _Fake()
    run(_manifest(tmp_path), system, tmp_path / "r.jsonl",
        documents=DOCS, cluster_members={"c": tuple(DOCS)})
    assert system.ingest_calls == 2
