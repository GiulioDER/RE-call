"""The builder: one question -> a paired family of instances, and the same manifest every time.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

Determinism is the load-bearing property. The manifest is released and cited; if two builds of one
corpus differ, there is no benchmark, only a run. So the digest is asserted equal across rebuilds
rather than assumed.
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmarks.ladder.build import build_instances, main
from benchmarks.ladder.manifest import (
    LABEL_ANSWERABLE,
    LABEL_UNANSWERABLE,
    RING_MAX,
    manifest_digest,
    read_manifest,
)
from benchmarks.ladder.rings import RingSpec
from benchmarks.ladder.sources.locomo import load_locomo

# A WIDER fixture than Task 5's SAMPLE, on purpose. Task 5's conversations have 2 and 1 turns, so
# a question there has at most ONE non-gold neighbour — and a shuffle of a one-element list equals
# its BM25 order, which would make `test_the_random_arm_differs_from_the_bm25_arm` fail every time
# rather than flakily. Eight turns gives the two orderings room to disagree.
WIDE_SAMPLE = [
    {
        "sample_id": "conv-0",
        "conversation": {
            "session_1_date_time": "7 May 2023",
            "session_1": [
                {"dia_id": "D1:1", "speaker": "Caroline", "text": "I went to the support group."},
                {"dia_id": "D1:2", "speaker": "Melanie", "text": "How was the support group?"},
                {"dia_id": "D1:3", "speaker": "Caroline", "text": "We talked about the group."},
                {"dia_id": "D1:4", "speaker": "Melanie", "text": "I ran a charity race."},
                {"dia_id": "D1:5", "speaker": "Caroline", "text": "The weather was cold."},
                {"dia_id": "D1:6", "speaker": "Melanie", "text": "Dinner plans for Friday."},
                {"dia_id": "D1:7", "speaker": "Caroline", "text": "A new job application."},
                {"dia_id": "D1:8", "speaker": "Melanie", "text": "The cat needed a vet."},
            ],
        },
        "qa": [
            {
                "question": "When did Caroline go to the support group?",
                "answer": "7 May 2023",
                "evidence": ["D1:1"],
                "category": 2,
            },
            {
                "question": "What did Melanie run?",
                "answer": "a charity race",
                "evidence": ["D1:4"],
                "category": 1,
            },
        ],
    }
]

SPEC = RingSpec(widths=(0, 1, 2, 3))


def _digest(instances) -> str:
    """The digest covers provenance as well as bodies, so every call must supply it.

    Held constant here on purpose: these tests compare manifests that differ in their INSTANCES,
    so varying the provenance too would let a test pass for the wrong reason.
    """
    return manifest_digest(instances, ring_widths=list(SPEC.widths), corpus_hashes={"locomo": "x"})


def _corpus(tmp_path: Path):
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(WIDE_SAMPLE), encoding="utf-8")
    return load_locomo(path)


def test_every_question_yields_one_answerable_original_plus_one_per_rung(tmp_path: Path):
    corpus = _corpus(tmp_path)
    instances = build_instances(corpus, SPEC, corpus_name="locomo")
    # 4 widths + RING_MAX + 1 answerable original = 6 per question
    assert len(instances) == len(corpus.questions) * 6


def test_the_answerable_original_excises_nothing(tmp_path: Path):
    instances = build_instances(_corpus(tmp_path), SPEC, corpus_name="locomo")
    originals = [i for i in instances if i.label == LABEL_ANSWERABLE]
    assert originals
    assert all(i.excised_doc_ids == () for i in originals)


def test_every_unanswerable_instance_excises_its_gold(tmp_path: Path):
    instances = build_instances(_corpus(tmp_path), SPEC, corpus_name="locomo")
    for inst in instances:
        if inst.label == LABEL_UNANSWERABLE:
            assert set(inst.gold_doc_ids) <= set(inst.excised_doc_ids)


def test_a_family_shares_one_pair_id(tmp_path: Path):
    instances = build_instances(_corpus(tmp_path), SPEC, corpus_name="locomo")
    by_question: dict[str, set[str]] = {}
    for inst in instances:
        by_question.setdefault(inst.source_question_id, set()).add(inst.pair_id)
    assert all(len(pairs) == 1 for pairs in by_question.values())


def test_instance_ids_are_unique(tmp_path: Path):
    instances = build_instances(_corpus(tmp_path), SPEC, corpus_name="locomo")
    ids = [i.instance_id for i in instances]
    assert len(ids) == len(set(ids))


def test_ring_max_instance_is_present_for_every_question(tmp_path: Path):
    corpus = _corpus(tmp_path)
    instances = build_instances(corpus, SPEC, corpus_name="locomo")
    at_max = [i for i in instances if i.ring == RING_MAX]
    assert len(at_max) == len(corpus.questions)


def test_two_builds_of_the_same_corpus_produce_the_same_digest(tmp_path: Path):
    corpus = _corpus(tmp_path)
    a = build_instances(corpus, SPEC, corpus_name="locomo")
    b = build_instances(corpus, SPEC, corpus_name="locomo")
    assert _digest(a) == _digest(b)


def test_the_random_seed_is_threaded_through_to_the_rings(tmp_path: Path):
    corpus = _corpus(tmp_path)
    a = build_instances(corpus, SPEC, corpus_name="locomo", random_seed=7)
    b = build_instances(corpus, SPEC, corpus_name="locomo", random_seed=7)
    c = build_instances(corpus, SPEC, corpus_name="locomo", random_seed=8)
    assert _digest(a) == _digest(b)
    assert _digest(a) != _digest(c)


def test_the_random_arm_differs_from_the_bm25_arm(tmp_path: Path):
    corpus = _corpus(tmp_path)
    bm25 = build_instances(corpus, SPEC, corpus_name="locomo")
    rand = build_instances(corpus, SPEC, corpus_name="locomo", random_seed=7)
    assert _digest(bm25) != _digest(rand)


def test_cli_writes_a_readable_manifest(tmp_path: Path):
    src = tmp_path / "locomo.json"
    src.write_text(json.dumps(WIDE_SAMPLE), encoding="utf-8")
    out = tmp_path / "manifest.jsonl"
    assert main(["--locomo", str(src), "--out", str(out), "--widths", "0,1"]) == 0
    instances, header = read_manifest(out)
    assert instances
    assert header["ring_widths"] == [0, 1]
    assert "locomo" in header["corpus_hashes"]


def test_sampling_is_deterministic_and_seed_dependent(tmp_path: Path):
    corpus = _corpus(tmp_path)
    a = build_instances(corpus, SPEC, corpus_name="locomo", sample=1, sample_seed=0)
    b = build_instances(corpus, SPEC, corpus_name="locomo", sample=1, sample_seed=0)
    assert _digest(a) == _digest(b)
    assert len({i.source_question_id for i in a}) == 1


def test_a_sample_larger_than_the_corpus_keeps_every_question(tmp_path: Path):
    corpus = _corpus(tmp_path)
    everything = build_instances(corpus, SPEC, corpus_name="locomo")
    huge = build_instances(corpus, SPEC, corpus_name="locomo", sample=9999)
    assert _digest(huge) == _digest(everything)


def test_sampling_does_not_change_the_rings_of_the_questions_it_keeps(tmp_path: Path):
    """The BM25 index spans the whole corpus, so drawing a subset must not reshape the x-axis."""
    corpus = _corpus(tmp_path)
    full = {i.instance_id: i.excised_doc_ids for i in build_instances(corpus, SPEC, corpus_name="locomo")}
    drawn = build_instances(corpus, SPEC, corpus_name="locomo", sample=1, sample_seed=0)
    assert drawn
    for inst in drawn:
        assert inst.excised_doc_ids == full[inst.instance_id]
