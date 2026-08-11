"""The committed artifacts agree with each other and with the census that produced them.

Properties:
  1. The gold manifest's positive row count equals the census `n_header_edges`.
  2. The manifest verifies against its own digest (read_manifest refuses a mismatch).
  3. Every positive carries exactly one successor label; every fixture negative carries none.
  3b. Every positive's label is the SUCCESSOR named in its own question_id, not the superseded
      document. Without this the suite cannot detect a wholesale direction inversion.
  4. Corpus-dependent recomputation runs only when RECALL_PEPS_DIR is set, and SKIPS loudly
     otherwise rather than passing vacuously.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from recall.eval.promotion.manifest import read_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
CENSUS = REPO_ROOT / "results" / "truth_extraction" / "census.json"
GOLD = REPO_ROOT / "benchmarks" / "labelling" / "truth_extraction" / "gold.manifest.jsonl"


@pytest.fixture(scope="module")
def census() -> dict:
    return json.loads(CENSUS.read_text(encoding="utf-8"))


def test_manifest_verifies_against_its_digest():
    # read_manifest recomputes and refuses a mismatch; reaching this line means it matched.
    questions, header = read_manifest(GOLD)
    assert header["digest"]
    assert questions


def test_positive_row_count_equals_census_header_edges(census: dict):
    questions, _ = read_manifest(GOLD)
    positives = [q for q in questions if q.expected_relevance_labels]
    assert len(positives) == census["n_header_edges"]


def test_every_positive_has_exactly_one_successor_label():
    questions, _ = read_manifest(GOLD)
    for question in questions:
        if question.expected_relevance_labels:
            assert len(question.expected_relevance_labels) == 1
            assert question.expected_relevance_labels[0].endswith(".rst")


def test_every_positive_label_is_the_SUCCESSOR_not_the_superseded():
    # `question_id` is "<superseded>-><successor>". Labelling a positive with the SUPERSEDED PEP
    # would make the gold set assert that the live document is the stale one — the inversion the
    # trust layer exists to prevent, baked into the labels every later number is scored against.
    #
    # This test exists because without it the suite cannot detect that inversion: a
    # `build_gold_questions` that swapped the two ends would still emit 47 positives, each with
    # exactly one `.rst` label, and pass every other test unchanged. Counting rows and checking a
    # suffix says nothing about direction.
    questions, _ = read_manifest(GOLD)
    positives = [q for q in questions if q.expected_relevance_labels]
    assert positives, "no positives in the manifest — this test would pass vacuously"
    for question in positives:
        superseded, _, successor = question.question_id.partition("->")
        assert successor, f"{question.question_id} is not an '<a>-><b>' identity"
        assert question.expected_relevance_labels[0] == f"{successor}.rst"


def test_fixture_negatives_are_frozen_with_no_labels():
    questions, _ = read_manifest(GOLD)
    negatives = [q for q in questions if not q.expected_relevance_labels]
    assert len(negatives) == 4
    assert all(q.corpus == "fix-transplant" for q in negatives)


def test_recall_ceiling_is_published_and_below_one(census: dict):
    # The number this set exists to publish. If it ever reads 1.0, the detector is matching
    # something that is not in the gold set.
    assert 0.0 < census["recall_ceiling"] < 1.0
    assert census["n_restated_in_prose"] <= census["n_header_edges"]


def test_census_recomputes_from_the_corpus(census: dict):
    peps_dir = os.environ.get("RECALL_PEPS_DIR")
    if not peps_dir:
        pytest.skip(
            "RECALL_PEPS_DIR unset — clone python/peps and point it at the nested peps/ dir. "
            "This test is SKIPPED, not passed: the corpus-dependent counts are unverified."
        )
    from benchmarks.labelling.truth_extraction.census import compute_census

    recomputed = compute_census(Path(peps_dir))
    assert recomputed.n_files == census["n_files"]
    assert recomputed.n_header_edges == census["n_header_edges"]
    assert recomputed.n_prose_marker_files == census["n_prose_marker_files"]
    assert recomputed.n_marker_without_header == census["n_marker_without_header"]
    assert recomputed.n_restated_in_prose == census["n_restated_in_prose"]


TRUST = REPO_ROOT / "recall" / "eval" / "peps_trust_queries.json"


def test_trust_set_is_between_40_and_70_queries():
    rows = json.loads(TRUST.read_text(encoding="utf-8"))
    assert 40 <= len(rows) <= 70, f"{len(rows)} queries — Wilson needs the shipped n=4 fixed"


def test_trust_set_matches_the_shipped_queries_schema():
    shipped = json.loads((REPO_ROOT / "recall" / "eval" / "queries.json").read_text(
        encoding="utf-8"))
    shipped_trust_keys = {k for e in shipped if e.get("trust") for k in e}
    rows = json.loads(TRUST.read_text(encoding="utf-8"))
    for row in rows:
        assert set(row) == shipped_trust_keys, f"{row['id']} does not match the shipped schema"


def test_successor_rows_have_a_successor_and_abstain_rows_do_not():
    rows = json.loads(TRUST.read_text(encoding="utf-8"))
    for row in rows:
        if row["expect"] == "successor":
            assert row["successor_ids"] and row["stale_ids"]
        else:
            assert row["expect"] == "abstain" and row["successor_ids"] == []


def test_successor_row_count_equals_census_header_edges(census: dict):
    rows = json.loads(TRUST.read_text(encoding="utf-8"))
    successors = [r for r in rows if r["expect"] == "successor"]
    assert len(successors) == census["n_header_edges"]


def test_stale_and_successor_are_not_inverted(census: dict):
    # `stale_ids` must hold the SUPERSEDED document and `successor_ids` the SUCCESSOR. Swapping
    # them scores a system correct exactly when it prefers the stale document — the failure the
    # trust layer exists to prevent, written into the labels every later number is graded on.
    #
    # This test exists because the suite was measured blind to it: an inverted builder still
    # emits 67 rows, 47 of them `successor`, with matching schema keys and non-empty
    # `successor_ids`, and passes every other test in this file. Counting rows and checking
    # shape says nothing about direction.
    #
    # The comparison is against `census.json`'s edge list, which is independently frozen and
    # whose own direction was verified against the PEP headers.
    edges = {(e["superseded"], e["successor"]) for e in census["edges"]}
    rows = json.loads(TRUST.read_text(encoding="utf-8"))

    def stem(chunk_id: str) -> str:
        name = chunk_id.rsplit(":", 1)[0]
        return name[:-4] if name.endswith(".rst") else name

    seen = set()
    for row in (r for r in rows if r["expect"] == "successor"):
        pair = (stem(row["stale_ids"][0]), stem(row["successor_ids"][0]))
        assert pair in edges, (
            f"{row['id']}: {pair[0]} -> {pair[1]} is not a census edge. "
            f"Reversed pair present in census: {(pair[1], pair[0]) in edges}"
        )
        seen.add(pair)
    assert seen == edges, "successor rows do not cover the census edges exactly"
