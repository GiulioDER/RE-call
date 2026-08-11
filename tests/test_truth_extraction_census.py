"""The committed artifacts agree with each other and with the census that produced them.

Properties:
  1. The gold manifest's positive row count equals the census `n_header_edges`.
  2. The manifest verifies against its own digest (read_manifest refuses a mismatch).
  3. Every positive carries exactly one successor label; every fixture negative carries none.
  3b. Every positive's label is the SUCCESSOR named in its own question_id, not the superseded
      document. Without this the suite cannot detect a wholesale direction inversion.
  4. Corpus-dependent recomputation runs only when RECALL_PEPS_DIR is set, and SKIPS loudly
     otherwise rather than passing vacuously.
  5. The published recall ceiling is a value strictly between 0 and 1, and the restated-in-prose
     count never exceeds the header-edge count it is a fraction of.
  6. The PEPs trust query set holds between 40 and 70 queries, replacing the shipped successor
     arm of n=4 that made a Wilson interval uninterpretable.
  7. Every trust query matches the schema of the shipped `queries.json`'s trust-flagged rows,
     key for key, so the trust arm can be consumed by the same code as the shipped set.
  8. Every `successor` row carries a non-empty `successor_ids` and `stale_ids`; every `abstain`
     row carries an empty `successor_ids` and no other `expect` value is allowed.
  9. The trust set's successor row count equals the number of DISTINCT superseded PEPs among the
     census edges, not the edge count itself. Five superseded PEPs have two successors each, so a
     row per edge would produce two rows sharing identical `query` text and `stale_ids` and
     differing only in successor, of which at most one could ever score correct: this is the
     property that made the arm's ceiling 42/47 without saying so. No two successor rows may
     share the same `stale_ids`.
  10. Every trust successor row's stale id and EVERY id in its successor_ids name a census edge
      in the SUCCESSOR direction, not the reverse, and together they cover the census edges
      exactly. This is the direction guard: an inverted builder still emits 62 rows with matching
      shape and passes every other test in this file, so counting rows and checking schema says
      nothing about direction.
  11. `build_gold.py`'s `--peps-sha` is refused when malformed, and refused when it disagrees
      with `census.json`'s own recorded `_provenance.peps_sha`: the two artifacts must describe
      the same corpus snapshot.
  12. An INDEPENDENT direction anchor, gated on RECALL_PEPS_DIR: across the census edges, no edge
      pairs a successor whose own `Status:` header reads `Superseded` with a superseded/predecessor
      whose `Status:` reads live (`Final`, `Active` or `Accepted`). The two existing direction
      guards (this file's #10, and `test_every_positive_label_is_the_SUCCESSOR_not_the_superseded`)
      are CONSISTENCY guards: `census.json`, the gold manifest and the trust set all derive from
      one `edges_from_fields`, so a wholesale inversion there flips all three together and every
      guard still passes. This test ties direction to a header field `edges_from_fields` never
      reads, so it cannot be fooled by the same inversion.
  13. The gold manifest's `corpus_hashes.census` names the exact bytes of the committed
      `census.json`, and the census's own recorded `peps_sha` matches what the manifest header
      carries. `corpus_hashes.census` is the manifest's only tie to the census it was frozen
      against, and re-freezing after a census edit is a manual step: without this test, an edited
      census leaves the manifest naming a file no longer on disk, and every other test in this
      file still passes.
  14. No abstain row's stale PEP appears in any census edge, in either position. A stale PEP that
      actually has a successor is not an abstention case.
"""
from __future__ import annotations

import hashlib
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


def test_check_peps_sha_accepts_the_recorded_sha(census: dict):
    from benchmarks.labelling.truth_extraction.build_gold import _check_peps_sha

    _check_peps_sha(census["_provenance"]["peps_sha"], CENSUS)  # must not raise


def test_check_peps_sha_rejects_a_malformed_sha():
    from benchmarks.labelling.truth_extraction.build_gold import _check_peps_sha

    with pytest.raises(ValueError, match="40-character"):
        _check_peps_sha("not-a-sha", CENSUS)


def test_check_peps_sha_reports_a_missing_census_file(tmp_path: Path):
    # A missing census.json (build_gold.py run before census.py) must not surface a bare
    # FileNotFoundError traceback naming no file; it is folded into the same ValueError the
    # caller already catches and turns into a SystemExit.
    from benchmarks.labelling.truth_extraction.build_gold import _check_peps_sha

    missing = tmp_path / "no-such-census.json"
    with pytest.raises(ValueError, match="does not exist"):
        _check_peps_sha("a" * 40, missing)


def test_check_peps_sha_reports_a_census_file_missing_the_sha_field(tmp_path: Path):
    from benchmarks.labelling.truth_extraction.build_gold import _check_peps_sha

    broken = tmp_path / "census.json"
    broken.write_text(json.dumps({"_provenance": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="could not be read"):
        _check_peps_sha("a" * 40, broken)


def test_check_peps_sha_reports_a_census_file_with_the_wrong_shape(tmp_path: Path):
    # Valid JSON, wrong shape: `_provenance` is not a dict, so `["peps_sha"]` raises TypeError
    # rather than KeyError. Both must be folded into the same reported ValueError.
    from benchmarks.labelling.truth_extraction.build_gold import _check_peps_sha

    broken = tmp_path / "census.json"
    broken.write_text(json.dumps({"_provenance": "oops"}), encoding="utf-8")
    with pytest.raises(ValueError, match="could not be read"):
        _check_peps_sha("a" * 40, broken)


def test_check_peps_sha_rejects_a_mismatch_against_the_census(census: dict):
    # A typo in --peps-sha is otherwise unfalsifiable: it is the sole corpus provenance in the
    # frozen gold manifest. Comparing against census.json's independently recorded peps_sha
    # catches it before the manifest is frozen.
    from benchmarks.labelling.truth_extraction.build_gold import _check_peps_sha

    wrong = "a" * 40
    assert wrong != census["_provenance"]["peps_sha"]
    with pytest.raises(ValueError, match="does not match"):
        _check_peps_sha(wrong, CENSUS)


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


def test_no_edge_pairs_a_superseded_successor_with_a_live_predecessor(census: dict):
    # An INDEPENDENT direction anchor. census.json, gold.manifest.jsonl and
    # peps_trust_queries.json all derive from one edges_from_fields, so a wholesale inversion
    # there flips all three together and the CONSISTENCY guards elsewhere in this suite (this
    # file's direction tests, and test_stale_and_successor_are_not_inverted) still pass: they
    # check the artifacts agree with each other, not that any of them is right.
    #
    # `Status:` is a header field edges_from_fields never reads, so it cannot be fooled by the
    # same inversion. If the graph were flipped, the successor half of many edges would carry a
    # live Status (it is the document actually superseded) while the predecessor half would not,
    # the opposite of the pattern actually observed, which is dominated by Superseded -> Final.
    peps_dir = os.environ.get("RECALL_PEPS_DIR")
    if not peps_dir:
        pytest.skip(
            "RECALL_PEPS_DIR unset: clone python/peps and point it at the nested peps/ dir. "
            "This test is SKIPPED, not passed: the direction anchor is unverified."
        )
    from benchmarks.labelling.truth_extraction.peps_header import header_fields, split_header

    root = Path(peps_dir)

    def status_of(stem: str) -> str:
        head, _ = split_header((root / f"{stem}.rst").read_text(encoding="utf-8", errors="replace"))
        return header_fields(head).get("Status", "")

    live = {"Final", "Active", "Accepted"}
    inverted = [
        (e["superseded"], e["successor"])
        for e in census["edges"]
        if status_of(e["successor"]) == "Superseded" and status_of(e["superseded"]) in live
    ]
    assert not inverted, f"successor marked Superseded, predecessor live: {inverted}"


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


def test_successor_row_count_equals_distinct_superseded_peps(census: dict):
    # NOT `n_header_edges`: five superseded PEPs have two successors each, so grouping by the
    # superseded PEP yields fewer rows than edges. This is the test that replaced the one that
    # pinned the defect in place: a row-per-edge builder passed the old, wrong version of it.
    distinct_superseded = {e["superseded"] for e in census["edges"]}
    rows = json.loads(TRUST.read_text(encoding="utf-8"))
    successors = [r for r in rows if r["expect"] == "successor"]
    assert len(successors) == len(distinct_superseded)


def test_no_two_successor_rows_share_stale_ids():
    # This is the property a row-per-edge builder violates: two rows with identical query text
    # and identical stale_ids, differing only in which single successor each expects. Identical
    # query text retrieves identically, so at most one of such a pair could ever score correct.
    rows = json.loads(TRUST.read_text(encoding="utf-8"))
    successors = [r for r in rows if r["expect"] == "successor"]
    stale_id_lists = [tuple(r["stale_ids"]) for r in successors]
    assert len(stale_id_lists) == len(set(stale_id_lists)), (
        "two successor rows share identical stale_ids, the property that made a pair of rows "
        "unwinnable by construction"
    )


def test_stale_and_successor_are_not_inverted(census: dict):
    # `stale_ids` must hold the SUPERSEDED document and `successor_ids` the SUCCESSOR(s).
    # Swapping them scores a system correct exactly when it prefers the stale document, the
    # failure the trust layer exists to prevent, written into the labels every later number is
    # graded on.
    #
    # This test exists because the suite was measured blind to it: an inverted builder still
    # emits 62 rows, 42 of them `successor`, with matching schema keys and non-empty
    # `successor_ids`, and passes every other test in this file. Counting rows and checking
    # shape says nothing about direction.
    #
    # The comparison is against `census.json`'s edge list, which is independently frozen and
    # whose own direction was verified against the PEP headers. Every row's stale id paired with
    # EVERY id in its successor_ids must be a census edge, and the union over all rows must equal
    # the census edge set exactly; a row's successor_ids can legitimately hold more than one id.
    edges = {(e["superseded"], e["successor"]) for e in census["edges"]}
    rows = json.loads(TRUST.read_text(encoding="utf-8"))

    def stem(chunk_id: str) -> str:
        name = chunk_id.rsplit(":", 1)[0]
        return name[:-4] if name.endswith(".rst") else name

    seen = set()
    for row in (r for r in rows if r["expect"] == "successor"):
        stale = stem(row["stale_ids"][0])
        for successor_id in row["successor_ids"]:
            pair = (stale, stem(successor_id))
            assert pair in edges, (
                f"{row['id']}: {pair[0]} -> {pair[1]} is not a census edge. "
                f"Reversed pair present in census: {(pair[1], pair[0]) in edges}"
            )
            seen.add(pair)
    assert seen == edges, "successor rows do not cover the census edges exactly"


def test_manifest_census_hash_matches_the_committed_census():
    # `corpus_hashes.census` is the manifest's only tie to the census it was frozen against, and
    # re-freezing after a census edit is a manual step. Without this, an edited census leaves the
    # manifest naming a file that is no longer on disk and every other test still passes.
    _, header = read_manifest(GOLD)
    assert header["corpus_hashes"]["census"] == hashlib.sha256(CENSUS.read_bytes()).hexdigest()
    assert header["corpus_hashes"]["peps_sha"] == json.loads(
        CENSUS.read_text(encoding="utf-8"))["_provenance"]["peps_sha"]


def test_no_abstain_stale_pep_appears_in_a_census_edge(census: dict):
    # An abstain row's premise is that its stale PEP has no successor. If that PEP actually names
    # or is named by an edge, in either position, it is not an abstention case: it is either a
    # superseded PEP with a successor (a successor case mislabelled abstain) or a live successor
    # (which should never have been sampled as a stale document at all).
    in_an_edge = {e["superseded"] for e in census["edges"]} | {e["successor"] for e in census["edges"]}
    rows = json.loads(TRUST.read_text(encoding="utf-8"))

    def stem(chunk_id: str) -> str:
        name = chunk_id.rsplit(":", 1)[0]
        return name[:-4] if name.endswith(".rst") else name

    for row in (r for r in rows if r["expect"] == "abstain"):
        stale = stem(row["stale_ids"][0])
        assert stale not in in_an_edge, (
            f"{row['id']}: {stale} appears in a census edge, so it is not an abstention case"
        )
