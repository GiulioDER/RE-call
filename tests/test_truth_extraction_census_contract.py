"""The census artifact boundary.

Properties, one test each:
  1. A payload missing `_provenance` is refused.
  2. A `_provenance` missing `peps_sha` is refused — the artifact would name no corpus version.
  3. A `_provenance` missing `recall_commit` is refused.
  4. A census whose `n_header_edges` disagrees with `len(edges)` is refused, because the two
     are the same fact written twice and a reader cannot tell which one is the typo.
  5. A census whose `n_restated_in_prose` disagrees with `len(restatements)` is refused.
  6. A census claiming more restatements than header edges is refused — the ceiling cannot
     exceed 100%.
  7. A ceiling of EXACTLY 100% is accepted. A corpus that restates every edge is legitimate, and
     refusing it would be a validator rejecting its own best possible input.
  8. A well-formed payload is NOT rejected.
  9. The write site calls the validator.
  10. The writer emits LF regardless of platform.
  11. The blind CSV exposes no arm / model / judge / score / rule / system column.
  12. The un-blinding key is a separate file, and every CSV item has an entry in it.
  13. The verdict column ships blank — blank is "undecidable", per score_beam_labels.py:29.
  14. Every row names a candidate target; the unprovable-target class is excluded, not guessed.
  15. The committed row count recomputes from the corpus (skips loudly without RECALL_PEPS_DIR).
  16. The PEPs SHA format validator accepts a real 40-hex SHA and refuses malformed input.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from benchmarks.labelling.truth_extraction.artifact_contract import validate_census
from benchmarks.labelling.truth_extraction.census import write_census


def _ok() -> dict:
    return {
        "n_files": 733,
        "n_header_edges": 2,
        "n_prose_marker_files": 209,
        "n_marker_without_header": 175,
        "n_restated_in_prose": 1,
        "edges": [
            {"superseded": "pep-0216", "successor": "pep-0287"},
            {"superseded": "pep-0386", "successor": "pep-0440"},
        ],
        "restatements": {"pep-0216->pep-0287": "It has been superseded by :pep:`287`."},
        "marker_without_header": ["pep-0001"],
        "file_digests": {"pep-0216.rst": "a" * 64},
        "_provenance": {
            "peps_sha": "5981b2a292610104eb30735423504c52fe454650",
            "clone_date": "2026-08-11",
            "recall_commit": "439717b",
            "generated_at": "2026-08-11T12:00:00+00:00",
            "model_stack": {},
            "invocation": "python -m benchmarks.labelling.truth_extraction.census ...",
        },
    }


def test_missing_provenance_is_refused():
    payload = _ok()
    del payload["_provenance"]
    with pytest.raises(ValueError, match="_provenance"):
        validate_census(payload)


def test_provenance_without_peps_sha_is_refused():
    payload = _ok()
    del payload["_provenance"]["peps_sha"]
    with pytest.raises(ValueError, match="peps_sha"):
        validate_census(payload)


def test_provenance_without_recall_commit_is_refused():
    payload = _ok()
    del payload["_provenance"]["recall_commit"]
    with pytest.raises(ValueError, match="recall_commit"):
        validate_census(payload)


def test_edge_count_disagreeing_with_edge_list_is_refused():
    payload = _ok()
    payload["n_header_edges"] = 3
    with pytest.raises(ValueError, match="n_header_edges"):
        validate_census(payload)


def test_restated_count_disagreeing_with_restatements_is_refused():
    payload = _ok()
    payload["n_restated_in_prose"] = 2
    with pytest.raises(ValueError, match="n_restated_in_prose"):
        validate_census(payload)


def test_ceiling_above_one_hundred_percent_is_refused():
    # Two restatements against one edge: more edges stated in prose than exist in the headers,
    # which means the restatement detector matched something outside the gold set.
    payload = _ok()
    payload["n_header_edges"] = 1
    payload["edges"] = [{"superseded": "pep-0216", "successor": "pep-0287"}]
    payload["restatements"] = {
        "pep-0216->pep-0287": "It has been superseded by :pep:`287`.",
        "pep-0386->pep-0440": "supersedes :pep:`386` even for metadata v1.",
    }
    payload["n_restated_in_prose"] = 2
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_census(payload)


def test_ceiling_of_exactly_one_hundred_percent_is_accepted():
    # A corpus where EVERY header edge is also stated in prose is a legitimate corpus, not a
    # malformed artifact. This pins the comparison at `>` and not `>=`: the loose form refuses a
    # perfect corpus, which is a validator rejecting its own best possible input.
    payload = _ok()
    payload["n_header_edges"] = 1
    payload["edges"] = [{"superseded": "pep-0216", "successor": "pep-0287"}]
    validate_census(payload)  # exactly 1 restatement, exactly 1 edge — must not raise


def test_well_formed_payload_is_accepted():
    validate_census(_ok())  # must not raise


def test_write_site_calls_the_validator(tmp_path: Path):
    payload = _ok()
    del payload["_provenance"]["peps_sha"]
    with pytest.raises(ValueError, match="peps_sha"):
        write_census(tmp_path / "census.json", payload)
    assert not (tmp_path / "census.json").exists(), "refused payload must not be written"


def test_writer_emits_lf_not_crlf(tmp_path: Path):
    path = tmp_path / "census.json"
    write_census(path, _ok())
    raw = path.read_bytes()
    assert b"\r\n" not in raw
    assert json.loads(raw.decode("utf-8"))["n_files"] == 733


def test_validate_peps_sha_format_accepts_a_real_sha():
    from benchmarks.labelling.truth_extraction.census import _validate_peps_sha_format

    _validate_peps_sha_format("5981b2a292610104eb30735423504c52fe454650")  # must not raise


def test_validate_peps_sha_format_rejects_a_malformed_sha():
    from benchmarks.labelling.truth_extraction.census import _validate_peps_sha_format

    with pytest.raises(ValueError, match="40-character"):
        _validate_peps_sha_format("not-a-sha")


_TE = Path(__file__).resolve().parents[1] / "benchmarks" / "labelling" / "truth_extraction"
CSV_PATH = _TE / "adjudication.csv"
KEY_PATH = _TE / "adjudication_key.json"


def _csv_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_blind_csv_leaks_no_arm_model_or_judge_column():
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    leaky = [
        column
        for column in header
        if any(token in column.lower() for token in ("arm", "model", "judge", "score", "rule", "system"))
    ]
    assert not leaky, f"blind CSV exposes {leaky} — the adjudicator would see what produced the row"


def test_the_key_is_a_separate_file_from_the_csv():
    assert KEY_PATH.exists() and KEY_PATH != CSV_PATH
    key = json.loads(KEY_PATH.read_text(encoding="utf-8"))
    assert key, "key file is empty — nothing could be un-blinded after labelling"


def test_every_csv_item_has_a_key_entry():
    items = {row["item"] for row in _csv_rows()}
    assert items == set(json.loads(KEY_PATH.read_text(encoding="utf-8")))


def test_the_builder_emits_a_blank_verdict_column():
    """Asserted on the BUILDER, not on the committed pack, because the pack is now labelled.

    This used to read `all(row["your_verdict_Y_or_N"] == "")` over the committed CSV, which was
    right while the sheet was waiting for an adjudicator and became false the moment one filled it
    in. Deleting it would have lost the property that still matters: a REBUILD must hand the next
    adjudicator an empty column, never carry a previous round's verdicts forward, which is how a
    labelling exercise quietly turns into a measurement of the last labelling exercise.

    The committed pack's own verdicts are pinned separately, in
    `tests/test_truth_extraction_adjudication.py`.
    """
    source = (
        Path(__file__).resolve().parent.parent
        / "benchmarks" / "labelling" / "truth_extraction" / "build_adjudication.py"
    ).read_text(encoding="utf-8")
    assert '"your_verdict_Y_or_N": ""' in source, (
        "the builder must emit an empty verdict column, or a rebuild ships a pre-labelled sheet"
    )


def test_every_row_names_a_candidate_target():
    # The pool is 175 FILES but only the 30 that name a target in the marker's sentence are
    # adjudicable. A row with an empty target would be asking a human to guess at an unprovable
    # one, which is the class fix.py refuses to guess at rather than the class it adjudicates.
    assert all(row["candidate_target"].startswith("pep-") for row in _csv_rows())


def test_row_count_recomputes_from_the_corpus():
    peps_dir = os.environ.get("RECALL_PEPS_DIR")
    if not peps_dir:
        pytest.skip(
            "RECALL_PEPS_DIR unset — the committed row count is UNVERIFIED against the corpus. "
            "Clone python/peps and point it at the nested peps/ dir."
        )
    from benchmarks.labelling.truth_extraction.build_adjudication import build_rows

    rebuilt, _ = build_rows(Path(peps_dir), seed=0, limit=None)
    assert len(rebuilt) == len(_csv_rows())
