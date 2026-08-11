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
"""
from __future__ import annotations

import json
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
