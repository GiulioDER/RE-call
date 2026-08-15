"""The evidence freeze: a script that runs ONCE against a live index, tested before it does.

Everything here is offline. The point is that the expensive run gets one attempt, so its refusal
paths and its digest canonicalisation are worth pinning in advance rather than discovering
afterwards that the artifact cannot be verified.

Properties, one test each:
  1. The digest is recomputable from the written file, which is what makes the pre-registered
     check A6 implementable at all.
  2. `_provenance` is outside the digest, so a regeneration's timestamp does not change it.
  3. Key order does not change the digest, so an unrelated refactor cannot invalidate a fixture.
  4. Changing any evidence byte DOES change it.
  5. `--verify` returns 0 on an honest fixture and 1 on a tampered one.
  6. An existing fixture is not silently overwritten.
  7. `--verify` needs no index, no dsn and no network.
  8. The row split is the experiment's design, so it is pinned rather than derived.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.freeze_supersession_evidence import (
    DIGEST_ALGORITHM,
    ROWS,
    canonical_evidence_json,
    evidence_digest,
    main,
    verify,
)

EVIDENCE = {
    "qst_0418": {"group": "A_supersession", "hits": [{"chunk_id": "c1", "score": 0.5,
                                                      "text": "v2 thresholds, naïve ünicode"}]},
    "qst_0310": {"group": "B_coverage", "hits": [{"chunk_id": "c2", "score": 0.25, "text": "x"}]},
}


def _fixture(tmp_path: Path, evidence: dict | None = None) -> Path:
    evidence = EVIDENCE if evidence is None else evidence
    payload = {
        "_provenance": {
            "generated_at": "2026-08-15T20:00:00+00:00",
            "evidence_sha256": evidence_digest(evidence),
            "digest_algorithm": DIGEST_ALGORITHM,
        },
        "evidence": evidence,
    }
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")
    return path


def test_the_digest_is_recomputable_from_the_written_file(tmp_path: Path):
    """The defect this closes: the digest was computed from bytes that appeared nowhere on disk.

    `sha256sum` of the file is a DIFFERENT value, so a verifier hashing the file concludes the
    fixture is corrupt. The recipe has to be recoverable from the artifact, and it is: the
    algorithm string travels in the provenance beside the digest.
    """
    import hashlib

    path = _fixture(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    # The recipe, applied by a reader who has ONLY the file and its `digest_algorithm` string.
    recomputed = hashlib.sha256(
        json.dumps(payload["evidence"], indent=1, sort_keys=True, ensure_ascii=False)
        .encode("utf-8")
    ).hexdigest()
    assert recomputed == payload["_provenance"]["evidence_sha256"]

    # ...and the file's own hash is NOT it, which is the trap the algorithm string exists to
    # prevent a verifier falling into.
    assert hashlib.sha256(path.read_bytes()).hexdigest() != recomputed
    assert "sort_keys=True" in DIGEST_ALGORITHM and "ensure_ascii=False" in DIGEST_ALGORITHM


def test_provenance_is_outside_the_digest(tmp_path: Path):
    """Otherwise every regeneration is a different fixture and 'unchanged between arms' is
    unfalsifiable, because provenance carries a timestamp."""
    a = _fixture(tmp_path)
    first = json.loads(a.read_text(encoding="utf-8"))
    second = dict(first)
    second["_provenance"] = {**first["_provenance"], "generated_at": "2099-01-01T00:00:00+00:00"}
    assert evidence_digest(first["evidence"]) == evidence_digest(second["evidence"])


def test_key_order_does_not_change_the_digest():
    reordered = {k: EVIDENCE[k] for k in reversed(list(EVIDENCE))}
    assert evidence_digest(reordered) == evidence_digest(EVIDENCE)
    assert list(reordered) != list(EVIDENCE), "the reordering must be real or this is vacuous"


def test_changing_one_evidence_byte_changes_the_digest():
    """The positive control for every test above. Without it they pass for a constant function."""
    tampered = json.loads(json.dumps(EVIDENCE))
    tampered["qst_0418"]["hits"][0]["text"] += "."
    assert evidence_digest(tampered) != evidence_digest(EVIDENCE)


def test_non_ascii_survives_canonicalisation():
    """`ensure_ascii=False` is load bearing: the corpus is not ASCII."""
    assert "naïve ünicode" in canonical_evidence_json(EVIDENCE)


def test_verify_accepts_an_honest_fixture_and_refuses_a_tampered_one(tmp_path: Path, capsys):
    path = _fixture(tmp_path)
    assert verify(path) == 0

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["evidence"]["qst_0418"]["hits"][0]["text"] = "silently edited"
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8",
                    newline="\n")
    assert verify(path) == 1
    assert "MISMATCH" in capsys.readouterr().out


def test_verify_needs_no_index_and_no_dsn(tmp_path: Path):
    """`--verify` is the check a third party runs, and they have neither."""
    path = _fixture(tmp_path)
    assert main(["--verify", str(path)]) == 0


def test_an_existing_fixture_is_not_silently_overwritten(tmp_path: Path):
    """It is the only copy and its digest is published; a silent clobber strands the
    pre-registration."""
    path = _fixture(tmp_path)
    with pytest.raises(SystemExit, match="--force"):
        main(["--questions", str(tmp_path / "q.jsonl"), "--dsn", "postgresql://x/y",
              "--out", str(path)])
    # ...and the file is untouched by the refusal.
    assert verify(path) == 0


def test_the_row_split_is_pinned_not_derived():
    """The A/B/C split IS the experiment's design. A silently different set would invalidate it."""
    assert ROWS["A_supersession"] == ("qst_0418", "qst_0419", "qst_0420", "qst_0425")
    assert len(ROWS["B_coverage"]) == 6
    assert ROWS["C_attribution"] == ("qst_0413",)
    everything = [q for ids in ROWS.values() for q in ids]
    assert len(everything) == len(set(everything)) == 11
