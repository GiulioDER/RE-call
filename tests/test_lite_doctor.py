"""`recall doctor` on a `sqlite:///` DSN: the file, the corpus and the calibration, writing nothing.

Invariants and the failure each one catches:
- D1 a lite DSN whose file does not exist is reported (with the command that creates it) and the
  file is NOT created: the doctor writes nothing.
- D2 an indexed, certified lite store reports its chunks and a certified calibration, and Docker as
  not needed.
- D3 a file holding another tenant is a blocking failure that names both tenants.
- D4 a calibration fitted to an earlier corpus whose threshold no longer re-verifies is reported
  stale, not certified.
- D5 a calibration carried forward to a changed corpus is reported as carried, with its delta,
  exactly as search serves it (the doctor and search must not disagree).

Red proof, 2026-10-07, each mutation alone, failing in its intended assertion (JUnit XML), then
restored byte for byte and green:
- Q1 (D1) `recall.doctor._lite_checks` opening the store through `LiteStore.from_dsn` first (the
  obvious way to read it, which creates the file and its schema): "the doctor created the store
  file". A first attempt, opening read-write in `inspect_lite_file`, SURVIVED: the existence check
  returns before any open, so it was the wrong mutation, not a passing guard.
- Q2 (D3) the tenant comparison in `recall.doctor._lite_checks` removed: "a file of another tenant
  was not refused".
- Q3 (D4) `inspect_lite_file` ignoring the corpus fingerprint: "a stale calibration was reported as
  certified".

Red proof, 2026-10-12, for the carry-forward changes (same method; JUnit XML on the test host):
- Q4 (D4) `judge_inherited_threshold` certifying on separability alone (M2 in
  `tests/test_lite_carry_forward.py`): "a stale calibration was reported as certified".
- Q5 (D5) `inspect_lite_file` without its carry (the pre-change rule, any change reads stale):
  "the doctor called a carried calibration stale".
"""

from __future__ import annotations

from pathlib import Path

from recall.doctor import run_checks
from recall.index import Indexer
from recall.lite import LiteStore
from recall.lite.calibration import auto_calibrate
from tests.test_lite_calibration import DIM, ContentWordEmbedder, _shift_gap_questions, _write_memos


def _checks(dsn: str, tenant: str = "default") -> dict[str, tuple[str, str]]:
    report = run_checks(dsn=dsn, embedder="hashing", tenant=tenant, project_root=None)
    return {c.name: (c.status, c.detail) for c in report.checks}


def _certified_store(tmp_path: Path, memos: int = 48) -> Path:
    db = tmp_path / "store" / "memory.db"
    with LiteStore(db, dim=DIM) as store:
        _write_memos(tmp_path / "m", memos)
        Indexer(store, ContentWordEmbedder(), env={}).index_path(tmp_path / "m")
        assert auto_calibrate(store, ContentWordEmbedder()).status.value == "certified"
    return db


def test_a_missing_file_is_reported_and_not_created(tmp_path: Path) -> None:
    """D1."""
    db = tmp_path / "nothing" / "memory.db"
    checks = _checks("sqlite:///" + db.as_posix())
    assert not db.exists(), "the doctor created the store file"
    assert checks["database"][0] == "warn" and "no file yet" in checks["database"][1]


def test_a_certified_store_is_reported_healthy(tmp_path: Path) -> None:
    """D2."""
    db = _certified_store(tmp_path)
    checks = _checks("sqlite:///" + db.as_posix())
    assert checks["docker"][0] == "skip"
    assert checks["database"][0] == "ok"
    assert checks["corpus"] == ("ok", f"48 chunk(s) from 48 source(s), {DIM}-wide embeddings")
    assert checks["calibration"][0] == "ok" and "certified" in checks["calibration"][1]


def test_another_tenants_file_is_refused(tmp_path: Path) -> None:
    """D3."""
    db = _certified_store(tmp_path)
    status, detail = _checks("sqlite:///" + db.as_posix(), tenant="acme")["corpus"]
    assert status == "fail" and "'default'" in detail and "'acme'" in detail, "a file of another tenant was not refused"


def test_a_calibration_for_an_earlier_corpus_is_stale(tmp_path: Path) -> None:
    """D4."""
    db = _certified_store(tmp_path)
    with LiteStore(db, dim=DIM) as store:
        _shift_gap_questions(store, tmp_path / "m", 6)
        Indexer(store, ContentWordEmbedder(), env={}).index_path(tmp_path / "m")
    status, detail = _checks("sqlite:///" + db.as_posix())["calibration"]
    assert status == "warn" and "earlier version" in detail, f"a stale calibration was reported as certified: {detail}"


def test_a_carried_calibration_is_reported_as_carried(tmp_path: Path) -> None:
    """D5."""
    db = _certified_store(tmp_path)
    with LiteStore(db, dim=DIM) as store:
        _write_memos(tmp_path / "m", 2, start=48)
        Indexer(store, ContentWordEmbedder(), env={}).index_path(tmp_path / "m")
        assert store.resolve_calibration().status.value == "certified"
    status, detail = _checks("sqlite:///" + db.as_posix())["calibration"]
    assert status == "ok", f"the doctor called a carried calibration stale: {detail}"
    assert "carried forward" in detail and "4%" in detail, detail


def _set_meta(db: Path, key: str, value: str) -> None:
    import sqlite3

    conn = sqlite3.connect(db)
    with conn:
        conn.execute("UPDATE meta SET value = ? WHERE key = ?", (value, key))
    conn.close()


def test_a_fit_for_another_model_names_that_model(tmp_path: Path) -> None:
    """CCA DOC-001 (run 20261107T115351Z). Invariant: when the doctor reports a calibration stale
    because the store is bound to another model than the fit was made for, it says so and names
    both models. Failure caught: it printed the only STALE message it had, which blames a changed
    corpus that never changed.

    Red proof, 2026-11-07 on the remote test host: with `and found.stale_for_model is not None`
    replaced by `and False` in `recall.doctor._lite_checks`, this failed at the detail assertion
    ("fitted to an earlier version ..."); restored, it passed."""
    db = _certified_store(tmp_path)
    with LiteStore(db, dim=DIM) as store:
        store.record_embedder_spec("another", model="another-model", dimension=DIM)
    status, detail = _checks("sqlite:///" + db.as_posix())["calibration"]
    assert status == "warn", detail
    assert "fitted for the model test-content-words" in detail and "another-model" in detail, detail


def test_the_doctor_reads_the_binding_as_search_does(tmp_path: Path) -> None:
    """CCA CODE-001 (run 20261107T115351Z). Invariant: the doctor and search read a store's bound
    width through one parser, so a width `int()` accepts (" 64") is current for both, and one it
    rejects ("--64") makes the doctor report STALE instead of raising. Failure caught: the doctor's
    own guard read " 64" as unknown (STALE while search answered) and let "--64" through to an
    `int()` that raised out of `inspect_lite_file`.

    Red proof, 2026-11-07 on the remote test host: with `bound_from_meta(meta)` in
    `recall.lite.inspect.inspect_lite_file` replaced by the earlier inline guard
    (`int(width) if width.lstrip("-").isdigit() else -1`), this failed at the first assertion
    (`STALE`); restored, it passed."""
    from recall.calibration_v2 import CalibrationStatus
    from recall.lite.inspect import inspect_lite_file

    db = _certified_store(tmp_path)
    _set_meta(db, "embedder_dimension", f" {DIM}")
    with LiteStore(db, dim=DIM) as store:
        assert store.resolve_calibration().status is CalibrationStatus.CERTIFIED
    assert inspect_lite_file(db).calibration is CalibrationStatus.CERTIFIED, "the report and search disagree"
    _set_meta(db, "embedder_dimension", f"--{DIM}")
    assert inspect_lite_file(db).calibration is CalibrationStatus.STALE
