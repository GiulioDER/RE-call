"""`recall doctor` on a `sqlite:///` DSN: the file, the corpus and the calibration, writing nothing.

Invariants and the failure each one catches:
- D1 a lite DSN whose file does not exist is reported (with the command that creates it) and the
  file is NOT created: the doctor writes nothing.
- D2 an indexed, certified lite store reports its chunks and a certified calibration, and Docker as
  not needed.
- D3 a file holding another tenant is a blocking failure that names both tenants.
- D4 a calibration fitted to an earlier corpus is reported stale, not certified.

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
"""

from __future__ import annotations

from pathlib import Path

from recall.doctor import run_checks
from recall.index import Indexer
from recall.lite import LiteStore
from recall.lite.calibration import auto_calibrate
from tests.test_lite_calibration import DIM, ContentWordEmbedder, _write_memos


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
        _write_memos(tmp_path / "m", 1, start=48)
        Indexer(store, ContentWordEmbedder(), env={}).index_path(tmp_path / "m")
    status, detail = _checks("sqlite:///" + db.as_posix())["calibration"]
    assert status == "warn" and "earlier version" in detail, f"a stale calibration was reported as certified: {detail}"
