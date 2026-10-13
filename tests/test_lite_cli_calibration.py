"""`recall calibration list/show/export` on a `sqlite:///` DSN; the rest refused by name.

On a lite DSN the whole group reached PostgreSQL (`CalibrationRepository`) and died in psycopg.
A lite store keeps each calibration as one checksummed artifact, without the query set and without
PostgreSQL's calibration table, so: `list` matches PostgreSQL field for field, `show` prints the
artifact, `export` writes the artifact alone (saying the query set is not kept), and `import`,
`calibrate`, `carry-forward`, `drift` and `auto` are refused by name, because a lite store
calibrates itself when it is indexed.

Invariants and the failure each one catches:
- C1 `list` returns every stored calibration, newest first, with PostgreSQL's keys, and marks an
  older one `superseded` (a lite store serves only its newest calibration).
- C2 `show` prints the artifact with its checksum; an unknown id exits naming it.
- C3 `export` writes the artifact, verifiable by its checksum, flagged as artifact-only.
- C4 the generation-bound subcommands are refused by name, and create no file.
- C5 `list` on a lite file that does not exist prints [] and does not create it.
- C6 the generation route is refused by name.

Red proof: recorded in each test's docstring.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from recall.index import Indexer
from recall.lite import LiteStore
from recall.lite.calibration import artifact_from_json, ensure_calibrated
from tests.test_lite_calibration import DIM, ContentWordEmbedder, _write_memos
from tests.test_lite_cli import _run

#: `CalibrationRepository.list_records`' keys, so a lite listing reads like a PostgreSQL one.
POSTGRES_LIST_KEYS = {
    "calibration_id", "generation_id", "lifecycle_state", "certified", "pipeline_fingerprint",
    "corpus_fingerprint", "query_set_digest", "created_at", "threshold_was_measured_here",
    "carried_forward_from", "corpus_delta",
}


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("RECALL_ENV", "RECALL_INDEX_MODE", "RECALL_SERVING_DSN", "RECALL_DSN", "RECALL_TRUST_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")


def _dsn(path: Path) -> str:
    return "sqlite:///" + path.as_posix()


def _twice_calibrated(tmp_path: Path) -> tuple[Path, list[str]]:
    """A lite file holding two calibrations: one fitted, then one refitted after more memos."""
    path, folder, embedder = tmp_path / "memory.db", tmp_path / "mem", ContentWordEmbedder()
    ids = []
    with LiteStore(path, dim=DIM) as store:
        for start, count in ((0, 40), (40, 8)):
            _write_memos(folder, count, start=start)
            Indexer(store, embedder, env={}).index_path(folder)
            outcome = ensure_calibrated(store, embedder)
            assert outcome.artifact is not None, outcome
            ids.append(outcome.artifact.calibration_id)
    return path, ids


def test_c1_list_matches_postgres_fields(tmp_path: Path) -> None:
    """Red: baseline `449aa0a1` raises from psycopg on the lite DSN; after the change, the older
    calibration left `published` (no supersession pass) fails the lifecycle assertion."""
    path, (first, second) = _twice_calibrated(tmp_path)
    out = _run(["--dsn", _dsn(path), "calibration", "list"])
    records = json.loads(out)
    assert [r["calibration_id"] for r in records] == [second, first], records
    assert all(set(r) == POSTGRES_LIST_KEYS for r in records), records
    assert [r["lifecycle_state"] for r in records] == ["published", "superseded"], records
    assert records[0]["threshold_was_measured_here"] is True and records[0]["carried_forward_from"] is None


def test_c2_show_prints_the_artifact(tmp_path: Path) -> None:
    """Red: `show_record` returning the newest artifact whatever the id: the first id shows the
    second calibration."""
    path, (first, _second) = _twice_calibrated(tmp_path)
    shown = json.loads(_run(["--dsn", _dsn(path), "calibration", "show", first]))
    assert shown["calibration_id"] == first and shown["checksum"] and shown["lifecycle_state"] == "superseded"
    missing = _run(["--dsn", _dsn(path), "calibration", "show", "cal_nope"])
    assert "EXIT" in missing and "cal_nope" in missing, missing


def test_c3_export_writes_the_artifact_alone(tmp_path: Path) -> None:
    """Red: `export_artifact` writing the bundle without the artifact-only flag: the flag assertion
    fails (a reader could not tell it from a PostgreSQL bundle missing its query set)."""
    path, (_first, second) = _twice_calibrated(tmp_path)
    target = tmp_path / "out.json"
    out = _run(["--dsn", _dsn(path), "calibration", "export", second, "--output", str(target)])
    assert str(target) in out and "query set" in out, out
    bundle = json.loads(target.read_text(encoding="utf-8"))
    assert bundle.get("artifact_only") is True and "query_set" not in bundle, f"not flagged artifact-only: {bundle}"
    artifact = artifact_from_json(json.dumps({k: v for k, v in bundle["artifact"].items()}))
    assert artifact.calibration_id == second


@pytest.mark.parametrize(
    "argv",
    [
        ["import", "bundle.json"],
        ["calibrate", "--generation", "g", "--queries", "q.json"],
        ["carry-forward", "--generation", "g"],
        ["drift", "--generation", "g"],
        ["auto", "--generation", "g"],
    ],
)
def test_c4_generation_bound_subcommands_are_refused(tmp_path: Path, argv: list[str]) -> None:
    """Red: the refusal removed for `argv[0]`: it reaches PostgreSQL and raises from psycopg."""
    path = tmp_path / "memory.db"
    out = _run(["--dsn", _dsn(path), "calibration", *argv])
    assert "EXIT" in out and f"calibration {argv[0]}" in out and "lite store" in out, out
    assert not path.exists(), "a refused subcommand created the lite file"


def test_c5_list_on_a_missing_file(tmp_path: Path) -> None:
    """Red: the existence check removed: listing creates the file."""
    path = tmp_path / "absent.db"
    assert json.loads(_run(["--dsn", _dsn(path), "calibration", "list"])) == []
    assert not path.exists(), "listing calibrations created a lite file"


def test_c6_the_generation_route_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Red: the route refusal removed: no refusal by name."""
    monkeypatch.setenv("RECALL_INDEX_MODE", "generation")
    out = _run(["--dsn", _dsn(tmp_path / "memory.db"), "calibration", "list"])
    assert "has no generations" in out, out


def _list(path: Path, *flags: str) -> str:
    return _run(["--dsn", _dsn(path), *flags, "calibration", "list"])


def _corrupt(path: Path, calibration_id: str) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "UPDATE calibrations SET payload = '{\"x\": 1}' WHERE json_extract(payload, '$.calibration_id') = ?",
            (calibration_id,),
        )
        conn.commit()
    finally:
        conn.close()


def test_c7_a_read_never_writes_a_file_that_is_not_a_lite_store(tmp_path: Path) -> None:
    """BUG-001 (P1). Red: the read path opening the store at `inspect_lite_file(path).dim or 1`
    (the code before this fix) stamps a 0-byte or foreign SQLite file with a lite `meta` table
    (dim 1) and lite tables, so `recall index` at the real width is refused afterwards."""
    empty = tmp_path / "empty.db"
    empty.write_bytes(b"")
    foreign = tmp_path / "foreign.db"
    conn = sqlite3.connect(foreign)
    conn.execute("CREATE TABLE notes (x TEXT)")
    conn.commit()
    conn.close()
    for path in (empty, foreign):
        before = path.read_bytes()
        out = _list(path)
        assert "EXIT" in out and "not a lite store" in out, out
        assert path.read_bytes() == before, f"a read wrote into {path.name}"


def test_c8_a_corrupt_calibration_is_reported_and_the_rest_still_list(tmp_path: Path) -> None:
    """BUG-002 and BUG-003. Red: with the newest calibration corrupt, the code before this fix
    died with "holds dim=..., not 1" (the width came from an inspect that stops at the bad
    calibration); with an older one corrupt, `list` died for every row."""
    path, (first, second) = _twice_calibrated(tmp_path)
    _corrupt(path, second)
    out = _list(path)
    assert out.lstrip().startswith("["), f"list failed on a corrupt newest calibration: {out}"
    listed = json.loads(out)
    assert [r["calibration_id"] for r in listed] == [first], listed
    # The store still serves the corrupt newest row, so nothing listed is in force.
    assert listed[0]["lifecycle_state"] == "superseded", f"an unserved calibration was reported in force: {listed}"
    _corrupt(path, first)
    assert json.loads(_list(path)) == [], "the listing did not survive two corrupt rows"
    shown = _run(["--dsn", _dsn(path), "calibration", "show", second])
    assert "EXIT" in shown and second in shown and "do not verify" in shown, shown


def test_c9_another_tenant_is_refused_by_name(tmp_path: Path) -> None:
    """BUG-006. Red: the code before this fix let `LiteStore` raise `LiteStoreError` out of the
    command, a traceback that never names `--tenant`."""
    path, _ids = _twice_calibrated(tmp_path)
    out = _list(path, "--tenant", "other")
    assert "EXIT" in out and "--tenant default" in out, out


def test_c10_listing_reads_only_the_width_not_the_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PERF-001. Red: the code before this fix called `inspect_lite_file`, which reads and hashes
    every chunk only to learn the width; with it made to fail, `list` failed."""
    path, (_first, second) = _twice_calibrated(tmp_path)

    def _no_corpus_scan(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("calibration list scanned the corpus")

    monkeypatch.setattr("recall.lite.inspect.inspect_lite_file", _no_corpus_scan)
    out = _list(path)
    assert out.lstrip().startswith("["), f"list did more than read the width: {out}"
    assert json.loads(out)[0]["calibration_id"] == second
