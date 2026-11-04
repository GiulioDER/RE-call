"""Fixes from the second DEEP CCA hunt of `recall/lite` (run 20261104T105718Z).

The first hunt (run 20261103T223409Z, PR 1165) left 44 findings unfixed; this run re-checked them
as auditor context and found new ones. Each test names the invariant it pins and the failure it
catches. Red proof: each was run on the test host against the pre-fix snapshot of the production
files (`.claude/audits/.prefix_snapshot/prefix/`, HEAD 3eb9c71f4) and failed in the assertion its
docstring quotes, then passed on the fixed tree.

- K1 (STAKES-001, P1) a calibration fit that outlives a forget of one of its sources keeps no
  evidence naming that source: its test questions quote the forgotten text.
- K2 (PERF-005) once a fit is in force, the evidence of older fits is deleted (only the newest
  fit's is ever read), so it does not grow by one row per fit.
- K3 (STAKES-002) bytes a re-index frees are zeroed, not only those a forget frees.
- K4 (STAKES-004) a forget's closing checkpoint blocked by a reader is reported, not only logged.
- K5 (BUG-004) a refused open closes its SQLite connection.
- K6 (DEPLOY-001) a SQLite too old for the schema is refused before the file is created.
- K7 (PERF-004) the fit in force is found through an index, not a scan and sort.
- K8 (ENV-001) a lane wait of nan or inf falls back to the default; empty means the default.
- K9 (CODE-018) the queue notice gives the wait in seconds, not the variable's name.
- K10 (ENV-002) doctor and the dashboard refuse a relative lite DSN as the server does.
- K11 (ENV-004) a refresh state file written for another store is not read as this one's.
- K12 (ENV-005) setup names the refresh state file even when the store's own pattern is ignored.
- K13 (ENV-008) the registered server carries the embedding cache settings setup ran with.
- K14 (STAKES-005) a memo id with a trailing newline names no file.
- K15 (SEC-003) a symbolic link at the authorship ledger is refused.
- K16 (SEC-002) the refresh source policy judges the name the indexer stores, relative to key_root.
- K17 (NUM-002) calibration status reports the carried calibration's separability.
- K18 (CODE-010) an operation receipt reads a naive moment as UTC.
- K19 (CODE-023) learned sparse retrieval raises `LiteUnsupported`.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import pytest

import recall.lite.calibration as cal
import recall.lite.store as store_mod
from recall.calibration_v2 import CalibrationStatus
from recall.index import Indexer
from recall.lite import LiteStore, LiteStoreError, LiteUnsupported
from recall.lite.setup import run_lite_setup
from recall.types import Chunk
from tests.test_lite_calibration import DIM, ContentWordEmbedder, _store, _write_memos


def _evidence(store: LiteStore) -> list[tuple[str, str]]:
    with store._lock:  # noqa: SLF001  # the test reads the table the module writes
        return [(str(a), str(b)) for a, b in store._conn.execute("SELECT calibration_id, sources FROM calibration_evidence")]  # noqa: SLF001


class _ForgetsDuringTheFit(ContentWordEmbedder):
    """Deletes one source the first time the fit embeds a question: a forget racing a refit."""

    def __init__(self, store: LiteStore, source: str) -> None:
        super().__init__()
        self._store: LiteStore | None = store
        self._source = source

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._store is not None:
            store, self._store = self._store, None
            store.delete_sources([self._source])
        return super().embed(texts)


def test_k1_a_fit_outliving_a_forget_keeps_no_evidence_of_it(tmp_path: Path) -> None:
    """Red against `save_calibration` without the source check: "the forgotten source's questions came back"."""
    store = _store(tmp_path, 48)
    forgotten = sorted({row.source for row in store.iter_chunks()})[0]
    cal.auto_calibrate(store, _ForgetsDuringTheFit(store, forgotten))
    assert all(forgotten not in json.loads(sources) for _cid, sources in _evidence(store)), (
        "the forgotten source's questions came back"
    )
    assert store.resolve_calibration().status is not CalibrationStatus.CERTIFIED or not _evidence(store)


def test_k2_only_the_fit_in_force_keeps_evidence(tmp_path: Path) -> None:
    """Red against `save_calibration` without the DELETE of other fits' evidence: "2 evidence rows"."""
    store = _store(tmp_path, 48)
    assert cal.auto_calibrate(store, ContentWordEmbedder()).status is CalibrationStatus.CERTIFIED
    _write_memos(tmp_path / "m", 40, start=48)
    Indexer(store, ContentWordEmbedder(), env={}).index_path(tmp_path / "m")
    second = cal.auto_calibrate(store, ContentWordEmbedder())
    assert second.status is CalibrationStatus.CERTIFIED and second.artifact is not None, second.reason
    rows = _evidence(store)
    assert len(rows) == 1, f"{len(rows)} evidence rows"
    assert rows[0][0] == second.artifact.calibration_id


def test_k3_a_reindex_zeroes_the_bytes_it_frees(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Red against no `secure_delete` outside `erasing()`: "the old version's bytes are still in the file".

    SQLite's default differs by build (on in Debian's, off in the Windows CPython one), so every
    connection here starts with it off, as on a build whose default is off.
    """
    real = sqlite3.connect

    def default_off(*args: object, **kwargs: object) -> sqlite3.Connection:
        conn = real(*args, **kwargs)  # type: ignore[arg-type]
        conn.execute("PRAGMA secure_delete = OFF")
        return conn

    monkeypatch.setattr(store_mod.sqlite3, "connect", default_off)
    marker = "zqxoldversionmarker"
    store = LiteStore(tmp_path / "memory.db", dim=DIM)
    old = Chunk(id="a", source="/m/a.md", text="first text", metadata={"file": "a.md", "note": marker * 200})
    store.replace_sources(["/m/a.md"], [old], [[1.0] + [0.0] * (DIM - 1)])
    new = Chunk(id="a", source="/m/a.md", text="second text", metadata={"file": "a.md"})
    store.replace_sources(["/m/a.md"], [new], [[1.0] + [0.0] * (DIM - 1)])
    assert store.checkpoint()
    store.close()
    raw = b"".join(p.read_bytes() for p in tmp_path.glob("memory.db*"))
    assert marker.encode() not in raw, "the old version's bytes are still in the file"


def test_k4_a_blocked_erase_checkpoint_is_reported(tmp_path: Path) -> None:
    """Red against `erasing()` discarding `checkpoint()`: "a blocked checkpoint was not reported" (None)."""
    store = _store(tmp_path, 6)
    with store._lock:  # noqa: SLF001
        store._conn.execute("PRAGMA busy_timeout = 100")  # noqa: SLF001  # the reader below never lets go
    reader = sqlite3.connect(store.path)
    try:
        reader.execute("BEGIN")
        reader.execute("SELECT count(*) FROM chunks").fetchone()
        with store.erasing():
            store.delete_sources([sorted({row.source for row in store.iter_chunks()})[0]])
        assert getattr(store, "erase_checkpoint_blocked", None) is True, "a blocked checkpoint was not reported"
        from recall_mcp.lifecycle import _with_checkpoint_note

        assert "write-ahead log" in _with_checkpoint_note(store, "Forgot 1 source.")
    finally:
        reader.close()
        store.close()


def test_k5_a_refused_open_closes_its_connection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Red against `__init__` without the close: "the refused open left its connection open"."""
    LiteStore(tmp_path / "memory.db", dim=DIM).close()
    opened: list[sqlite3.Connection] = []
    real = sqlite3.connect

    def spy(*args: object, **kwargs: object) -> sqlite3.Connection:
        conn = real(*args, **kwargs)  # type: ignore[arg-type]
        opened.append(conn)
        return conn

    monkeypatch.setattr(store_mod.sqlite3, "connect", spy)
    with pytest.raises(LiteStoreError):
        LiteStore(tmp_path / "memory.db", dim=DIM * 2)
    assert opened, "no connection was opened"
    try:
        opened[-1].execute("SELECT 1")
    except sqlite3.ProgrammingError:
        return
    pytest.fail("the refused open left its connection open")


@pytest.fixture
def _fresh_sqlite_check() -> object:
    check = getattr(store_mod, "_require_sqlite", None)
    if check is not None:
        check.cache_clear()
    yield
    if check is not None:
        check.cache_clear()


@pytest.mark.usefixtures("_fresh_sqlite_check")
def test_k6_an_old_sqlite_is_refused_before_the_file_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Red against `__init__` without `_require_sqlite()`: DID NOT RAISE, and the file appeared."""
    monkeypatch.setattr(store_mod.sqlite3, "sqlite_version_info", (3, 24, 0))
    with pytest.raises(LiteStoreError, match="SQLite 3.25"):
        LiteStore(tmp_path / "new" / "memory.db", dim=DIM)
    assert not (tmp_path / "new" / "memory.db").exists(), "a store was created on a SQLite too old for it"


def test_k7_the_fit_in_force_is_found_through_an_index(tmp_path: Path) -> None:
    """Red against the table without `calibrations_latest`: the plan was "SCAN calibrations" plus a temp B-tree."""
    store = LiteStore(tmp_path / "memory.db", dim=DIM)
    with store._lock:  # noqa: SLF001
        plan = " ".join(
            str(row[-1])
            for row in store._conn.execute(  # noqa: SLF001
                f"EXPLAIN QUERY PLAN SELECT payload FROM calibrations {store_mod.LATEST_CALIBRATION_ORDER} LIMIT 1"  # noqa: S608
            )
        )
    assert "calibrations_latest" in plan and "TEMP B-TREE" not in plan, plan


def test_k8_a_non_finite_lane_wait_means_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Red against `max(0.0, float(raw))`: "nan read as a 0 s wait" (0.0 == 300.0)."""
    from recall.lite.writer_lock import _wait_default

    for value in ("nan", "inf", ""):
        monkeypatch.setenv("RECALL_LITE_WRITE_WAIT", value)
        assert _wait_default("write") == 300.0, f"nan read as a 0 s wait ({value!r})"
    for value, expected in (("0", 0.0), ("-5", 0.0), ("12.5", 12.5)):  # controls: unchanged values
        monkeypatch.setenv("RECALL_LITE_WRITE_WAIT", value)
        assert _wait_default("write") == expected
    monkeypatch.setenv("RECALL_LITE_WRITE_WAIT", "2m")
    with pytest.raises(LiteStoreError):
        _wait_default("write")


def test_k9_the_queue_notice_gives_the_wait_in_seconds(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Red against the notice printing `WAIT_ENV[lane]`: "It gives up after RECALL_LITE_WRITE_WAIT seconds"."""
    from recall.lite.writer_lock import queued_notice

    monkeypatch.delenv("RECALL_LITE_WRITE_WAIT", raising=False)
    queued_notice("recall index")(2, None)
    err = capsys.readouterr().err
    assert "after 300 seconds" in err, err


def test_k10_doctor_and_dashboard_refuse_a_relative_lite_dsn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Red against the prefix slice in `_lite_checks`: doctor warned "no file yet" and named `recall index`."""
    from recall.dashboard.db import DatabaseUnavailable
    from recall.dashboard.lite_db import store_path
    from recall.doctor import _lite_checks

    monkeypatch.chdir(tmp_path)
    first = next(iter(_lite_checks("sqlite:///home/me/memory.db", tenant="default")))
    assert first.status == "fail" and "absolute" in first.detail, first
    with pytest.raises(DatabaseUnavailable, match="absolute"):
        store_path("sqlite:///home/me/memory.db")


def test_k11_another_stores_refresh_state_is_not_read(tmp_path: Path) -> None:
    """Red against `read_state` ignoring the recorded store: memory.sqlite read memory.db's state."""
    from recall.lite.refresh import STATE_SCHEMA, read_state, state_path

    first, second = tmp_path / "memory.db", tmp_path / "memory.sqlite"
    assert state_path(first) == state_path(second)
    state_path(first).write_text(json.dumps({"schema": STATE_SCHEMA, "store": str(first.resolve()), "last": {}}), encoding="utf-8")
    assert read_state(first), "a store's own state was not read"
    assert read_state(second) == {}, "another store's refresh state was read as this one's"


def test_k12_setup_names_the_state_file_when_the_store_is_ignored(tmp_path: Path) -> None:
    """Red against the state note nested in `if not listed`: "the refresh state file was not named"."""
    project = tmp_path / "project"
    project.mkdir()
    _write_memos(project / "notes", 30)
    (project / ".gitignore").write_text("data/memory.db*\n", encoding="utf-8")
    report = run_lite_setup(
        project_root=project, memory=project / "notes", embedder=ContentWordEmbedder(),
        embedder_name="content-words", store_path=project / "data" / "memory.db", register=False,
    )
    assert any("data/memory.refresh.json" in note for note in report.notes), "the refresh state file was not named"
    assert not any(note.startswith("add `data/memory.db*`") for note in report.notes), "an ignored store was named"


def test_k13_the_registered_server_keeps_setups_cache_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Red against `local_env` without the cache keys: "the server env lost RECALL_EMBED_CACHE"."""
    pytest.importorskip("mcp")
    project = tmp_path / "project"
    project.mkdir()
    _write_memos(project / "notes", 30)
    config = tmp_path / "claude.json"
    config.write_text(json.dumps({"projects": {}}), encoding="utf-8")
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    monkeypatch.setenv("RECALL_EMBED_CACHE_MAX_MB", "64")
    run_lite_setup(
        project_root=project, memory=project / "notes", embedder=ContentWordEmbedder(),
        embedder_name="content-words", config_path=config,
    )
    written = config.read_text(encoding="utf-8")
    assert '"RECALL_EMBED_CACHE": "off"' in written, "the server env lost RECALL_EMBED_CACHE"
    assert '"RECALL_EMBED_CACHE_MAX_MB": "64"' in written


def test_k14_a_memo_id_with_a_trailing_newline_names_no_file(tmp_path: Path) -> None:
    """Red against `MEMO_ID.match` in `path_of`: it returned a path for "note-0001\\n"."""
    from recall.lite.assistant_memos import MemoFolder

    folder = MemoFolder(tmp_path)
    assert folder.path_of("note-0001") is not None
    assert folder.path_of("note-0001\n") is None, "a memo id with a trailing newline named a file"


def test_k15_a_linked_authorship_ledger_is_refused(tmp_path: Path) -> None:
    """Red against `load_ledger` following the link: it read the target's authors (DID NOT RAISE)."""
    from recall.lite import memo_requests

    target = tmp_path / "elsewhere.json"
    target.write_text(json.dumps({"version": 1, "authors": {"note-0001": "assistant"}}), encoding="utf-8")
    ledger = tmp_path / "memos" / ".recall-authors.json"
    ledger.parent.mkdir()
    try:
        ledger.symlink_to(target)
    except OSError:
        pytest.skip("this platform cannot create a symbolic link here")
    with pytest.raises(ValueError, match="symbolic link"):
        memo_requests.load_ledger(ledger)


def test_k16_the_refresh_policy_judges_the_stored_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Red against the pre-filter judging names relative to the walked folder: every memo was denied (FAILED)."""
    from recall.lite.refresh import Outcome, refresh, refresh_access_context
    from recall.security_policy import load_source_policy

    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    project = tmp_path / "project"
    (project / "memory").mkdir(parents=True)
    _write_memos(project / "memory", 3)
    _write_memos(project / "memory" / "notes", 1)
    policy_file = tmp_path / "policy.json"
    policy_file.write_text(
        json.dumps({"rules": [{"prefix": "memory/notes", "principals": [], "purposes": ["retrieval"], "redactions": []}]}),
        encoding="utf-8",
    )
    policy = load_source_policy({"RECALL_SOURCE_POLICY_FILE": str(policy_file)})
    with LiteStore(project / ".recall" / "memory.db", dim=DIM) as store:
        report = refresh(
            store, [project / "memory"], ContentWordEmbedder(), key_root=project,
            security_policy=policy, security_context=refresh_access_context(store.tenant),
        )
        assert report.outcome is not Outcome.FAILED, f"{report.outcome}: {report.reason}"
        assert report.denied == 3, report.denied
        sources = {row.source for row in store.iter_chunks()}
    assert sources and all("notes" in Path(s).parts for s in sources), sorted(sources)


def test_k17_status_reports_the_carried_separability(tmp_path: Path) -> None:
    """Red against `status_report` copying only the delta counts: "no carried separability"."""
    store = _store(tmp_path, 48)
    assert cal.auto_calibrate(store, ContentWordEmbedder()).status is CalibrationStatus.CERTIFIED
    _write_memos(tmp_path / "m", 2, start=48)
    Indexer(store, ContentWordEmbedder(), env={}).index_path(tmp_path / "m")
    carried = cal.carry_forward(store)
    assert carried.artifact is not None, carried.reason
    report = cal.status_report(store)
    block = report.get("carried_forward")
    assert isinstance(block, dict)
    assert block.get("separability") == carried.artifact.separability, "no carried separability"


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="needs time.tzset to set a non-UTC local zone")
def test_k18_a_receipt_reads_a_naive_moment_as_utc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Red against `moment.astimezone(UTC)`: under TZ=Asia/Tokyo the naive midnight became 15:00 the day before."""
    from recall.lite.receipts import _stamp

    monkeypatch.setenv("TZ", "Asia/Tokyo")
    time.tzset()
    try:
        assert _stamp(datetime(2026, 1, 1, 0, 0)) == "2026-01-01T00:00:00.000000+00:00"
    finally:
        monkeypatch.delenv("TZ", raising=False)
        os.environ.pop("TZ", None)
        time.tzset()


def test_k19_learned_sparse_retrieval_is_unsupported(tmp_path: Path) -> None:
    """Red against `raise LiteStoreError(...)`: the call escaped `pytest.raises(LiteUnsupported)`."""
    store = LiteStore(tmp_path / "memory.db", dim=DIM)
    with pytest.raises(LiteUnsupported):
        store.query_learned_sparse("q", k=1)
    with pytest.raises(LiteUnsupported):
        store.sparse_covered_sources("p")
