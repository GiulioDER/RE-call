"""`recall.lite.refresh` and `recall lite refresh`: the lite store follows its memo folder.

- R1 an unchanged folder short-circuits: no lane is taken, nothing is indexed.
- R2 an added memo, and an edit that keeps the file's size, are each refreshed.
- R3 a deleted memo is pruned and counted.
- R4 a refresh already queued for the store makes this one `coalesced`, at once.
- R5 a held lane with `wait_s=0` gives `busy`, naming the holder.
- R6 a cold start (below the calibration floor) is `refreshed`, with the calibration's own reason.
- R7 `check` writes nothing and tells current from changed.
- R8 a failed refresh keeps the last success.
- R11 the CLI's exit codes: 0 refreshed or unchanged, 3 busy; `--check` 0 current, 1 changed.
- R12 a memo folder that contains the store does not refresh forever on the store's own files.
- R13 a missing memo folder is `failed`, takes no lane and prunes nothing; R13b the same when it
  disappears while the refresh queues.
- R14 a corrupt state file reads as empty: one refresh, then unchanged.

Red proof, 2026-10-12, on Linux: each mutation alone, the named test failing in its own assertion,
then restored and green (12 passed):
- M1 (R1) `refresh`: the unchanged short-circuit disabled (`False and not force`): "an unchanged refresh took the write lane"
- M2 (R2) `snapshot`: size only, no `st_mtime_ns`: the same-size edit came back `unchanged`
- M3 (R3) `_refresh_locked`: `report.deleted += 0`: deleted 0, not 1
- M4 (R4) `refresh`: `coalesce=False`: `Outcome.BUSY ... refresh waited 2.0s`, not coalesced
- M5 (R5) `refresh`: the busy holder dropped: `AssertionError: None`
- M6 (R6) `_refresh_locked`: a `missing` calibration reported as `failed`: `Outcome.FAILED`
- M7 (R7) `check`: writes the state file back: "check wrote something"
- M8 (R8) `_refresh_locked`: a failure recorded as the last success: "a failed refresh replaced the last success"
- M11 (R11) `lite_cmd`: `raise SystemExit(EXIT_BUSY)` made `pass`: exit 0, not 3
- M12 (R12) `recall.index._DEFAULT_INDEX_EXCLUDED_SUFFIXES` without `.json`: "the store's own files moved the snapshot"
- M13 (R13) `refresh`: the missing-folder check disabled: "a refresh of a missing folder took the write lane"
  (it survived the first R13, which only asserted the outcome; the in-lane check of R13b caught it)
- M13b (R13b) `_refresh_locked`: the in-lane missing-folder check disabled: outcome `refreshed` over 0 files
- M14 (R14) `read_state`: `ValueError` not caught: "raised JSONDecodeError"
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

import recall.lite.refresh as refresh_mod
from recall.cli import main as cli_main
from recall.lite import LiteStore
from recall.lite.refresh import Outcome, check, read_state, refresh, state_path
from recall.lite.writer_lock import lite_lane_status
from tests.test_lite_calibration import DIM, ContentWordEmbedder, _write_memos
from tests.test_lite_writer_lock import _at, _spawn, _wait_for


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    _write_memos(tmp_path / "memory", 3)
    return tmp_path


@pytest.fixture
def store(project: Path) -> Iterator[LiteStore]:
    with LiteStore(project / ".recall" / "memory.db", dim=DIM) as opened:
        yield opened


def _run(store: LiteStore, project: Path, **kwargs: object) -> refresh_mod.RefreshReport:
    return refresh(store, [project / "memory"], ContentWordEmbedder(), **kwargs)  # type: ignore[arg-type]


def _holder(db: Path, tmp_path: Path, tag: str = "holder", purpose: str = "index") -> tuple[subprocess.Popen[bytes], Path]:
    """Another process holding the write lane until the returned release file exists."""
    log, release = tmp_path / f"{tag}.log", tmp_path / f"{tag}.release"
    proc = _spawn(db, log, tag, purpose=purpose, hold_s=60, release=release)
    _wait_for(lambda: _at(log, tag, "acquired") is not None)
    return proc, release


def _files(root: Path) -> dict[str, tuple[int, int]]:
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


def test_an_unchanged_folder_takes_no_lane(store: LiteStore, project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R1."""
    first = _run(store, project)
    assert first.outcome is Outcome.REFRESHED, first.reason
    chunks = store.count()

    def no_lane(*args: object, **kwargs: object) -> None:
        raise AssertionError("an unchanged refresh took the write lane")

    monkeypatch.setattr(refresh_mod, "lite_lane", no_lane)
    second = _run(store, project)
    assert second.outcome is Outcome.UNCHANGED, f"{second.outcome}: {second.reason}"
    assert store.count() == chunks


def test_an_added_memo_and_a_same_size_edit_are_refreshed(store: LiteStore, project: Path) -> None:
    """R2."""
    _run(store, project)
    _write_memos(project / "memory", 1, start=3)
    added = _run(store, project)
    assert (added.outcome, added.indexed_files) == (Outcome.REFRESHED, 1), added.to_dict()
    assert any("memo-003.md" in c.source for c in store.iter_chunks())

    memo = project / "memory" / "memo-000.md"
    text = memo.read_text(encoding="utf-8")
    edited = text.replace("spring", "autumn")  # same length: only the mtime can tell
    assert len(edited) == len(text) and edited != text
    before = memo.stat().st_mtime_ns
    memo.write_text(edited, encoding="utf-8")
    os.utime(memo, ns=(before + 1_000_000, before + 1_000_000))
    after = _run(store, project)
    assert (after.outcome, after.indexed_files) == (Outcome.REFRESHED, 1), after.to_dict()


def test_a_deleted_memo_is_pruned(store: LiteStore, project: Path) -> None:
    """R3."""
    _run(store, project)
    (project / "memory" / "memo-001.md").unlink()
    report = _run(store, project)
    assert (report.outcome, report.deleted) == (Outcome.REFRESHED, 1), report.to_dict()
    assert not any("memo-001.md" in c.source for c in store.iter_chunks())


def test_a_queued_refresh_coalesces_this_one(store: LiteStore, project: Path, tmp_path: Path) -> None:
    """R4: a real holder, a real queued refresh in another process, then this one."""
    db = Path(store.path)
    holder, release = _holder(db, tmp_path)
    queued = _spawn(db, tmp_path / "q.log", "queued", purpose="refresh", wait_s=60)
    try:
        _wait_for(lambda: any(w.get("purpose") == "refresh" for w in lite_lane_status(db).waiters))
        started = time.monotonic()
        report = _run(store, project, wait_s=2.0)
        assert report.outcome is Outcome.COALESCED, f"{report.outcome}: {report.reason}"
        assert time.monotonic() - started < 1.5, "a coalesced refresh waited for the lane"
        assert read_state(db)["last"]["outcome"] == "coalesced"
        assert read_state(db).get("last_success") is None
    finally:
        release.touch()
        holder.wait(30)
        queued.wait(30)


def test_a_held_lane_gives_busy_and_names_the_holder(store: LiteStore, project: Path, tmp_path: Path) -> None:
    """R5."""
    holder, release = _holder(Path(store.path), tmp_path)
    try:
        report = _run(store, project, wait_s=0)
    finally:
        release.touch()
        holder.wait(30)
    assert report.outcome is Outcome.BUSY, f"{report.outcome}: {report.reason}"
    assert report.holder is not None and report.holder.get("purpose") == "index", report.holder


def test_a_cold_start_is_refreshed_with_the_calibration_reason(store: LiteStore, project: Path) -> None:
    """R6: three memos are below the calibration floor."""
    report = _run(store, project)
    assert report.outcome is Outcome.REFRESHED, f"{report.outcome}: {report.reason}"
    assert report.calibration == "missing"
    assert report.calibration_reason.startswith("Trusted answers turn on once memory holds enough text"), report.calibration_reason
    assert report.calibration_reason in report.render()
    assert read_state(store.path)["last"]["calibration_reason"] == report.calibration_reason


def test_check_writes_nothing_and_tells_current_from_changed(store: LiteStore, project: Path) -> None:
    """R7."""
    db, roots = Path(store.path), [project / "memory"]
    _run(store, project)
    before = _files(project)
    current = check(db, roots)
    assert _files(project) == before, "check wrote something"
    assert current.current, current.render()
    _write_memos(project / "memory", 1, start=3)
    before = _files(project)
    changed = check(db, roots)
    assert _files(project) == before, "check wrote something"
    assert not changed.current, changed.render()
    assert "changed since the last refresh" in changed.render()
    assert "Trusted answers turn on" in changed.render(), "the cold start reason was not reported"


def test_a_failure_keeps_the_last_success(store: LiteStore, project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R8."""
    _run(store, project)
    success = read_state(store.path)["last_success"]
    _write_memos(project / "memory", 1, start=3)

    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("the embedder is down")

    monkeypatch.setattr("recall.index.Indexer.index_path", broken)
    report = _run(store, project)
    assert report.outcome is Outcome.FAILED and "the embedder is down" in report.reason
    state = read_state(store.path)
    assert state["last"]["outcome"] == "failed"
    assert state["last_success"] == success, "a failed refresh replaced the last success"


def test_the_cli_exit_codes(project: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """R11, with the hashing embedder the CLI can resolve by name."""
    db = project / ".recall" / "memory.db"
    LiteStore(db, dim=64).close()
    base = ["lite", "refresh", "--store", str(db), "--memory", str(project / "memory"), "--embedder", "hashing"]

    def code(*extra: str) -> int:
        try:
            cli_main([*base, *extra])
        except SystemExit as exc:
            return int(exc.code or 0) if not isinstance(exc.code, str) else -1
        return 0

    assert code() == 0, capsys.readouterr()
    assert code() == 0 and "unchanged" in capsys.readouterr().out
    assert code("--check") == 0
    _write_memos(project / "memory", 1, start=3)
    assert code("--check") == 1
    holder, release = _holder(db, tmp_path)
    try:
        assert code("--wait", "0") == 3, capsys.readouterr()
    finally:
        release.touch()
        holder.wait(30)


def test_a_folder_holding_the_store_does_not_refresh_forever(project: Path) -> None:
    """R12: the store, its WAL, its lanes and the state file all live inside the memo folder."""
    root = project / "memory"
    with LiteStore(root / "memory.db", dim=DIM) as inside:
        first = refresh(inside, [root], ContentWordEmbedder())  # type: ignore[arg-type]
        assert first.outcome is Outcome.REFRESHED, first.reason
        assert state_path(inside.path).exists()
        second = refresh(inside, [root], ContentWordEmbedder())  # type: ignore[arg-type]
        assert second.outcome is Outcome.UNCHANGED, f"the store's own files moved the snapshot: {second.to_dict()}"


def test_a_missing_folder_fails_and_prunes_nothing(
    store: LiteStore, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R13."""
    _run(store, project)
    chunks = store.count()
    (project / "memory").rename(project / "memory-moved")

    def no_lane(*args: object, **kwargs: object) -> None:
        raise AssertionError("a refresh of a missing folder took the write lane")

    monkeypatch.setattr(refresh_mod, "lite_lane", no_lane)
    report = _run(store, project)
    assert report.outcome is Outcome.FAILED and "memory folder not found" in report.reason, report.to_dict()
    assert store.count() == chunks


def test_a_corrupt_state_file_costs_one_refresh(store: LiteStore, project: Path) -> None:
    """R14."""
    _run(store, project)
    state_path(store.path).write_text("{not json", encoding="utf-8")
    try:
        outcome: object = _run(store, project).outcome
    except ValueError as exc:  # asserted below, so a crash on the file reads as this test's failure
        outcome = f"raised {type(exc).__name__}: {exc}"
    assert outcome is Outcome.REFRESHED, outcome
    assert _run(store, project).outcome is Outcome.UNCHANGED
    assert json.loads(state_path(store.path).read_text(encoding="utf-8"))["schema"] == 1


def test_a_folder_gone_while_queued_fails_and_prunes_nothing(
    store: LiteStore, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R13b: the folder is there for the first snapshot and gone by the one taken inside the lane."""
    _run(store, project)
    chunks = store.count()
    _write_memos(project / "memory", 1, start=3)  # so the first snapshot sees a change
    real = refresh_mod.snapshot
    calls: list[int] = []

    def moving(roots: object) -> refresh_mod.Snapshot:
        calls.append(1)
        if len(calls) == 2:
            (project / "memory").rename(project / "memory-moved")
        return real(roots)  # type: ignore[arg-type]

    monkeypatch.setattr(refresh_mod, "snapshot", moving)
    report = _run(store, project)
    assert report.outcome is Outcome.FAILED and "memory folder not found" in report.reason, report.to_dict()
    assert store.count() == chunks
