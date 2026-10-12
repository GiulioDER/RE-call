"""`lite_lane`: one writer per lite file and lane, first come first served, across processes.

Invariants and the failure each one catches (multi-process ones use `tests/lite_lane_worker.py`
as a real separate process, so the OS lock, not Python state, is what is tested):
- W1 two processes in the write lane never hold it at the same time.
- W2 waiters acquire in arrival order.
- W3 a killed waiter is pruned; the next waiter does not sit out the dead one's timeout.
- W4 a killed holder releases the lane at once.
- W5 `wait_s=0` reports busy at once; `on_busy="skip"` yields None instead of raising.
- W6 coalesce returns at once only for a same-purpose WAITING ticket; a same-purpose holder does
  not count; another purpose queues.
- W7 nesting in one thread is reentrant; another thread in the same process must wait.
- W8 status reports holder and waiters and never touches a lane lock.
- W9 the write and refit lanes are independent; write inside refit in one thread is refused.
- P1 two spellings of one path share one lane; P2 an exception in the body releases the lane.

Red proof: see the end of this docstring (filled in when run).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from recall.lite import writer_lock as wl
from recall.lite.writer_lock import (
    LaneOrderError,
    LiteWriterBusy,
    LiteWriterCoalesced,
    lite_lane,
    lite_lane_status,
)


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "store" / "memory.db"


def _try(db: Path, *, purpose: str = "index", coalesce: bool = False) -> str:
    """One zero-wait attempt at the write lane, reported as a word."""
    try:
        with lite_lane(db, purpose=purpose, wait_s=0, coalesce=coalesce) as ticket:
            return "acquired" if ticket is not None else "skipped"
    except LiteWriterCoalesced:
        return "coalesced"
    except LiteWriterBusy:
        return "busy"


def _hold_in_thread(db: Path, *, lane: str = "write", purpose: str = "index") -> tuple[threading.Event, threading.Thread]:
    """Hold a lane in another thread until the returned event is set."""
    acquired, release = threading.Event(), threading.Event()

    def run() -> None:
        with lite_lane(db, lane=lane, purpose=purpose, wait_s=5):  # type: ignore[arg-type]
            acquired.set()
            release.wait(10)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert acquired.wait(5), "the holder thread never acquired the lane"
    return release, thread


def test_wait_zero_reports_busy_and_skip_yields_none(db: Path) -> None:
    """W5 (in-process half)."""
    release, thread = _hold_in_thread(db)
    try:
        with pytest.raises(LiteWriterBusy) as busy:
            with lite_lane(db, purpose="ingest", wait_s=0):
                pytest.fail("acquired a lane another thread holds")
        assert busy.value.holder and busy.value.holder["purpose"] == "index", busy.value.holder
        with lite_lane(db, purpose="ingest", wait_s=0, on_busy="skip") as ticket:
            assert ticket is None, "on_busy='skip' yielded a ticket for a held lane"
    finally:
        release.set()
        thread.join(5)


def test_nesting_is_reentrant_and_another_thread_waits(db: Path) -> None:
    """W7."""
    with lite_lane(db, purpose="index", wait_s=0) as outer:
        with lite_lane(db, purpose="index", wait_s=0) as inner:
            assert inner is outer, "nesting in one thread took a second ticket"
        seen: list[str] = []
        other = threading.Thread(target=lambda: seen.append(_try(db)), daemon=True)
        other.start()
        other.join(5)
        assert seen == ["busy"], f"another thread was not kept out: {seen}"
    assert _try(db) == "acquired", "the lane was not released after the outer block"


def test_lanes_are_independent_and_write_inside_refit_is_refused(db: Path) -> None:
    """W9."""
    release, thread = _hold_in_thread(db, lane="write")
    try:
        with lite_lane(db, lane="refit", purpose="calibrate", wait_s=0) as refit:
            assert refit is not None, "a held write lane blocked the refit lane"
            with pytest.raises(LaneOrderError):
                with lite_lane(db, lane="write", purpose="index", wait_s=0):
                    pass
    finally:
        release.set()
        thread.join(5)


def test_two_spellings_of_one_path_share_a_lane(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P1."""
    release, thread = _hold_in_thread(db)
    try:
        monkeypatch.chdir(db.parent)
        assert _try(Path("memory.db")) == "busy", "a relative spelling got its own lane"
        assert _try(Path(str(db).replace(os.sep, "/"))) == "busy", "a slash spelling got its own lane"
    finally:
        release.set()
        thread.join(5)


def test_an_exception_in_the_body_releases_the_lane(db: Path) -> None:
    """P2."""
    with pytest.raises(RuntimeError):
        with lite_lane(db, purpose="index", wait_s=0):
            raise RuntimeError("body failed")
    assert _try(db) == "acquired", "a failed body left the lane held"
    assert not list((wl.lanes_dir(db) / "write.queue").glob("*.ticket")), "a failed body left its ticket"


def test_status_touches_no_lane_lock(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """W8 (in-process half): status reads the holder and never tries a lane lock file."""
    release, thread = _hold_in_thread(db, purpose="refresh")
    try:
        tried: list[str] = []
        real = wl._try_lock

        def spy(fh: object) -> bool:
            tried.append(str(getattr(fh, "name", "")))
            return real(fh)  # type: ignore[arg-type]

        monkeypatch.setattr(wl, "_try_lock", spy)
        status = lite_lane_status(db)
        assert status.holder and status.holder["purpose"] == "refresh", status
        lane_locks = [name for name in tried if name.endswith(("write.lock", "refit.lock"))]
        assert not lane_locks, f"status tried a lane lock: {lane_locks}"
    finally:
        release.set()
        thread.join(5)
