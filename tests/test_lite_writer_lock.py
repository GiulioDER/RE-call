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

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
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


_REPO = Path(__file__).resolve().parents[1]


def _spawn(db: Path, log: Path, tag: str, *, lane: str = "write", purpose: str = "index",
           hold_s: float = 0.0, wait_s: float = 30.0, coalesce: bool = False) -> subprocess.Popen[bytes]:
    """A real separate process that takes the lane and logs to `log` (see lite_lane_worker)."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(_REPO), os.environ.get("PYTHONPATH", "")])}
    return subprocess.Popen(
        [sys.executable, str(_REPO / "tests" / "lite_lane_worker.py"), str(db), lane, purpose,
         str(hold_s), str(wait_s), str(log), tag, "1" if coalesce else "0"],
        env=env,
    )


def _events(log: Path) -> list[dict[str, object]]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]


def _wait_for(predicate: Callable[[], bool], timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition not reached within the timeout")


def _at(log: Path, tag: str, event: str) -> float | None:
    times = [float(e["t"]) for e in _events(log) if e["tag"] == tag and e["event"] == event]  # type: ignore[arg-type]
    return times[0] if times else None


def _waiting(db: Path) -> int:
    return len(lite_lane_status(db).waiters)


def test_two_processes_never_hold_the_write_lane_together(db: Path, tmp_path: Path) -> None:
    """W1."""
    log = tmp_path / "w1.log"
    procs = [_spawn(db, log, f"p{i}", hold_s=0.4) for i in range(3)]
    for proc in procs:
        assert proc.wait(60) == 0
    spans = sorted((_at(log, f"p{i}", "acquired"), _at(log, f"p{i}", "released")) for i in range(3))  # type: ignore[type-var]
    assert all(a is not None and r is not None for a, r in spans), _events(log)
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert start >= end, f"two processes held the write lane at once: {spans}"  # type: ignore[operator]


def test_waiters_acquire_in_arrival_order(db: Path, tmp_path: Path) -> None:
    """W2."""
    log = tmp_path / "w2.log"
    holder = _spawn(db, log, "a", hold_s=1.5)
    _wait_for(lambda: _at(log, "a", "acquired") is not None)
    procs = [holder]
    for n, tag in enumerate(["b", "c", "d"], start=1):
        procs.append(_spawn(db, log, tag, hold_s=0.1))
        _wait_for(lambda n=n: _waiting(db) == n)  # type: ignore[misc]
    for proc in procs:
        assert proc.wait(60) == 0
    order = sorted(["b", "c", "d"], key=lambda t: _at(log, t, "acquired") or 0.0)
    assert order == ["b", "c", "d"], f"waiters were not served in arrival order: {order}"


def test_a_killed_waiter_is_pruned_without_a_timeout(db: Path, tmp_path: Path) -> None:
    """W3."""
    log = tmp_path / "w3.log"
    holder = _spawn(db, log, "a", hold_s=1.5)
    _wait_for(lambda: _at(log, "a", "acquired") is not None)
    doomed = _spawn(db, log, "b", wait_s=60)
    _wait_for(lambda: _waiting(db) == 1)
    after = _spawn(db, log, "c", wait_s=60)
    _wait_for(lambda: _waiting(db) == 2)
    doomed.kill()
    doomed.wait(10)
    assert holder.wait(30) == 0 and after.wait(30) == 0
    released, got = _at(log, "a", "released"), _at(log, "c", "acquired")
    assert released is not None and got is not None, _events(log)
    assert got - released < 3.0, f"the next waiter sat behind a dead one for {got - released:.1f}s"


def test_a_killed_holder_releases_the_lane(db: Path, tmp_path: Path) -> None:
    """W4."""
    log = tmp_path / "w4.log"
    holder = _spawn(db, log, "a", hold_s=120)
    _wait_for(lambda: _at(log, "a", "acquired") is not None)
    waiter = _spawn(db, log, "b", wait_s=60)
    _wait_for(lambda: _waiting(db) == 1)
    killed_at = time.time()
    holder.kill()
    holder.wait(10)
    assert waiter.wait(30) == 0
    got = _at(log, "b", "acquired")
    assert got is not None and got - killed_at < 3.0, f"a dead holder kept the lane: {_events(log)}"


def test_coalesce_only_on_a_same_purpose_waiter(db: Path, tmp_path: Path) -> None:
    """W6."""
    log = tmp_path / "w6.log"
    holder = _spawn(db, log, "a", purpose="refresh", hold_s=3)
    _wait_for(lambda: _at(log, "a", "acquired") is not None)
    assert _try(db, purpose="refresh", coalesce=True) == "busy", "a same-purpose HOLDER coalesced a caller"
    waiter = _spawn(db, log, "b", purpose="refresh", wait_s=30)
    _wait_for(lambda: _waiting(db) == 1)
    assert _try(db, purpose="refresh", coalesce=True) == "coalesced", "a same-purpose waiter did not coalesce"
    assert _try(db, purpose="index", coalesce=True) == "busy", "another purpose coalesced"
    assert holder.wait(30) == 0 and waiter.wait(30) == 0


def test_status_lists_holder_and_waiters(db: Path, tmp_path: Path) -> None:
    """W8 (multi-process half): purposes and order are readable while their files are locked."""
    log = tmp_path / "w8.log"
    holder = _spawn(db, log, "a", purpose="refresh", hold_s=2)
    _wait_for(lambda: _at(log, "a", "acquired") is not None)
    waiter = _spawn(db, log, "b", purpose="ingest", wait_s=30)
    _wait_for(lambda: _waiting(db) == 1)
    first, second = lite_lane_status(db), lite_lane_status(db)
    assert first.holder and first.holder["purpose"] == "refresh", first
    assert [w["purpose"] for w in first.waiters] == ["ingest"], first
    assert first.waiters == second.waiters, "reading the status changed the queue"
    assert holder.wait(30) == 0 and waiter.wait(30) == 0
