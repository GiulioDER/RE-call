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
- P3 a ticket that cannot be opened for a reason other than being gone is kept and counted live.
- P4 a failure after the ticket is created releases it at once, even while the traceback is kept.
- W12 a released lane passes to the next waiter within a short poll (measured 2026-10-12: a poll
  backing off to 0.5 s left 40 writes of ~0.1 s each taking 16.7 s on Windows, ~12 s of it idle).

Red proof, 2026-10-12, each mutation alone, run on Linux, failing in the named assertion, then
restored byte for byte and green (W2 needed equal polling first: M2 survived the original test):
- M1 (W1) `_acquire`: `position == 1 and _try_lock(lane_fh)` replaced by `True` (no head check, no lock); `test_two_processes_never_hold_the_write_lane_together`: "AssertionError: two processes held the write lane at once: [(1791813168.3088408, 1791813168.71447), (179181316..."
- M2 (W2) `_acquire`: the head check dropped, any waiter tries the lane lock; `test_waiters_acquire_in_arrival_order`: "AssertionError: waiters were not served in arrival order: ['b', 'd', 'c', 'f', 'e']"
- M3 (W3) `_live_tickets`: every ticket counted live, so dead ones are never pruned; `test_a_killed_waiter_is_pruned_without_a_timeout`: "AssertionError: the next waiter never acquired: [{'tag': 'a', 'event': 'start', 't': 1791813278.4855647}, {'ta..."
- M4 (W4) `_acquire`: also requires no `<lane>.holder` file (trusting the file over the lock); `test_a_killed_holder_releases_the_lane`: "AssertionError: a dead holder kept the lane: [{'tag': 'a', 'event': 'start', 't': 1791813342.2527637}, {'tag':..."
- M5 (W5) `_acquire`: the deadline ignored (`budget + 3600`); `test_wait_zero_reports_busy_and_skip_yields_none`: "Failed: acquired a lane another thread holds"
- M6 (W6) coalesce: the holder no longer excluded; `test_coalesce_only_on_a_same_purpose_waiter`: "AssertionError: a same-purpose HOLDER coalesced a caller"
- M7 (W6) coalesce: the purpose comparison dropped; `test_coalesce_only_on_a_same_purpose_waiter`: "AssertionError: another purpose coalesced"
- M8 (W7) `lite_lane`: the reentrant branch disabled; `test_nesting_is_reentrant_and_another_thread_waits`: "AssertionError: nesting in one thread did not reuse its ticket: None"
- M9 (W8) `lite_lane_status`: tries the lane lock before reading; `test_status_touches_no_lane_lock`: "AssertionError: status tried a lane lock: ['/tmp/pytest-of-<user>/pytest-1859/test_status_touches_no_lane_lo0/..."
- M10 (W9) `_acquire`: every lane opens `write.lock`; `test_lanes_are_independent_and_write_inside_refit_is_refused`: "AssertionError: a held write lane blocked the refit lane"
- M11 (W9) `lite_lane`: the `LaneOrderError` check removed; `test_lanes_are_independent_and_write_inside_refit_is_refused`: "Failed: DID NOT RAISE LaneOrderError"
- M12 (P1) `lanes_dir`: `.resolve()` dropped; `test_two_spellings_of_one_path_share_a_lane`: "AssertionError: another spelling of the same file queued behind itself: None"
- M13 (P2) `lite_lane`: release moved out of `finally`; `test_an_exception_in_the_body_releases_the_lane`: "AssertionError: a failed body left its ticket"
After the final review, 2026-10-12, same method:
- P3 pre-fix `_ticket_alive` (any OSError on open meant dead): "an unopenable live ticket was not
  counted live: []".
- P4 pre-fix `_acquire` (ticket created outside the cleanup): "a failed acquire left a live ticket
  blocking the queue: [...000000000001-<pid>.ticket]".
- M1 re-proved after W1 moved to a start file (the blocker version failed in setup under M1, not in
  its assertion): "two processes held the write lane at once".
- M22 (W12) the 0.5 s poll restored (`POLL_HEAD_S = POLL_MAX_S = 0.5`): survived the chained-gap
  W12 (sequential spawns poll in phase order), red after the per-round offsets: "released lanes sat
  idle before the waiter took them: [0.204, 0.082, 0.462, 0.376, 0.278]".
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
        with lite_lane(db, purpose="index", wait_s=0, on_busy="skip") as inner:
            assert inner is outer, f"nesting in one thread did not reuse its ticket: {inner}"
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
        with lite_lane(db, lane="refit", purpose="calibrate", wait_s=0, on_busy="skip") as refit:
            assert refit is not None, "a held write lane blocked the refit lane"
    finally:
        release.set()
        thread.join(5)
    with lite_lane(db, lane="refit", purpose="calibrate", wait_s=0):
        with pytest.raises(LaneOrderError):
            with lite_lane(db, lane="write", purpose="index", wait_s=0):
                pass


def test_two_spellings_of_one_path_share_a_lane(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P1: a nested hold through another spelling of the same file is the same hold, not a
    second waiter queued behind itself (which would deadlock, or report busy with wait_s=0)."""
    db.parent.mkdir(parents=True)
    monkeypatch.chdir(db.parent)
    other = Path("..") / db.parent.name / "memory.db"
    with lite_lane(db, purpose="index", wait_s=0) as outer:
        with lite_lane(other, purpose="index", wait_s=0, on_busy="skip") as inner:
            assert inner is outer, f"another spelling of the same file queued behind itself: {inner}"


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
           hold_s: float = 0.0, wait_s: float = 30.0, coalesce: bool = False,
           release: Path | None = None, equal_poll: bool = False,
           start_file: Path | None = None) -> subprocess.Popen[bytes]:
    """A real separate process that takes the lane and logs to `log` (see lite_lane_worker)."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(_REPO), os.environ.get("PYTHONPATH", "")])}
    if equal_poll:
        env["LITE_LANE_EQUAL_POLL"] = "1"
    if start_file is not None:
        env["LITE_LANE_START_FILE"] = str(start_file)
    return subprocess.Popen(
        [sys.executable, str(_REPO / "tests" / "lite_lane_worker.py"), str(db), lane, purpose,
         str(hold_s), str(wait_s), str(log), tag, "1" if coalesce else "0", *([str(release)] if release else [])],
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
    """W1: three started processes ask for the lane at the same moment (a start file)."""
    log, go = tmp_path / "w1.log", tmp_path / "w1.go"
    procs = [_spawn(db, log, f"p{n}", hold_s=0.4, start_file=go) for n in range(3)]
    _wait_for(lambda: all(_at(log, f"p{n}", "start") is not None for n in range(3)))
    go.touch()
    for proc in procs:
        assert proc.wait(60) == 0
    spans = sorted((_at(log, f"p{i}", "acquired"), _at(log, f"p{i}", "released")) for i in range(3))  # type: ignore[type-var]
    assert all(a is not None and r is not None for a, r in spans), _events(log)
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert start >= end, f"two processes held the write lane at once: {spans}"  # type: ignore[operator]


def test_waiters_acquire_in_arrival_order(db: Path, tmp_path: Path) -> None:
    """W2: five waiters polling at one rate, so only the head check can keep arrival order
    (by chance, 1 in 120)."""
    log, release = tmp_path / "w2.log", tmp_path / "w2.release"
    holder = _spawn(db, log, "a", hold_s=60, release=release)
    _wait_for(lambda: _at(log, "a", "acquired") is not None)
    procs = [holder]
    tags = ["b", "c", "d", "e", "f"]
    for n, tag in enumerate(tags, start=1):
        procs.append(_spawn(db, log, tag, hold_s=0.1, equal_poll=True))
        _wait_for(lambda n=n: _waiting(db) == n)  # type: ignore[misc]
    release.touch()
    for proc in procs:
        assert proc.wait(60) == 0
    order = sorted(tags, key=lambda t: _at(log, t, "acquired") or 0.0)
    assert order == tags, f"waiters were not served in arrival order: {order}"


def test_a_killed_waiter_is_pruned_without_a_timeout(db: Path, tmp_path: Path) -> None:
    """W3."""
    log, release = tmp_path / "w3.log", tmp_path / "w3.release"
    holder = _spawn(db, log, "a", hold_s=60, release=release)
    _wait_for(lambda: _at(log, "a", "acquired") is not None)
    doomed = _spawn(db, log, "b", wait_s=60)
    _wait_for(lambda: _waiting(db) == 1)
    after = _spawn(db, log, "c", wait_s=10)
    _wait_for(lambda: _waiting(db) == 2)
    doomed.kill()
    doomed.wait(10)
    release.touch()
    assert holder.wait(30) == 0 and after.wait(30) == 0
    released, got = _at(log, "a", "released"), _at(log, "c", "acquired")
    assert released is not None and got is not None, f"the next waiter never acquired: {_events(log)}"
    assert got - released < 3.0, f"the next waiter sat behind a dead one for {got - released:.1f}s"


def test_a_killed_holder_releases_the_lane(db: Path, tmp_path: Path) -> None:
    """W4."""
    log = tmp_path / "w4.log"
    holder = _spawn(db, log, "a", hold_s=120)
    _wait_for(lambda: _at(log, "a", "acquired") is not None)
    waiter = _spawn(db, log, "b", wait_s=10)
    _wait_for(lambda: _waiting(db) == 1)
    killed_at = time.time()
    holder.kill()
    holder.wait(10)
    assert waiter.wait(30) == 0
    got = _at(log, "b", "acquired")
    assert got is not None and got - killed_at < 3.0, f"a dead holder kept the lane: {_events(log)}"


def test_coalesce_only_on_a_same_purpose_waiter(db: Path, tmp_path: Path) -> None:
    """W6."""
    log, release = tmp_path / "w6.log", tmp_path / "w6.release"
    holder = _spawn(db, log, "a", purpose="refresh", hold_s=60, release=release)
    _wait_for(lambda: _at(log, "a", "acquired") is not None)
    assert _try(db, purpose="refresh", coalesce=True) == "busy", "a same-purpose HOLDER coalesced a caller"
    waiter = _spawn(db, log, "b", purpose="refresh", wait_s=30)
    _wait_for(lambda: _waiting(db) == 1)
    assert _try(db, purpose="refresh", coalesce=True) == "coalesced", "a same-purpose waiter did not coalesce"
    assert _try(db, purpose="index", coalesce=True) == "busy", "another purpose coalesced"
    release.touch()
    assert holder.wait(30) == 0 and waiter.wait(30) == 0


def test_status_lists_holder_and_waiters(db: Path, tmp_path: Path) -> None:
    """W8 (multi-process half): purposes and order are readable while their files are locked."""
    log, release = tmp_path / "w8.log", tmp_path / "w8.release"
    holder = _spawn(db, log, "a", purpose="refresh", hold_s=60, release=release)
    _wait_for(lambda: _at(log, "a", "acquired") is not None)
    waiter = _spawn(db, log, "b", purpose="ingest", wait_s=30)
    _wait_for(lambda: _waiting(db) == 1)
    first, second = lite_lane_status(db), lite_lane_status(db)
    assert first.holder and first.holder["purpose"] == "refresh", first
    assert [w["purpose"] for w in first.waiters] == ["ingest"], first
    assert first.waiters == second.waiters, "reading the status changed the queue"
    release.touch()
    assert holder.wait(30) == 0 and waiter.wait(30) == 0


def test_the_lane_passes_to_the_next_waiter_promptly(db: Path, tmp_path: Path) -> None:
    """W12: five handoffs cost under 0.6 s in total. Each round releases at a different offset
    (0.0 to 0.4 s) into one waiter's polling, so a slow poll costs about its period whatever its
    phase (a 0.5 s poll sums to about 1.0 s here), while a 10 ms head poll sums to about 0.05 s."""
    gaps = []
    for i in range(5):
        log = tmp_path / f"w12-{i}.log"
        release, thread = _hold_in_thread(db, purpose="hold")
        waiter = _spawn(db, log, "b", wait_s=60)
        _wait_for(lambda: _waiting(db) == 1)
        time.sleep(2.0 + 0.1 * i)  # past any backoff ramp, then a different phase each round
        released_at = time.time()
        release.set()
        thread.join(5)
        assert waiter.wait(30) == 0
        got = _at(log, "b", "acquired")
        assert got is not None, _events(log)
        gaps.append(got - released_at)
    assert sum(gaps) < 0.6, f"released lanes sat idle before the waiter took them: {[round(g, 3) for g in gaps]}"


def test_a_ticket_that_cannot_be_opened_is_kept_live(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P3: only a ticket that is GONE is dead. Out of descriptors, or a sharing violation from a
    scanner, must not make a live waiter's ticket look dead and get it deleted from the queue."""
    release, thread = _hold_in_thread(db)
    try:
        root = wl.lanes_dir(db)
        held = sorted((root / "write.queue").glob("*.ticket"))
        assert len(held) == 1, held

        def refuse(path: object, *args: object, **kwargs: object) -> object:
            raise OSError(24, "Too many open files")

        monkeypatch.setattr(wl, "open", refuse, raising=False)
        live = wl._live_tickets(root, "write", prune=True)
        monkeypatch.undo()
        assert live == held, f"an unopenable live ticket was not counted live: {live}"
        assert held[0].exists(), "an unopenable live ticket was deleted from the queue"
    finally:
        release.set()
        thread.join(5)


def test_a_failure_after_the_ticket_is_made_releases_it(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P4: the lane lock file cannot be opened after the ticket exists. The ticket must be
    unlocked and gone before the exception leaves, not when the traceback is collected."""
    real_open = wl._open

    def no_lane_lock(path: Path) -> object:
        if path.name.endswith(".lock"):
            raise OSError(28, "No space left on device")
        return real_open(path)

    monkeypatch.setattr(wl, "_open", no_lane_lock)
    with pytest.raises(OSError) as failed:
        with lite_lane(db, purpose="index", wait_s=0):
            pytest.fail("acquired without a lane lock")
    monkeypatch.undo()
    assert failed.value is not None  # the traceback, and every frame in it, is still referenced here
    left = wl._live_tickets(wl.lanes_dir(db), "write", prune=False)
    assert not left, f"a failed acquire left a live ticket blocking the queue: {left}"
