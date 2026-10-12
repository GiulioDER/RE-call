"""One writer at a time per lite file: two lanes, each a first come, first served queue.

Several MCP servers (Claude Code's, Codex's, an agent's memory provider) open one lite file. SQLite
makes each transaction atomic (`LiteStore._write`), but a write OPERATION spans many: indexing reads
the stored hashes, embeds, writes and prunes, and two interleaved runs can replace rows the other
has just written. `lite_lane` makes such operations take turns.

Two lanes per file, independent of each other: `write` (index, ingest, forget, refresh, setup),
held for the whole operation, and `refit` (a calibration fit), so a fit of several seconds never
blocks a write. A write inside a refit in one thread is refused (lock order is write, then refit).

The design is the embedding host's queue (`embed_queue.sh`), ported to Windows, macOS and Linux
with the standard library. Beside the database, `<db>.lanes/` holds per lane:

- `<lane>.lock`  whoever holds an OS lock on it holds the lane;
- `<lane>.mutex` held for milliseconds around every queue edit, so nobody sees a half-made ticket;
- `<lane>.seq`   a counter that orders tickets;
- `<lane>.queue/<seq>-<pid>.ticket`  one per waiter and holder, OS-locked by its creator for its
  whole life, so a ticket is live exactly when its lock cannot be taken. The OS drops the lock
  when the process dies, however it dies, so no pid or start time is read and pid reuse is moot;
- `<lane>.holder` which ticket holds the lane, for `lite_lane_status`.

Only the oldest live ticket ever tries the lane lock, so arrival order is kept; dead tickets are
pruned by whoever edits the queue next. Every thread takes its own ticket: `flock` conflicts
between two descriptors of one process and `LockFileEx` between two handles, so threads queue in
the same file queue as processes. Nesting in one thread is reentrant.

On Windows a byte lock is mandatory, so locking byte 0 of a ticket would stop others reading it;
the lock sits at `_WIN_LOCK_OFFSET`, past any content. A lite file on a network filesystem is not
supported: file locks there are unreliable.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Literal

from recall.lite.store import LiteStoreError

__all__ = [
    "LaneOrderError",
    "LaneStatus",
    "LaneTicket",
    "LiteWriterBusy",
    "LiteWriterCoalesced",
    "lanes_dir",
    "lite_lane",
    "lite_lane_status",
    "queued_notice",
]

_log = logging.getLogger(__name__)

Lane = Literal["write", "refit"]
LANES: tuple[str, ...] = ("write", "refit")
DEFAULT_WAIT_S = {"write": 300.0, "refit": 120.0}
WAIT_ENV = {"write": "RECALL_LITE_WRITE_WAIT", "refit": "RECALL_LITE_REFIT_WAIT"}
# The head of the queue polls fast: it is the only one that can take the lane, and a slow poll
# there is idle time for every waiter behind it (measured 2026-10-12, a 0.5 s backoff left 40
# writes of ~0.1 s each taking 16.7 s on Windows, ~12 s of it idle). The others back off a little.
POLL_HEAD_S = 0.01
POLL_START_S = 0.02
POLL_MAX_S = 0.1
MUTEX_WAIT_S = 30.0
_WIN_LOCK_OFFSET = 1 << 30

if sys.platform == "win32":
    import msvcrt

    def _try_lock(fh: IO[bytes]) -> bool:
        fh.seek(_WIN_LOCK_OFFSET)
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _unlock(fh: IO[bytes]) -> None:
        fh.seek(_WIN_LOCK_OFFSET)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(fh: IO[bytes]) -> bool:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def _unlock(fh: IO[bytes]) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


class LiteWriterBusy(LiteStoreError):
    """The lane was not free within the wait. Names who holds it, so the caller can retry."""

    def __init__(self, lane: str, purpose: str, waited_s: float, position: int, holder: dict[str, object] | None):
        self.lane, self.purpose, self.waited_s, self.position, self.holder = lane, purpose, waited_s, position, holder
        held = (
            f"held by {holder.get('purpose')} (pid {holder.get('pid')} on {holder.get('host')}, "
            f"for {max(0.0, time.time() - float(holder.get('since', time.time()))):.0f}s)"  # type: ignore[arg-type]
            if holder
            else "held by a process that has not recorded itself yet"
        )
        super().__init__(
            f"the lite store's {lane} lane is busy: {held}; {purpose} waited {waited_s:.1f}s at "
            f"position {position}. Retry when it finishes."
        )


class LiteWriterCoalesced(LiteStoreError):
    """A waiter with the same purpose is already queued and will see this caller's changes."""

    def __init__(self, lane: str, purpose: str):
        self.lane, self.purpose = lane, purpose
        super().__init__(f"a {purpose} is already queued for the {lane} lane; this one is not needed")


class LaneOrderError(LiteStoreError):
    """A thread holding the refit lane asked for the write lane (the order is write, then refit)."""


@dataclass(frozen=True)
class LaneTicket:
    lane: str
    purpose: str
    pid: int
    host: str
    seq: int
    acquired_at: float
    waited_s: float


@dataclass
class LaneStatus:
    lane: str
    holder: dict[str, object] | None
    waiters: list[dict[str, object]] = field(default_factory=list)


@dataclass
class _Held:
    ticket: LaneTicket
    depth: int


_local = threading.local()


def _held() -> dict[tuple[str, str], _Held]:
    held: dict[tuple[str, str], _Held] | None = getattr(_local, "held", None)
    if held is None:
        held = {}
        _local.held = held
    return held


def lanes_dir(path: str | Path) -> Path:
    """The sidecar directory for a database file, the same for every spelling of its path."""
    resolved = Path(path).resolve()
    return resolved.with_name(resolved.name + ".lanes")


def _open(path: Path) -> IO[bytes]:
    # "a+b": create if missing, never truncate, and `.name` is the path (tests filter on it).
    # Append mode also puts a ticket's JSON at offset 0 on Windows after the lock's seek.
    return open(path, "a+b")  # noqa: SIM115  # the callers close it


def _remove(path: Path) -> None:
    # Windows refuses to delete a file another process has open (a status read, a prune probe);
    # the ticket is unlocked by then, so the next prune removes it.
    with suppress(FileNotFoundError, PermissionError):
        os.remove(path)


@contextmanager
def _mutex(root: Path, lane: str) -> Iterator[None]:
    fh = _open(root / f"{lane}.mutex")
    try:
        deadline = time.monotonic() + MUTEX_WAIT_S
        while not _try_lock(fh):
            if time.monotonic() > deadline:
                raise LiteStoreError(f"the {lane} queue mutex in {root} was not free for {MUTEX_WAIT_S:.0f}s")
            time.sleep(0.002)
        try:
            yield
        finally:
            _unlock(fh)
    finally:
        fh.close()


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _ticket_alive(path: Path) -> bool:
    """A ticket is live when its creator still holds its lock. Call under the mutex.

    Only a ticket that is GONE is dead. Any other failure to open it (out of descriptors, a
    scanner's sharing violation on Windows) counts it live for this round: calling it dead would
    delete a live waiter from the queue, and it would then wait out its whole budget.
    """
    try:
        fh = open(path, "r+b")  # noqa: SIM115  # closed below on every path
    except FileNotFoundError:
        return False
    except OSError:
        return True
    try:
        if _try_lock(fh):
            _unlock(fh)
            return False
        return True
    finally:
        fh.close()


def _live_tickets(root: Path, lane: str, *, prune: bool) -> list[Path]:
    """Live tickets, oldest first; with `prune`, dead ones are deleted. Call under the mutex."""
    queue = root / f"{lane}.queue"
    live = []
    for path in sorted(queue.glob("*.ticket")):
        if _ticket_alive(path):
            live.append(path)
        elif prune:
            _remove(path)
    return live


def _holder(root: Path, lane: str, live: list[Path]) -> dict[str, object] | None:
    record = _read_json(root / f"{lane}.holder")
    if record is None or str(record.get("ticket")) not in {p.name for p in live}:
        return None
    return record


def _next_seq(root: Path, lane: str) -> int:
    """The next ticket number. Call under the mutex.

    The counter file is rewritten in place, so a process killed mid-write leaves it empty. Every
    ticket still queued carries its number in its name, so the next number is taken past those
    too: a reset counter can then neither reuse a live ticket's name nor sort a newcomer ahead of
    a waiter already queued.
    """
    path = root / f"{lane}.seq"
    try:
        current = int(path.read_text(encoding="ascii").strip() or 0)
    except (OSError, ValueError):
        current = 0
    for ticket in (root / f"{lane}.queue").glob("*.ticket"):
        with suppress(ValueError):
            current = max(current, int(ticket.name.split("-", 1)[0]))
    path.write_text(str(current + 1), encoding="ascii")
    return current + 1


def _wait_default(lane: str) -> float:
    raw = os.environ.get(WAIT_ENV[lane])
    if raw is None:
        return DEFAULT_WAIT_S[lane]
    try:
        return max(0.0, float(raw))
    except ValueError:
        raise LiteStoreError(f"{WAIT_ENV[lane]} must be a number of seconds, got {raw!r}") from None


@contextmanager
def lite_lane(
    path: str | Path,
    *,
    lane: Lane = "write",
    purpose: str,
    wait_s: float | None = None,
    coalesce: bool = False,
    on_busy: Literal["raise", "skip"] = "raise",
    on_queued: Callable[[int, dict[str, object] | None], None] | None = None,
) -> Iterator[LaneTicket | None]:
    """Hold `lane` of the lite file at `path` for the block. See the module docstring.

    Yields a `LaneTicket`, or None when `on_busy="skip"` and the lane was not acquired (busy past
    `wait_s`, coalesced, or not attemptable: a malformed wait variable, an uncreatable lanes
    directory). With `on_busy="raise"` those raise `LiteWriterBusy`, `LiteWriterCoalesced`, or the
    underlying `LiteStoreError` / `OSError`. Exceptions from the block itself always propagate.
    `on_queued(position, holder)` is called once, when the first attempt finds the lane taken and
    the call is about to wait (see `queued_notice` for a command a person is watching). `wait_s=0` tries once; None takes the lane's default
    (`RECALL_LITE_WRITE_WAIT`, 300 s; `RECALL_LITE_REFIT_WAIT`, 120 s).
    """
    if lane not in LANES:
        raise ValueError(f"unknown lane {lane!r}; expected one of {LANES}")
    root = lanes_dir(path)
    key = (str(root), lane)
    held = _held()
    if key in held:
        held[key].depth += 1
        try:
            yield held[key].ticket
        finally:
            held[key].depth -= 1
        return
    if lane == "write" and (str(root), "refit") in held:
        raise LaneOrderError(
            f"this thread holds the refit lane of {root} and asked for the write lane; take write first"
        )
    try:
        budget = _wait_default(lane) if wait_s is None else max(0.0, wait_s)
        acquired = _acquire(root, lane, purpose, budget, coalesce, on_queued)
    except (LiteWriterBusy, LiteWriterCoalesced) as exc:
        if on_busy == "raise":
            raise
        _log.info("lite %s lane not taken for %s: %s", lane, purpose, exc)
        yield None
        return
    except (LiteStoreError, OSError) as exc:
        # The lane could not even be attempted (a malformed wait variable, an uncreatable lanes
        # directory, a mutex nobody releases). "skip" promises the caller no exception from the
        # lane itself, which is what lets a background fit run without ever raising into a write.
        if on_busy == "raise":
            raise
        _log.warning("lite %s lane could not be attempted for %s: %s", lane, purpose, exc)
        yield None
        return
    ticket, ticket_fh, lane_fh, ticket_path = acquired
    held[key] = _Held(ticket, 1)
    try:
        yield ticket
    finally:
        del held[key]
        _release(root, lane, ticket_fh, lane_fh, ticket_path)


def _acquire(
    root: Path,
    lane: str,
    purpose: str,
    budget: float,
    coalesce: bool,
    on_queued: Callable[[int, dict[str, object] | None], None] | None = None,
) -> tuple[LaneTicket, IO[bytes], IO[bytes], Path]:
    queue = root / f"{lane}.queue"
    queue.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    pid, host = os.getpid(), socket.gethostname()
    ticket_fh: IO[bytes] | None = None
    lane_fh: IO[bytes] | None = None
    ticket_path: Path | None = None
    # Everything from the ticket's creation on sits inside this try, so a failure at any step
    # (a full disk on the write, an unopenable lane lock) unlocks and removes the ticket before the
    # exception leaves, rather than when a retained traceback is finally collected.
    try:
        with _mutex(root, lane):
            live = _live_tickets(root, lane, prune=True)
            if coalesce:
                holder = _holder(root, lane, live)
                for path in live:
                    record = _read_json(path) or {}
                    if path.name != (holder or {}).get("ticket") and record.get("purpose") == purpose:
                        raise LiteWriterCoalesced(lane, purpose)
            seq = _next_seq(root, lane)
            made = ticket_path = queue / f"{seq:012d}-{pid}.ticket"
            mine = ticket_fh = _open(made)
            if not _try_lock(mine):  # a new file nobody else can know yet: cannot happen
                raise LiteStoreError(f"could not lock a new ticket {made}")
            record = {"seq": seq, "pid": pid, "host": host, "purpose": purpose, "lane": lane, "enqueued_at": time.time()}
            mine.write(json.dumps(record).encode("utf-8"))
            mine.flush()
        lock = lane_fh = _open(root / f"{lane}.lock")
        poll = POLL_START_S
        told = False
        while True:
            with _mutex(root, lane):
                live = _live_tickets(root, lane, prune=True)
                position = live.index(made) + 1 if made in live else len(live) + 1
                if position == 1 and _try_lock(lock):
                    now = time.time()
                    (root / f"{lane}.holder").write_text(
                        json.dumps({**record, "ticket": made.name, "since": now}), encoding="utf-8"
                    )
                    waited = time.monotonic() - started
                    return (LaneTicket(lane, purpose, pid, host, seq, now, waited), mine, lock, made)
                holder = _holder(root, lane, live)
            waited = time.monotonic() - started
            if waited >= budget:
                raise LiteWriterBusy(lane, purpose, waited, position, holder)
            if on_queued is not None and not told:
                told = True
                on_queued(position, holder)
            poll = POLL_HEAD_S if position == 1 else min(poll * 2, POLL_MAX_S)
            time.sleep(min(poll, max(0.0, budget - waited)))
    except BaseException:
        if ticket_fh is not None:
            _let_go(ticket_fh, lane_fh)
        elif lane_fh is not None:
            lane_fh.close()
        if ticket_path is not None:
            # A mutex that cannot be had must not replace the error being raised; an unlocked
            # ticket left behind is pruned by the next queue edit.
            with suppress(LiteStoreError), _mutex(root, lane):
                _remove(ticket_path)
        raise


def _let_go(ticket_fh: IO[bytes], lane_fh: IO[bytes] | None) -> None:
    """Unlock and close this holder's handles, whatever else has failed."""
    if lane_fh is not None:
        with suppress(OSError):
            _unlock(lane_fh)
        lane_fh.close()
    with suppress(OSError):
        _unlock(ticket_fh)
    ticket_fh.close()


def _release(root: Path, lane: str, ticket_fh: IO[bytes], lane_fh: IO[bytes], ticket_path: Path) -> None:
    try:
        with _mutex(root, lane):
            _remove(root / f"{lane}.holder")
            _let_go(ticket_fh, lane_fh)
            _remove(ticket_path)
    except BaseException:
        # No mutex (a suspended process holds it): free the lane anyway rather than when a
        # retained traceback is collected. The holder record and ticket file stay behind, but
        # both are ignored once the ticket's lock is gone, and the next prune removes the ticket.
        if not ticket_fh.closed:
            _let_go(ticket_fh, lane_fh)
        raise


def lite_lane_status(path: str | Path, lane: str = "write") -> LaneStatus:
    """Who holds `lane` and who waits, oldest first. Takes the queue mutex, never a lane lock."""
    if lane not in LANES:
        raise ValueError(f"unknown lane {lane!r}; expected one of {LANES}")
    root = lanes_dir(path)
    if not (root / f"{lane}.queue").is_dir():
        return LaneStatus(lane, None, [])
    with _mutex(root, lane):
        live = _live_tickets(root, lane, prune=False)
        holder = _holder(root, lane, live)
        waiters = []
        for ticket in live:
            if holder is not None and ticket.name == holder.get("ticket"):
                continue
            record = _read_json(ticket)
            if record is not None:
                waiters.append({k: record.get(k) for k in ("seq", "pid", "host", "purpose", "enqueued_at")})
    return LaneStatus(lane, holder, waiters)


def queued_notice(command: str, lane: str = "write") -> Callable[[int, dict[str, object] | None], None]:
    """An `on_queued` for a command a person is watching: one line on stderr, naming the holder.

    A queued write can wait minutes, and a silent terminal reads as a hang.
    """

    def say(position: int, holder: dict[str, object] | None) -> None:
        who = f"{holder.get('purpose')} (pid {holder.get('pid')})" if holder else "another writer"
        print(
            f"{command}: waiting for the lite store's {lane} lane, held by {who}; position {position}. "
            f"It gives up after {WAIT_ENV.get(lane, 'the wait')} seconds.",
            file=sys.stderr,
            flush=True,
        )

    return say
