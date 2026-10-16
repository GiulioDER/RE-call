"""An evaluation clock: a lite server that reads "now" from a file instead of the wall clock.

An evaluation that replays weeks of a person's life in minutes needs the memory to age with the
replay: a fact valid until Friday must read expired on the simulated Saturday, a memo written on the
simulated Monday must carry Monday as its write time, and a fact applied that day must be asserted
then. `RECALL_EVAL_CLOCK=<file>` on a lite server does that. The file holds one ISO 8601 time with
an offset; the driver rewrites it whenever its simulated time moves, and every call reads it again.

While it is installed, four defaults that read the wall clock read it instead, and nothing else
changes; a caller that passes its own time keeps it:

| default | where |
|---|---|
| the time a lite row is written (indexed_at, audit events) | `recall.lite.store._now` |
| `now` and `known_as_of` of a trusted search, the fact controller's fresh search included | `recall.trust.trusted_search` |
| the time a fact is applied and its cards are checked | `recall.provenance_controller.ProvenanceController` |
| `as_of` of the current facts | `recall_mcp.provenance.current_facts_memory` |

It fails closed: a missing, unreadable or offset-less time raises `EvalClockError` in the call that
needed it, rather than falling back to the wall clock and answering from the wrong day. It is
process-wide, which a lite server (one store, one client, stdio) is; `install` refuses a second
clock rather than replacing one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from recall.errors import RecallError

__all__ = ["EVAL_CLOCK_ENV", "EvalClockError", "file_clock", "from_env", "install", "installed", "now", "uninstall"]

EVAL_CLOCK_ENV = "RECALL_EVAL_CLOCK"

_source: Callable[[], datetime] | None = None


class EvalClockError(ValueError, RecallError):
    """The evaluation clock is configured but cannot say what time it is."""


def file_clock(path: str | Path) -> Callable[[], datetime]:
    """A clock that reads one ISO 8601 time with an offset from `path` on every call."""
    target = Path(path)

    def read() -> datetime:
        try:
            raw = target.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise EvalClockError(f"{EVAL_CLOCK_ENV} file {str(target)!r} cannot be read: {exc}") from exc
        try:
            when = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise EvalClockError(f"{EVAL_CLOCK_ENV} file {str(target)!r} holds {raw!r}, not an ISO time") from exc
        if when.tzinfo is None:
            raise EvalClockError(f"{EVAL_CLOCK_ENV} file {str(target)!r} holds {raw!r}, a time with no offset")
        return when.astimezone(UTC)

    return read


def install(source: Callable[[], datetime]) -> None:
    global _source
    if _source is not None:
        raise EvalClockError("an evaluation clock is already installed in this process")
    _source = source


def uninstall() -> None:
    global _source
    _source = None


def installed() -> bool:
    return _source is not None


def now() -> datetime | None:
    """The evaluation time, or None when no clock is installed (callers then use the wall clock)."""
    source = _source
    return None if source is None else source()


def from_env(env: Mapping[str, str], *, lite: bool) -> Callable[[], datetime] | None:
    """The clock `RECALL_EVAL_CLOCK` asks for, or None when it is unset. Lite stores only."""
    raw = env.get(EVAL_CLOCK_ENV, "").strip()
    if not raw:
        return None
    if not lite:
        raise EvalClockError(
            f"{EVAL_CLOCK_ENV} is served on a lite store only (a sqlite:/// DSN): a shared server "
            "must never answer from a simulated day"
        )
    clock = file_clock(raw)
    clock()  # refuse at startup, not on the first search
    return clock
