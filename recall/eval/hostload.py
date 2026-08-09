"""Is this host quiet enough to time anything on?

A latency benchmark measures the machine as much as the code. VPS2 was at load average 33.7 on
12 cores when this was written, with several unrelated python processes and four Postgres
backends on it. Every leg of a split measured there carries queueing delay, in an amount nobody
can attribute afterwards, and the artifact would read as a property of the store.

So the reading is taken, recorded, and checked. Recorded even when it passes: an undated,
unloaded latency artifact cannot be compared against itself.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import cast

_log = logging.getLogger(__name__)

#: Default ceiling, in load average per core. This is a JUDGEMENT CALL, not a measurement: it
#: leaves roughly seventy percent of cores free, which is where queueing stops being visible in
#: wall-clock in my experience of this box. It is exposed as a flag precisely because it is not
#: derived from anything, and a caller with a measurement should override it.
#:
#: ⚠️ It is also UNREACHABLE on at least one real host, which is worth knowing before you treat a
#: refusal as a reason to wait. VPS2, the box this guard was written for, was sampled six times
#: over a minute on 2026-08-07 and read 12.85, 12.03, 12.39, 11.82, 12.41 and 15.94 on 12 cores.
#: Its FLOOR is about 1.0 per core, because it runs 64 services and 170 timers continuously. A
#: 0.30 ceiling there is not a bar the host clears when it is quiet, it is a bar the host never
#: clears, and a guard that can only refuse tells you about itself rather than about the run.
#:
#: So a caller measuring on a busy production host should pass a ceiling derived from THAT host's
#: floor and publish the readings, rather than waiting for a number that is not coming. What the
#: guard still buys in that case is the honest label: the artifact records what the load was, and
#: a figure taken at a host's floor may not be read as a quiet-machine measurement.
DEFAULT_MAX_LOAD_PER_CORE = 0.30


class HostTooBusyError(RuntimeError):
    """The host is under load that would be indistinguishable from the cost being measured."""

    def __init__(self, message: str, *, load: float) -> None:
        super().__init__(message)
        #: The exact reading that triggered the refusal, not the two decimal places in the
        #: message. A caller that wants the number back should read this, not parse the sentence.
        self.load = load


def read_load_per_core() -> float | None:
    """One minute load average divided by core count, or `None` where that is unknowable.

    `None` is a real answer, not a placeholder, but it is not returned for the same reason in
    both cases it can occur. `os.getloadavg` does not exist on Windows: that is `AttributeError`,
    a documented and accepted platform limit, and the guard genuinely cannot fire there. `OSError`
    is a `/proc` read failing on a platform that does support it, which on the host this guard
    exists to protect means a transient failure, not a missing capability, so it is logged as a
    warning: the caller still gets `None` back, but the log says the quiescence guard just went
    dark rather than that it was never available. The caller records the `None` either way and
    the artifact carries JSON null.
    """
    getloadavg = getattr(os, "getloadavg", None)
    if getloadavg is None:  # no getloadavg on this platform (Windows)
        return None
    try:
        one_minute = cast(Callable[[], tuple[float, float, float]], getloadavg)()[0]
    except OSError as exc:
        _log.warning(
            "could not read host load average (%s); the quiescence guard is not protecting "
            "this run",
            exc,
        )
        return None
    cores = os.cpu_count() or 1
    return one_minute / cores


def assert_host_quiet(
    ceiling: float = DEFAULT_MAX_LOAD_PER_CORE, *, allow_busy: bool = False
) -> float | None:
    """Return the load per core, refusing above `ceiling` unless `allow_busy`.

    The reading is returned on every path that returns: when it is `None`, when it passes under
    `ceiling`, and when `allow_busy` overrides a reading above `ceiling`, because the caller
    stamps it into provenance either way. A run that was allowed to proceed on a busy host must
    still say how busy. On the one path that does not return, the refusal, the exact reading is
    not lost: it is carried on the raised `HostTooBusyError` as `.load`, alongside the two decimal
    places already in the message.
    """
    load = read_load_per_core()
    if load is None or allow_busy or load <= ceiling:
        return load
    raise HostTooBusyError(
        f"host load is {load:.2f} per core, above the {ceiling:.2f} ceiling. Every leg of a "
        f"latency split measured here would carry queueing delay that cannot be attributed "
        f"afterwards, so the artifact would describe the host rather than the store. Wait for "
        f"the box, or pass --allow-busy-host to publish a contended measurement as one.",
        load=load,
    )
