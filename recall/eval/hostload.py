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

_log = logging.getLogger(__name__)

#: Default ceiling, in load average per core. This is a JUDGEMENT CALL, not a measurement: it
#: leaves roughly seventy percent of cores free, which is where queueing stops being visible in
#: wall-clock in my experience of this box. It is exposed as a flag precisely because it is not
#: derived from anything, and a caller with a measurement should override it.
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
    try:
        one_minute = os.getloadavg()[0]
    except AttributeError:  # no getloadavg on this platform (Windows)
        return None
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
