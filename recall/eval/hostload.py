"""Is this host quiet enough to time anything on?

A latency benchmark measures the machine as much as the code. VPS2 was at load average 33.7 on
12 cores when this was written, with several unrelated python processes and four Postgres
backends on it. Every leg of a split measured there carries queueing delay, in an amount nobody
can attribute afterwards, and the artifact would read as a property of the store.

So the reading is taken, recorded, and checked. Recorded even when it passes: an undated,
unloaded latency artifact cannot be compared against itself.
"""

from __future__ import annotations

import os

#: Default ceiling, in load average per core. This is a JUDGEMENT CALL, not a measurement: it
#: leaves roughly seventy percent of cores free, which is where queueing stops being visible in
#: wall-clock in my experience of this box. It is exposed as a flag precisely because it is not
#: derived from anything, and a caller with a measurement should override it.
DEFAULT_MAX_LOAD_PER_CORE = 0.30


class HostTooBusyError(RuntimeError):
    """The host is under load that would be indistinguishable from the cost being measured."""


def read_load_per_core() -> float | None:
    """One minute load average divided by core count, or `None` where that is unknowable.

    `None` is a real answer, not a placeholder. `os.getloadavg` does not exist on Windows, and
    inventing a zero there would turn a guard that cannot fire into a guard that reports all
    clear. The caller records the `None` and the artifact carries JSON null.
    """
    try:
        one_minute = os.getloadavg()[0]
    except (OSError, AttributeError):  # no getloadavg (Windows), or /proc unreadable
        return None
    cores = os.cpu_count() or 1
    return one_minute / cores


def assert_host_quiet(
    ceiling: float = DEFAULT_MAX_LOAD_PER_CORE, *, allow_busy: bool = False
) -> float | None:
    """Return the load per core, refusing above `ceiling` unless `allow_busy`.

    The reading is returned in every case, including when it refuses to be the reason to stop and
    including when the override is set, because the caller stamps it into provenance either way.
    A run that was allowed to proceed on a busy host must still say how busy.
    """
    load = read_load_per_core()
    if load is None or allow_busy or load <= ceiling:
        return load
    raise HostTooBusyError(
        f"host load is {load:.2f} per core, above the {ceiling:.2f} ceiling. Every leg of a "
        f"latency split measured here would carry queueing delay that cannot be attributed "
        f"afterwards, so the artifact would describe the host rather than the store. Wait for "
        f"the box, or pass --allow-busy-host to publish a contended measurement as one."
    )
