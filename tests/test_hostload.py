"""The quiescence guard. A latency artifact from a contended host is not a measurement."""

from __future__ import annotations

import pytest

from recall.eval import hostload
from recall.eval.hostload import HostTooBusyError, assert_host_quiet, read_load_per_core


def test_a_busy_host_is_refused(monkeypatch) -> None:
    """Shown FIRING, not shown running.

    VPS2 sat at load 33.7 on 12 cores while this was being designed. That is 2.8 per core, and
    every leg of a latency split measured there would carry queueing delay nobody can attribute.
    """
    monkeypatch.setattr(hostload, "read_load_per_core", lambda: 2.81)

    with pytest.raises(HostTooBusyError, match="2.81"):
        assert_host_quiet(0.30, allow_busy=False)


def test_a_quiet_host_passes_and_returns_the_reading(monkeypatch) -> None:
    monkeypatch.setattr(hostload, "read_load_per_core", lambda: 0.11)

    assert assert_host_quiet(0.30, allow_busy=False) == 0.11


def test_allow_busy_overrides_but_still_returns_the_reading(monkeypatch) -> None:
    """The override does not hide the number. The artifact still gets stamped with it."""
    monkeypatch.setattr(hostload, "read_load_per_core", lambda: 2.81)

    assert assert_host_quiet(0.30, allow_busy=True) == 2.81


def test_an_unavailable_reading_does_not_refuse(monkeypatch) -> None:
    """`os.getloadavg` is Unix only. On Windows this guard CANNOT fire, and that is recorded
    rather than papered over: the field serialises as JSON null and the published artifact comes
    from Linux."""
    monkeypatch.setattr(hostload, "read_load_per_core", lambda: None)

    assert assert_host_quiet(0.30, allow_busy=False) is None


def test_read_load_per_core_is_none_or_a_positive_float() -> None:
    """Runs on whatever host the suite runs on, so it asserts the CONTRACT, not a value."""
    value = read_load_per_core()

    assert value is None or (isinstance(value, float) and value >= 0.0)
