"""`provider_stub` must shut its listener down even when a tracked client's close goes wrong.

`_stop` promises this in prose already: "Each step is guarded: a client whose ``close`` raises must
not be able to skip the server shutdown behind it, or one bad teardown leaks the listening socket
into the rest of the session." The close loop catches `Exception`, so the promise holds for the
ordinary case and fails for the one that costs most. A `KeyboardInterrupt` or `SystemExit` raised
while closing a client walks straight past that `except` and takes the shutdown with it, leaving
the listening socket and its accept loop alive for the remainder of the session, in a helper three
test modules share.

The distinction matters because of WHERE those two arrive. `KeyboardInterrupt` is delivered at an
arbitrary bytecode boundary in the main thread, so a developer pressing Ctrl-C during a slow
teardown lands inside the close loop as easily as anywhere else, and pytest raises `SystemExit`
through `--exitfirst` style paths. Neither is exotic, and neither is something a test author can
route around: the leak happens inside the helper.

The property is asserted through the SOCKET rather than through `_stop`'s return value. A leak
report is a string the helper chooses to emit; a listener that still accepts connections is the
defect itself, and it stays observable no matter how the reporting is later reworked.
"""
from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from tests.provider_stub_helpers import EMBEDDINGS_OK, provider_stub


class _CloseRaises:
    """A tracked client whose `close` raises what the close loop does not catch."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc
        self.closed = False

    def close(self) -> None:
        self.closed = True
        raise self._exc


@contextmanager
def _no_thread_may_die_on_an_exception() -> Iterator[list[Any]]:
    """Capture anything that escapes a thread, which is how a skipped `shutdown()` shows up.

    `server_close()` releases the socket; only `shutdown()` stops the accept loop. Deleting
    `shutdown()` does NOT leave a thread running, which is why probing for a surviving
    `serve_forever` thread caught nothing: the loop dies anyway, on an `OSError` from a selector
    whose socket was closed underneath it (WinError 10038 here). So the observable difference is
    not a thread that lives, it is a thread that dies BADLY, and pytest reports that only as a
    warning nobody fails on.

    Asserting on it directly is also the stronger property: an orderly teardown must not leave any
    thread unwinding through an exception, whatever the cause.
    """
    escaped: list[Any] = []
    previous = threading.excepthook

    def _record(args: Any) -> None:
        # Filtered HERE rather than after the fact, for two reasons. The hook is a process-wide
        # global, so for the duration of the window it sees EVERY thread in the process, and an
        # unfiltered list would let anything from any source land under a message blaming this
        # teardown. And anything not ours is handed on to the previous hook rather than swallowed,
        # so this capture cannot hide a failure another test depends on seeing.
        #
        # Filtering in the context manager's `finally` instead was tried and is WRONG: the body
        # reads the list before that `finally` runs, so it read an always-empty list and every
        # test passed vacuously.
        #
        # Matched on the thread NAME, not on `_target`. `Thread.run` deletes `_target` in its own
        # `finally`, which runs before `_bootstrap_inner` reaches the excepthook, so a `_target`
        # test is always False here and silently disables this whole detector. Python names a
        # thread after its target ("Thread-3 (serve_forever)"), and that survives.
        if "serve_forever" in (args.thread.name if args.thread else ""):
            escaped.append(args)
        else:
            previous(args)

    threading.excepthook = _record
    try:
        yield escaped
    finally:
        threading.excepthook = previous


def _assert_probe_can_see_a_live_listener(port: int) -> None:
    """Prove the oracle answers True for a listener that IS up, before its False is believed.

    Without this the whole file could pass on an oracle that always answers False: every probe
    would report "nothing is listening" and every assertion below would be satisfied by a broken
    instrument rather than by the fix. Called inside the block, on the same port, in the same
    test, so it cannot drift from the probe it vouches for.
    """
    assert _still_listening(port), (
        "the probe cannot see the stub's own live listener, so its later False proves nothing "
        "about the teardown"
    )


def _port_of(stub: Any) -> int:
    return int(stub.base_url.rsplit(":", 1)[1].split("/", 1)[0])


def _still_listening(port: int) -> bool:
    """True if anything still accepts on the port, which is the leak stated plainly.

    A refusal and a timeout both mean "nothing accepted", and on this platform a dead loopback
    port is NOT refused: measured here, every post-teardown probe consumed the full timeout rather
    than returning WSAECONNREFUSED. Narrowing to `ConnectionRefusedError` therefore does not
    harden the oracle, it breaks it.

    The real hazard the narrow version was reaching for is that "no answer" could mean "the probe
    failed" rather than "nothing is there", and no choice of exception settles that. What settles
    it is making the oracle prove it can see a LIVE listener, in the same test, on the same port,
    before the absence is believed: see `_assert_probe_can_see_a_live_listener` at each call site.
    Anything that is neither a refusal nor a timeout propagates rather than quietly deciding it.
    """
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2.0):
            return True
    except (ConnectionRefusedError, TimeoutError):
        return False


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit()], ids=["ctrl-c", "sysexit"])
def test_the_listener_is_shut_down_even_if_a_close_raises_a_base_exception(
    exc: BaseException,
) -> None:
    client = _CloseRaises(exc)
    port: int | None = None
    # The exception is re-raised, deliberately: swallowing a `KeyboardInterrupt` in a teardown is
    # its own defect, and the contract here is about the listener, not about suppression.
    with _no_thread_may_die_on_an_exception() as escaped:
        with pytest.raises(type(exc)):
            with provider_stub(EMBEDDINGS_OK) as stub:
                port = _port_of(stub)
                _assert_probe_can_see_a_live_listener(port)
                stub.track(client)
        # INSIDE the capture, deliberately. A thread dies on its own schedule, and settling after
        # the capture context had already restored `threading.excepthook` sent the very exception
        # this is looking for to the real hook instead: measured, it took the detector from
        # catching the mutation to missing it 3 times out of 3.
        time.sleep(0.3)
        died = [
            f"{e.thread.name if e.thread else '<unknown thread>'}: {e.exc_value!r}"
            for e in escaped
        ]

    assert client.closed, "the close was never attempted, so this proves nothing about the guard"
    assert port is not None
    assert not _still_listening(port), (
        "the listening socket outlived the block: a raising close skipped the server shutdown, "
        "which is exactly what `_stop`'s docstring says cannot happen"
    )
    # The socket and the loop are two properties, and `server_close()` delivers only the first.
    #
    # A thread dies on its own schedule, so this arm is a PROBABILISTIC detector where the two
    # above are deterministic: the settle window makes it reliable, not certain. It is kept
    # because nothing deterministic observes a missing `shutdown()` from outside the helper, and
    # a probabilistic guard that says so is worth more than a deterministic one that cannot see
    # the case at all.
    assert died == [], (
        "a thread unwound through an exception during teardown: `server_close()` released the "
        "socket while the accept loop was still polling it, which is what `server.shutdown()` "
        "exists to prevent"
    )


def test_a_close_raising_an_ordinary_exception_is_reported_not_swallowed() -> None:
    """The ordinary case, and the one whose CONTRACT changed: a raising close is now named.

    It used to pass silently. That left an operator reading "a stub handler thread outlived the
    test" about a connection nobody could close, five seconds of joins away from the real cause
    and pointing at the wrong subsystem. The listener still comes down either way; what is new is
    that the teardown names WHICH client refused to close, by index and type. Two clients raising
    the same exception used to produce two identical strings.

    `_stop` also puts close failures at the FRONT of the leak list, ahead of any thread report,
    and that ordering is NOT pinned here: with a single leak entry the list cannot express an
    order, and `pytest.raises(match=...)` searches rather than anchors. Pinning it needs a case
    carrying both a raising close and a genuinely leaked handler thread, which costs the join
    timeout on every run. Said plainly rather than left to be assumed from the docstring above it.
    """
    client = _CloseRaises(RuntimeError("connection reset by peer"))
    port: int | None = None

    with pytest.raises(AssertionError, match=r"closing tracked client 0 \(_CloseRaises\) raised"):
        with provider_stub(EMBEDDINGS_OK) as stub:
            port = _port_of(stub)
            _assert_probe_can_see_a_live_listener(port)
            stub.track(client)

    assert client.closed
    assert port is not None
    assert not _still_listening(port)


class _CloseRecords:
    """A tracked client whose close succeeds, so a later close can be shown to still happen."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_an_ordinary_close_failure_does_not_stop_the_remaining_closes() -> None:
    """The loop continues past a raising close, which nothing else in the suite observes.

    Every other call site tracks exactly one client per stub, so `stub.clients[:1]` is a literal
    no-op across the whole suite and the loop's second iteration was pinned by nothing.
    """
    first = _CloseRaises(RuntimeError("connection reset by peer"))
    second = _CloseRecords()

    with pytest.raises(AssertionError, match=r"closing tracked client 0 \(_CloseRaises\) raised"):
        with provider_stub(EMBEDDINGS_OK) as stub:
            stub.track(first)
            stub.track(second)

    assert first.closed
    assert second.closed, "a raising close swallowed the close behind it"


def test_a_recorded_problem_rides_out_on_an_escaping_base_exception() -> None:
    """`add_note` carries the close failures out when the teardown cannot return them normally.

    This is the shape the `except BaseException` limb exists for and the only one that reaches it:
    a close that RECORDS a problem, followed by one that escapes. Without it the note attachment
    is a behaviour this commit adds that no test can tell from its absence, because the tests that
    record a problem never escape, and the tests that escape never record one first.

    It also documents the residual `_stop` claims: the client behind the escaping one is NOT
    closed, because a BaseException skips the remaining closes.
    """
    recorded = _CloseRaises(RuntimeError("connection reset by peer"))
    escaping = _CloseRaises(KeyboardInterrupt())
    behind = _CloseRecords()

    with pytest.raises(KeyboardInterrupt) as caught:
        with provider_stub(EMBEDDINGS_OK) as stub:
            stub.track(recorded)
            stub.track(escaping)
            stub.track(behind)

    notes = getattr(caught.value, "__notes__", [])
    assert any("closing tracked client 0 (_CloseRaises) raised" in n for n in notes), (
        f"the recorded close failure was lost behind the escaping exception; notes={notes!r}"
    )
    assert not behind.closed, "the documented residual is wrong: closes behind the escape ran"
