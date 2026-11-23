"""SEC-001: framing is checked before stream consumption or application dispatch.

Red proof: these invalid-framing nodes fail their intended assertions against
970dbeb7b. The valid-framing control is red-proved by a production cap mutation.
The real handler is obtained through serve; transport and listener are fakes.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from threading import BoundedSemaphore

import pytest

from recall.dashboard import server


def _request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, length: str, payload: bytes = b""):
    app = server.DashboardApp(tmp_path, port=8765, token="proof")
    dispatched: list[bytes] = []
    monkeypatch.setattr(app, "handle", lambda method, path, headers, body: dispatched.append(body) or server.Response(200, b"ok", ()))
    monkeypatch.setattr(server, "_ExclusiveServer", lambda address, handler: handler)
    handler_type = server.serve(app)
    handler = object.__new__(handler_type)
    handler.headers = {"Content-Length": length}
    handler.path = "/"
    handler.command = "POST"
    handler.request_version = "HTTP/1.1"
    handler.wfile = BytesIO()
    responses: list[int] = []
    consumed: list[int] = []
    handler.send_response = lambda status, *args: responses.append(status)
    handler.send_header = lambda *args: None
    handler.end_headers = lambda: None

    class BoundedInput:
        def read(self, size: int) -> bytes:
            assert size >= 0, "SEC-001: handler requested an unbounded read"
            consumed.append(size)
            return payload[:size]

    handler.rfile = BoundedInput()
    error = None
    try:
        handler._respond("POST")
    except ValueError as exc:
        error = exc
    assert error is None, "SEC-001: malformed framing escaped as an exception"
    return responses, consumed, dispatched


@pytest.mark.parametrize("length,status", [("-1", 400), ("invalid", 400), ("1000001", 413)])
def test_invalid_framing_is_refused_before_reading_or_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, length: str, status: int,
) -> None:
    responses, consumed, dispatched = _request(tmp_path, monkeypatch, length)
    assert responses == [status], "SEC-001: invalid framing was accepted"
    assert consumed == [], "SEC-001: invalid framing consumed the input stream"
    assert dispatched == [], "SEC-001: invalid framing reached the application"


@pytest.mark.parametrize("payload", [b"", b"csrf=proof"])
def test_valid_framing_preserves_the_exact_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: bytes,
) -> None:
    """Compatibility control: a cap mutation to -1 must fail this assertion."""
    responses, _consumed, dispatched = _request(tmp_path, monkeypatch, str(len(payload)), payload)
    assert responses == [200] and dispatched == [payload]


def test_dashboard_declines_a_request_when_worker_slots_are_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = object.__new__(server._ExclusiveServer)
    listener._request_slots = BoundedSemaphore(1)
    assert listener._request_slots.acquire(blocking=False)
    dispatched = []
    closed = []
    request = object()
    monkeypatch.setattr(
        server.ThreadingHTTPServer, "process_request",
        lambda self, request, address: dispatched.append(request),
    )
    monkeypatch.setattr(listener, "shutdown_request", closed.append)

    listener.process_request(request, ("127.0.0.1", 12345))

    assert dispatched == []
    assert closed == [request]


def test_dashboard_closes_a_request_after_the_read_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = server.DashboardApp(tmp_path, port=8765, token="proof")
    monkeypatch.setattr(server, "_ExclusiveServer", lambda address, handler: handler)
    handler = object.__new__(server.serve(app))
    configured_timeouts = []

    class TimedInput:
        def readline(self, limit: int) -> bytes:
            raise TimeoutError("fixture read exceeded the timeout")

    class Connection:
        def settimeout(self, seconds: float) -> None:
            configured_timeouts.append(seconds)

        def makefile(self, mode: str, buffer_size: int) -> TimedInput:
            assert mode == "rb"
            return TimedInput()

    handler.request = Connection()
    handler.close_connection = False
    handler.log_error = lambda *args: None
    handler.setup()
    handler.handle_one_request()

    assert configured_timeouts == [10]
    assert handler.close_connection is True


def test_dashboard_enforces_the_whole_request_read_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = server.DashboardApp(tmp_path, port=8765, token="proof")
    monkeypatch.setattr(server, "_ExclusiveServer", lambda address, handler: handler)
    handler = object.__new__(server.serve(app))
    scheduled = []
    shutdown = []

    class Deadline:
        def __init__(self, seconds: float, expire) -> None:
            assert seconds == 10
            self.expire = expire
            self.cancelled = False

        def start(self) -> None:
            scheduled.append(self)

        def cancel(self) -> None:
            self.cancelled = True

    class Input:
        def readline(self, limit: int) -> bytes:
            assert len(scheduled) == 1
            scheduled[0].expire()
            return b""

    class Connection:
        def shutdown(self, mode: int) -> None:
            shutdown.append(mode)

    monkeypatch.setattr(server.threading, "Timer", Deadline)
    handler.rfile = Input()
    handler.connection = Connection()
    handler.handle_one_request()

    assert shutdown == [server.socket.SHUT_RD]
    assert scheduled[0].cancelled
    assert handler.close_connection
