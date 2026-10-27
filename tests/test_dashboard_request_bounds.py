"""SEC-001: framing is checked before stream consumption or application dispatch.

Red proof: these invalid-framing nodes fail their intended assertions against
970dbeb7b. The valid-framing control is red-proved by a production cap mutation.
The real handler is obtained through serve; transport and listener are fakes.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

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
