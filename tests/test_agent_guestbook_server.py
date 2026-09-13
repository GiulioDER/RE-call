from __future__ import annotations

import json
import select
import socket
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

from infra.agent_guestbook.agent_guestbook_server import GuestbookStore, make_server


@contextmanager
def _running_server(database: Path) -> Iterator[str]:
    server = make_server("127.0.0.1", 0, database)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _json(url: str) -> tuple[int, dict[str, object]]:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, json.loads(response.read())


def _post(url: str, body: bytes = b"") -> tuple[int, dict[str, object]]:
    request = urllib.request.Request(url, data=body, method="POST")
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read())


def _raw_request(host: str, port: int, request: bytes) -> bytes:
    with socket.create_connection((host, port), timeout=2) as connection:
        connection.sendall(request)
        connection.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        while chunk := connection.recv(4096):
            chunks.append(chunk)
    return b"".join(chunks)


def test_empty_post_records_only_an_aggregate_greeting(tmp_path: Path) -> None:
    database = tmp_path / "guestbook.sqlite3"
    with _running_server(database) as base_url:
        status, before = _json(f"{base_url}/agent-hello")
        assert status == 200
        assert before["hello_count"] == 0

        status, greeting = _post(f"{base_url}/agent-hello")
        assert status == 201
        assert greeting == {
            "message": "hello",
            "protocol": "recall-agent-hello/1",
            "visitor": 1,
        }

        _, after = _json(f"{base_url}/agent-hello")
        assert after["hello_count"] == 1
        assert isinstance(after["last_hello_at"], str)

    with sqlite3.connect(database) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(guestbook_state)").fetchall()
        }
        row = connection.execute(
            "SELECT singleton, hello_count, last_hello_at FROM guestbook_state"
        ).fetchone()
    assert columns == {"singleton", "hello_count", "last_hello_at"}
    assert row is not None
    assert row[0:2] == (1, 1)


def test_request_data_is_refused_without_recording_a_greeting(tmp_path: Path) -> None:
    database = tmp_path / "guestbook.sqlite3"
    with _running_server(database) as base_url:
        with pytest.raises(urllib.error.HTTPError) as error:
            _post(f"{base_url}/agent-hello", b"hello")
        assert error.value.code == 400

        with pytest.raises(urllib.error.HTTPError) as error:
            _json(f"{base_url}/agent-hello?model=anything")
        assert error.value.code == 404

        _, state = _json(f"{base_url}/agent-hello")
        assert state["hello_count"] == 0


def test_ambiguous_content_length_is_refused_without_recording_a_greeting(
    tmp_path: Path,
) -> None:
    """Red on 3969157c: duplicate framing recorded a greeting from an ambiguous request."""
    database = tmp_path / "guestbook.sqlite3"
    with _running_server(database) as base_url:
        host, port_text = base_url.removeprefix("http://").split(":")
        response = _raw_request(
            host,
            int(port_text),
            (
                b"POST /agent-hello HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"Content-Length: 0\r\n"
                b"Content-Length: 5\r\n"
                b"Connection: close\r\n\r\nhello"
            ),
        )
        assert b" 400 " in response.split(b"\r\n", 1)[0]
        _, state = _json(f"{base_url}/agent-hello")
        assert state["hello_count"] == 0


def test_incomplete_request_is_closed_after_the_server_timeout(tmp_path: Path) -> None:
    """Red on 3969157c: a partial request held its worker indefinitely."""
    database = tmp_path / "guestbook.sqlite3"
    server = make_server("127.0.0.1", 0, database)
    server.request_timeout = 0.1
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with socket.create_connection(server.server_address, timeout=2) as connection:
            connection.sendall(b"GET /agent-hello HTTP/1.1\r\nHost: localhost\r\n")
            time.sleep(0.3)
            readable, _, _ = select.select([connection], [], [], 1)
            assert readable, "the server left the incomplete request open"
            assert connection.recv(1) == b""
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_aggregate_count_survives_a_server_restart(tmp_path: Path) -> None:
    database = tmp_path / "guestbook.sqlite3"
    first = GuestbookStore(database)
    assert first.record_hello().visitor == 1

    second = GuestbookStore(database)
    assert second.snapshot().hello_count == 1
    assert second.record_hello().visitor == 2
