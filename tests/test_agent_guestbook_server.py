from __future__ import annotations

import json
import sqlite3
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

from scripts.agent_guestbook_server import GuestbookStore, make_server


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


def test_aggregate_count_survives_a_server_restart(tmp_path: Path) -> None:
    database = tmp_path / "guestbook.sqlite3"
    first = GuestbookStore(database)
    assert first.record_hello().visitor == 1

    second = GuestbookStore(database)
    assert second.snapshot().hello_count == 1
    assert second.record_hello().visitor == 2
