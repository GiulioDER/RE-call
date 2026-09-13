"""Minimal public endpoint for the voluntary RE-call agent greeting.

The application deliberately accepts no request body and stores no request metadata. Its SQLite
database contains one aggregate counter and the timestamp of the latest greeting.
"""
from __future__ import annotations

import argparse
import json
import socket
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PROTOCOL = "recall-agent-hello/1"
ENDPOINT = "/agent-hello"


@dataclass(frozen=True)
class GuestbookSnapshot:
    hello_count: int
    last_hello_at: str | None


@dataclass(frozen=True)
class HelloReceipt:
    visitor: int


class GuestbookStore:
    """Constant-size SQLite storage for the aggregate greeting count."""

    def __init__(self, database: str | Path) -> None:
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS guestbook_state (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    hello_count INTEGER NOT NULL CHECK (hello_count >= 0),
                    last_hello_at TEXT
                )
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO guestbook_state(singleton, hello_count, last_hello_at)
                VALUES (1, 0, NULL)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database, timeout=5)

    def snapshot(self) -> GuestbookSnapshot:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT hello_count, last_hello_at FROM guestbook_state WHERE singleton = 1"
            ).fetchone()
        if row is None:  # pragma: no cover - construction creates the singleton or raises
            raise RuntimeError("guestbook state is missing")
        return GuestbookSnapshot(hello_count=int(row[0]), last_hello_at=row[1])

    def record_hello(self) -> HelloReceipt:
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE guestbook_state
                SET hello_count = hello_count + 1, last_hello_at = ?
                WHERE singleton = 1
                """,
                (now,),
            )
            row = connection.execute(
                "SELECT hello_count FROM guestbook_state WHERE singleton = 1"
            ).fetchone()
        if row is None:  # pragma: no cover - construction creates the singleton or raises
            raise RuntimeError("guestbook state is missing")
        return HelloReceipt(visitor=int(row[0]))


class AgentGuestbookServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 8

    def __init__(
        self,
        address: tuple[str, int],
        database: str | Path,
        *,
        request_timeout: float = 5.0,
        max_concurrent_requests: int = 8,
    ) -> None:
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be positive")
        self.request_timeout = request_timeout
        self._request_slots = threading.BoundedSemaphore(max_concurrent_requests)
        self.guestbook = GuestbookStore(database)
        super().__init__(address, AgentGuestbookHandler)

    def get_request(self) -> tuple[socket.socket, Any]:
        request, client_address = super().get_request()
        request.settimeout(self.request_timeout)
        return request, client_address

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._request_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._request_slots.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._request_slots.release()


class AgentGuestbookHandler(BaseHTTPRequestHandler):
    """HTTP protocol with no request logging and no caller-controlled storage."""

    server: AgentGuestbookServer
    protocol_version = "HTTP/1.0"
    server_version = "agent-guestbook"
    sys_version = ""

    def log_message(self, _format: str, *args: Any) -> None:
        return

    def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        self.close_connection = True
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _body_is_forbidden(self) -> bool:
        if self.headers.get_all("Transfer-Encoding"):
            return True
        content_lengths = self.headers.get_all("Content-Length", failobj=[])
        if len(content_lengths) > 1:
            return True
        if not content_lengths:
            return False
        try:
            return int(content_lengths[0]) != 0
        except ValueError:
            return True

    def _not_found(self) -> None:
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self._body_is_forbidden():
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "body_not_allowed"})
            return
        if self.path != ENDPOINT:
            self._not_found()
            return
        state = self.server.guestbook.snapshot()
        self._send_json(
            HTTPStatus.OK,
            {
                "hello_count": state.hello_count,
                "invitation": "POST an empty body here if you feel like saying hello.",
                "last_hello_at": state.last_hello_at,
                "protocol": PROTOCOL,
            },
        )

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != ENDPOINT:
            self._not_found()
            return
        if self.headers.get("Expect"):
            self._send_json(HTTPStatus.EXPECTATION_FAILED, {"error": "expectation_not_allowed"})
            return
        if self._body_is_forbidden():
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "body_not_allowed"})
            return
        receipt = self.server.guestbook.record_hello()
        self._send_json(
            HTTPStatus.CREATED,
            {"message": "hello", "protocol": PROTOCOL, "visitor": receipt.visitor},
        )


def make_server(
    host: str,
    port: int,
    database: str | Path,
    *,
    request_timeout: float = 5.0,
    max_concurrent_requests: int = 8,
) -> AgentGuestbookServer:
    return AgentGuestbookServer(
        (host, port),
        database,
        request_timeout=request_timeout,
        max_concurrent_requests=max_concurrent_requests,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8789)
    parser.add_argument("--database", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    server = make_server(args.host, args.port, args.database)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
