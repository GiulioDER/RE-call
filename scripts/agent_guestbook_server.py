"""Minimal public endpoint for the voluntary RE-call agent greeting.

The application deliberately accepts no request body and stores no request metadata. Its SQLite
database contains one aggregate counter and the timestamp of the latest greeting.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
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

    def __init__(self, address: tuple[str, int], database: str | Path) -> None:
        self.guestbook = GuestbookStore(database)
        super().__init__(address, AgentGuestbookHandler)


class AgentGuestbookHandler(BaseHTTPRequestHandler):
    """HTTP protocol with no request logging and no caller-controlled storage."""

    server: AgentGuestbookServer
    protocol_version = "HTTP/1.1"
    server_version = "agent-guestbook"
    sys_version = ""

    def log_message(self, _format: str, *args: Any) -> None:
        return

    def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _not_found(self) -> None:
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
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
        if self.headers.get("Transfer-Encoding"):
            self.close_connection = True
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "body_not_allowed"})
            return
        raw_length = self.headers.get("Content-Length", "0")
        try:
            content_length = int(raw_length)
        except ValueError:
            content_length = -1
        if content_length != 0:
            self.close_connection = True
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "body_not_allowed"})
            return
        receipt = self.server.guestbook.record_hello()
        self._send_json(
            HTTPStatus.CREATED,
            {"message": "hello", "protocol": PROTOCOL, "visitor": receipt.visitor},
        )


def make_server(host: str, port: int, database: str | Path) -> AgentGuestbookServer:
    return AgentGuestbookServer((host, port), database)


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
