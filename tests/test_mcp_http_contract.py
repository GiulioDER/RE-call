"""One end-to-end contract for the authenticated MCP HTTP serving boundary.

The narrower HTTP tests are useful diagnostics, but they can each pass while the complete request
path is broken at a seam between authentication, tenant routing, tool authorisation, or metering.
This test keeps those concerns in one real streamable-http process and drives a real tool call over
HTTP, using a NOSUPERUSER/NOBYPASSRLS role so tenant isolation cannot pass vacuously.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import requires_db

httpx = pytest.importorskip("httpx")

HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
READ_BUDGET = 1


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _initialize_body(client_name: str, request_id: int = 1) -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": client_name, "version": "1.0"},
        },
    }


def _rpc_payload(response: Any) -> dict[str, Any]:
    """Decode either the JSON or SSE response form accepted by streamable-http."""
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" not in content_type:
        payload = response.json()
        assert isinstance(payload, dict)
        return payload
    for line in response.text.splitlines():
        if line.startswith("data:"):
            payload = json.loads(line.removeprefix("data:").strip())
            assert isinstance(payload, dict)
            return payload
    raise AssertionError(f"MCP response contained no JSON-RPC event: {response.text!r}")


def _tool_text(payload: dict[str, Any]) -> str:
    result = payload.get("result")
    if not isinstance(result, dict):
        return json.dumps(payload, sort_keys=True)
    content = result.get("content", [])
    if not isinstance(content, list):
        return json.dumps(payload, sort_keys=True)
    return "\n".join(
        str(item["text"])
        for item in content
        if isinstance(item, dict) and isinstance(item.get("text"), str)
    )


@pytest.fixture(scope="module")
def live_server(
    tmp_path_factory: pytest.TempPathFactory, unprivileged_dsn: str
) -> Iterator[tuple[str, dict[str, str], Path]]:
    """Start one authenticated HTTP server serving two isolated tenants."""
    tokens = {name: secrets.token_urlsafe(32) for name in ("writer", "reader")}
    tmp = tmp_path_factory.mktemp("mcp-contract")
    index_root = tmp / "index-root"
    index_root.mkdir()
    token_file = tmp / "tokens.json"
    token_file.write_text(
        json.dumps(
            {
                "principals": [
                    {
                        "name": "writer",
                        "token": tokens["writer"],
                        "tenant": "tenant-a",
                        "scopes": ["recall:read", "recall:write"],
                    },
                    {
                        "name": "reader",
                        "token": tokens["reader"],
                        "tenant": "tenant-b",
                        "scopes": ["recall:read"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    token_file.chmod(0o600)
    port = _free_port()
    env = {
        **os.environ,
        "RECALL_TRANSPORT": "streamable-http",
        "RECALL_MCP_STATELESS": "0",
        "RECALL_EMBEDDER": "hashing",
        "RECALL_DSN": unprivileged_dsn,
        "RECALL_AUTH_TOKENS_FILE": str(token_file),
        "RECALL_AUTH_ISSUER_URL": f"http://127.0.0.1:{port}",
        "RECALL_AUTH_RESOURCE_URL": f"http://127.0.0.1:{port}",
        "RECALL_HOST": "127.0.0.1",
        "RECALL_PORT": str(port),
        "RECALL_INDEX_ROOT": str(index_root),
        "RECALL_RATE_READ_PER_MIN": str(READ_BUDGET),
    }
    log_path = tmp / "server.log"
    log = log_path.open("w+", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "recall_mcp.server"],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}/mcp/"
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            if process.poll() is not None:
                log.seek(0)
                pytest.fail(f"server exited early:\n{log.read()}")
            try:
                # An HTTP response, including an expected 401, proves the listener is ready.
                httpx.post(url, json=_initialize_body("readiness"), headers=HEADERS, timeout=2)
                break
            except httpx.RequestError:
                time.sleep(0.5)
        else:  # pragma: no cover
            pytest.fail("server did not start within 60s")
        yield url, tokens, log_path
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:  # pragma: no cover
            process.kill()
        log.close()


def _session(url: str, token: str, client_name: str) -> dict[str, str]:
    headers = {**HEADERS, "Authorization": f"Bearer {token}"}
    response = httpx.post(
        url,
        json=_initialize_body(client_name),
        headers=headers,
        timeout=30,
        follow_redirects=True,
    )
    assert response.status_code == 200, response.text
    session_id = response.headers.get("mcp-session-id")
    assert session_id, response.text
    headers["mcp-session-id"] = session_id
    initialized = httpx.post(
        url,
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        headers=headers,
        timeout=30,
        follow_redirects=True,
    )
    assert initialized.status_code in {200, 202}, initialized.text
    return headers


def _call(
    url: str, headers: dict[str, str], name: str, arguments: dict[str, object]
) -> dict[str, Any]:
    response = httpx.post(
        url,
        json={
            "jsonrpc": "2.0",
            "id": secrets.randbelow(1_000_000),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
        headers=headers,
        timeout=60,
        follow_redirects=True,
    )
    assert response.status_code == 200, response.text
    return _rpc_payload(response)


@requires_db
def test_authenticated_http_request_preserves_the_whole_security_contract(
    live_server: tuple[str, dict[str, str], Path],
) -> None:
    """Authentication, isolation, scopes, rate limiting, and execution hold on one wire path."""
    url, tokens, log_path = live_server

    unknown = httpx.post(
        url,
        json=_initialize_body("unknown"),
        headers={**HEADERS, "Authorization": f"Bearer {secrets.token_urlsafe(32)}"},
        timeout=30,
        follow_redirects=True,
    )
    assert unknown.status_code == 401

    writer = _session(url, tokens["writer"], "writer")
    reader = _session(url, tokens["reader"], "reader")
    private_body = "tenant-a-only-7f8e2c1d"
    encoded = base64.b64encode(private_body.encode("utf-8")).decode("ascii")
    ingest = _call(
        url,
        writer,
        "recall_ingest",
        {"files": [{"name": "private.md", "content_b64": encoded}]},
    )
    ingest_result = json.loads(_tool_text(ingest))
    assert ingest_result["state"] == "completed"
    assert ingest_result["chunks"] > 0

    writer_inventory = _call(url, writer, "recall_inventory", {})
    writer_entries = json.loads(_tool_text(writer_inventory))["entries"]
    assert any("private.md" in entry["source"] for entry in writer_entries)

    forbidden_write = _call(
        url,
        reader,
        "recall_ingest",
        {"files": [{"name": "forbidden.md", "content_b64": encoded}]},
    )
    assert forbidden_write["result"]["isError"] is True
    # The SDK deliberately redacts PermissionError details from the client-facing result. The
    # generic error envelope proves the tool was stopped, while the server audit line proves it was
    # stopped by the missing write scope rather than by a later filesystem or database failure.
    assert _tool_text(forbidden_write) == "Error executing tool recall_ingest"
    assert "denied for scope recall:write" in log_path.read_text(encoding="utf-8")

    reader_inventory = _call(url, reader, "recall_inventory", {})
    assert json.loads(_tool_text(reader_inventory))["entries"] == []

    rate_limited = _call(url, writer, "recall_inventory", {})
    assert rate_limited["result"]["isError"] is True
    assert "rate limit exceeded" in _tool_text(rate_limited)
