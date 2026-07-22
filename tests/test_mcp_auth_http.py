"""End-to-end: does an unauthenticated HTTP request actually get rejected?

Why this exists when `test_mcp_auth_wiring.py` already passes
-------------------------------------------------------------
Those tests prove `RecallTokenVerifier.verify_token` returns None for a bad token. They do NOT
prove the SDK ever calls it. Every one of them would stay green if `token_verifier` were dropped
from the `FastMCP(...)` call, if the auth middleware were never mounted, or if the transport
served tool requests before reaching it — and in each of those cases the server would be wide
open while the suite reported success.

So this file starts a real server in a subprocess and makes real HTTP requests. It exercises the
REJECTION path, not just the green one: a run where only the authenticated request were checked
could not distinguish "auth works" from "auth is disabled and everything succeeds".
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import sys
import time

import pytest

from tests.conftest import TEST_DSN, requires_db

httpx = pytest.importorskip("httpx")

BODY = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    """A real `streamable-http` server with one read-only principal."""
    token = secrets.token_urlsafe(32)
    tmp = tmp_path_factory.mktemp("auth")
    tokens = tmp / "tokens.json"
    tokens.write_text(
        json.dumps(
            {"principals": [
                {"name": "agent", "token": token, "tenant": "team-a", "scopes": ["recall:read"]}
            ]}
        ),
        encoding="utf-8",
    )
    port = _free_port()
    env = {
        **os.environ,
        "RECALL_TRANSPORT": "streamable-http",
        "RECALL_EMBEDDER": "hashing",  # no model download; this test is about auth, not retrieval
        "RECALL_DSN": TEST_DSN,
        "RECALL_AUTH_TOKENS_FILE": str(tokens),
        "RECALL_AUTH_ISSUER_URL": f"http://127.0.0.1:{port}",
        "RECALL_AUTH_RESOURCE_URL": f"http://127.0.0.1:{port}",
        "RECALL_HOST": "127.0.0.1",
        "RECALL_PORT": str(port),
    }
    log = open(tmp / "server.log", "w+", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "recall_mcp.server"], env=env, stdout=log, stderr=subprocess.STDOUT
    )
    url = f"http://127.0.0.1:{port}/mcp/"
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            if proc.poll() is not None:
                log.seek(0)
                pytest.fail(f"server exited early:\n{log.read()}")
            try:
                httpx.post(url, json=BODY, headers=HEADERS, timeout=2)
                break  # any HTTP response means it is listening
            except httpx.RequestError:
                time.sleep(0.5)
        else:
            pytest.fail("server did not start within 60s")
        yield url, token, tmp / "server.log"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()
        log.close()


def post(url, token=None):
    headers = dict(HEADERS)
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    # follow_redirects: the transport 307s between /mcp and /mcp/, and a redirect counted as
    # "not 401" would make an unauthenticated request look accepted.
    return httpx.post(url, json=BODY, headers=headers, timeout=30, follow_redirects=True)


@requires_db
def test_a_request_with_no_credentials_is_rejected(live_server):
    """The single most important assertion in the suite."""
    url, _, _ = live_server
    assert post(url).status_code == 401


@requires_db
def test_a_request_with_an_unknown_token_is_rejected(live_server):
    url, _, _ = live_server
    assert post(url, secrets.token_urlsafe(32)).status_code == 401


@requires_db
def test_a_request_with_a_malformed_authorization_header_is_rejected(live_server):
    url, _, _ = live_server
    resp = httpx.post(
        url, json=BODY, headers={**HEADERS, "Authorization": "Bearer"}, timeout=30,
        follow_redirects=True,
    )
    assert resp.status_code == 401


@requires_db
def test_the_valid_token_is_accepted(live_server):
    """The control. Without it, a server rejecting EVERYTHING would pass the tests above."""
    url, token, _ = live_server
    resp = post(url, token)
    assert resp.status_code != 401, resp.text
    assert resp.status_code < 400, resp.text


@requires_db
def test_the_server_never_writes_a_token_to_its_log(live_server):
    """Logs travel further than anyone expects; even a prefix shrinks a brute-force space."""
    _, token, log_path = live_server
    contents = log_path.read_text(encoding="utf-8", errors="replace")
    assert token not in contents
    assert token[:8] not in contents
