"""Rate limiting, end to end, through a real authenticated server.

`test_limits.py` proves the bucket arithmetic. It cannot prove the server *calls* it — and that
is the half that was actually broken while the unit tests were green. These drive a real
`streamable-http` server over HTTP, with a real token, through a real MCP session, and assert on
what a client receives.

The budget is set to 2 calls/min via the environment, so the refusal arrives on the third request
rather than after a wait that would make this test slow and flaky.
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

HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
READ_BUDGET = 2


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _init_body(name: str) -> dict:
    return {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": name, "version": "1.0"}},
    }


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    """One server, three principals on three separate tenants, read budget of 2/min."""
    tokens = {name: secrets.token_urlsafe(32) for name in ("a", "b", "c")}
    tmp = tmp_path_factory.mktemp("ratelimit")
    token_file = tmp / "tokens.json"
    token_file.write_text(
        json.dumps({"principals": [
            {"name": f"agent-{n}", "token": t, "tenant": f"tenant-{n}",
             "scopes": ["recall:read"]}
            for n, t in tokens.items()
        ]}),
        encoding="utf-8",
    )
    port = _free_port()
    env = {
        **os.environ,
        "RECALL_TRANSPORT": "streamable-http",
        "RECALL_EMBEDDER": "hashing",  # no model download; this is about metering, not retrieval
        "RECALL_DSN": TEST_DSN,
        "RECALL_AUTH_TOKENS_FILE": str(token_file),
        "RECALL_AUTH_ISSUER_URL": f"http://127.0.0.1:{port}",
        "RECALL_AUTH_RESOURCE_URL": f"http://127.0.0.1:{port}",
        "RECALL_HOST": "127.0.0.1",
        "RECALL_PORT": str(port),
        "RECALL_RATE_READ_PER_MIN": str(READ_BUDGET),
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
                httpx.post(url, json=_init_body("probe"), headers=HEADERS, timeout=2)
                break
            except httpx.RequestError:
                time.sleep(0.5)
        else:  # pragma: no cover
            pytest.fail("server did not start within 60s")
        yield url, tokens
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()
        log.close()


def _session(url: str, token: str) -> dict:
    """Handshake, and return headers carrying the session id a tool call requires."""
    headers = {**HEADERS, "Authorization": f"Bearer {token}"}
    resp = httpx.post(url, json=_init_body("test"), headers=headers, timeout=30,
                      follow_redirects=True)
    assert resp.status_code == 200, resp.text
    session_id = resp.headers.get("mcp-session-id")
    assert session_id, "server returned no session id"
    headers["mcp-session-id"] = session_id
    httpx.post(url, json={"jsonrpc": "2.0", "method": "notifications/initialized"},
               headers=headers, timeout=30, follow_redirects=True)
    return headers


def _search(url: str, headers: dict, n: int = 1) -> str:
    resp = httpx.post(
        url,
        json={"jsonrpc": "2.0", "id": 100 + n, "method": "tools/call",
              "params": {"name": "recall_search", "arguments": {"query": "anything", "k": 1}}},
        headers=headers, timeout=30, follow_redirects=True,
    )
    assert resp.status_code == 200, resp.text
    return resp.text


@requires_db
def test_a_tenant_is_throttled_once_its_call_budget_is_spent(live_server):
    url, tokens = live_server
    headers = _session(url, tokens["a"])

    for i in range(READ_BUDGET):
        assert "rate limit exceeded" not in _search(url, headers, i), f"throttled early at call {i}"

    refused = _search(url, headers, READ_BUDGET)
    assert "rate limit exceeded" in refused
    # The client is told when to come back; without it the only strategy is to retry immediately,
    # which is the behaviour the limit exists to stop.
    assert "Retry in" in refused


@requires_db
def test_exhausting_one_tenant_does_not_throttle_another(live_server):
    """Per-TENANT is the claim, so one tenant hitting its ceiling must not affect a neighbour."""
    url, tokens = live_server
    b = _session(url, tokens["b"])
    for i in range(READ_BUDGET):
        _search(url, b, i)
    assert "rate limit exceeded" in _search(url, b, READ_BUDGET)

    c = _session(url, tokens["c"])
    assert "rate limit exceeded" not in _search(url, c, 0)


@requires_db
def test_the_refusal_never_echoes_the_bearer_token(live_server):
    """A throttling path is a new place to leak a credential into an error string."""
    url, tokens = live_server
    headers = _session(url, tokens["a"])
    for i in range(READ_BUDGET + 2):
        body = _search(url, headers, i)
        assert tokens["a"] not in body


# --------------------------------------------------------------------------------------------
# The byte quota, through the server
# --------------------------------------------------------------------------------------------
#
# A separate server because this one needs the write scope, a corpus to index, and a byte budget
# small enough to trip. Worth the second process: with only the service-level test, deleting
# `on_measured=_debit` from the tool body left the whole suite green — the quota was verified as
# a contract nobody was holding.

INDEX_BYTE_BUDGET = 2500  # two 1000-byte memos fit; the third request does not


@pytest.fixture(scope="module")
def indexing_server(tmp_path_factory):
    token = secrets.token_urlsafe(32)
    tmp = tmp_path_factory.mktemp("quota")
    corpus = tmp / "memory"
    corpus.mkdir()
    for i in range(1):
        (corpus / f"memo{i}.md").write_text("x" * 1000, encoding="utf-8")

    token_file = tmp / "tokens.json"
    token_file.write_text(
        json.dumps({"principals": [
            {"name": "writer", "token": token, "tenant": "tenant-w",
             "scopes": ["recall:read", "recall:write"]}
        ]}),
        encoding="utf-8",
    )
    port = _free_port()
    env = {
        **os.environ,
        "RECALL_TRANSPORT": "streamable-http",
        "RECALL_EMBEDDER": "hashing",
        "RECALL_DSN": TEST_DSN,
        "RECALL_AUTH_TOKENS_FILE": str(token_file),
        "RECALL_AUTH_ISSUER_URL": f"http://127.0.0.1:{port}",
        "RECALL_AUTH_RESOURCE_URL": f"http://127.0.0.1:{port}",
        "RECALL_HOST": "127.0.0.1",
        "RECALL_PORT": str(port),
        "RECALL_INDEX_ROOT": str(corpus),
        "RECALL_INDEX_BYTES_PER_HOUR": str(INDEX_BYTE_BUDGET),
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
                httpx.post(url, json=_init_body("probe"), headers=HEADERS, timeout=2)
                break
            except httpx.RequestError:
                time.sleep(0.5)
        else:  # pragma: no cover
            pytest.fail("server did not start within 60s")
        yield url, token, str(corpus)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()
        log.close()


@requires_db
def test_repeated_indexing_is_stopped_by_the_byte_quota(indexing_server):
    """Each request is individually legal; only the aggregate spend is not.

    This is the gap the issue described — "a client within the per-call budget can still call
    repeatedly". Re-indexing the same unchanged corpus is the cheapest possible way to express
    it: the work is skipped, but the bytes are still measured and charged, because a caller must
    not be able to dodge the quota by pointing at content the server happens to have seen.
    """
    url, token, path = indexing_server
    headers = _session(url, token)

    def index() -> str:
        resp = httpx.post(
            url,
            json={"jsonrpc": "2.0", "id": 200, "method": "tools/call",
                  "params": {"name": "recall_index", "arguments": {"path": path}}},
            headers=headers, timeout=60, follow_redirects=True,
        )
        assert resp.status_code == 200, resp.text
        return resp.text

    assert "rate limit" not in index().lower(), "throttled on the first 1000-byte request"
    assert "rate limit" not in index().lower(), "throttled on the second, still within budget"

    refused = index()
    assert "rate limit exceeded" in refused
    assert "index_bytes" in refused
