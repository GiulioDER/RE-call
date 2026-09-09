"""Application level recovery smoke test for a restored database."""

from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from recall.errors import RecallError


HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


class ApplicationSmokeError(RuntimeError, RecallError):
    """Raised when the restored application cannot complete its authenticated MCP request."""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_json(
    url: str,
    body: dict[str, object] | None = None,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 10,
) -> tuple[int, dict[str, str], bytes]:
    request = Request(
        url,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers=headers or {},
        method="GET" if body is None else "POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers.items()), response.read()
    except HTTPError as exc:
        return exc.code, dict(exc.headers.items()), exc.read()
    except URLError as exc:
        raise ApplicationSmokeError(f"application smoke request failed: {exc.reason}") from exc


def _rpc_payload(headers: dict[str, str], body: bytes) -> dict[str, object]:
    content_type = headers.get("Content-Type", "")
    if not content_type:
        content_type = next(
            (value for key, value in headers.items() if key.lower() == "content-type"), ""
        )
    text = body.decode("utf-8")
    if "text/event-stream" in content_type:
        for line in text.splitlines():
            if line.startswith("data:"):
                payload = json.loads(line.removeprefix("data:").strip())
                if isinstance(payload, dict):
                    return payload
        raise ApplicationSmokeError("MCP response contained no JSON-RPC event")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ApplicationSmokeError("MCP response was not a JSON-RPC object")
    return payload


def _smoke_environment(
    dsn: str, tenant: str, token_file: Path, port: int, token: str
) -> dict[str, str]:
    """Build an isolated app environment from the restored DSN and smoke credentials."""
    environment = {key: value for key, value in os.environ.items() if not key.startswith("RECALL_")}
    base_url = f"http://127.0.0.1:{port}"
    environment.update(
        {
            # Restore smoke uses the development trust and static-token path, but must still use
            # the production-shaped generation route. `restore-drill` is not a runtime environment
            # accepted by `resolve_runtime_route`, so leaving that label here makes the generated
            # process exit during lifespan startup before `/readyz` can ever become healthy.
            "RECALL_ENV": "development",
            "RECALL_TRANSPORT": "streamable-http",
            "RECALL_MCP_STATELESS": "0",
            "RECALL_SERVING_DSN": dsn,
            "RECALL_TENANT": tenant,
            "RECALL_INDEX_MODE": "generation",
            "RECALL_EMBEDDER": os.environ.get("RECALL_RESTORE_SMOKE_EMBEDDER", "fastembed"),
            "RECALL_MCP_TOOLS": "search",
            # Static tokens are a development only recovery harness. Relaxing the trust gate here
            # lets the smoke verify application startup and retrieval plumbing even when the
            # restored fixture intentionally contains no calibration artifact. Production serving
            # remains strict and uses OIDC in the ECS task definition.
            "RECALL_TRUST_MODE": "development",
            "RECALL_RATE_LIMIT_BACKEND": "off",
            "RECALL_AUTH_MODE": "static",
            "RECALL_AUTH_TOKENS_FILE": str(token_file),
            "RECALL_AUTH_ISSUER_URL": base_url,
            "RECALL_AUTH_RESOURCE_URL": base_url,
            "RECALL_HOST": "127.0.0.1",
            "RECALL_PORT": str(port),
        }
    )
    token_file.write_text(
        json.dumps(
            {
                "principals": [
                    {
                        "name": "restore-smoke",
                        "token": token,
                        "tenant": tenant,
                        "scopes": ["recall:read"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    token_file.chmod(0o600)
    return environment


def run_application_smoke(
    *, dsn: str, tenant: str, representative_chunk_id: str
) -> dict[str, object]:
    """Start the shipped HTTP app and complete one authenticated retrieval request."""
    token = secrets.token_urlsafe(32)
    port = _free_port()
    with tempfile.TemporaryDirectory(prefix="recall-restore-smoke-") as directory:
        token_file = Path(directory) / "tokens.json"
        environment = _smoke_environment(dsn, tenant, token_file, port, token)
        process = subprocess.Popen(
            [sys.executable, "-m", "recall_mcp.server"],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        base_url = f"http://127.0.0.1:{port}"
        # Starlette's canonical MCP route is `/mcp`; `/mcp/` returns a 307 redirect. The smoke
        # client intentionally uses the standard library and does not follow redirects, so use
        # the canonical path to exercise the actual authenticated protocol rather than its slash
        # normalisation response.
        mcp_url = f"{base_url}/mcp"
        timeout = float(os.environ.get("RECALL_RESTORE_SMOKE_TIMEOUT_SECONDS", "180"))
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise ApplicationSmokeError("MCP application exited before readiness")
                try:
                    status, _, _ = _http_json(f"{base_url}/readyz", timeout=3)
                except ApplicationSmokeError:
                    status = 0
                if status == 200:
                    break
                time.sleep(0.5)
            else:
                raise ApplicationSmokeError("MCP application did not become ready")

            auth_headers = {**HEADERS, "Authorization": f"Bearer {token}"}
            initialize = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "restore-smoke", "version": "1.0"},
                },
            }
            status, response_headers, response_body = _http_json(
                mcp_url, initialize, headers=auth_headers, timeout=30
            )
            if status != 200:
                raise ApplicationSmokeError(f"MCP initialize returned HTTP {status}")
            initialized = _rpc_payload(response_headers, response_body)
            if "error" in initialized:
                raise ApplicationSmokeError("MCP initialize returned a JSON-RPC error")
            session_id = response_headers.get("mcp-session-id")
            if not session_id:
                raise ApplicationSmokeError("MCP initialize did not issue a session")
            session_headers = {**auth_headers, "mcp-session-id": session_id}

            status, _, _ = _http_json(
                mcp_url,
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                headers=session_headers,
                timeout=30,
            )
            if status not in {200, 202}:
                raise ApplicationSmokeError("MCP initialized notification was rejected")

            status, response_headers, response_body = _http_json(
                mcp_url,
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
                headers=session_headers,
                timeout=30,
            )
            if status != 200:
                raise ApplicationSmokeError("MCP tools/list was rejected")
            tools_payload = _rpc_payload(response_headers, response_body)
            result = tools_payload.get("result")
            tools = result.get("tools", []) if isinstance(result, dict) else []
            tool_names = sorted(
                item["name"] for item in tools if isinstance(item, dict) and "name" in item
            )
            if tool_names != ["recall_evidence", "recall_search"]:
                raise ApplicationSmokeError(f"restore smoke served unexpected tools: {tool_names}")

            status, response_headers, response_body = _http_json(
                mcp_url,
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "recall_search",
                        "arguments": {"query": representative_chunk_id, "k": 1},
                    },
                },
                headers=session_headers,
                timeout=60,
            )
            if status != 200:
                raise ApplicationSmokeError("MCP recall_search was rejected")
            search_payload = _rpc_payload(response_headers, response_body)
            result = search_payload.get("result")
            if "error" in search_payload or (isinstance(result, dict) and result.get("isError")):
                raise ApplicationSmokeError(
                    "MCP recall_search returned an application error: "
                    + json.dumps(search_payload, sort_keys=True)
                )
            return {
                "passed": True,
                "transport": "streamable-http",
                "authenticated": True,
                "tools": tool_names,
                "search": "recall_search",
            }
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


__all__ = ["ApplicationSmokeError", "run_application_smoke"]
