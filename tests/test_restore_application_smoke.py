from __future__ import annotations

import json
from pathlib import Path

import recall.ops.restore_smoke as smoke
from recall.ops.restore_smoke import _rpc_payload, _smoke_environment


def test_restore_smoke_uses_the_restored_dsn_and_exact_search_surface(
    tmp_path: Path, monkeypatch
) -> None:
    token_file = tmp_path / "tokens.json"
    monkeypatch.setenv("RECALL_RESTORE_SMOKE_EMBEDDER", "hashing")

    environment = _smoke_environment(
        "postgresql://restore.example/recall",
        "tenant-a",
        token_file,
        8123,
        "t" * 43,
    )

    assert environment["RECALL_SERVING_DSN"] == "postgresql://restore.example/recall"
    assert environment["RECALL_MCP_TOOLS"] == "search"
    assert environment["RECALL_AUTH_MODE"] == "static"
    assert environment["RECALL_EMBEDDER"] == "hashing"
    assert json.loads(token_file.read_text(encoding="utf-8"))["principals"][0]["tenant"] == "tenant-a"


def test_restore_smoke_decodes_json_and_sse_mcp_responses() -> None:
    payload = {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}

    assert _rpc_payload({"Content-Type": "application/json"}, json.dumps(payload).encode()) == payload
    assert _rpc_payload(
        {"Content-Type": "text/event-stream"},
        f"event: message\ndata: {json.dumps(payload)}\n\n".encode(),
    ) == payload


def test_restore_smoke_runs_readiness_handshake_surface_and_search(monkeypatch) -> None:
    """The receipt path must exercise the shipped app protocol, not just database helpers."""

    class Process:
        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

    responses = iter(
        [
            (200, {}, b""),
            (200, {"Content-Type": "application/json", "mcp-session-id": "session"}, json.dumps({"result": {}}).encode()),
            (202, {}, b""),
            (200, {"Content-Type": "application/json"}, json.dumps({"result": {"tools": [{"name": "recall_search"}, {"name": "recall_evidence"}]}}).encode()),
            (200, {"Content-Type": "application/json"}, json.dumps({"result": {"content": [{"text": "{}"}]}}).encode()),
        ]
    )
    requests: list[tuple[str, dict[str, object] | None, dict[str, str] | None]] = []

    monkeypatch.setattr(smoke, "_free_port", lambda: 8123)
    monkeypatch.setattr(smoke.subprocess, "Popen", lambda *args, **kwargs: Process())

    def fake_http(url, body=None, *, headers=None, timeout=10):
        requests.append((url, body, headers))
        return next(responses)

    monkeypatch.setattr(smoke, "_http_json", fake_http)

    result = smoke.run_application_smoke(
        dsn="postgresql://restore.example/recall",
        tenant="tenant-a",
        representative_chunk_id="chunk-a",
    )

    assert result["passed"] is True
    assert [body.get("method") if body else None for _, body, _ in requests] == [
        None,
        "initialize",
        "notifications/initialized",
        "tools/list",
        "tools/call",
    ]
    assert requests[-1][1]["params"]["name"] == "recall_search"
