from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

import psycopg

import recall.ops.restore_smoke as smoke
from recall.embeddings import HashingEmbedder
from recall.generation_build import BuildRequest, pipeline_for
from recall.generations import GenerationManager
from recall.lineage import IndexManifestV1, ManifestObjectV1
from recall.manifest import LocalObjectReader
from recall.runtime_route import resolve_runtime_route
from recall.ops.restore_smoke import _rpc_payload, _smoke_environment
from tests.conftest import requires_db


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
    assert environment["RECALL_ENV"] == "development"
    assert resolve_runtime_route(environment).mode == "generation"
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


@requires_db
def test_restore_smoke_reaches_readyz_in_the_real_application(
    tmp_path: Path, unprivileged_dsn: str, monkeypatch
) -> None:
    """The real subprocess must boot the generated environment and complete retrieval.

    Proof record: baseline is the current generated ``RECALL_ENV=restore-drill`` value, target
    symbol is ``run_application_smoke``, and the expected baseline failure is route resolution
    before ``/readyz``. This is intentionally a real process and a real generation, not a mocked
    ``Popen`` or HTTP transport.
    """
    tenant = f"restore-smoke-{uuid4().hex[:12]}"
    body = b"restore smoke representative evidence"
    source = tmp_path / "memo.md"
    source.write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    manifest = IndexManifestV1(
        tenant,
        "restore-smoke-v1",
        (
            ManifestObjectV1(
                source.as_uri(), digest, "text/markdown", len(body), digest
            ),
        ),
    )
    manager = GenerationManager(unprivileged_dsn, tenant, actor="pytest", environment="test")
    embedder = HashingEmbedder()
    request = BuildRequest(commit_root=None)
    chunker, pipeline = pipeline_for(embedder, request)
    generation = manager.create(manifest, pipeline)
    manager.build(
        generation.generation_id,
        LocalObjectReader([tmp_path]),
        embedder,
        chunker,
    )
    manager.validate(generation.generation_id)
    manager.promote(generation.generation_id, unsafe_development=True)
    monkeypatch.setenv("RECALL_RESTORE_SMOKE_EMBEDDER", "hashing")

    try:
        result = smoke.run_application_smoke(
            dsn=unprivileged_dsn,
            tenant=tenant,
            representative_chunk_id="restore smoke representative evidence",
        )
    finally:
        with psycopg.connect(unprivileged_dsn, autocommit=True) as connection:
            connection.execute(
                "SELECT set_config('recall.tenant_id', %s, false)", (tenant,)
            )
            for table in (
                "recall_source_tombstones",
                "recall_audit_events",
                "recall_ingest_jobs",
                "recall_tenant_state",
                "recall_generations",
                "recall_chunks_v1",
            ):
                connection.execute(f"DELETE FROM {table} WHERE tenant_id = %s", (tenant,))

    assert result["passed"] is True
