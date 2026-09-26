from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest

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


def _search_tool_result(document: dict[str, object]) -> dict[str, object]:
    """Shape a ``recall_search`` tools/call result exactly as the shipped app returns it.

    The tool returns a JSON string, so the search document is serialised inside
    ``structuredContent["result"]`` and repeated as the first text block. Measured against the
    real application in ``test_restore_smoke_reaches_readyz_in_the_real_application``.
    """
    text = json.dumps(document)
    return {"structuredContent": {"result": text}, "content": [{"type": "text", "text": text}]}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tenant_id", "tenant-other"),
        ("generation_id", "generation-other"),
    ],
)
def test_restore_smoke_rejects_wrong_returned_identity(field, value) -> None:
    """A search answered from the wrong tenant or generation must fail the smoke.

    Invariant: the restored application must answer from the tenant and generation the drill
    restored. Failure mode: a healthy HTTP process serves a different tenant or a stale
    generation while the smoke reports success.

    Proof record: target symbol ``_validate_search_identity``. Mutation 1 deletes the tenant
    ``raise``; node ``[tenant_id-tenant-other]`` fails with ``DID NOT RAISE``. Mutation 2 deletes
    the generation ``raise``; node ``[generation_id-generation-other]`` fails the same way. Both
    pass with the guards restored.
    """
    document = {"tenant_id": "tenant-a", "generation_id": "generation-a", "hits": []}
    document[field] = value

    with pytest.raises(smoke.ApplicationSmokeError, match=field.split("_")[0]):
        smoke._validate_search_identity(
            _search_tool_result(document),
            tenant="tenant-a",
            expected_generation="generation-a",
        )


def test_restore_smoke_reads_identity_from_the_text_block_when_unstructured() -> None:
    """A client that drops ``structuredContent`` still carries the document as text."""
    document = {"tenant_id": "tenant-a", "generation_id": "generation-a"}
    result = _search_tool_result(document)
    del result["structuredContent"]

    assert smoke._validate_search_identity(
        result, tenant="tenant-a", expected_generation="generation-a"
    ) == {"tenant_id": "tenant-a", "generation_id": "generation-a"}


def test_restore_smoke_rejects_a_search_result_without_a_document() -> None:
    with pytest.raises(smoke.ApplicationSmokeError, match="no search document"):
        smoke._validate_search_identity(
            {"content": []}, tenant="tenant-a", expected_generation="generation-a"
        )


def test_restore_smoke_runs_readiness_handshake_surface_and_search(monkeypatch) -> None:
    """The receipt path must exercise the shipped app protocol and report the served identity."""

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
            (
                200,
                {"Content-Type": "application/json"},
                json.dumps(
                    {
                        "result": _search_tool_result(
                            {"tenant_id": "tenant-a", "generation_id": "generation-a", "hits": []}
                        )
                    }
                ).encode(),
            ),
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
        expected_generation="generation-a",
        representative_chunk_id="chunk-a",
    )

    assert result["passed"] is True
    assert result["tenant_id"] == "tenant-a"
    assert result["generation_id"] == "generation-a"
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
            expected_generation=generation.generation_id,
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
    assert result["tenant_id"] == tenant
    assert result["generation_id"] == generation.generation_id
