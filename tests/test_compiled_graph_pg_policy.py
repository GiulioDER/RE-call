"""SEC-001 on a real PostgreSQL generation: only the sources the policy admits for ``compile`` are sent.

Invariant: a generation built under a source policy stamps its digest on every chunk, and the
compiled graph built for that generation before promotion (``ensure_for_generation``) sends only
the sources whose rule lists ``compile``. The provider is a fake transport.

Red proof is NOT recorded yet. The mutation to run is ``recall.compiled_graph.service._admitted``
returning every chunk, which should fail on the withheld-count assertion. The test needs a
database, and neither the author's workstation (which runs no tests) nor the remote test host
(which has none) could run it on 2026-11-28; the Lite twin in ``tests/test_compiled_graph_lite_hooks.py`` carries the
same invariant with its red proof recorded.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import psycopg
import pytest

from recall.compiled_graph import service
from recall.generation_build import BuildRequest
from recall.lineage import IndexManifestV1, ManifestObjectV1
from recall.manifest import LocalObjectReader
from tests.conftest import TEST_DSN, _db_available, db_unreachable_reason, requires_db
from tests.test_compiled_graph_lite_hooks import _artifacts, _cite_first_anchor, _compile_env, _sources

DOCS = {
    "keep.md": "# Keep\n\nThe release checklist is owned by the platform team.\n",
    "skip.md": "# Skip\n\nThe incident rota is owned by the support team.\n",
}


class _Embedder:
    dim = 64
    name = "model-a"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(index % 7) for index in range(self.dim)] for _ in texts]


@pytest.fixture
def manager():  # type: ignore[no-untyped-def]  # yields a GenerationManager for a throwaway tenant
    from recall.generations import GenerationManager

    if not _db_available():
        pytest.skip(db_unreachable_reason())
    tenant = "cg-policy-" + uuid.uuid4().hex[:10]
    value = GenerationManager(TEST_DSN, tenant, actor="pytest", environment="test")
    yield value
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        conn.execute("DELETE FROM recall_source_tombstones WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_audit_events WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_ingest_jobs WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_tenant_state WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_generations WHERE tenant_id = %s", (tenant,))


@requires_db
def test_a_generation_compiles_only_policy_admitted_sources(  # type: ignore[no-untyped-def]  # manager is a fixture
    manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.generation_build import build_generation
    from recall.security_policy import AccessContext, load_source_policy

    monkeypatch.setattr(service, "urllib_transport", _cite_first_anchor)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    entries = []
    for name, text in DOCS.items():
        data = text.encode("utf-8")
        (corpus / name).write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        entries.append(ManifestObjectV1((corpus / name).as_uri(), sha, "text/markdown", len(data), sha))
    rules = [
        {"prefix": "keep.md", "principals": [], "purposes": ["retrieval", "compile"], "redactions": []},
        {"prefix": "skip.md", "principals": [], "purposes": ["retrieval"], "redactions": []},
    ]
    policy_file = tmp_path / "policy.json"
    policy_file.write_text(json.dumps({"default_deny": True, "rules": rules}), encoding="utf-8")
    root = tmp_path / "compiled"
    env = {**_compile_env(root), "RECALL_SOURCE_POLICY_FILE": str(policy_file)}
    policy = load_source_policy(env)
    assert policy is not None
    manifest = IndexManifestV1(manager.tenant_id, "policy-v1", tuple(entries))
    stats = build_generation(
        manager, manifest, LocalObjectReader([corpus]), _Embedder(), BuildRequest(revision="fixture"),
        security_policy=policy, security_context=AccessContext("cli", manager.tenant_id),
    )
    # As a promotion does: build, validate, then the compiled graph, then promote.
    manager.validate(stats.generation_id)
    message = service.ensure_for_generation(TEST_DSN, manager.tenant_id, stats.generation_id, _Embedder, env)
    assert message.startswith("compiled graph built for "), message
    assert "1 source(s) withheld by the source policy for purpose compile" in message, message
    built = _artifacts(root)
    assert len(built) == 1, built
    assert {Path(source).name for source in _sources(built[0])} == {"keep.md"}
