"""STAKES-002: dependency readiness and temporal state cannot outlive their inputs."""
# Imported pytest fixtures are intentionally shadowed by their injected values.
# ruff: noqa: F811
from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from recall.generation_store import GenerationStore
from recall.lineage import IndexManifestV1, ManifestObjectV1
from recall.manifest import S3Allowlist, S3ObjectReader
from tests.conftest import TEST_DSN, requires_db
from tests.test_generations import _Embedder, _S3, _pipeline, _ready, manager  # noqa: F401


def _active_dependency_generation(manager) -> tuple[str, str]:
    documents = {
        "dependent.md": (
            b'---\nrecall_graph: {"authority":"policy","depends_on":["prerequisite.md"]}\n---\n'
            b"The policy depends on the prerequisite observation.\n"
        ),
        "prerequisite.md": (
            b'---\nrecall_graph: {"authority":"tool_observation"}\nvalid_until: 2030-01-02\n---\n'
            b"The prerequisite observation is valid for a bounded window.\n"
        ),
    }
    entries = tuple(
        ManifestObjectV1(
            f"s3://approved/corpora/{manager.tenant_id}/{name}",
            "object-v1",
            "text/markdown",
            len(data),
            hashlib.sha256(data).hexdigest(),
        )
        for name, data in sorted(documents.items())
    )
    manifest = IndexManifestV1(manager.tenant_id, "dependency-cache-proof", entries)
    objects = {
        ("approved", f"corpora/{manager.tenant_id}/{name}", "object-v1"): data
        for name, data in documents.items()
    }
    reader = S3ObjectReader(_S3(objects), S3Allowlist.parse("approved/corpora/"))
    gid = _ready(manager, manifest, _pipeline("model-a"), reader, _Embedder(1))
    manager.promote(gid, unsafe_development=True)
    prerequisite_uri = f"s3://approved/corpora/{manager.tenant_id}/prerequisite.md"
    return gid, prerequisite_uri


@requires_db
def test_implicit_now_projection_recomputes_after_prerequisite_expiry(manager, monkeypatch):
    """PgVectorStore.dependency_projection must evaluate each implicit-current request.

    Red target: baseline 76a0b36ea, unchanged node
    tests/test_pr543_dependency_cache.py::test_implicit_now_projection_recomputes_after_prerequisite_expiry.
    The intended failure is the final assertion: the cached pre-expiry closure leaves the
    dependent healthy after its authoritative prerequisite expires. The clock is injected at the
    project_current_state boundary; real generation rows, temporal parsing and closure execute.
    """
    import recall.current_state as current_state

    gid, _ = _active_dependency_generation(manager)
    clock = [datetime(2030, 1, 1, tzinfo=UTC)]
    real_project = current_state.project_current_state

    def project_at_clock(store, *, as_of=None, **kwargs):
        return real_project(store, as_of=as_of or clock[0], **kwargs)

    monkeypatch.setattr(current_state, "project_current_state", project_at_clock)
    with GenerationStore(TEST_DSN, 64, tenant=manager.tenant_id, dependency_mode="enforce") as store:
        first = store.dependency_projection()
        assert first is not None, "fixture must have a ready persisted projection"
        assert first.generation_id == gid
        assert first.reason_for("dependent.md") is None
        assert [(edge.dependent, edge.prerequisite) for edge in first.edges] == [
            ("dependent.md", "prerequisite.md")
        ]
        clock[0] = datetime(2030, 1, 3, tzinfo=UTC)
        second = store.dependency_projection()
    reason = None if second is None else second.reason_for("dependent.md")
    assert reason is not None and reason.cause == "expired", (
        "implicit-current dependency projection retained the pre-expiry healthy closure"
    )


@requires_db
def test_explicit_time_projection_rechecks_readiness(manager):
    """PgVectorStore.dependency_projection must revalidate readiness at identical times.

    Red target: baseline 76a0b36ea, unchanged node
    tests/test_pr543_dependency_cache.py::test_explicit_time_projection_rechecks_readiness.
    The final assertion fails because the old early cache hit bypasses the revoked marker.
    """
    gid, _ = _active_dependency_generation(manager)
    now = datetime(2030, 1, 1, tzinfo=UTC)
    with GenerationStore(TEST_DSN, 64, tenant=manager.tenant_id, dependency_mode="enforce") as store:
        first = store.dependency_projection(as_of=now)
        assert first is not None and first.reason_for("dependent.md") is None
        with manager._connect() as conn:
            conn.execute(
                "UPDATE recall_generations "
                "SET validation_summary = validation_summary - 'dependency_projection' "
                "WHERE tenant_id = %s AND generation_id = %s",
                (manager.tenant_id, gid),
            )
        assert store.dependency_projection(as_of=now) is None, (
            "cached dependency projection bypassed readiness revocation"
        )


@requires_db
def test_explicit_time_projection_rechecks_same_generation_erasure(manager):
    """PgVectorStore.dependency_projection cannot retain an erased prerequisite.

    Red target: baseline 76a0b36ea, unchanged node
    tests/test_pr543_dependency_cache.py::test_explicit_time_projection_rechecks_same_generation_erasure.
    The final assertion fails when the same generation's pre-erasure healthy closure is reused.
    A missing readiness marker may refuse the projection, which is a valid closed outcome.
    """
    gid, prerequisite_uri = _active_dependency_generation(manager)
    now = datetime(2030, 1, 1, tzinfo=UTC)
    with GenerationStore(TEST_DSN, 64, tenant=manager.tenant_id, dependency_mode="enforce") as store:
        first = store.dependency_projection(as_of=now)
        assert first is not None and first.reason_for("dependent.md") is None
        before = store.generation_binding()["corpus_fingerprint"]
        manager.forget(prerequisite_uri)
        assert store.generation_id == gid
        assert store.generation_binding()["corpus_fingerprint"] != before
        second = store.dependency_projection(as_of=now)
    reason = None if second is None else second.reason_for("dependent.md")
    assert second is None or (reason is not None and reason.cause == "unresolved"), (
        "same-generation erasure reused a healthy dependency on the erased source"
    )
