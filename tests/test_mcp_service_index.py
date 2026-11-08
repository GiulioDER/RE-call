import os

import pytest

from recall.embeddings import HashingEmbedder
from recall.security_policy import AccessContext, SourceRule, SourceSecurityPolicy
from recall_mcp.indexing import IndexPreflightError
from recall_mcp.service import index_memory

from tests.conftest import dev_search_memory, requires_db


def test_production_refuses_local_filesystem_ingestion_before_reading(monkeypatch) -> None:
    monkeypatch.setenv("RECALL_ENV", "production")
    with pytest.raises(ValueError, match="development-only"):
        index_memory(object(), object(), "private/local/path")  # type: ignore[arg-type]


class _TenantIndexSink:
    tenant = "reader"

    def __init__(self):
        self.texts = []

    def source_content_hashes(self):
        return {}

    def replace_sources(self, sources, chunks, embeddings):
        self.texts.extend(chunk.text for chunk in chunks)

    def analyze_if_stale(self, modified):
        return True


@pytest.mark.parametrize("request_path", ["uploads/victim", "uploads/victim/private.md", "."])
def test_index_cannot_copy_another_tenants_staged_upload(tmp_path, monkeypatch, request_path):
    """At 65b7a791c the ancestor imported victim text and both direct requests failed to refuse."""
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path))
    private = tmp_path / "uploads" / "victim" / "private.md"
    private.parent.mkdir(parents=True)
    private.write_text("private material from the victim tenant", encoding="utf-8")
    own = tmp_path / "uploads" / "reader" / "own.md"
    own.parent.mkdir()
    own.write_text("ordinary material from the reader tenant", encoding="utf-8")
    store = _TenantIndexSink()
    embedder = HashingEmbedder(dim=64)

    if request_path == ".":
        index_memory(store, embedder, str(tmp_path))  # type: ignore[arg-type]
        assert store.texts == ["ordinary material from the reader tenant"]
    else:
        with pytest.raises(IndexPreflightError, match="another tenant"):
            index_memory(store, embedder, str(tmp_path / request_path))  # type: ignore[arg-type]
        assert not store.texts
        index_memory(store, embedder, str(own))  # type: ignore[arg-type]
        assert store.texts == ["ordinary material from the reader tenant"]


@pytest.mark.parametrize("request_path", ["restricted", "restricted/public/memo.md"])
def test_index_policy_uses_the_operator_root(tmp_path, monkeypatch, request_path):
    """At 65b7a791c both paths stripped the denied prefix and indexed the restricted file."""
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path))
    restricted = tmp_path / "restricted" / "public"
    restricted.mkdir(parents=True)
    (restricted / "memo.md").write_text("private text", encoding="utf-8")
    public = tmp_path / "public"
    public.mkdir()
    (public / "memo.md").write_text("public text", encoding="utf-8")
    policy = SourceSecurityPolicy(
        tuple(SourceRule(name, purposes=frozenset({"indexing"})) for name in ("public", "memo.md"))
    )
    store = _TenantIndexSink()
    embedder = HashingEmbedder(dim=64)
    context = AccessContext("reader", store.tenant, purpose="indexing")
    with pytest.raises(IndexPreflightError, match="denied"):
        index_memory(
            store, embedder, str(tmp_path / request_path),  # type: ignore[arg-type]
            security_policy=policy, security_context=context,
        )
    assert not store.texts
    index_memory(
        store, embedder, str(public),  # type: ignore[arg-type]
        security_policy=policy, security_context=context,
    )
    assert store.texts == ["public text"]


@requires_db
def test_index_then_search(tmp_path, make_store, monkeypatch):
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path))
    (tmp_path / "note.md").write_text("the caching decision was adopted", encoding="utf-8")
    store = make_store(64)
    emb = HashingEmbedder(dim=64)
    stats = index_memory(store, emb, str(tmp_path))
    assert stats.chunks == 1
    assert stats.files == 1
    result = dev_search_memory(store, emb, "caching")
    assert any("caching" in h.text for h in result.hits)


@requires_db
def test_index_rejects_path_outside_root(tmp_path, make_store, monkeypatch):
    root = tmp_path / "allowed"
    root.mkdir()
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(root))
    store = make_store(64)
    emb = HashingEmbedder(dim=64)
    # tmp_path is the PARENT of the allowed root -> must be rejected before any read.
    # The message deliberately no longer echoes the RESOLVED root back to the caller — that was a
    # filesystem-mapping oracle for a path probe. See tests/test_error_path_disclosure.py.
    with pytest.raises(ValueError, match="outside the directory this server is allowed to index"):
        index_memory(store, emb, str(tmp_path))


class _CountingEmbedder(HashingEmbedder):
    """HashingEmbedder that records every text it is asked to embed."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__(dim=dim)
        self.embedded: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return super().embed(texts)


@requires_db
def test_mcp_index_reuses_the_shared_embedding_cache(tmp_path, make_store, monkeypatch):
    """The MCP `recall_index` path writes through the shared embedding cache and reads it back.

    Invariant: identical chunk text reaching `index_memory` a second time, under a different
    source, is served from the shared cache and never re-embedded. Every other indexing entry
    point (`recall index`, `generation build`, setup, seeding) already opens it; this one passed
    no `cache`, so `Indexer` fell back to None and every call re-embedded.

    Red proof, 2026-09-25: run against `recall_mcp/indexing.py` at `bdffd482`, where the
    `Indexer(...)` call in `index_memory` passes no `cache`. It failed on the final assertion
    with the second source's chunk sent to the embedder again (`embedded == [text]`).
    """
    cache_file = tmp_path / "cache" / "embeddings.sqlite"
    monkeypatch.setenv("RECALL_EMBED_CACHE", str(cache_file))
    root = tmp_path / "root"
    text = "the shared cache decision was adopted for every indexing entry point"
    for name in ("first", "second"):
        (root / name).mkdir(parents=True)
        (root / name / "note.md").write_text(text, encoding="utf-8")
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(root))
    store = make_store(64)
    emb = _CountingEmbedder(dim=64)

    first = index_memory(store, emb, str(root / "first"))
    assert first.chunks == 1
    assert emb.embedded, "the first index must embed; otherwise the test observes nothing"

    emb.embedded.clear()
    second = index_memory(store, emb, str(root / "second"))
    assert second.chunks == 1
    assert store.count() == 2
    assert emb.embedded == [], "identical text under a new source must come from the cache"


# --------------------------------------------------------------------------------------------
# Indexing budget caps (SECURITY.md "Indexing is client-callable and unbounded")
# --------------------------------------------------------------------------------------------


def _write_files(root, count, body="x" * 50):
    for i in range(count):
        (root / f"note{i}.md").write_text(body, encoding="utf-8")


@requires_db
def test_index_under_both_limits_still_works(tmp_path, make_store, monkeypatch):
    """The happy path is unaffected: a small tree under both caps indexes normally."""
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path))
    _write_files(tmp_path, 3)
    store = make_store(64)
    emb = HashingEmbedder(dim=64)
    stats = index_memory(store, emb, str(tmp_path))
    assert stats.files == 3
    assert stats.chunks == 3
    assert store.count() == 3


@requires_db
def test_index_over_file_count_limit_is_refused_before_anything_is_written(
    tmp_path, make_store, monkeypatch
):
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path))
    monkeypatch.setenv("RECALL_INDEX_MAX_FILES", "5")
    _write_files(tmp_path, 6)  # one over the limit
    store = make_store(64)
    emb = HashingEmbedder(dim=64)
    with pytest.raises(ValueError, match="exceeds the file-count budget"):
        index_memory(store, emb, str(tmp_path))
    # The whole point of a PRE-FLIGHT cap: nothing was embedded or written for the refused request.
    assert store.count() == 0


@requires_db
def test_index_over_byte_limit_is_refused_before_anything_is_written(
    tmp_path, make_store, monkeypatch
):
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path))
    monkeypatch.setenv("RECALL_INDEX_MAX_BYTES", "100")
    _write_files(tmp_path, 3, body="y" * 50)  # 150 bytes total > 100
    store = make_store(64)
    emb = HashingEmbedder(dim=64)
    with pytest.raises(ValueError, match="exceeds the byte budget"):
        index_memory(store, emb, str(tmp_path))
    assert store.count() == 0


@requires_db
def test_index_limits_are_configurable_and_a_raised_limit_lets_the_same_tree_through(
    tmp_path, make_store, monkeypatch
):
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path))
    _write_files(tmp_path, 6)
    store = make_store(64)
    emb = HashingEmbedder(dim=64)

    monkeypatch.setenv("RECALL_INDEX_MAX_FILES", "5")
    with pytest.raises(ValueError, match="exceeds the file-count budget"):
        index_memory(store, emb, str(tmp_path))
    assert store.count() == 0

    monkeypatch.setenv("RECALL_INDEX_MAX_FILES", "10")  # raised -> same tree now clears the cap
    stats = index_memory(store, emb, str(tmp_path))
    assert stats.files == 6
    assert store.count() == 6


@requires_db
def test_budget_error_names_the_limit_the_measured_value_and_the_env_var(
    tmp_path, make_store, monkeypatch
):
    """Same shape as the existing out-of-root error: name the limit, the measured value, and
    the environment variable a caller can raise."""
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path))
    monkeypatch.setenv("RECALL_INDEX_MAX_FILES", "2")
    _write_files(tmp_path, 4)
    store = make_store(64)
    emb = HashingEmbedder(dim=64)
    with pytest.raises(ValueError) as exc:
        index_memory(store, emb, str(tmp_path))
    msg = str(exc.value)
    assert "4 candidate file(s)" in msg  # measured value
    assert "limit 2" in msg  # the configured limit
    assert "RECALL_INDEX_MAX_FILES" in msg  # the variable to raise it
    assert store.count() == 0

def test_protected_generation_ingest_refuses_relabeling_before_create(tmp_path, monkeypatch):
    """CCA SEC-001. Before the fix the restricted carry-forward reached generation creation."""
    import hashlib
    from contextlib import nullcontext
    from types import SimpleNamespace

    from recall.embeddings import HashingEmbedder
    from recall.lineage import IndexManifestV1, ManifestObjectV1
    from recall.security_policy import AccessContext, SourceRule, SourceSecurityPolicy
    from recall_mcp import desktop_ingest
    from recall_mcp.indexing import IndexPreflightError

    tenant_root = tmp_path / "uploads" / "team"
    job = tenant_root / "new-job"
    job.mkdir(parents=True)
    (job / "new.md").write_bytes(b"new public information")
    old_files = []
    for name in ("keep.md", "private/secret.md"):
        path = tenant_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        content = b"previous document content"
        path.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        old_files.append(ManifestObjectV1(
            uri=path.as_uri(), media_type="text/markdown", version_id=digest,
            size=len(content), sha256=digest,
        ))
    manifest = IndexManifestV1(tenant_id="team", corpus_version="operator-build", objects=tuple(old_files))

    def create(*args, **kwargs):
        raise AssertionError("restricted source reached generation creation")

    manager = SimpleNamespace(
        tenant_ingest_lock=nullcontext, servable_manifest=lambda: manifest, create=create,
    )
    monkeypatch.setattr(desktop_ingest, "GenerationManager", lambda *args, **kwargs: manager)
    purposes = frozenset({"indexing"})
    policy = SourceSecurityPolicy((
        SourceRule("keep.md", purposes=purposes),
        SourceRule("uploads/team", purposes=purposes),
        SourceRule("private", purposes=purposes, principals=frozenset({"admin"})),
    ))
    with pytest.raises(IndexPreflightError):
        desktop_ingest.generation_ingest(
            SimpleNamespace(tenant="team", _dsn="unused"), HashingEmbedder(dim=64), str(job), "memory",
            security_policy=policy, security_context=AccessContext("writer", "team", purpose="indexing"),
            env={"RECALL_INDEX_ROOT": str(tmp_path)},
        )


@pytest.mark.parametrize("alias", [True, False])
def test_index_refuses_case_alias_of_upload_namespace(tmp_path, monkeypatch, alias):
    """CCA SEC-002. With POSIX lexical spelling a direct alternate-case target indexed the file."""
    from pathlib import Path

    from recall_mcp import indexing

    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path))
    target = tmp_path / "Uploads" / "victim" / "private.md"
    target.parent.mkdir(parents=True)
    target.write_text("private material from another tenant", encoding="utf-8")
    original = indexing.within

    def lexical(root, path):
        # On a case-insensitive POSIX volume the file opens but realpath retains "Uploads".
        if Path(path).parts == target.parts:
            return target if root == tmp_path else None
        if Path(path).parts == (tmp_path / "uploads").parts:
            return tmp_path / "uploads"
        return original(root, path)

    monkeypatch.setattr(indexing, "within", lexical)
    monkeypatch.setattr(Path, "samefile", lambda self, other: alias)
    store = _TenantIndexSink()
    if not alias:
        try:
            index_memory(store, HashingEmbedder(dim=64), str(target))
        except IndexPreflightError as exc:
            pytest.fail(f"an ordinary directory was treated as an upload alias: {exc}")
        assert store.texts == ["private material from another tenant"]
        return
    with pytest.raises(IndexPreflightError, match="another tenant"):
        index_memory(store, HashingEmbedder(dim=64), str(target))
    assert not store.texts


def test_erase_legacy_single_component_upload_tenant(tmp_path, monkeypatch):
    """CCA STAKES-001. The staging validator previously rejected legacy tenant cleanup."""
    from recall.uploads import delete_staged_sources

    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path))
    source = tmp_path / "uploads" / "research team" / "job" / "memo.md"
    source.parent.mkdir(parents=True)
    source.write_text("existing staged content", encoding="utf-8")
    try:
        removed = delete_staged_sources("research team", [str(source)])
    except ValueError:
        removed = 0
    assert removed == 1
    assert not source.exists()


@pytest.mark.skipif(os.name != "posix", reason="native POSIX tenant names")
@pytest.mark.parametrize("tenant", ["research.", "research:team", r"research\team"])
def test_erase_legacy_posix_upload_name(tmp_path, monkeypatch, tenant):
    """FIX-003 L6: cleanup rejected an existing native POSIX tenant directory."""
    from recall.uploads import delete_staged_sources

    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path))
    source = tmp_path / "uploads" / tenant / "job" / "memo.md"
    source.parent.mkdir(parents=True)
    source.write_text("legacy staged data", encoding="utf-8")
    try:
        removed = delete_staged_sources(tenant, [str(source)])
    except ValueError as exc:
        pytest.fail(f"legacy cleanup rejected the native directory: {exc}")
    assert removed == 1
    assert not source.exists()
