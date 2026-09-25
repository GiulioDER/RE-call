import pytest

from recall.embeddings import HashingEmbedder
from recall_mcp.service import index_memory

from tests.conftest import dev_search_memory, requires_db


def test_production_refuses_local_filesystem_ingestion_before_reading(monkeypatch) -> None:
    monkeypatch.setenv("RECALL_ENV", "production")
    with pytest.raises(ValueError, match="development-only"):
        index_memory(object(), object(), "private/local/path")  # type: ignore[arg-type]


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
