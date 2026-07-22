import pytest

from recall.embeddings import HashingEmbedder
from recall_mcp.service import index_memory, search_memory

from tests.conftest import requires_db


@requires_db
def test_index_then_search(tmp_path, make_store, monkeypatch):
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path))
    (tmp_path / "note.md").write_text("the caching decision was adopted", encoding="utf-8")
    store = make_store(64)
    emb = HashingEmbedder(dim=64)
    stats = index_memory(store, emb, str(tmp_path))
    assert stats.chunks == 1
    assert stats.files == 1
    result = search_memory(store, emb, "caching")
    assert any("caching" in h.text for h in result.hits)


@requires_db
def test_index_rejects_path_outside_root(tmp_path, make_store, monkeypatch):
    root = tmp_path / "allowed"
    root.mkdir()
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(root))
    store = make_store(64)
    emb = HashingEmbedder(dim=64)
    # tmp_path is the PARENT of the allowed root -> must be rejected before any read.
    with pytest.raises(ValueError, match="outside the allowed index root"):
        index_memory(store, emb, str(tmp_path))


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
