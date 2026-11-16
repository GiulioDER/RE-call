"""Committed DELETE payloads reach cache cleanup even when the earlier read was stale.

The database transport is doubled; these tests do not establish PostgreSQL SQL or lock behavior.
Red proof uses mutations of the collector propagation and lifecycle digest union, as recorded
with the release validation. The public build's separate regression uses the real SQLite cache.
"""
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

from recall.cache import EMBEDDING_TEXT_DIGEST_KEY, EmbeddingCache, text_digest
from recall.generation_store import GenerationStore
from recall.generations import ErasureResult, GenerationManager
from recall_mcp import lifecycle


@pytest.mark.parametrize("contextual", [False, True])
def test_forget_purges_text_committed_after_its_initial_read(tmp_path, monkeypatch, contextual):
    """Public forget must purge actual DELETE rows, including contextual embedding digests.

    Removing the generation-specific lifecycle collector branch retains the late cache row.
    """
    store = object.__new__(GenerationStore)
    store._tenant, store._dsn, store._migration_target = "synthetic", "postgresql://unused", None
    source, old, late = "source.md", "old body", "new body committed while deletion waited"
    embedded = "context plus " + late if contextual else late
    metadata = {EMBEDDING_TEXT_DIGEST_KEY: text_digest(embedded)} if contextual else {}
    path = tmp_path / "embeddings.sqlite"
    monkeypatch.setenv("RECALL_EMBED_CACHE", str(path))
    with EmbeddingCache(path) as cache:
        cache.put_many([("old-key", [1.0]), ("late-key", [2.0])], texts=[old, embedded])
        assert len(cache.get_many(["old-key", "late-key"])) == 2
    monkeypatch.setattr(store, "sources_for_identifiers", lambda _: {source: [source]})
    monkeypatch.setattr(store, "texts_for_sources", lambda _: [(old, {})])
    monkeypatch.setattr(lifecycle, "_forget_retractions", lambda *args: None)
    monkeypatch.setattr(lifecycle, "_erasure_lock", lambda _: nullcontext())
    rescue_texts = []
    monkeypatch.setattr(GenerationManager, "list_generations", lambda _: [])
    monkeypatch.setattr(GenerationManager, "collected_generation_ids", lambda _: set())
    monkeypatch.setattr("recall.generation_rescue.erase_forgotten",
                        lambda _, texts, *args, **kwargs: (rescue_texts.extend(texts) or 0, 0))

    def forget(_manager, uri, *, erased_texts=None, **kwargs):
        # Models the rows DELETE actually removed, after the initially observed old version.
        if erased_texts is not None:
            erased_texts.append((late, metadata))
        return ErasureResult(uri, ("generation",), 1, "event")

    monkeypatch.setattr(GenerationManager, "forget", forget)
    result = lifecycle.forget_memory(store, [source], erase_memo_requests=False)
    with EmbeddingCache(path) as cache:
        retained = cache.get_many(["old-key", "late-key"])
    assert result.chunks_removed == 1
    assert not retained, "forget left a cache vector published while it waited for deletion"
    assert late in rescue_texts, "the rescue cleanup omitted the actual removed text"


@pytest.mark.parametrize("commit_fails", [False, True])
def test_manager_exposes_deleted_text_only_after_commit(monkeypatch, commit_fails):
    """Actual manager and legacy DELETE decoders collect rows only after successful commit.

    Dropping collector publication fails the success assertion; moving it into the transaction
    fails the transaction-boundary assertion in both cases. Rollback is a preservation control.
    """
    manager = GenerationManager("postgresql://unused", "synthetic")
    previous = ("previous", {})
    captured = [previous]
    returned = [("generation body", {}), ("adopted body", {"file": "source.md"})]

    class Connection:
        @contextmanager
        def transaction(self):
            yield
            assert captured == [previous], "uncommitted deleted text escaped the transaction"
            if commit_fails:
                raise RuntimeError("synthetic commit failure")

        def execute(self, query, params=None):
            sql = query if isinstance(query, str) else query.as_string()
            rows = []
            if sql.startswith("DELETE FROM recall_chunks_v1"):
                rows = [("chunk", *returned[0])] if "text, metadata" in sql else [("chunk",)]
            elif sql.startswith("WITH removed AS"):
                rows = [("legacy", *([None] * 6), *returned[1])]
            elif "to_regclass" in sql:
                rows = [(True,)]
            return SimpleNamespace(fetchall=lambda: rows, fetchone=lambda: rows[0] if rows else None)

    @contextmanager
    def connect():
        yield Connection()

    monkeypatch.setattr(manager, "_connect", connect)
    monkeypatch.setattr(manager, "_source_lock", lambda *args: None)
    monkeypatch.setattr(manager, "_lock_tenant_state", lambda _: ("generation",))
    monkeypatch.setattr(manager, "_audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(manager, "_scrub_sparse_rows", lambda *args: None)
    monkeypatch.setattr("recall.generations.pg_record_removals", lambda *args, **kwargs: None)
    monkeypatch.setattr("recall.supersession_records.pg_record_removals", lambda *args, **kwargs: None)
    error = None
    try:
        result = manager.forget("source.md", legacy_table="legacy_chunks", erased_texts=captured)
    except RuntimeError as exc:
        error = exc
    if commit_fails:
        assert error is not None and str(error) == "synthetic commit failure"
        assert captured == [previous], "a rolled-back deletion exposed text for cleanup"
    else:
        assert error is None and result.chunks_removed == 2
        assert captured == [previous, *returned], "committed generation or legacy text was omitted"


def test_later_source_failure_cleans_already_committed_erasure(tmp_path, monkeypatch):
    """A partial batch failure must preserve its error and purge already committed cache rows.

    Red before the partial-failure handler: the late first-source vector survives without rows.
    The untouched second source is a cache-preservation control.
    """
    store = object.__new__(GenerationStore)
    store._tenant, store._dsn, store._migration_target = "synthetic", "postgresql://unused", None
    path = tmp_path / "embeddings.sqlite"
    monkeypatch.setenv("RECALL_EMBED_CACHE", str(path))
    late, untouched = "late first source", "untouched second source"
    with EmbeddingCache(path) as cache:
        cache.put_many([("late-key", [1.0]), ("untouched-key", [2.0])], texts=[late, untouched])
    monkeypatch.setattr(store, "sources_for_identifiers", lambda _: {s: [s] for s in ("a.md", "b.md")})
    monkeypatch.setattr(store, "texts_for_sources", lambda _: [("old first source", {}), (untouched, {})])
    monkeypatch.setattr(lifecycle, "_forget_retractions", lambda *args: None)
    monkeypatch.setattr(lifecycle, "_erasure_lock", lambda _: nullcontext())
    monkeypatch.setattr(GenerationManager, "list_generations", lambda _: [])
    monkeypatch.setattr(GenerationManager, "collected_generation_ids", lambda _: set())
    rescued, committed = [], []
    monkeypatch.setattr("recall.generation_rescue.erase_forgotten",
                        lambda _, texts, *args, **kwargs: (rescued.extend(texts) or 0, 0))

    def forget(_manager, uri, *, erased_texts=None, **kwargs):
        if uri == "b.md":
            raise RuntimeError("synthetic later deletion failure")
        committed.append(uri)
        erased_texts.append((late, {}))
        return ErasureResult(uri, ("generation",), 1, "event")

    monkeypatch.setattr(GenerationManager, "forget", forget)
    with pytest.raises(RuntimeError, match="synthetic later deletion failure"):
        lifecycle.forget_memory(store, ["a.md", "b.md"], erase_memo_requests=False)
    assert committed == ["a.md"], "negative control: the first deletion did not commit"
    with EmbeddingCache(path) as cache:
        retained = cache.get_many(["late-key", "untouched-key"])
    assert retained == {"untouched-key": [2.0]}, "partial batch failure orphaned a committed source's cache"
    assert rescued == [late], "partial batch failure skipped committed rescue cleanup"
