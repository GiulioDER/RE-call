"""Ordinary generation cache writes share the source lock that serializes forget.

BUG-004 red proof uses the public build entry point with its pre-fix cache write outside
the final source transaction. Database state is doubled, while embedding and erasure use
the real content-addressed cache. The already-tombstoned case is a baseline control.
"""

from contextlib import contextmanager
from pathlib import Path
from threading import Event, Lock, Thread, get_ident
from types import SimpleNamespace

import pytest

from recall import generations
from recall.cache import EmbeddingCache, cache_key, text_digest
from recall.embeddings import embedding_profile
from recall.generations import GenerationManager
from recall.lineage import GenerationState
from recall.semantic_graph import build_semantic_graph
from tests.test_generations import _Embedder, _manifest, _pipeline, _reader


@pytest.fixture
def build_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    data = b"The archive keeps the harbour ledger."
    manifest = _manifest("cache-erasure", data)
    embedder = _Embedder(1)
    pipeline = _pipeline(embedder.name)
    manager = GenerationManager("postgresql://recall:recall@127.0.0.1:1/recall", manifest.tenant_id)
    state = SimpleNamespace(lock=Lock(), owner=None, checks=0, tombstone_at=None, publications=[], written=[],
                            commit_fails=False, manager=manager)
    cache_path = tmp_path / "embeddings.sqlite"

    @contextmanager
    def connect():  # noqa: ANN202
        connection = SimpleNamespace(locked=False)

        @contextmanager
        def transaction():  # noqa: ANN202
            try:
                yield connection
                if state.commit_fails:
                    raise RuntimeError("synthetic commit failure")
            finally:
                if connection.locked:
                    connection.locked = False
                    state.owner = None
                    state.lock.release()

        connection.transaction = transaction
        yield connection

    def source_lock(connection, tenant, source):  # noqa: ANN001, ANN202
        assert tenant == manifest.tenant_id and source == manifest.objects[0].uri
        state.lock.acquire()
        connection.locked = True
        state.owner = get_ident()

    def tombstoned(connection, source):  # noqa: ANN001, ANN202
        state.checks += 1
        return state.tombstone_at is not None and state.checks >= state.tombstone_at

    def write_source(connection, generation_id, source, version, digest, chunks, vectors, language):  # noqa: ANN001, ANN202
        state.written.extend(chunks)
        return len(chunks)

    put_many = EmbeddingCache.put_many

    def record_publication(cache, pairs, *, texts=None):  # noqa: ANN001, ANN202
        state.publications.append(state.owner == get_ident())
        return put_many(cache, pairs, texts=texts)

    graph = build_semantic_graph([], tenant_id=manifest.tenant_id, generation_id="gen-test")
    monkeypatch.setattr(manager, "_connect", connect)
    monkeypatch.setattr(manager, "_source_lock", source_lock)
    monkeypatch.setattr(manager, "_is_tombstoned", tombstoned)
    monkeypatch.setattr(manager, "_require_generation", lambda *args: SimpleNamespace(state=GenerationState.BUILDING))
    monkeypatch.setattr(manager, "_load_identity", lambda *args: (manifest, pipeline))
    monkeypatch.setattr(manager, "_reuse_source", lambda *args, **kwargs: 0)
    monkeypatch.setattr(manager, "_write_source", write_source)
    monkeypatch.setattr(manager, "_build_semantic_graph", lambda *args: (graph, {}))
    monkeypatch.setattr(manager, "_mark_validating", lambda *args, **kwargs: None)
    monkeypatch.setattr(manager, "_refresh_chunk_statistics", lambda: True)
    monkeypatch.setattr(generations, "open_default_cache", lambda: EmbeddingCache(cache_path))
    monkeypatch.setattr(EmbeddingCache, "put_many", record_publication)
    state.build = lambda: manager.build("gen-test", _reader(manifest, data), embedder, lambda text: [text])
    state.embedder, state.cache_path = embedder, cache_path
    return state


@pytest.mark.parametrize("failure", ["write", "commit"])
def test_failed_source_write_does_not_leave_orphan_cache_vectors(build_case, monkeypatch, failure):
    """Cache publication followed by a failed source transaction must not leave unowned vectors.

    Red on the locked-publication patch: the intended final cache assertion retained the row.
    PostgreSQL write and commit failures are injected at the transport boundary.
    """
    state, texts = build_case, []
    embed = state.embedder.embed

    def record(inputs):
        texts.extend(inputs)
        return embed(inputs)

    def write(*args):
        if failure == "write":
            raise RuntimeError("synthetic write failure")
        state.commit_fails = True
        return 1

    monkeypatch.setattr(state.embedder, "embed", record)
    monkeypatch.setattr(state.manager, "_write_source", write)
    monkeypatch.setattr(state.manager, "fail", lambda *args: None)
    caught = None
    try:
        state.build()
    except RuntimeError as exc:
        caught = exc
    assert caught is not None and str(caught) == f"synthetic {failure} failure"
    assert texts and state.publications == [True], "negative control: no cache publication before the failure"
    keys = [cache_key(embedding_profile(state.embedder), state.embedder.dim, text, "passage") for text in texts]
    with EmbeddingCache(state.cache_path) as cache:
        assert not cache.get_many(keys), "failed source transaction left orphan cache vectors"


def test_a_build_serializes_cache_publication_with_forget(build_case, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    """Cache publication holds the source lock, so a waiting forget erases its result.

    Pre-fix build fails the final publication assertion after the eraser runs before
    embedding returns. The transaction double models the existing advisory source lock.
    """
    state = build_case
    attempted, acquired = Event(), Event()
    observed = {}
    workers = []
    embed = state.embedder.embed

    def embed_while_forgetting(texts):  # noqa: ANN001, ANN202
        observed["texts"] = list(texts)

        def erase():  # noqa: ANN202
            attempted.set()
            with state.lock:
                acquired.set()
                with EmbeddingCache(state.cache_path) as cache:
                    keys = [cache_key(embedding_profile(state.embedder), state.embedder.dim, text, "passage")
                            for text in texts]
                    observed["before_erasure"] = bool(cache.get_many(keys))
                    cache.erase_digests({text_digest(text) for text in texts})

        worker = Thread(target=erase, daemon=True)
        workers.append(worker)
        worker.start()
        assert attempted.wait(5), "the eraser did not attempt the source lock"
        observed["waited"] = not acquired.wait(0.1)
        return embed(texts)

    monkeypatch.setattr(state.embedder, "embed", embed_while_forgetting)
    try:
        stats = state.build()
    finally:
        for worker in workers:
            worker.join(5)
    assert workers and all(not worker.is_alive() for worker in workers), "the eraser did not finish"
    keys = [cache_key(embedding_profile(state.embedder), state.embedder.dim, text, "passage")
            for text in observed["texts"]]
    with EmbeddingCache(state.cache_path) as cache:
        retained = cache.get_many(keys)
    assert stats.chunks == 1 and state.embedder.calls == 1
    assert state.publications == [True] and observed["waited"] and observed["before_erasure"] and not retained, (
        "the build published cache vectors outside its source lock after forget could finish"
    )


@pytest.mark.parametrize("tombstone_at", [1, 2], ids=["already-forgotten-control", "forgotten-after-first-check"])
def test_a_tombstoned_source_is_not_embedded(build_case, tombstone_at: int) -> None:  # noqa: ANN001
    """A final-check tombstone blocks embedding and cache publication. First-check is a control."""
    state = build_case
    state.tombstone_at = tombstone_at
    stats = state.build()
    assert stats.tombstoned_objects == 1 and stats.chunks == 0 and not state.written
    assert state.embedder.calls == 0 and not state.publications, "a tombstoned source was embedded into the cache"
