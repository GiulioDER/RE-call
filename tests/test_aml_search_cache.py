"""Parity and staleness proofs for the hosted Search caches, ``recall_aml.search_cache``.

The invariant every test here guards: a C9 Search served through the caches returns exactly what
the uncached code returned, byte for byte, including after every kind of write to the tenant and
including writes made by another process that never told this one. The failure modes are a
ranking that differs from ``rank_bm25_chunks`` (a mis-counted term, a dropped document, a summation
in another order) and a cached value served after the rows under it changed.

Red proofs, run 2026-09-26 against deliberate mutations of the production lines named, each
reverted afterwards; every one failed in the assertion named, not in setup or collection. The
node ids are all in this file.

* M1, ``Bm25Snapshot.extend`` replacing a known term's postings with the new documents instead of
  merging them (an incremental update that drops documents):
  ``test_snapshot_ranks_bit_identically_to_the_reference_built_whole_or_in_parts`` failed at the
  ``_exact(ranked) == expected`` assertion (hit 0 score ``0x1.eac24ca0aeaeep+2`` against the
  reference's), ``test_cached_bm25_is_the_reference_through_every_kind_of_write`` failed at the
  parity assertion after the first incremental insert (``new-1`` ranked first against the
  reference's ``id-893176931-43``), and
  ``test_c9_search_is_byte_identical_to_the_uncached_code_across_adds_and_deletes`` failed at
  ``served == expected``.
* M2, ``extend`` taking a document's length as ``len(counts)`` (distinct terms, a mis-count):
  the snapshot parity test failed at the same assertion (``0x1.5963abb0c376ep+2``).
* M3, ``Bm25Snapshot.rank`` summing over ``dict.fromkeys(query_terms)`` instead of
  ``query_terms`` (a repeated query term counted once): the snapshot parity test failed at the
  same assertion (``0x1.96fd0b599c534p+1``).
* M4, the fingerprint without its ``sum(xmin)`` term, in the SQL and in ``_fingerprint_of`` and
  the supersession statement alike: ``test_an_update_that_keeps_count_and_latest_time_is_still_seen``
  failed with ``'hit' == 'rebuild'``, a stale index served. Mutating only the SQL half does NOT go
  red, correctly: the two halves then never agree, so every Search rebuilds, which is slow and safe.
* M5, the hit test in ``TenantSearchCache.rank_bm25`` dropping its fingerprint comparison:
  ``test_cached_bm25_is_the_reference_through_every_kind_of_write`` failed at the parity assertion
  after the other store's update, and the end to end test failed at ``served == expected``.
* M6, ``superseded_ids`` keeping empty ids (``if value is not None`` for ``if value``):
  ``test_cached_supersession_is_the_store_scan_through_writes`` failed with ``''`` as an extra
  item. M6b, its hit test dropping the fingerprint comparison: the same test failed with
  ``'old-4'`` missing. The end to end test stayed green under M6b, since no superseded record was
  among the rendered hits, so it is not claimed as a proof of that line.
* M7, ``rank`` serving the cached ``Chunk`` itself instead of ``_fresh``:
  ``test_a_served_hit_is_a_private_copy_the_next_search_cannot_see_changed`` failed with
  ``99 == 0``.
* M8, ``rank_bm25`` refreshing without ``entry.build_lock``:
  ``test_concurrent_cold_searches_build_the_index_once`` failed with ``8 == 1`` rebuilds.
* M9, the service passing ``query_vector=None`` to the sidecar:
  ``test_the_code_route_embeds_its_query_once_and_the_sidecar_reuses_it`` failed with the query
  embedded twice. M10, passing ``run.query_vector`` whatever the route:
  ``test_the_context_route_sidecar_still_embeds_with_the_code_embedder`` failed with ``[] ==
  [query]`` (the Context vector reused for the Code sidecar). The end to end test stayed green
  under M10, so it is not claimed for that line either.
* M11, ``_corpus_status`` not registering its flight:
  ``test_concurrent_status_misses_share_one_computation`` failed with ``8 == 1`` computations.
* M12, ``_invalidate_corpus_status`` not forgetting the flight in progress:
  ``test_a_write_during_a_status_computation_is_never_hidden_by_it`` failed with ``1 == 2``, a
  Search after the write joining a read made before it.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
import hashlib
import random
import threading
from pathlib import Path

import psycopg
import pytest

from recall.pool import SharedPool
from recall.store import PgVectorStore
from recall.types import Chunk, ScoredChunk
from recall_aml.code4 import rank_bm25_chunks
from recall_aml.embedding_lock import CachedEmbedder
from recall_aml.identity import tenant_for
from recall_aml.models import (
    AddRequest,
    CodingMemoryRecord,
    EvidenceSpan,
    Message,
    SearchRequest,
    SearchResponse,
)
from recall_aml.retrieval import HostedRetriever, RetrievalRun
from recall_aml.search_cache import Bm25Snapshot, TenantSearchCache
from recall_aml.service import HostedService
from recall_aml.storage import PgHostedRepository
from recall_aml.variants import variant
from tests.conftest import TEST_DSN, requires_db


C9 = variant("C9_routed_specialists_grounded_graph_atomic")
_WORDS = [
    "parser", "regression", "cache", "eviction", "policy", "the", "a", "is", "to", "x",
    "widget_factory", "retry", "timeout", "deploy", "config", "2fa", "b2", "index", "query",
    "merge", "conflict", "sql", "schema", "migration", "flaky", "test", "pytest", "json",
    "handler", "error", "Error", "PARSER", "gc", "heap", "thread", "lock", "async", "await",
]


def _window(chunk_id: str, text: str, *, session: str = "s", segment: int = 0) -> Chunk:
    return Chunk(
        id=chunk_id,
        source=f"aml://{session}",
        text=text,
        metadata={"source_session_id": session, "segment": segment, "record_type": "raw"},
    )


def _corpus(rng: random.Random, count: int) -> list[Chunk]:
    chunks = []
    for index in range(count):
        words = [rng.choice(_WORDS) for _ in range(rng.randint(0, 60))]
        chunks.append(
            _window(
                f"id-{rng.randint(0, 10**9):09d}-{index}",
                " ".join(words),
                session=f"sessions/{rng.randint(0, 5)}.jsonl",
                segment=rng.randint(0, 40),
            )
        )
    return chunks


def _query(rng: random.Random) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(rng.randint(1, 9)))


def _exact(hits: list[ScoredChunk]) -> list[tuple[object, ...]]:
    """Everything a hit carries, with the score as its exact bit pattern."""
    return [
        (hit.chunk, hit.score.hex(), hit.score_kind, hit.indexed_at, hit.first_indexed_at)
        for hit in hits
    ]


def _split(rng: random.Random, chunks: list[Chunk]) -> list[list[Chunk]]:
    cuts = sorted(rng.sample(range(len(chunks) + 1), k=min(2, len(chunks) + 1)))
    bounds = [0, *cuts, len(chunks)]
    return [chunks[start:end] for start, end in zip(bounds, bounds[1:])]


# Pure snapshot parity: no database.


def test_snapshot_ranks_bit_identically_to_the_reference_built_whole_or_in_parts() -> None:
    """Whole builds and incremental builds rank exactly as ``rank_bm25_chunks`` does.

    Three hundred random corpora, each queried with repeated terms, stopwords, mixed case and
    every width from 1 to past the corpus size, with both tie orders. Scores compare by ``hex``.
    """
    rng = random.Random(20260926)
    for _ in range(300):
        chunks = _corpus(rng, rng.randint(0, 80))
        whole = Bm25Snapshot.build(((chunk, 1) for chunk in chunks), (0, None, 0))
        parts = Bm25Snapshot(fingerprint=(0, None, 0))
        for part in _split(rng, chunks):
            parts = parts.extend(((chunk, 1) for chunk in part), (0, None, 0))
        for _ in range(8):
            query = _query(rng)
            k = rng.choice([1, 3, 10, 100, 1_000])
            stable = rng.random() < 0.5
            expected = _exact(rank_bm25_chunks(chunks, query, k=k, stable_ties=stable))
            for snapshot in (whole, parts):
                ranked = snapshot.rank(query, k=k, stable_ties=stable)
                assert ranked is not None
                assert _exact(ranked) == expected


def test_snapshot_defers_to_the_reference_only_where_the_reference_raises() -> None:
    """A positive candidate without a Code4 window identity: None, so the caller re-raises."""
    chunks = [
        _window("a", "parser fix"),
        Chunk("b", "source", "parser regression", {}),
        _window("c", "unrelated words", segment=1),
    ]
    snapshot = Bm25Snapshot.build(((chunk, 1) for chunk in chunks), (0, None, 0))

    assert snapshot.rank("parser", k=100, stable_ties=True) is None
    with pytest.raises(ValueError, match="source_session_id"):
        rank_bm25_chunks(chunks, "parser", k=100, stable_ties=True)
    # The same malformed chunk is harmless where it does not score, in both implementations.
    ranked = snapshot.rank("unrelated", k=100, stable_ties=True)
    assert ranked is not None
    assert _exact(ranked) == _exact(rank_bm25_chunks(chunks, "unrelated", k=100, stable_ties=True))
    assert [hit.chunk.id for hit in ranked] == ["c"]


def test_a_served_hit_is_a_private_copy_the_next_search_cannot_see_changed() -> None:
    chunk = _window("a", "parser fix")
    snapshot = Bm25Snapshot.build([(chunk, 1)], (0, None, 0))

    first = snapshot.rank("parser", k=10, stable_ties=True)
    assert first is not None
    first[0].chunk.metadata["segment"] = 99
    second = snapshot.rank("parser", k=10, stable_ties=True)

    assert second is not None
    assert second[0].chunk.metadata["segment"] == 0


# Against a real store: every write path, and a writer this process never hears about.


def _reference(store: PgVectorStore, query: str, *, stable: bool = True) -> list[tuple[object, ...]]:
    return _exact(rank_bm25_chunks(list(store.iter_chunks()), query, k=100, stable_ties=stable))


def _vectors(chunks: list[Chunk]) -> list[list[float]]:
    return [[1.0, 0.0, 0.0] for _ in chunks]


_QUERIES = [
    "parser regression",
    "cache eviction policy the",
    "retry retry timeout",
    "widget_factory config",
    "brand new words",
]


def _assert_parity(cache: TenantSearchCache, store: PgVectorStore) -> list[str]:
    statuses = []
    for query in _QUERIES:
        ranked, status = cache.rank_bm25(store, query, k=100, stable_ties=True)
        statuses.append(status)
        assert _exact(ranked) == _reference(store, query)
    return statuses


@requires_db
def test_cached_bm25_is_the_reference_through_every_kind_of_write(make_store) -> None:
    """Insert, in-process invalidation, update and delete, each by this store or by another one."""
    store = make_store(3)
    other = PgVectorStore(TEST_DSN, dim=3, table=store.table)  # a writer in another "process"
    try:
        rng = random.Random(7)
        first = _corpus(rng, 60)
        store.upsert(first, _vectors(first))
        cache = TenantSearchCache()

        assert _assert_parity(cache, store) == ["rebuild", "hit", "hit", "hit", "hit"]

        added = [
            _window("new-1", "brand new parser words"),
            _window("new-2", "retry cache eviction", segment=3),
        ]
        store.upsert(added, _vectors(added))
        assert _assert_parity(cache, store)[0] == "incremental"

        cache.invalidate([store._tenant])
        assert _assert_parity(cache, store)[0] == "revalidated"

        other.upsert([_window("new-1", "entirely different text now")], [[1.0, 0.0, 0.0]])
        assert _assert_parity(cache, store)[0] == "rebuild"

        more = [_window("new-3", "brand new words again", segment=7)]
        other.upsert(more, _vectors(more))
        assert _assert_parity(cache, store)[0] == "incremental"

        other.delete_sources([first[0].source])
        assert _assert_parity(cache, store)[0] == "rebuild"

        cache.invalidate([store._tenant], drop=True)
        assert _assert_parity(cache, store)[0] == "rebuild"
    finally:
        other.close()


@requires_db
def test_an_update_that_keeps_count_and_latest_time_is_still_seen(make_store) -> None:
    """A backdated update moves neither the row count nor ``max(indexed_at)``; ``xmin`` must.

    The upsert writes ``indexed_at = now()``, the start of ITS transaction. A transaction that
    began before the newest row was written therefore updates a row to an older time than the
    newest one, and the count does not change either.
    """
    store = make_store(3)
    base = [_window("a", "parser fix"), _window("b", "cache eviction", segment=1)]
    store.upsert(base, _vectors(base))
    cache = TenantSearchCache()
    with psycopg.connect(TEST_DSN) as early:
        early.execute("SELECT set_config('recall.tenant_id', %s, true)", (store._tenant,))
        early.execute("SELECT now()")  # this transaction's now() is fixed from here
        late = [_window("c", "retry timeout", segment=2)]
        store.upsert(late, _vectors(late))
        assert _assert_parity(cache, store)[0] == "rebuild"
        early.execute(
            f"UPDATE {store.table} SET text = %s, indexed_at = now() "  # noqa: S608
            "WHERE tenant_id = %s AND id = %s",
            ("widget_factory config brand new words", store._tenant, "a"),
        )
        early.commit()
    latest = store._with_retry(
        lambda conn: conn.execute(
            f"SELECT count(*), max(indexed_at) = (SELECT indexed_at FROM {store.table} "  # noqa: S608
            "WHERE tenant_id = %s AND id = 'c') FROM " + store.table + " WHERE tenant_id = %s",
            (store._tenant, store._tenant),
        ).fetchone()
    )
    assert latest == (3, True)  # the scenario really left both unchanged

    ranked, status = cache.rank_bm25(store, "widget_factory config", k=100, stable_ties=True)

    assert status == "rebuild"
    assert _exact(ranked) == _reference(store, "widget_factory config")
    assert [hit.chunk.id for hit in ranked] == ["a"]


@requires_db
def test_a_missing_window_identity_raises_exactly_what_the_reference_raises(make_store) -> None:
    store = make_store(3)
    rows = [_window("a", "parser fix"), Chunk("b", "src", "parser regression", {"segment": 1})]
    store.upsert(rows, _vectors(rows))
    cache = TenantSearchCache()

    with pytest.raises(ValueError) as reference:
        rank_bm25_chunks(list(store.iter_chunks()), "parser", k=100, stable_ties=True)
    with pytest.raises(ValueError) as cached:
        cache.rank_bm25(store, "parser", k=100, stable_ties=True)

    assert str(cached.value) == str(reference.value)
    ranked, _ = cache.rank_bm25(store, "parser", k=100, stable_ties=False)
    assert _exact(ranked) == _reference(store, "parser", stable=False)


@requires_db
def test_concurrent_cold_searches_build_the_index_once(make_store) -> None:
    store = make_store(3)
    rows = _corpus(random.Random(3), 3_000)
    store.upsert(rows, _vectors(rows))
    cache = TenantSearchCache()
    barrier = threading.Barrier(8)
    results: list[list[tuple[object, ...]]] = []

    def search(view: PgVectorStore) -> None:
        barrier.wait()
        ranked, _ = cache.rank_bm25(view, "parser retry", k=100, stable_ties=True)
        results.append(_exact(ranked))

    # One store per thread, since the fixture's store holds a single connection. They share a
    # DSN, table and tenant, so they share one cache entry, exactly as tenant views do in serving.
    views = [PgVectorStore(TEST_DSN, dim=3, table=store.table) for _ in range(8)]
    try:
        threads = [threading.Thread(target=search, args=(view,)) for view in views]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        for view in views:
            view.close()

    assert cache.stats["bm25_rebuild"] == 1
    assert len(results) == 8 and all(result == results[0] for result in results)
    assert results[0] == _reference(store, "parser retry")


def _supersession_rows() -> list[Chunk]:
    def row(chunk_id: str, supersedes: object, segment: int) -> Chunk:
        metadata: dict[str, object] = {"source_session_id": "s", "segment": segment}
        if supersedes is not ...:
            metadata["supersedes"] = supersedes
        return Chunk(chunk_id, "src", f"text {chunk_id}", metadata)

    return [
        row("m1", ["old-1", "old-2", "old-1"], 0),
        row("m2", ["", None, "old-3"], 1),
        row("m3", "not-an-array", 2),
        row("m4", {"old-9": True}, 3),
        row("m5", [], 4),
        row("m6", ..., 5),
    ]


@requires_db
def test_cached_supersession_is_the_store_scan_through_writes(make_store) -> None:
    store = make_store(3)
    other = PgVectorStore(TEST_DSN, dim=3, table=store.table)
    try:
        rows = _supersession_rows()
        store.upsert(rows, _vectors(rows))
        cache = TenantSearchCache()

        first = cache.superseded_ids(store)
        assert first == store.explicit_superseded_chunk_ids() == {"old-1", "old-2", "old-3"}
        assert cache.superseded_ids(store) == first
        assert cache.stats["supersession_hit"] == 1

        newer = [Chunk("m7", "src", "text m7", {"supersedes": ["old-4"]})]
        other.upsert(newer, _vectors(newer))  # another process: nobody invalidates this cache
        assert cache.superseded_ids(store) == store.explicit_superseded_chunk_ids()
        assert "old-4" in cache.superseded_ids(store)

        other.delete_sources(["src"])
        assert cache.superseded_ids(store) == store.explicit_superseded_chunk_ids() == frozenset()
    finally:
        other.close()


# The graph sidecar reuses the primary query vector only when it is the same embedder's.


class _CountingEmbedder:
    dim = 3

    def __init__(self, name: str, vector: list[float]) -> None:
        self.name = name
        self.vector = vector
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return list(self.vector)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [list(self.vector) for _ in texts]


class _Reranker:
    def rerank(self, _query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        return hits


class _SpyStore:
    """A tenant store whose reads are fixed; it records the vector each dense read was given."""

    def __init__(self, repository: "_SpyRepository", tenant: str) -> None:
        self.repository = repository
        self.tenant = tenant

    def _rows(self) -> list[Chunk]:
        return sorted(self.repository.chunks[self.tenant].values(), key=lambda row: row.id)

    def query_dense_exact(self, vector: list[float], k: int) -> list[ScoredChunk]:
        self.repository.dense_vectors.append((self.tenant, list(vector)))
        return [ScoredChunk(row, 0.5) for row in self._rows()[:k]]

    query_dense = query_dense_exact

    def query_sparse(self, query: str, k: int, vec: list[float] | None = None) -> list[ScoredChunk]:
        return []

    def iter_chunks(self, batch_size: int = 256):
        yield from self._rows()

    def explicit_superseded_chunk_ids(self) -> frozenset[str]:
        return frozenset()

    def chunks_by_ids(self, ids):
        rows = self.repository.chunks[self.tenant]
        return {chunk_id: rows[chunk_id] for chunk_id in ids if chunk_id in rows}


class _SpyRepository:
    def __init__(self) -> None:
        self.chunks: dict[str, dict[str, Chunk]] = defaultdict(dict)
        self.dense_vectors: list[tuple[str, list[float]]] = []

    def tenant_store(self, tenant: str) -> _SpyStore:
        return _SpyStore(self, tenant)

    def specialist_store(self, tenant: str, profile: str) -> _SpyStore:
        return _SpyStore(self, "specialist:" + tenant)

    def graph_store(self, tenant: str) -> _SpyStore:
        return _SpyStore(self, "graph:" + tenant)

    def atomic_view_store(self, scope: str) -> _SpyStore:
        return _SpyStore(self, "views:" + scope)

    def corpus_status(self, tenant: str) -> dict[str, object]:
        return {"generation_id": "g", "corpus_sha256": "f" * 64}

    def graph_corpus_status(self, tenant: str) -> dict[str, object]:
        raise RuntimeError("not modelled")


def _spy_service() -> tuple[HostedService, _SpyRepository, _CountingEmbedder, _CountingEmbedder]:
    repository = _SpyRepository()
    code = _CountingEmbedder("code", [1.0, 0.0, 0.0])
    context = _CountingEmbedder("context", [0.0, 1.0, 0.0])
    service = HostedService(
        repository,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        HostedRetriever(code, _Reranker()),  # type: ignore[arg-type]
        behavior=C9,
        multimodal_embedder=object(),  # type: ignore[arg-type]
        specialist_retrievers={
            C9.context_embedding_profile: HostedRetriever(context, _Reranker())  # type: ignore[arg-type]
        },
    )
    return service, repository, code, context


def _graph_vectors(repository: _SpyRepository) -> list[list[float]]:
    return [vector for tenant, vector in repository.dense_vectors if tenant.startswith("graph:")]


@pytest.mark.anyio
async def test_the_code_route_embeds_its_query_once_and_the_sidecar_reuses_it() -> None:
    service, repository, code, context = _spy_service()

    response = await service.search(SearchRequest(query="fix the parser bug", user_id="u", top_k=5))

    assert response.specialist_route == "code"
    assert response.graph_attempted is True and response.graph_fallback is False
    assert code.queries == ["fix the parser bug"]
    assert context.queries == []
    assert _graph_vectors(repository) == [[1.0, 0.0, 0.0]]


@pytest.mark.anyio
async def test_the_context_route_sidecar_still_embeds_with_the_code_embedder() -> None:
    service, repository, code, context = _spy_service()

    response = await service.search(
        SearchRequest(query="what did we decide in the meeting", user_id="u", top_k=5)
    )

    assert response.specialist_route == "context"
    assert response.graph_attempted is True and response.graph_fallback is False
    assert context.queries == ["what did we decide in the meeting"]
    assert code.queries == ["what did we decide in the meeting"]
    assert _graph_vectors(repository) == [[1.0, 0.0, 0.0]]


# Corpus status: one computation per miss, and never cached across a write.


class _StatusRepository:
    def __init__(self) -> None:
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()

    def corpus_status(self, tenant: str) -> dict[str, object]:
        with self._lock:
            self.calls += 1
            call = self.calls
        self.started.set()
        assert self.release.wait(10)
        return {"generation_id": "g", "corpus_sha256": f"{call:064d}", "call": call}


def _status_service() -> tuple[HostedService, _StatusRepository]:
    repository = _StatusRepository()
    service = HostedService(
        repository,  # type: ignore[arg-type]
        None,
        HostedRetriever(_CountingEmbedder("code", [1.0, 0.0, 0.0]), _Reranker()),  # type: ignore[arg-type]
        behavior=variant("A0_raw"),
    )
    return service, repository


@pytest.mark.anyio
async def test_concurrent_status_misses_share_one_computation() -> None:
    service, repository = _status_service()

    pending = [asyncio.ensure_future(service.corpus_status("u")) for _ in range(8)]
    await asyncio.to_thread(repository.started.wait, 10)
    await asyncio.sleep(0.05)
    repository.release.set()
    results = await asyncio.gather(*pending)

    assert repository.calls == 1
    assert all(result == results[0] for result in results)
    assert await service.corpus_status("u") == results[0]
    assert repository.calls == 1


@pytest.mark.anyio
async def test_a_write_during_a_status_computation_is_never_hidden_by_it() -> None:
    """A status read before a write must not be cached, nor joined by a Search after the write."""
    service, repository = _status_service()
    tenant = tenant_for("u")

    before = asyncio.ensure_future(service.corpus_status("u"))
    await asyncio.to_thread(repository.started.wait, 10)
    service._invalidate_corpus_status(tenant)  # an Add finished while the read was running
    after = asyncio.ensure_future(service.corpus_status("u"))
    await asyncio.sleep(0.05)
    repository.release.set()

    assert (await before)["call"] == 1
    assert (await after)["call"] == 2
    assert (await service.corpus_status("u"))["call"] == 2
    assert repository.calls == 2


# End to end on a real database: the served C9 response is byte-identical to the uncached code.


def _hash_vector(salt: str, text: str, dim: int) -> list[float]:
    """Deterministic, and deliberately NOT representable in float32, like a provider vector."""
    digest = hashlib.sha256((salt + "\0" + text).encode("utf-8")).digest()
    return [
        int.from_bytes(digest[4 * i : 4 * i + 4], "little") / 2**32 * 2 - 1 + 1e-12
        for i in range(dim)
    ]


class _HashEmbedder:
    def __init__(self, name: str, dim: int) -> None:
        self.name = name
        self.dim = dim
        self.queries = 0

    def embed_query(self, text: str) -> list[float]:
        self.queries += 1
        return _hash_vector(self.name + ":query", text, self.dim)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [_hash_vector(self.name + ":passage", text, self.dim) for text in texts]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_passages(texts)


class _RecordCompiler:
    """One grounded record per Add, superseding the session's earlier records."""

    def compile_anchored_v3(self, messages, session_id, prior):
        quote = messages[0].content[:40]
        return [
            CodingMemoryRecord(
                kind="successful repair",
                problem=f"problem seen in {session_id}",
                action="changed the parser retry policy",
                outcome="the flaky test passed",
                entities=["parser"],
                evidence_spans=[
                    EvidenceSpan(message_ordinal=0, start=0, end=len(quote), quote=quote)
                ],
                source_session_id=session_id,
                supersedes=[record.id for record in prior][:2],
            )
        ]


class _ReferenceCache:
    """The uncached reads, exactly as ``HostedRetriever`` made them before the caches existed."""

    def rank_bm25(self, store, query, *, k, stable_ties):
        return rank_bm25_chunks(list(store.iter_chunks()), query, k=k, stable_ties=stable_ties), "none"

    def superseded_ids(self, store):
        return store.explicit_superseded_chunk_ids()

    def invalidate(self, tenants, *, drop=False):
        return None


class _ReferenceRetriever(HostedRetriever):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._search_cache = _ReferenceCache()  # type: ignore[assignment]

    def apply_graph_sidecar(self, store, query, run: RetrievalRun, *, query_vector=None):
        # The old sidecar embedded the query itself, whatever the caller already had.
        return super().apply_graph_sidecar(store, query, run)


def _messages(session: int, count: int) -> list[Message]:
    rng = random.Random(session)
    return [
        Message(
            role="user" if index % 2 == 0 else "assistant",
            content=" ".join(rng.choice(_WORDS) for _ in range(rng.randint(20, 90)))
            + f" step {session}-{index}.",
        )
        for index in range(count)
    ]


def _everything(response: SearchResponse) -> dict[str, object]:
    """Every field, including the diagnostics ``model_dump`` excludes from the wire body."""
    fields = {
        name: getattr(response, name) for name in type(response).model_fields if name != "data"
    }
    return {"data": response.model_dump(mode="json")["data"], **fields}


_E2E_QUERIES = [
    "fix the parser regression in the retry handler",
    "what did we decide in the meeting about cache eviction",
    "json schema migration conflict",
    "remember the timeout policy we agreed",
]


@requires_db
def test_c9_search_is_byte_identical_to_the_uncached_code_across_adds_and_deletes(
    make_store, tmp_path: Path
) -> None:
    fixture_store = make_store(8)
    pool = SharedPool(TEST_DSN, min_size=1, max_size=8)
    serving_store = PgVectorStore(
        TEST_DSN,
        8,
        table=fixture_store.table,
        tenant="aml_service_readiness",
        shared_pool=pool,
        owns_pool=True,
    )
    profile = C9.context_embedding_profile

    def embedders(label: str) -> tuple[CachedEmbedder, CachedEmbedder]:
        # A fresh cache per service, as production has one: the first embedding of a query is a
        # miss (full precision) and the uncached sidecar's second one a hit (float32 from SQLite).
        return (
            CachedEmbedder(_HashEmbedder("code", 8), tmp_path / f"{label}-code.sqlite"),  # type: ignore[arg-type]
            CachedEmbedder(_HashEmbedder("context", 8), tmp_path / f"{label}-context.sqlite"),  # type: ignore[arg-type]
        )

    def service(label: str, retriever_type: type[HostedRetriever]) -> HostedService:
        code, context = embedders(label)
        return HostedService(
            repository,
            _RecordCompiler(),  # type: ignore[arg-type]
            retriever_type(code, _Reranker()),  # type: ignore[arg-type]
            behavior=C9,
            multimodal_embedder=object(),  # type: ignore[arg-type]
            specialist_retrievers={profile: retriever_type(context, _Reranker())},  # type: ignore[arg-type]
        )

    try:
        code, context = embedders("repository")
        repository = PgHostedRepository(serving_store, code, specialist_embedders={profile: context})
        cached = service("cached", HostedRetriever)
        reference = service("reference", _ReferenceRetriever)
        user = f"e2e-{fixture_store.table}"
        observed: list[dict[str, object]] = []

        def add(through: HostedService, request_id: str, session: str, count: int) -> None:
            asyncio.run(
                through.add(
                    AddRequest(
                        request_id=request_id,
                        user_id=user,
                        session_id=session,
                        messages=_messages(int(hashlib.sha256(request_id.encode()).hexdigest(), 16) % 1000, count),
                    )
                )
            )

        def compare() -> None:
            for query in _E2E_QUERIES:
                # Each instance's corpus STATUS cache is invalidated only by its own writes, before
                # and after this change alike, so the reference instance would report a stale
                # ``corpus_sha256`` after an Add made through the other one. That field is a
                # diagnostic excluded from the wire body; starting both cold keeps it comparable,
                # and the status cache has its own tests above.
                cached._corpus_status_cache.clear()
                reference._corpus_status_cache.clear()
                request = SearchRequest(query=query, user_id=user, top_k=100)
                served = _everything(asyncio.run(cached.search(request)))
                expected = _everything(asyncio.run(reference.search(request)))
                assert served == expected
                observed.append(served)

        compare()  # an empty tenant
        add(cached, "r1", "session-a", 6)
        compare()
        compare()  # served from warm caches
        add(cached, "r2", "session-a", 5)  # supersedes r1's record
        compare()
        add(reference, "r3", "session-b", 7)  # a writer that never tells the cached service
        compare()
        add(cached, "r4", "session-b", 3)
        compare()
        assert asyncio.run(cached.delete_user(user)) > 0
        compare()
        add(reference, "r5", "session-c", 4)  # re-created behind the cached service's back
        compare()

        # Not vacuous: results, graph promotions and a warm cache all happened.
        stats = (
            cached._retriever._search_cache.stats  # type: ignore[attr-defined]
            + cached._specialist_retrievers[profile]._search_cache.stats  # type: ignore[attr-defined]
        )
        assert stats["bm25_hit"] and stats["bm25_incremental"] and stats["supersession_hit"], stats
        assert any(item["graph_promoted_count"] for item in observed)
        assert sum(len(item["data"]) for item in observed) > 100  # type: ignore[arg-type]
    finally:
        serving_store.close()
