"""Behavior proofs for C9's atomic views, built inside Add and read by Search.

The defect these guard against: C8's atomic stage read a file artifact built by hand after ingest
and bound to the served corpus fingerprint, so under AML's Add then Search contract (no pause, a new
fingerprint after every Add) it fell back on every query. C9 must rescue from views that the Add
itself wrote, with no artifact anywhere, including a Search between two Adds.

Red proofs, each a deliberate mutation of production code, run 2026-09-23 and reverted; every one
failed in the named assertion, not in setup:

* M1, deleting the ``await self._persist_atomic_views(tenant, chunks)`` call in
  ``HostedService._add_once``: ``test_c9_search_rescues_from_views_written_by_add_with_no_artifact``
  failed at the ``view_writes`` assertion (``[] == [...]``), and
  ``test_a_search_between_two_adds_sees_the_second_adds_views`` failed at
  ``after.atomic_rescue_candidate_available is True``, the behavioural one.
* M2, returning from ``_persist_atomic_views`` before the Context specialist write:
  ``test_a_search_between_two_adds_sees_the_second_adds_views`` failed at the same
  ``candidate_available`` assertion, so the Context-routed Search depends on the specialist views.
* M3, deleting the protected-parent ``continue`` in ``select_view_rescue``:
  ``test_selection_excludes_protected_parents_and_breaks_ties_deterministically`` failed with
  ``'parent-0' == 'parent-7'``.
* M4, swapping each view's parent with its neighbour window in ``build_view_chunks``:
  ``test_every_view_is_an_exact_word_range_inside_its_parent_window`` failed at the containment
  assertion (``0 <= -120``). Two cruder mappings (every view to window 0; every view one window
  back) were refused earlier by the builder's own per-parent bound, ``BuildRefusal``, so they are
  not counted as proofs of this assertion.
* M5, passing ``self._embedder`` for a specialist profile in
  ``PgHostedRepository.persist_atomic_views``:
  ``test_repository_embeds_views_with_the_scope_embedder_in_an_isolated_tenant`` failed with
  ``[1.0, 0.0, 0.0] == [0.0, 1.0, 0.0]``.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
import threading

import pytest

from recall.atomic_rescue import AtomicRescueSelectionError
from recall.pool import SharedPool
from recall.store import PgVectorStore
from recall.types import Chunk, ScoredChunk
from recall_aml.atomic_views import (
    ATOMIC_VIEW_RECORD_TYPE,
    build_view_chunks,
    max_views_per_parent,
    select_view_rescue,
    view_query_width,
)
from recall_aml.identity import atomic_view_tenant, specialist_tenant, tenant_for
from recall_aml.models import AddRequest, Message, SearchRequest
from recall_aml.retrieval import HostedRetriever
from recall_aml.service import HostedService, build_chunks
from recall_aml.storage import PgHostedRepository
from recall_aml.variants import variant
from tests.conftest import TEST_DSN, requires_db


C9 = variant("C9_routed_specialists_grounded_graph_atomic")
NEEDLE = "zanzibar"


def _session_messages(count: int, *, needle_in: int | None = None) -> list[str]:
    messages = []
    for index in range(count):
        words = [f"m{index}w{position}" for position in range(90)]
        if needle_in == index:
            words[40:44] = [NEEDLE, "cache", "eviction", "policy"]
        messages.append(" ".join(words) + f" finished step {index}.")
    return messages


def _raw_windows(session: str, messages: list[str]) -> list[Chunk]:
    request = AddRequest(
        request_id=f"req-{session}",
        user_id="user",
        session_id=session,
        messages=[Message(role="user", content=content) for content in messages],
    )
    chunks = build_chunks(
        request,
        [],
        word_window_size=C9.word_window_size,
        word_window_stride=C9.word_window_stride,
        content_only_windows=C9.content_only_windows,
        stable_window_identity=C9.stable_window_order,
    )
    return [chunk for chunk in chunks if chunk.metadata.get("record_type") == "raw"]


def test_every_view_is_an_exact_word_range_inside_its_parent_window() -> None:
    windows = _raw_windows("s-views", _session_messages(8))
    by_id = {chunk.id: chunk for chunk in windows}
    views = build_view_chunks(windows)

    assert views
    assert {view.metadata["parent_chunk_id"] for view in views} == set(by_id)
    for view in views:
        parent = by_id[view.metadata["parent_chunk_id"]]
        start = view.metadata["word_start"] - parent.metadata["word_start"]
        end = view.metadata["word_end"] - parent.metadata["word_start"]
        assert 0 <= start < end <= len(parent.text.split())
        assert parent.text.split()[start:end] == view.text.split()
        assert view.metadata["record_type"] == ATOMIC_VIEW_RECORD_TYPE
        assert view.source == parent.source


def test_view_ids_are_content_addressed_and_the_query_width_bounds_each_parent() -> None:
    windows = _raw_windows("s-bound", _session_messages(12))
    first = build_view_chunks(windows)
    second = build_view_chunks(list(reversed(windows)))
    assert [view.id for view in first] == [view.id for view in second]

    owned: dict[str, int] = defaultdict(int)
    for view in first:
        owned[view.metadata["parent_chunk_id"]] += 1
    assert max(owned.values()) <= max_views_per_parent(C9.word_window_size)
    assert view_query_width(C9.word_window_size) == 5 * max_views_per_parent(160) + 1


def test_a_request_too_short_for_any_view_yields_none_rather_than_failing() -> None:
    assert build_view_chunks(_raw_windows("s-short", ["ok."])) == []
    assert build_view_chunks([]) == []


def _view_hit(view_id: str, parent: str, score: float, *, parent_ordinal: int = 0) -> ScoredChunk:
    return ScoredChunk(
        Chunk(
            id=view_id,
            source="aml://session/x",
            text=view_id,
            metadata={
                "record_type": ATOMIC_VIEW_RECORD_TYPE,
                "parent_chunk_id": parent,
                "parent_ordinal": parent_ordinal,
                "view_ordinal": 0,
            },
        ),
        score,
    )


def _dense(count: int) -> list[ScoredChunk]:
    return [
        ScoredChunk(Chunk(id=f"parent-{index}", source="s", text="t", metadata={}), 1 - index / 100)
        for index in range(count)
    ]


def test_selection_excludes_protected_parents_and_breaks_ties_deterministically() -> None:
    views = [
        _view_hit("view-a", "parent-0", 0.99),
        _view_hit("view-b", "parent-8", 0.90, parent_ordinal=8),
        _view_hit("view-c", "parent-7", 0.90, parent_ordinal=7),
        _view_hit("view-d", "parent-9", 0.50, parent_ordinal=9),
    ]
    selection = select_view_rescue(views, _dense(10))

    assert selection.chunk_id == "parent-7"
    assert selection.score == pytest.approx(0.90)
    with pytest.raises(AtomicRescueSelectionError, match="five dense"):
        select_view_rescue(views, _dense(4))
    with pytest.raises(AtomicRescueSelectionError, match="outside dense top five"):
        select_view_rescue([_view_hit("view-a", "parent-0", 0.99)], _dense(10))


# A service-level fake. Raw windows holding the needle rank LAST, so the parent the rescue must
# find is outside every top five; a view scores by whether it holds the needle, which stands in for
# a vector. The queries share no word with the corpus, so BM25 cannot lift the needle either.


class _Embedder:
    dim = 3

    def __init__(self, name: str) -> None:
        self.name = name

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def embed_passages(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class _Reranker:
    def rerank(self, query, hits):
        return hits


class _Compiler:
    def compile_anchored_v3(self, messages, session_id, prior):
        raise RuntimeError("no model in this test; C9 drops the deterministic fallback")


class _Store:
    def __init__(self, repository: "_Repository", tenant: str) -> None:
        self.repository = repository
        self.tenant = tenant

    def _rows(self) -> list[Chunk]:
        return sorted(self.repository.chunks[self.tenant].values(), key=lambda item: item.id)

    def query_dense_exact(self, vector, k):
        rows = self._rows()
        if rows and rows[0].metadata.get("record_type") == ATOMIC_VIEW_RECORD_TYPE:
            scored = [ScoredChunk(row, 0.95 if NEEDLE in row.text else 0.1) for row in rows]
            scored.sort(key=lambda hit: (-hit.score, hit.chunk.id))
            return scored[:k]
        rows.sort(key=lambda row: (NEEDLE in row.text, row.id))
        return [ScoredChunk(chunk, 0.9 - index / 1000) for index, chunk in enumerate(rows[:k])]

    query_dense = query_dense_exact

    def query_sparse(self, query, k, vec=None):
        return []

    def iter_chunks(self, batch_size=256):
        yield from self._rows()

    def explicit_superseded_chunk_ids(self):
        return frozenset()

    def chunks_by_ids(self, ids):
        return {i: self.repository.chunks[self.tenant][i] for i in ids if i in self.repository.chunks[self.tenant]}


class _Repository:
    def __init__(self) -> None:
        self.chunks = defaultdict(dict)
        self.receipts = {}
        self.locks = defaultdict(threading.Lock)
        self.view_writes: list[tuple[str, str | None]] = []

    def tenant_store(self, tenant):
        return _Store(self, tenant)

    def specialist_store(self, tenant, profile):
        return self.tenant_store(specialist_tenant(tenant, profile))

    def graph_store(self, tenant):
        return self.tenant_store("graph:" + tenant)

    def atomic_view_store(self, scope_tenant):
        return self.tenant_store(atomic_view_tenant(scope_tenant))

    def acquire_request_lock(self, tenant, request_id):
        lock = self.locks[(tenant, request_id)]
        lock.acquire()
        return lock

    def release_request_lock(self, handle):
        handle.release()

    def get_receipt(self, tenant, request_id, fingerprint):
        return self.receipts.get((tenant, request_id, fingerprint))

    def record_receipt(self, tenant, request_id, fingerprint, result):
        self.receipts[(tenant, request_id, fingerprint)] = result

    def prior_records(self, tenant, source, *, graph_sidecar=False):
        return []

    def persist(self, tenant, chunks):
        for chunk in chunks:
            self.chunks[tenant][chunk.id] = chunk
        return len(chunks)

    def persist_graph(self, tenant, chunks):
        return self.persist("graph:" + tenant, chunks)

    def persist_specialist(self, tenant, profile, chunks):
        return self.persist(specialist_tenant(tenant, profile), chunks)

    def persist_atomic_views(self, tenant, profile, chunks):
        scope = tenant if profile is None else specialist_tenant(tenant, profile)
        self.view_writes.append((tenant, profile))
        return self.persist(atomic_view_tenant(scope), chunks)

    def corpus_status(self, tenant):
        return {"generation_id": "c9-test", "corpus_sha256": "f" * 64}

    def graph_corpus_status(self, tenant):
        raise RuntimeError("graph status is not modelled here")


def _c9_service() -> tuple[HostedService, _Repository]:
    repository = _Repository()
    service = HostedService(
        repository,  # type: ignore[arg-type]
        _Compiler(),  # type: ignore[arg-type]
        HostedRetriever(_Embedder("code"), _Reranker()),  # type: ignore[arg-type]
        behavior=C9,
        multimodal_embedder=object(),  # type: ignore[arg-type]
        specialist_retrievers={
            C9.context_embedding_profile: HostedRetriever(_Embedder("context"), _Reranker())  # type: ignore[arg-type]
        },
    )
    return service, repository


def _add(service: HostedService, session: str, messages: list[str]) -> None:
    asyncio.run(
        service.add(
            AddRequest(
                request_id=f"req-{session}",
                user_id="c9-user",
                session_id=session,
                messages=[Message(role="user", content=content) for content in messages],
            )
        )
    )


def _search(service: HostedService, query: str):
    return asyncio.run(
        service.search(SearchRequest(query=query, user_id="c9-user", top_k=100))
    )


def test_c9_search_rescues_from_views_written_by_add_with_no_artifact(monkeypatch) -> None:
    for name in (
        "RECALL_ATOMIC_RESCUE_MODE",
        "RECALL_ATOMIC_RESCUE_PLACEMENT",
        "RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)
    service, repository = _c9_service()
    _add(service, "s-one", _session_messages(12, needle_in=10))

    tenant = tenant_for("c9-user")
    assert repository.view_writes == [(tenant, None), (tenant, C9.context_embedding_profile)]
    response = _search(service, "Fix the parser.py regression")

    assert response.specialist_route == "code"
    assert response.atomic_rescue_attempted is True
    assert response.atomic_rescue_active is True
    assert response.atomic_rescue_fallback is False
    assert response.atomic_rescue_candidate_available is True
    rescued = response.data[5].content
    assert NEEDLE in rescued
    assert all(NEEDLE not in item.content for item in response.data[:5])


def test_a_search_between_two_adds_sees_the_second_adds_views(monkeypatch) -> None:
    monkeypatch.delenv("RECALL_ATOMIC_RESCUE_MODE", raising=False)
    monkeypatch.delenv("RECALL_ATOMIC_RESCUE_PLACEMENT", raising=False)
    service, _ = _c9_service()
    _add(service, "s-first", _session_messages(12))

    before = _search(service, "what did we decide in yesterday's meeting?")
    assert before.specialist_route == "context"
    assert all(NEEDLE not in item.content for item in before.data)

    _add(service, "s-second", _session_messages(3, needle_in=1))
    after = _search(service, "what did we decide in yesterday's meeting?")

    assert after.atomic_rescue_candidate_available is True
    assert after.atomic_rescue_fallback is False
    assert NEEDLE in after.data[5].content


def test_c8_under_the_same_flow_falls_back_which_is_the_defect_c9_removes(monkeypatch) -> None:
    monkeypatch.setenv("RECALL_ATOMIC_RESCUE_MODE", "active")
    monkeypatch.setenv("RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT", "/nonexistent-atomic-artifacts")
    repository = _Repository()
    c8 = variant("C8_routed_specialists_grounded_graph")
    service = HostedService(
        repository,  # type: ignore[arg-type]
        _Compiler(),  # type: ignore[arg-type]
        HostedRetriever(_Embedder("code"), _Reranker()),  # type: ignore[arg-type]
        behavior=c8,
        multimodal_embedder=object(),  # type: ignore[arg-type]
        specialist_retrievers={
            c8.context_embedding_profile: HostedRetriever(_Embedder("context"), _Reranker())  # type: ignore[arg-type]
        },
    )
    _add(service, "s-one", _session_messages(12, needle_in=10))
    response = _search(service, "Fix the parser.py regression")

    assert repository.view_writes == []
    assert response.atomic_rescue_active is False
    assert response.atomic_rescue_fallback is True


def test_version_reports_the_effective_atomic_stage(monkeypatch) -> None:
    monkeypatch.delenv("RECALL_ATOMIC_RESCUE_MODE", raising=False)
    monkeypatch.delenv("RECALL_ATOMIC_RESCUE_PLACEMENT", raising=False)
    service, _ = _c9_service()
    assert service.atomic_rescue_profile == {
        "enabled": True,
        "source": "add-time-views",
        "view_profile": "aml-add-micro-v1",
        "mode": "active",
        "placement": "fused",
    }
    monkeypatch.setenv("RECALL_ATOMIC_RESCUE_MODE", "off")
    assert service.atomic_rescue_profile["mode"] == "off"


class _EmbedderRecorder:
    dim = 3

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.name = "recorder"

    def embed_passages(self, texts):
        return [list(self.vector) for _ in texts]


class _UpsertStore:
    def __init__(self, root: "_BaseStore", tenant: str) -> None:
        self.root = root
        self.tenant = tenant

    def upsert(self, chunks, vectors):
        self.root.upserts[self.tenant] = (list(chunks), [list(vector) for vector in vectors])
        return len(chunks)


class _BaseStore:
    def __init__(self) -> None:
        self.upserts: dict[str, tuple[list[Chunk], list[list[float]]]] = {}

    def for_tenant(self, tenant):
        return _UpsertStore(self, tenant)


def test_repository_embeds_views_with_the_scope_embedder_in_an_isolated_tenant() -> None:
    base = _BaseStore()
    profile = C9.context_embedding_profile
    repository = PgHostedRepository(
        base,  # type: ignore[arg-type]
        _EmbedderRecorder([1.0, 0.0, 0.0]),  # type: ignore[arg-type]
        specialist_embedders={profile: _EmbedderRecorder([0.0, 1.0, 0.0])},  # type: ignore[dict-item]
    )
    views = build_view_chunks(_raw_windows("s-repo", _session_messages(4)))
    tenant = tenant_for("repo-user")

    repository.persist_atomic_views(tenant, None, views)
    repository.persist_atomic_views(tenant, profile, views)

    code_scope = atomic_view_tenant(tenant)
    context_scope = atomic_view_tenant(specialist_tenant(tenant, profile))
    assert set(base.upserts) == {code_scope, context_scope}
    assert tenant not in base.upserts
    assert base.upserts[code_scope][1][0] == [1.0, 0.0, 0.0]
    assert base.upserts[context_scope][1][0] == [0.0, 1.0, 0.0]
    assert {chunk.metadata["embedding_profile"] for chunk in base.upserts[context_scope][0]} == {
        profile
    }


# Real pgvector: rows round-trip through JSONB, live beside the corpus without entering it, answer
# the exact dense query Search issues, and die with the user.


@requires_db
def test_postgres_c9_add_writes_views_search_rescues_and_delete_removes_them(make_store) -> None:
    """The whole C9 path against a real store, with the three embedders faked.

    Red proof, 2026-09-23: removing ``atomic_view_tenant(tenant)`` and the specialist view tenants
    from ``PgHostedRepository.delete_tenant`` fails the final ``count() == 0`` assertion, because
    a deleted user's views would otherwise survive in the shared table.
    """
    fixture_store = make_store(3)
    pool = SharedPool(TEST_DSN, min_size=1, max_size=8)
    serving_store = PgVectorStore(
        TEST_DSN,
        3,
        table=fixture_store.table,
        tenant="aml_service_readiness",
        shared_pool=pool,
        owns_pool=True,
    )
    profile = C9.context_embedding_profile
    try:
        repository = PgHostedRepository(
            serving_store,
            _Embedder("code"),  # type: ignore[arg-type]
            specialist_embedders={profile: _Embedder("context")},  # type: ignore[dict-item]
        )
        service = HostedService(
            repository,
            _Compiler(),  # type: ignore[arg-type]
            HostedRetriever(_Embedder("code"), _Reranker()),  # type: ignore[arg-type]
            behavior=C9,
            multimodal_embedder=object(),  # type: ignore[arg-type]
            specialist_retrievers={
                profile: HostedRetriever(_Embedder("context"), _Reranker())  # type: ignore[arg-type]
            },
        )
        user = f"c9-pg-{fixture_store.table}"
        tenant = tenant_for(user)
        asyncio.run(
            service.add(
                AddRequest(
                    request_id="c9-pg-request",
                    user_id=user,
                    session_id="c9-pg-session",
                    messages=[
                        Message(role="user", content=content)
                        for content in _session_messages(12, needle_in=10)
                    ],
                )
            )
        )
        code_views = repository.atomic_view_store(tenant)
        context_views = repository.atomic_view_store(specialist_tenant(tenant, profile))
        raw_rows = list(repository.tenant_store(tenant).iter_chunks())

        assert code_views.count() > len(raw_rows) >= 6
        assert context_views.count() == code_views.count()
        assert all(row.metadata["record_type"] == "raw" for row in raw_rows)
        nearest = code_views.query_dense_exact([1.0, 0.0, 0.0], k=view_query_width(160))
        assert {hit.chunk.metadata["parent_chunk_id"] for hit in nearest} <= {
            row.id for row in raw_rows
        }

        found = asyncio.run(
            service.search(SearchRequest(query="Fix the parser.py regression", user_id=user, top_k=100))
        )
        assert found.atomic_rescue_active is True
        assert found.atomic_rescue_fallback is False
        assert found.atomic_rescue_candidate_available is True

        asyncio.run(service.delete_user(user))
        assert code_views.count() == 0
        assert context_views.count() == 0
    finally:
        serving_store.close()


def test_a_deterministic_view_refusal_does_not_fail_the_add(monkeypatch) -> None:
    """A retry cannot fix a refusal, so it must not turn one request into a failed official job.

    Red proof, 2026-09-23: removing the ``except BuildRefusal`` branch in
    ``HostedService._persist_atomic_views`` makes ``service.add`` raise ``BuildRefusal`` here.
    """
    import recall_aml.service as service_module
    from recall_aml.atomic_views import BuildRefusal

    def refuse(_chunks):
        raise BuildRefusal("windows disagree on their overlap")

    monkeypatch.setattr(service_module, "build_view_chunks", refuse)
    service, repository = _c9_service()
    _add(service, "s-refused", _session_messages(8))

    assert repository.view_writes == []
    assert repository.chunks[tenant_for("c9-user")]
