"""Atomic rescue runs on the unscoped primary search only, never inside an expansion's search.

`trusted_search` hands the same retriever to document expansion, which searches again once per
expanded source with `source=...`. The rescue selects from the whole tenant, so running it there
would add its cost per expanded source, could insert a parent from another source into a list that
is meant to hold one, and raises outright on a source with fewer than five dense hits.

Red proof (2026-09-23), against the implementation before this change (`cdcfa23d`, where
`HybridRetriever._retrieve_legs` and `HybridRetriever.search` applied the transforms
unconditionally):

* ``tests/test_atomic_rescue_expansion_scope.py::test_rescue_runs_once_when_document_expansion_searches_again[dense]``
  failed with ``assert [None, 'a.md', 'b.md'] == [None]``: the dense transform ran on the primary
  search and again inside both source-scoped expansion searches.
* ``...[fused]`` failed the same way for the post-fusion transform.
* ``tests/test_atomic_rescue_expansion_scope.py::test_a_scoped_search_leaves_the_rescue_out``
  failed with ``assert ['a.md'] == []``.

Each was then run against a deliberate mutation of the fix. Removing the `is_empty` gate from
`HybridRetriever._retrieve_legs` fails the ``[dense]`` case and
``test_a_scoped_search_leaves_the_rescue_out``; removing it from `HybridRetriever.search` fails the
``[fused]`` case. Each failed with the same assertion as above.
"""

from __future__ import annotations

import pytest

from recall.calibration import Calibration
from recall.retriever import DocumentExpansionPolicy, HybridRetriever
from recall.trust import TrustPolicy, trusted_search
from recall.types import Chunk, ScoredChunk
from tests.test_atomic_rescue_production_shadow import _ActiveStore, _Embedder


def _chunk(chunk_id: str, source: str) -> Chunk:
    return Chunk(chunk_id, source, chunk_id, {"file": source})


class _TwoSourceStore(_ActiveStore):
    """Seven chunks over two sources, and a dense leg that honours `source=` as a real store does."""

    CHUNKS = [
        *(_chunk(f"a-{index}", "a.md") for index in range(1, 5)),
        *(_chunk(f"b-{index}", "b.md") for index in range(1, 4)),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.dense_sources: list[str | None] = []

    def query_dense(self, vector, k, source=None, scope=None):
        del vector, scope
        self.dense_sources.append(source)
        return [
            ScoredChunk(chunk, 1.0 - index / 100)
            for index, chunk in enumerate(self.CHUNKS)
            if source is None or chunk.source == source
        ][:k]

    def cosines_for(self, ids, vec):
        del vec
        return {chunk_id: 0.4 for chunk_id in ids}


def _search(
    monkeypatch, tmp_path, placement: str
) -> tuple[list[str | None], list[str], list[str | None]]:
    from recall import trust

    class Artifact:
        def assert_lineage(self, **kwargs):
            del kwargs

    store = _TwoSourceStore()
    ran_on: list[str | None] = []

    def scope_of(dense):
        # The primary search sees both sources; an expansion search sees exactly one.
        sources = {hit.chunk.source for hit in dense}
        return None if len(sources) > 1 else next(iter(sources))

    def dense_insert(artifact, query_vector, dense, loader):
        del artifact, query_vector, loader
        ran_on.append(scope_of(dense))
        return list(dense)

    def fused_insert(artifact, query_vector, dense, ranked, loader):
        del artifact, query_vector, loader
        ran_on.append(scope_of(dense))
        return list(ranked)

    monkeypatch.setattr(trust, "load_atomic_rescue_artifact", lambda path: Artifact())
    monkeypatch.setattr(trust, "insert_atomic_rescue_dense", dense_insert)
    monkeypatch.setattr(trust, "insert_atomic_rescue_fused", fused_insert)
    result = trusted_search(
        store,
        _Embedder(),
        "query",
        k=7,
        candidate_k=7,
        calibration=Calibration(embedder="test-profile", threshold=0.1, scale=0.1),
        policy=TrustPolicy.development(),
        env={
            "RECALL_ATOMIC_RESCUE_MODE": "active",
            "RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT": str(tmp_path),
            "RECALL_ATOMIC_RESCUE_PLACEMENT": placement,
        },
        document_expansion=DocumentExpansionPolicy(
            enabled=True, max_sources=2, chunks_per_source=3, relational_query_only=False
        ),
    )
    return ran_on, [hit.chunk.id for hit in result.hits], store.dense_sources


@pytest.mark.parametrize("placement", ["dense", "fused"])
def test_rescue_runs_once_when_document_expansion_searches_again(
    monkeypatch, tmp_path, placement
) -> None:
    ran_on, hits, dense_sources = _search(monkeypatch, tmp_path, placement)
    # The expansion really did search each source again, so a pass is not an idle expansion.
    assert dense_sources == [None, "a.md", "b.md"]
    assert ran_on == [None]
    assert hits


def test_a_scoped_search_leaves_the_rescue_out() -> None:
    store = _TwoSourceStore()
    ran_on: list[str] = []

    def dense_transform(vector, dense):
        del vector
        ran_on.append(dense[0].chunk.source)
        return list(dense)

    retriever = HybridRetriever(store, _Embedder(), dense_transform=dense_transform)
    result = retriever.search("query", k=3, source="a.md")
    assert [hit.chunk.id for hit in result.hits] == ["a-1", "a-2", "a-3"]
    assert ran_on == []
