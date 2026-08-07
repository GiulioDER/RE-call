"""Is the SERVED system the MEASURED system?

`search_fused` cites +0.0084 nDCG@5 and +0.0842 R@100 from `mq_nested2_nogold`. If the retriever's
outer fusion ever diverges from the benchmark's, those numbers describe something the code no
longer computes, and nothing else in the suite would notice. This is the serving analogue of the
`validate` gate the benchmark itself had to have.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recall.retriever import HybridRetriever
from tests.fakes import FakeEmbedder, FakeStore

FIXTURE = Path(__file__).parent / "fixtures" / "fused_parity.json"


class _PassThroughReranker:
    """Preserves the fused order, so the test compares FUSION rather than reranking."""

    def rerank(self, query, hits):
        return list(hits)


class _ScriptedStore(FakeStore):
    """Returns the fixture's leg rankings, selecting the variant per call.

    Keyed on the query VECTOR, not on a variant set by an earlier call. `_retrieve_legs` calls
    `query_dense` BEFORE `query_sparse`, so a store that selected its variant in `query_sparse`
    and read it in `query_dense` would read a stale value and silently serve the wrong legs.
    """

    def __init__(self, legs: dict, qvec_last: list[float], qvec_full: list[float]) -> None:
        super().__init__()
        self._by_vec = {tuple(qvec_last): legs["last"], tuple(qvec_full): legs["full"]}
        self._by_text: dict[str, dict] = {}

    def bind_text(self, query: str, history: str, legs: dict) -> None:
        self._by_text = {query: legs["last"], history: legs["full"]}

    def _rows(self, ids: list[str]) -> list:
        return self._hits([(cid, 0.5) for cid in ids])

    def query_dense(self, vector, k, source=None):
        return self._rows(self._by_vec[tuple(vector)]["dense"][:k])

    def query_sparse(self, text, k, source=None, vec=None):
        self.sparse_vec_used = vec
        return self._rows(self._by_text[text]["splade"][:k])


@pytest.mark.parametrize("index", range(4))
def test_serving_fusion_reproduces_the_benchmark_arm(index: int) -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    case = data["cases"][index]
    query, history_turn = "the question", "an earlier turn of different length"

    embedder = FakeEmbedder()
    qvec_last = embedder.embed([query])[0]
    qvec_full = embedder.embed([history_turn])[0]
    assert qvec_last != qvec_full, (
        "the fixture needs two distinguishable query vectors; adjust the strings' lengths"
    )

    store = _ScriptedStore(case["legs"], qvec_last, qvec_full)
    store.bind_text(query, history_turn, case["legs"])

    retriever = HybridRetriever(
        store, embedder, reranker=_PassThroughReranker(), sparse_backend="lexical",
        candidate_k=100,
    )

    result = retriever.search_fused(query, [history_turn], k=100)

    assert [h.chunk.id for h in result.hits] == case["expected"][: len(result.hits)]
