"""Is the SERVED system the MEASURED system?

`search_fused` cites +0.0084 nDCG@5 and +0.0842 R@100 from `mq_nested2_nogold`. If the retriever's
outer fusion ever diverges from the benchmark's, those numbers describe something the code no
longer computes, and nothing else in the suite would notice. This is the serving analogue of the
`validate` gate the benchmark itself had to have.

Two checks live here, not one. `test_serving_fusion_reproduces_the_benchmark_arm` pins the
retriever against the fixture; `test_fixture_matches_the_live_benchmark_arm` pins the fixture
against the live `fuse_arm`. The fixture is a frozen snapshot, so only the second check would
notice the benchmark's own fusion logic drifting out from under it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.mtrag.multiquery import POST_HOC_ARMS, fuse_arm
from recall.retriever import HybridRetriever
from tests.fakes import FakeEmbedder, FakeStore

FIXTURE = Path(__file__).parent / "fixtures" / "fused_parity.json"

#: The measured arm this fixture pins. Its two variants are "last" and "full", and its two legs
#: are named "dense" and "splade", the LEARNED sparse leg, not the lexical one: `HybridRetriever`
#: only reaches `query_learned_sparse` under `sparse_backend="splade"`.
_ARM_NAME = "mq_nested2_nogold"


class _PassThroughReranker:
    """Preserves the fused order, so the test compares FUSION rather than reranking."""

    def rerank(self, query, hits):
        return list(hits)


class _FakeSparseEncoder:
    """A minimal stand-in for `SpladeEncoder`: deterministic, non-empty term weights.

    `search_fused`'s SPLADE leg is reached only through `encoder.encode(...)` and
    `encoder.profile.profile_id`; this supplies both without loading a real model. Weights are
    keyed by text length modulo 7, the same trick `FakeEmbedder` already uses to keep two texts
    distinguishable, so the query and the history in this test always produce two different,
    non-empty weight dicts.
    """

    class _Profile:
        profile_id = "fake-sparse-v1"

    profile = _Profile()

    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        return [{1: float(len(t) % 7) + 1.0} for t in texts]


class _ScriptedStore(FakeStore):
    """Returns the fixture's leg rankings, selecting the variant per call.

    Keyed on the query VECTOR for the dense leg, and on the query TEXT (by way of its encoded
    SPLADE weights) for the learned sparse leg, never on a variant set by an earlier call.
    `_retrieve_legs` calls `query_dense` before `query_learned_sparse`, so a store that selected
    its variant in one and read it in the other would read a stale value and silently serve the
    wrong legs.
    """

    def __init__(self, legs: dict, qvec_last: list[float], qvec_full: list[float]) -> None:
        super().__init__()
        self._by_vec = {tuple(qvec_last): legs["last"], tuple(qvec_full): legs["full"]}
        self._by_weights: dict[frozenset, dict] = {}

    def bind_text(
        self, query: str, history: str, legs: dict, encoder: _FakeSparseEncoder
    ) -> None:
        q_weights = encoder.encode([query])[0]
        h_weights = encoder.encode([history])[0]
        assert q_weights != h_weights, (
            "the fixture needs two distinguishable SPLADE encodings; adjust the strings' lengths"
        )
        self._by_weights = {
            frozenset(q_weights.items()): legs["last"]["splade"],
            frozenset(h_weights.items()): legs["full"]["splade"],
        }

    def _rows(self, ids: list[str]) -> list:
        return self._hits([(cid, 0.5) for cid in ids])

    def query_dense(self, vector, k, source=None):
        return self._rows(self._by_vec[tuple(vector)]["dense"][:k])

    def query_learned_sparse(self, weights, k, profile_id, source=None, vec=None):
        self.learned_vec_used = vec
        return self._rows(self._by_weights[frozenset(weights.items())][:k])


#: Case 4 in the fixture is engineered, not sampled from a real dump. `RRF_K` at 60 sets every
#: id's Reciprocal Rank Fusion term to 1 / (60 + rank + 1); with the shallow ranks the other four
#: cases use, a one point change to `RRF_K` moves every term by too little to cross any of the
#: real gaps between them, so the cross-check below would stay green under that mutation even
#: though it is fusing something else. Case 4 places two ids, "rrfk_single_top" (one term, at
#: rank 0 of one variant) and "rrfk_full98" (two terms, at ranks 38 and 98 across both variants),
#: whose scores agree to five decimal places at RRF_K=60 and swap order at RRF_K=61, so a change
#: to that constant is guaranteed to move this case's output and is not guaranteed to move the
#: other four.
_FIXTURE_CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("index", range(len(_FIXTURE_CASES)))
def test_serving_fusion_reproduces_the_benchmark_arm(index: int) -> None:
    case = _FIXTURE_CASES[index]
    query, history_turn = "the question", "an earlier turn of different length"

    embedder = FakeEmbedder()
    qvec_last = embedder.embed([query])[0]
    qvec_full = embedder.embed([history_turn])[0]
    assert qvec_last != qvec_full, (
        "the fixture needs two distinguishable query vectors; adjust the strings' lengths"
    )

    encoder = _FakeSparseEncoder()
    store = _ScriptedStore(case["legs"], qvec_last, qvec_full)
    store.bind_text(query, history_turn, case["legs"], encoder)

    retriever = HybridRetriever(
        store, embedder, reranker=_PassThroughReranker(), sparse_backend="splade",
        sparse_encoder=encoder, candidate_k=100,
    )

    result = retriever.search_fused(query, [history_turn], k=100)

    # Full-length comparison, not a prefix: slicing `expected` to `len(result.hits)` would let a
    # regression that quietly drops trailing hits pass on a matching prefix, which is precisely
    # the drift this gate exists to catch.
    actual = [h.chunk.id for h in result.hits]
    assert len(actual) == len(case["expected"]), (
        f"search_fused returned {len(actual)} hits, the benchmark arm produced "
        f"{len(case['expected'])}; a shortfall here is a regression even where the shared "
        f"prefix still matches"
    )
    assert actual == case["expected"]


@pytest.mark.parametrize("index", range(len(_FIXTURE_CASES)))
def test_fixture_matches_the_live_benchmark_arm(index: int) -> None:
    """The fixture's `expected` is a frozen snapshot; check it against the LIVE `fuse_arm`.

    Without this, a change to `fuse_arm`, or to a constant such as `RRF_K` it depends on, that
    silently changes what `mq_nested2_nogold` computes would leave the fixture, and therefore the
    test above, asserting parity against a stale snapshot. This closes that gap: drift in either
    the retriever or the benchmark now goes red.
    """
    case = _FIXTURE_CASES[index]
    arm = next(a for a in POST_HOC_ARMS if a.name == _ARM_NAME)
    assert fuse_arm(arm, case["legs"]) == case["expected"]
