"""The lexical baseline, and the streaming read it is built on.

These tests exist because the BM25 arm is what every published retrieval number in this repo is
now measured *against*. A baseline that is quietly wrong does not fail loudly — it makes the
system under test look better or worse by a margin nobody can see, which is worse than having no
baseline at all. So they assert the ranking property, the IDF property, and the two ways the
underlying read could silently return the wrong set of chunks.
"""
from __future__ import annotations

import pytest

from recall.eval.bm25 import BM25Retriever, tokenize
from recall.retriever import HybridRetriever
from recall.store import PgVectorStore
from recall.types import Chunk

from tests.conftest import TEST_DSN, requires_db


def _vec(i: int, dim: int = 4) -> list[float]:
    v = [0.0] * dim
    v[i % dim] = 1.0
    return v


# ------------------------------------------------------------------------------------------
# iter_chunks: the read the baseline is built from
# ------------------------------------------------------------------------------------------


@requires_db
def test_iter_chunks_yields_every_chunk_once(make_store):
    store = make_store(4)
    store.upsert(
        [Chunk(f"c{i}", f"s{i}.md", f"body {i}", {"file": f"s{i}.md"}) for i in range(25)],
        [_vec(i) for i in range(25)],
    )
    ids = [c.id for c in store.iter_chunks(batch_size=4)]

    # Exactly once each: a batch size that does not divide the row count is where an off-by-one
    # in cursor paging would drop or repeat the final partial batch.
    assert sorted(ids) == sorted(f"c{i}" for i in range(25))
    assert len(ids) == len(set(ids))


@requires_db
def test_iter_chunks_carries_text_and_metadata(make_store):
    store = make_store(4)
    store.upsert([Chunk("c1", "a.md", "the body text", {"file": "a.md", "ord": 3})], [_vec(0)])
    (chunk,) = list(store.iter_chunks())
    assert chunk.text == "the body text"
    assert chunk.metadata["file"] == "a.md"
    assert chunk.metadata["ord"] == 3


@requires_db
def test_iter_chunks_is_tenant_scoped(make_store):
    """The other tenant's row must EXIST before we assert it is invisible.

    Without that assertion a silently failed write makes this test green while proving nothing —
    the same trap the cross-tenant retrieval tests document.
    """
    store = make_store(4)
    other = PgVectorStore(TEST_DSN, dim=4, table=store.table, tenant="other-tenant")
    try:
        other.ensure_schema()
        store.upsert([Chunk("mine", "a.md", "my memory", {"file": "a.md"})], [_vec(0)])
        other.upsert([Chunk("theirs", "b.md", "their memory", {"file": "b.md"})], [_vec(1)])

        assert other.count() == 1, "the other tenant's row was not written; the test is vacuous"
        assert [c.id for c in store.iter_chunks()] == ["mine"]
        assert [c.id for c in other.iter_chunks()] == ["theirs"]
    finally:
        other.close()


@requires_db
def test_iter_chunks_rejects_a_nonpositive_batch_size(make_store):
    store = make_store(4)
    # A generator body does not run until first iteration, so the guard has to be observable
    # through consumption — asserting on the bare call would pass even with no guard at all.
    with pytest.raises(ValueError, match="batch_size"):
        list(store.iter_chunks(batch_size=0))


# ------------------------------------------------------------------------------------------
# BM25 itself
# ------------------------------------------------------------------------------------------


def test_tokenize_lowercases_and_splits_on_non_alphanumeric():
    assert tokenize("Rate-Limits: 100/sec (v2)!") == ["rate", "limits", "100", "sec", "v2"]


@requires_db
def test_bm25_ranks_the_keyword_match_first(make_store):
    store = make_store(4)
    store.upsert(
        [
            Chunk("c1", "cache.md", "we chose redis for the cache layer", {"file": "cache.md"}),
            Chunk("c2", "auth.md", "tokens are verified against the registry", {"file": "auth.md"}),
            Chunk("c3", "infra.md", "the servers run in one region", {"file": "infra.md"}),
        ],
        [_vec(0), _vec(1), _vec(2)],
    )
    hits = BM25Retriever(store).search("which cache did we choose", k=3).hits
    assert hits[0].chunk.metadata["file"] == "cache.md"
    assert hits[0].score > 0.0


@requires_db
def test_bm25_weights_a_corpus_wide_term_far_below_a_rare_one(make_store):
    """The IDF property — the one that separates BM25 from raw term frequency.

    Note what is asserted and what is not. In the Lucene/Robertson variant used here, a term
    present in EVERY document does not score exactly zero — `ln(1 + 0.5/(N + 0.5))` is small and
    positive, which is the point of the `+1` inside the log (it keeps a common term from going
    negative). Asserting `== 0` would be asserting a different formula's behaviour.

    The comparison is made WITHIN ONE DOCUMENT, at one occurrence each, so the only thing that
    differs between the two numbers is the IDF. Comparing across documents instead would fold in
    term frequency and length normalisation — a doc stuffed with "the" scores 0.18× a rare term
    rather than 0.14×, and a test written that way is really asserting a tf-saturation constant.
    """
    store = make_store(4)
    store.upsert(
        [
            Chunk("c1", "a.md", "the the the the alpha", {"file": "a.md"}),
            Chunk("c2", "b.md", "the beta", {"file": "b.md"}),
            Chunk("c3", "c.md", "the gamma", {"file": "c.md"}),
        ],
        [_vec(0), _vec(1), _vec(2)],
    )
    bm25 = BM25Retriever(store)
    # `iter_chunks` orders by id, so index 1 is c2 — "the beta", one occurrence of each term.
    common, rare = bm25.score("the")[1], bm25.score("beta")[1]
    assert 0.0 < common < 0.2 * rare

    # The consequence that actually matters: the document stuffed with "the" must not win a
    # query for a rare term it does not contain.
    hits = bm25.search("beta", k=3).hits
    assert hits[0].chunk.metadata["file"] == "b.md"


@requires_db
def test_bm25_returns_nothing_rather_than_dividing_by_zero_on_an_empty_corpus(make_store):
    store = make_store(4)
    assert len(BM25Retriever(store)) == 0
    assert BM25Retriever(store).search("anything", k=5).hits == []


@requires_db
def test_bm25_filters_by_source(make_store):
    store = make_store(4)
    store.upsert(
        [
            Chunk("c1", "a.md", "the cache is redis", {"file": "a.md"}),
            Chunk("c2", "b.md", "the cache is redis", {"file": "b.md"}),
        ],
        [_vec(0), _vec(1)],
    )
    hits = BM25Retriever(store).search("cache", k=5, source="b.md").hits
    assert [h.chunk.source for h in hits] == ["b.md"]


@requires_db
def test_bm25_rejects_a_nonpositive_k(make_store):
    with pytest.raises(ValueError, match="k must be >= 1"):
        BM25Retriever(make_store(4)).search("q", k=0)


# ------------------------------------------------------------------------------------------
# The ablation switch the baseline table needs
# ------------------------------------------------------------------------------------------


def test_a_retriever_with_neither_leg_is_refused():
    """`use_dense=False, use_sparse=False` would retrieve nothing and report no gap.

    Silently returning an empty result would read in an evaluation table as an arm that scored
    0.00 — a measurement — rather than as a misconfiguration.
    """
    with pytest.raises(ValueError, match="use_dense"):
        HybridRetriever(None, None, use_dense=False, use_sparse=False)  # type: ignore[arg-type]


@requires_db
def test_sparse_only_retrieval_finds_a_lexical_match(make_store):
    """The sparse-only arm has to actually retrieve, or its row in the table is a floor of 0."""
    from recall.embeddings import HashingEmbedder

    emb = HashingEmbedder(dim=4)
    store = make_store(4)
    store.upsert(
        [
            Chunk("c1", "a.md", "the deployment uses kubernetes", {"file": "a.md"}),
            Chunk("c2", "b.md", "unrelated prose about gardening", {"file": "b.md"}),
        ],
        [_vec(0), _vec(1)],
    )
    result = HybridRetriever(store, emb, use_dense=False).search("kubernetes", k=2)
    assert [h.chunk.metadata["file"] for h in result.hits][:1] == ["a.md"]
    # Documented consequence, pinned so nobody later "fixes" it into silence: `gap_warning` is
    # computed from dense scores, of which this arm produces none, and an empty candidate set is
    # a gap. So it fires even on this clean lexical hit. Fail-closed, and uninformative here — it
    # reports "no dense evidence gathered", not "the corpus lacks an answer".
    assert result.gap_warning is True
