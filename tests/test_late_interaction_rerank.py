import numpy as np
import pytest

from recall.rerank import (
    DEFAULT_LATE_INTERACTION_MODEL,
    LATE_INTERACTION_MODELS,
    PERMISSIVE_LICENCES,
    LateInteractionReranker,
    Reranker,
    late_interaction_licence,
    maxsim,
)
from recall.types import Chunk, ScoredChunk


def test_maxsim_matches_hand_computed_value():
    # q0 . d0 = 1.0, q0 . d1 = 0.6  -> max 1.0
    # q1 . d0 = 0.0, q1 . d1 = 0.8  -> max 0.8
    # sum = 1.8
    query = np.array([[1.0, 0.0], [0.0, 1.0]])
    doc = np.array([[1.0, 0.0], [0.6, 0.8]])
    assert maxsim(query, doc) == pytest.approx(1.8)


def test_maxsim_is_max_not_mean():
    """The mutation check in G5 relies on these differing. If they ever coincide the gate is
    vacuous, so the difference is pinned here rather than assumed."""
    query = np.array([[1.0, 0.0], [0.0, 1.0]])
    doc = np.array([[1.0, 0.0], [0.6, 0.8]])
    mean_version = float((query @ doc.T).mean(axis=1).sum())
    assert mean_version == pytest.approx(1.2)
    assert maxsim(query, doc) != pytest.approx(mean_version)


def test_maxsim_refuses_empty_document():
    """A document with no tokens cannot be scored. Returning 0.0 would place it mid-ranking,
    which is the same silent-corruption shape `rerank_order` refuses a missing score for."""
    query = np.array([[1.0, 0.0]])
    with pytest.raises(ValueError, match="no tokens"):
        maxsim(query, np.zeros((0, 2)))


def test_maxsim_refuses_empty_query():
    with pytest.raises(ValueError, match="no tokens"):
        maxsim(np.zeros((0, 2)), np.array([[1.0, 0.0]]))


def test_maxsim_refuses_dimension_mismatch():
    with pytest.raises(ValueError, match="dimension"):
        maxsim(np.array([[1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]))


def test_mit_is_permissive():
    """The load-bearing correction to `sparse.py:195`, which gates on `!= "apache-2.0"` and would
    therefore refuse the MIT primary arm under its own guard."""
    assert "mit" in PERMISSIVE_LICENCES
    assert late_interaction_licence("colbert-ir/colbertv2.0") == "mit"


def test_apache_is_permissive():
    assert late_interaction_licence("answerdotai/answerai-colbert-small-v1") == "apache-2.0"


def test_default_model_is_permissive():
    assert LATE_INTERACTION_MODELS[DEFAULT_LATE_INTERACTION_MODEL] in PERMISSIVE_LICENCES


def test_noncommercial_refused_without_optin():
    with pytest.raises(ValueError, match="cc-by-nc-4.0"):
        late_interaction_licence("jinaai/jina-colbert-v2")


def test_noncommercial_allowed_with_optin():
    assert late_interaction_licence(
        "jinaai/jina-colbert-v2", accept_noncommercial_license=True
    ) == "cc-by-nc-4.0"


def test_unknown_checkpoint_refused():
    """An unrecorded licence is exactly what this check exists to prevent, so an unknown model
    raises even though it might be perfectly permissive."""
    with pytest.raises(ValueError, match="unknown late-interaction model"):
        late_interaction_licence("some/unrecorded-colbert")


def test_unknown_checkpoint_refused_even_with_optin():
    """The opt-in waives the LICENCE check, not the REGISTRY check."""
    with pytest.raises(ValueError, match="unknown late-interaction model"):
        late_interaction_licence("some/unrecorded-colbert", accept_noncommercial_license=True)




class _FakeEncoder:
    """Returns pre-set token matrices by text. Records which method each text went through, so a
    test can prove queries use `query_embed` and documents use `passage_embed`."""

    def __init__(self, table: dict[str, list[list[float]]]) -> None:
        self._table = table
        self.query_calls: list[str] = []
        self.passage_calls: list[str] = []

    def query_embed(self, texts):
        texts = list(texts)
        self.query_calls.extend(texts)
        return [np.array(self._table[t]) for t in texts]

    def passage_embed(self, texts):
        texts = list(texts)
        self.passage_calls.extend(texts)
        return [np.array(self._table[t]) for t in texts]


def _hit(cid: str, text: str, score: float) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(id=cid, source="f", text=text), score=score)


def _reranker(table):
    return LateInteractionReranker(_FakeEncoder(table), model_name="colbert-ir/colbertv2.0")


def test_satisfies_the_reranker_protocol():
    assert isinstance(_reranker({}), Reranker)


def test_reorders_by_maxsim():
    table = {
        "q": [[1.0, 0.0]],
        "far": [[0.0, 1.0]],   # maxsim 0.0
        "near": [[1.0, 0.0]],  # maxsim 1.0
    }
    hits = [_hit("far", "far", 0.9), _hit("near", "near", 0.1)]
    out = _reranker(table).rerank("q", hits)
    assert [h.chunk.id for h in out] == ["near", "far"]


def test_preserves_dense_cosine_score():
    """THE load-bearing invariant. trust.py:292 thresholds on `score` and trust.py:536 feeds it to
    cal.confidence(). A MaxSim value is an unbounded sum in different units; leaking it into
    `score` would corrupt calibrated confidence for every hit. Same hazard rerank.py:84 documents
    for the cross-encoder."""
    table = {
        "q": [[1.0, 0.0]],
        "far": [[0.0, 1.0]],
        "near": [[1.0, 0.0]],
    }
    hits = [_hit("far", "far", 0.9), _hit("near", "near", 0.1)]
    out = _reranker(table).rerank("q", hits)
    by_id = {h.chunk.id: h.score for h in out}
    assert by_id == {"far": 0.9, "near": 0.1}
    assert sorted(h.score for h in out) == sorted(h.score for h in hits)


def test_preserves_indexed_at_and_first_indexed_at():
    from datetime import datetime, timezone

    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first = datetime(2025, 1, 1, tzinfo=timezone.utc)
    table = {"q": [[1.0, 0.0]], "a": [[1.0, 0.0]]}
    hits = [
        ScoredChunk(
            chunk=Chunk(id="a", source="f", text="a"),
            score=0.5,
            indexed_at=stamp,
            first_indexed_at=first,
        )
    ]
    out = _reranker(table).rerank("q", hits)
    assert out[0].indexed_at == stamp
    assert out[0].first_indexed_at == first


def test_uses_query_embed_for_query_and_passage_embed_for_documents():
    """ColBERT prepends distinct [Q]/[D] markers and pads queries with [MASK]. Using `embed` for
    both sides produces wrong scores that still look like plausible numbers."""
    table = {"q": [[1.0, 0.0]], "a": [[1.0, 0.0]]}
    encoder = _FakeEncoder(table)
    LateInteractionReranker(encoder, model_name="colbert-ir/colbertv2.0").rerank(
        "q", [_hit("a", "a", 0.5)]
    )
    assert encoder.query_calls == ["q"]
    assert encoder.passage_calls == ["a"]


def test_empty_hits_returns_empty():
    assert _reranker({}).rerank("q", []) == []


def test_ties_preserve_input_order():
    table = {"q": [[1.0, 0.0]], "a": [[1.0, 0.0]], "b": [[1.0, 0.0]]}
    hits = [_hit("a", "a", 0.1), _hit("b", "b", 0.2)]
    out = _reranker(table).rerank("q", hits)
    assert [h.chunk.id for h in out] == ["a", "b"]


def test_output_is_a_permutation_of_input():
    table = {"q": [[1.0, 0.0]], "a": [[0.0, 1.0]], "b": [[1.0, 0.0]]}
    hits = [_hit("a", "a", 0.1), _hit("b", "b", 0.2)]
    out = _reranker(table).rerank("q", hits)
    assert len(out) == len(hits)
    assert {h.chunk.id for h in out} == {"a", "b"}


def test_records_its_licence():
    rr = _reranker({})
    assert rr.model_name == "colbert-ir/colbertv2.0"
    assert rr.licence == "mit"


def test_construction_refuses_an_unregistered_checkpoint():
    with pytest.raises(ValueError, match="unknown late-interaction model"):
        LateInteractionReranker(_FakeEncoder({}), model_name="some/unrecorded")


def test_unscoreable_document_sorts_last_instead_of_aborting_the_batch():
    """`maxsim` refuses a zero-token document, and for one document that is right. For a BATCH it
    is not: raising would break reranking for every hit in the request over one malformed chunk.
    Last is not mid-pool, so the original objection to scoring 0.0 is still honoured."""
    table = {"q": [[1.0, 0.0]], "empty": [], "ok": [[1.0, 0.0]], "weak": [[0.0, 1.0]]}
    hits = [_hit("empty", "empty", 0.9), _hit("weak", "weak", 0.5), _hit("ok", "ok", 0.1)]
    out = _reranker(table).rerank("q", hits)
    assert [h.chunk.id for h in out] == ["ok", "weak", "empty"]


def test_unscoreable_documents_keep_their_input_order_among_themselves():
    table = {"q": [[1.0, 0.0]], "e1": [], "e2": [], "ok": [[1.0, 0.0]]}
    hits = [_hit("e1", "e1", 0.9), _hit("e2", "e2", 0.5), _hit("ok", "ok", 0.1)]
    out = _reranker(table).rerank("q", hits)
    assert [h.chunk.id for h in out] == ["ok", "e1", "e2"]


def test_unscoreable_document_still_keeps_its_dense_cosine():
    """The reorder-only invariant must hold for the salvaged case too."""
    table = {"q": [[1.0, 0.0]], "empty": [], "ok": [[1.0, 0.0]]}
    hits = [_hit("empty", "empty", 0.9), _hit("ok", "ok", 0.1)]
    out = _reranker(table).rerank("q", hits)
    assert {h.chunk.id: h.score for h in out} == {"empty": 0.9, "ok": 0.1}


def test_empty_query_still_raises():
    """Deliberately NOT salvaged. With no query tokens there is no evidence to rank anything by,
    so unlike a single bad document there is no partial ordering worth returning."""
    table = {"q": [], "ok": [[1.0, 0.0]]}
    with pytest.raises(ValueError, match="query has no tokens"):
        _reranker(table).rerank("q", [_hit("ok", "ok", 0.1)])


def test_empty_query_raises_even_when_every_document_is_also_unscoreable():
    """The empty-query guarantee must not depend on the document mix. `maxsim` runs only for
    documents that have tokens, so with every document unscoreable it would never be reached and
    a query carrying no evidence would return an order rather than raising."""
    table = {"q": [], "e1": [], "e2": []}
    with pytest.raises(ValueError, match="query has no tokens"):
        _reranker(table).rerank("q", [_hit("e1", "e1", 0.9), _hit("e2", "e2", 0.5)])
