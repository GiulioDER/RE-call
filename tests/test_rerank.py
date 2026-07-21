import pytest

from recall.rerank import CrossEncoderReranker, NoOpReranker, Reranker
from recall.types import Chunk, ScoredChunk


def _hit(cid: str, text: str, score: float = 0.0) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(id=cid, source="f", text=text), score=score)


def test_noop_preserves_order():
    hits = [_hit("a", "alpha"), _hit("b", "beta")]
    assert NoOpReranker().rerank("q", hits) == hits
    assert isinstance(NoOpReranker(), Reranker)


try:
    import sentence_transformers  # noqa: F401

    _HAS_ST = True
except ImportError:
    _HAS_ST = False


@pytest.mark.skipif(not _HAS_ST, reason="sentence-transformers not installed (recall[rerank])")
def test_cross_encoder_reorders_relevant_first():
    hits = [
        _hit("irrelevant", "the weather in antarctica is cold and windy"),
        _hit("relevant", "python generators and list comprehensions explained"),
    ]
    rr = CrossEncoderReranker()
    assert isinstance(rr, Reranker)
    reranked = rr.rerank("how do python generators work", hits)
    assert reranked[0].chunk.id == "relevant"


@pytest.mark.skipif(not _HAS_ST, reason="sentence-transformers not installed (recall[rerank])")
def test_cross_encoder_preserves_cosine_score_and_indexed_at():
    # the reranker must REORDER only: leaking raw cross-encoder logits into `score` would
    # corrupt the trust layer, which reads that field as a dense cosine
    from datetime import datetime, timezone

    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    hits = [
        ScoredChunk(chunk=Chunk(id="a", source="f", text="the weather in antarctica"),
                    score=0.41, indexed_at=ts),
        ScoredChunk(chunk=Chunk(id="b", source="f", text="python generators explained"),
                    score=0.41, indexed_at=ts),
    ]
    reranked = CrossEncoderReranker().rerank("how do python generators work", hits)
    assert {h.chunk.id for h in reranked} == {"a", "b"}
    for h in reranked:
        assert h.score == 0.41          # dense cosine preserved
        assert h.indexed_at == ts       # provenance preserved


def test_default_reranker_model_is_pinned_to_a_hub_revision(monkeypatch):
    """An unpinned Hub reference is mutable — pin the reranker like the entailment judge does.

    Same supply-chain reasoning as `recall.entailment.DEFAULT_QNLI_REVISION`: whoever controls
    the repo can swap the weights and every consumer picks them up on the next cold cache.
    """
    import recall.rerank as rerank_mod

    seen = {}

    class FakeCrossEncoder:
        def __init__(self, model, revision=None):
            seen["model"], seen["revision"] = model, revision

    monkeypatch.setattr(
        rerank_mod, "_load_cross_encoder", lambda m, r: FakeCrossEncoder(m, revision=r)
    )
    CrossEncoderReranker()
    assert seen["model"] == rerank_mod.DEFAULT_RERANK_MODEL
    assert seen["revision"] == rerank_mod.DEFAULT_RERANK_REVISION
    assert len(rerank_mod.DEFAULT_RERANK_REVISION) == 40  # a real commit sha, not a branch


def test_custom_reranker_model_does_not_inherit_the_default_pin(monkeypatch):
    import recall.rerank as rerank_mod

    seen = {}
    monkeypatch.setattr(
        rerank_mod, "_load_cross_encoder", lambda m, r: seen.update(model=m, revision=r)
    )
    CrossEncoderReranker(model="some-org/other-reranker")
    assert seen["revision"] is None  # the default pin belongs to the default model only
