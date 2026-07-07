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
