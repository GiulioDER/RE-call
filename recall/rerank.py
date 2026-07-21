from __future__ import annotations

from typing import Protocol, runtime_checkable

from recall.types import ScoredChunk


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]: ...


class NoOpReranker:
    def rerank(self, query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        return hits


DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
#: Pinned Hub commit of the DEFAULT model, for the same reason as
#: `recall.entailment.DEFAULT_QNLI_REVISION`: an unpinned Hub reference is mutable, so the repo
#: owner (or a compromise) can swap the weights and every consumer silently picks them up on
#: the next cold cache. Pinning makes the resolved artifact immutable.
DEFAULT_RERANK_REVISION = "c5ee24cb16019beea0893ab7796b1df96625c6b8"


def _load_cross_encoder(model: str, revision: str | None):
    """Import + construct the cross-encoder (a seam so the wiring is testable without weights)."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("CrossEncoderReranker requires: pip install recall[rerank]") from exc
    return CrossEncoder(model, revision=revision)


class CrossEncoderReranker:
    """Reorder hits by cross-encoder relevance. Requires `pip install recall[rerank]`.

    The default model is pinned to a Hub revision; if you supply your own `model`, pin your
    own `revision` too.
    """

    def __init__(
        self,
        model: str = DEFAULT_RERANK_MODEL,
        revision: str | None = DEFAULT_RERANK_REVISION,
    ) -> None:
        if model != DEFAULT_RERANK_MODEL and revision == DEFAULT_RERANK_REVISION:
            revision = None  # the default pin belongs to the default model only
        self._model = _load_cross_encoder(model, revision)

    def rerank(self, query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        if not hits:
            return hits
        scores = self._model.predict([(query, h.chunk.text) for h in hits])
        order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
        # Reorder ONLY — each hit keeps its dense cosine `score` and `indexed_at`. The
        # cross-encoder logit is an unbounded relevance score in different units; leaking it
        # into `score` would corrupt every downstream consumer that reads it as a cosine
        # (the trust layer's thresholds and calibrated confidence in particular).
        return [hits[i] for i in order]
