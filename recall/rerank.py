from __future__ import annotations

from typing import Protocol, runtime_checkable

from recall.types import ScoredChunk


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]: ...


class NoOpReranker:
    def rerank(self, query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        return hits


class CrossEncoderReranker:
    """Reorder hits by cross-encoder relevance. Requires `pip install recall[rerank]`."""

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError("CrossEncoderReranker requires: pip install recall[rerank]") from exc
        self._model = CrossEncoder(model)

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
