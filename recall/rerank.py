from __future__ import annotations

from typing import Protocol, runtime_checkable

from recall.types import ScoredChunk


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]: ...


class NoOpReranker:
    def rerank(self, query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        return hits
