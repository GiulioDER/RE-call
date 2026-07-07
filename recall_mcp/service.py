from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field

from recall.embeddings import Embedder, HashingEmbedder
from recall.guards import staleness
from recall.index import Indexer
from recall.retriever import HybridRetriever
from recall.store import PgVectorStore


def make_embedder(name: str) -> Embedder:
    """Return the embedder backend by name ('fastembed' local default, or offline 'hashing')."""
    if name == "hashing":
        return HashingEmbedder(dim=64)
    if name == "fastembed":
        from recall.embeddings import FastEmbedEmbedder

        return FastEmbedEmbedder()
    raise ValueError(f"unknown embedder: {name!r} (use 'fastembed' or 'hashing')")


class SearchHit(BaseModel):
    source: str = Field(description="Where this memory came from (file/source id).")
    score: float = Field(description="Dense cosine similarity in [-1, 1]; 0.0 if sparse-only.")
    text: str = Field(description="The retrieved memory chunk.")


class SearchResult(BaseModel):
    query: str
    gap_warning: bool = Field(description="True when the memory probably lacks a relevant answer.")
    stale: bool = Field(description="True when the memory index is older than the freshness window.")
    advice: str = Field(description="What the agent should do with this result.")
    hits: list[SearchHit]


def search_memory(
    store: PgVectorStore,
    embedder: Embedder,
    query: str,
    source: str | None = None,
    k: int = 5,
) -> SearchResult:
    """Run a hybrid search and format it into actionable self-recall guidance."""
    result = HybridRetriever(store, embedder).search(query, k=k, source=source)
    hits = [
        SearchHit(source=h.chunk.source, score=round(h.score, 4), text=h.chunk.text)
        for h in result.hits
    ]
    if result.gap_warning:
        advice = (
            "Probable corpus gap — the memory likely does NOT contain a relevant answer; "
            "treat these hits as unreliable and do not rely on them."
        )
    else:
        advice = (
            f"{len(hits)} relevant memory hit(s). Consult before re-proposing: if a closed "
            "decision or falsified hypothesis appears here, do not re-litigate it."
        )
    if result.staleness.stale:
        advice += " NOTE: the memory index is stale — consider re-indexing."
    return SearchResult(
        query=query,
        gap_warning=result.gap_warning,
        stale=result.staleness.stale,
        advice=advice,
        hits=hits,
    )


def index_memory(store: PgVectorStore, embedder: Embedder, path: str) -> dict:
    """Index a markdown file or folder into memory; return counts + a human message."""
    stats = Indexer(store, embedder).index_path(path)
    return {
        "files": stats.files,
        "chunks": stats.chunks,
        "message": f"Indexed {stats.chunks} chunk(s) from {stats.files} file(s) into memory.",
    }


def memory_stats(store: PgVectorStore, max_age: timedelta = timedelta(days=2)) -> dict:
    """Report memory size and freshness."""
    newest = store.newest_indexed_at()
    stale = staleness(newest, datetime.now(timezone.utc), max_age).stale
    return {
        "chunks": store.count(),
        "newest_indexed_at": newest.isoformat() if newest else None,
        "stale": stale,
    }
