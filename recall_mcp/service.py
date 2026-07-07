from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from recall.embeddings import Embedder, HashingEmbedder
from recall.guards import staleness
from recall.index import Indexer
from recall.retriever import HybridRetriever
from recall.store import PgVectorStore

HASHING_DIM = 64  # offline HashingEmbedder width; matches the eval/test default
MAX_SEARCH_K = 50  # upper bound on hits per search — clamps untrusted client input


def make_embedder(name: str) -> Embedder:
    """Return the embedder backend by name ('fastembed' local default, or offline 'hashing')."""
    if name == "hashing":
        return HashingEmbedder(dim=HASHING_DIM)
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


class IndexResult(BaseModel):
    files: int = Field(description="Number of files indexed.")
    chunks: int = Field(description="Number of chunks written to memory.")
    message: str = Field(description="Human-readable summary of what was indexed.")


class MemoryStatsResult(BaseModel):
    chunks: int = Field(description="Total chunks currently in memory.")
    newest_indexed_at: str | None = Field(
        description="ISO-8601 timestamp of the newest chunk, or null if memory is empty."
    )
    stale: bool = Field(description="True when the newest chunk is older than the freshness window.")


def search_memory(
    store: PgVectorStore,
    embedder: Embedder,
    query: str,
    source: str | None = None,
    k: int = 5,
) -> SearchResult:
    """Run a hybrid search and format it into actionable self-recall guidance.

    `k` is clamped to [1, MAX_SEARCH_K] so an untrusted client cannot request an unbounded result set.
    """
    k = max(1, min(k, MAX_SEARCH_K))
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


def index_memory(store: PgVectorStore, embedder: Embedder, path: str) -> IndexResult:
    """Index a markdown file or folder into memory; return counts + a human message.

    `path` is confined to RECALL_INDEX_ROOT (default: the current working directory) so a client
    cannot read arbitrary files off the server's filesystem. Re-indexing overwrites a file's chunks
    in place; if a file shrinks, orphaned trailing chunks are not garbage-collected.
    """
    root = Path(os.environ.get("RECALL_INDEX_ROOT", ".")).resolve()
    target = Path(path).resolve()
    if not target.is_relative_to(root):
        raise ValueError(
            f"path {path!r} is outside the allowed index root {str(root)!r}; "
            "set RECALL_INDEX_ROOT to widen it."
        )
    if not target.exists():
        raise ValueError(f"path not found: {path!r}")
    stats = Indexer(store, embedder).index_path(target)
    return IndexResult(
        files=stats.files,
        chunks=stats.chunks,
        message=f"Indexed {stats.chunks} chunk(s) from {stats.files} file(s) into memory.",
    )


def memory_stats(
    store: PgVectorStore, max_age: timedelta = timedelta(days=2)
) -> MemoryStatsResult:
    """Report memory size and freshness (`stale` is True when the newest chunk is older than `max_age`, default 2 days)."""
    newest = store.newest_indexed_at()
    stale = staleness(newest, datetime.now(timezone.utc), max_age).stale
    return MemoryStatsResult(
        chunks=store.count(),
        newest_indexed_at=newest.isoformat() if newest else None,
        stale=stale,
    )
