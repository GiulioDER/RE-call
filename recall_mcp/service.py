from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from recall.calibration import Calibration
from recall.embeddings import Embedder, HashingEmbedder
from recall.guards import staleness
from recall.index import Indexer
from recall.store import PgVectorStore
from recall.timing import TimedEmbedder
from recall.trust import trusted_search

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
    score: float = Field(description="True dense cosine similarity in [-1, 1].")
    confidence: float = Field(
        description="Calibrated confidence in [0, 1]; 0.5 sits exactly at the abstention "
        "threshold. Uncalibrated when the result says calibrated=false."
    )
    verdict: str = Field(
        description="Trust verdict: ok | superseded | expired | not_yet_valid | low_confidence "
        "| invalid_metadata. Only 'ok' hits should be relied on. (The library also defines "
        "not_entailed for the opt-in entailment stage, which this server does not enable.)"
    )
    superseded_by: str | None = Field(
        default=None, description="File of the memory that replaces this one, when superseded."
    )
    valid_until: str | None = Field(
        default=None, description="ISO end of this memory's validity window, when declared."
    )
    indexed_at: str | None = Field(
        default=None, description="ISO timestamp of when this memory entered the index."
    )
    text: str = Field(description="The retrieved memory chunk.")


class SearchResult(BaseModel):
    query: str
    abstained: bool = Field(
        description="True when NO valid hit survived — say you don't know instead of answering."
    )
    reason: str = Field(description="Why the search abstained; empty otherwise.")
    calibrated: bool = Field(
        description="True when a per-embedder calibration was applied to threshold/confidence."
    )
    gap_warning: bool = Field(description="True when the memory probably lacks a relevant answer.")
    stale: bool = Field(description="True when the memory index is older than the freshness window.")
    advice: str = Field(description="What the agent should do with this result.")
    embed_ms: float | None = Field(
        default=None,
        description="Query-embedding latency in milliseconds (cost/latency metadata; null if "
        "not measured). Additive — clients that ignore it are unaffected.",
    )
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
    calibration: Calibration | None = None,
) -> SearchResult:
    """Run a trust-evaluated hybrid search and format it into actionable self-recall guidance.

    Every hit carries confidence + provenance + validity; superseded or out-of-window memories
    are demoted below valid ones, and when no valid hit remains the result abstains.
    `k` is clamped to [1, MAX_SEARCH_K] so an untrusted client cannot request an unbounded result set.
    """
    k = max(1, min(k, MAX_SEARCH_K))
    timed = TimedEmbedder(embedder)  # measure embedding latency without altering trusted_search
    result = trusted_search(store, timed, query, k=k, source=source, calibration=calibration)
    hits = [
        SearchHit(
            source=h.provenance.file or h.chunk.source,
            score=round(h.cosine, 4),
            confidence=round(h.confidence, 4),
            verdict=h.verdict,
            superseded_by=h.validity.superseded_by,
            valid_until=h.validity.valid_until.isoformat() if h.validity.valid_until else None,
            indexed_at=h.provenance.indexed_at.isoformat() if h.provenance.indexed_at else None,
            text=h.chunk.text,
        )
        for h in result.hits
    ]
    superseded = [h for h in hits if h.verdict == "superseded"]
    if result.abstained:
        advice = (
            f"No trustworthy memory for this query — say you don't know and do NOT answer "
            f"from these hits. Reason: {result.reason}."
        )
    elif superseded:
        names = ", ".join(f"{h.source} -> {h.superseded_by}" for h in superseded)
        advice = (
            f"{sum(1 for h in hits if h.verdict == 'ok')} valid memory hit(s). NOTE: some "
            f"matches are superseded ({names}) — rely only on the current version. Consult "
            "before re-proposing: if a closed decision appears here, do not re-litigate it."
        )
    else:
        advice = (
            f"{len(hits)} relevant memory hit(s). Consult before re-proposing: if a closed "
            "decision or falsified hypothesis appears here, do not re-litigate it."
        )
    if not result.calibrated:
        advice += (
            " NOTE: confidence is UNCALIBRATED (default threshold) — run `recall calibrate` "
            "against a labeled query set for this embedder."
        )
    if result.staleness.stale:
        advice += " NOTE: the memory index is stale — consider re-indexing."
    return SearchResult(
        query=query,
        abstained=result.abstained,
        reason=result.reason,
        calibrated=result.calibrated,
        gap_warning=result.gap_warning,
        stale=result.staleness.stale,
        advice=advice,
        embed_ms=round(timed.stats.total_ms, 2),
        hits=hits,
    )


def index_memory(store: PgVectorStore, embedder: Embedder, path: str) -> IndexResult:
    """Index a markdown file or folder into memory; return counts + a human message.

    `path` is confined to RECALL_INDEX_ROOT (default: the current working directory) so a client
    cannot read arbitrary files off the server's filesystem. Re-indexing REPLACES each file's
    chunks completely, so a shrunk file leaves no stale chunks behind.
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
