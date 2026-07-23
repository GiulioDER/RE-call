from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from recall.calibration import Calibration
from recall.embeddings import Embedder, HashingEmbedder
from recall.guards import staleness
from recall.observability import METRICS
from recall.index import Indexer, candidate_files
from recall.store import PgVectorStore
from recall.timing import TimedEmbedder
from recall.trust import trusted_search

HASHING_DIM = 64  # offline HashingEmbedder width; matches the eval/test default
MAX_SEARCH_K = 50  # upper bound on hits per search — clamps untrusted client input

# Indexing budget caps (SECURITY.md "Indexing is client-callable and unbounded").
# `recall_index` is client-callable and, once past the RECALL_INDEX_ROOT confinement check below,
# had no ceiling on how much of that root it would walk, read and send to the embedder — with a
# paid embedder configured that is uncapped cloud spend per call. These two limits are enforced by
# `index_memory` BEFORE `Indexer.index_path` touches a single file: the candidate set is walked and
# measured first (`candidate_files` + `Path.stat`, no reads), and the whole request is refused if
# it exceeds either one. A cap that trips mid-walk, after some files are already embedded, is not a
# budget cap — it just makes the overspend partial instead of total.
#
# Defaults were chosen from this project's own real workloads, measured directly rather than
# guessed, so a legitimate `recall_index` call on any of them clears both limits with headroom:
#   - `make demo` indexes `corpus/`: 5 files, ~1.6 KB total.
#   - `recall code` indexes RE-call's own package (`recall/`): 30 files, ~242 KB total.
#   - The real eval corpus this project measures retrieval against (docs/CASE_STUDY.md,
#     re-measured for this change): 796 markdown memos, ~4.1 MB of content (5.6 MB on disk
#     including directory overhead).
# 2000 files / 20 MB give the largest of those (the 796-file, ~4-6 MB real corpus) roughly 2.5x
# headroom on file count and 3.5-5x headroom on bytes, while still refusing a client that points
# `recall_index` at something categorically bigger than a memory corpus — a vendored dependency
# tree, a build output directory, a whole home directory.
DEFAULT_MAX_INDEX_FILES = 2000
DEFAULT_MAX_INDEX_BYTES = 20_000_000  # 20 MB


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
        "| ambiguous_supersession "
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
    files: int = Field(
        description="Number of files (re)indexed by this call. Unchanged files are counted in "
        "`skipped`, not here, so a no-op re-index reports 0 — that does not mean the index is empty."
    )
    chunks: int = Field(description="Number of chunks written to memory.")
    skipped: int = Field(
        default=0,
        description="Files whose content was unchanged since the last index, so they were not "
        "re-embedded.",
    )
    deleted: int = Field(
        default=0,
        description="Sources permanently removed because their files are gone from disk. "
        "Re-indexing is destructive in this one respect; reported so a caller can see it rather "
        "than discovering it later as missing memory.",
    )
    message: str = Field(description="Human-readable summary of what was indexed.")


class ForgetResult(BaseModel):
    chunks_removed: int = Field(
        description="Number of chunks permanently deleted, across every matched source."
    )
    sources_removed: list[str] = Field(
        description="Requested sources that had at least one chunk and were deleted."
    )
    sources_not_found: list[str] = Field(
        default_factory=list,
        description="Requested sources that matched no chunk for this tenant — a typo, or a "
        "source that was already forgotten. Reported separately from sources_removed so a "
        "caller can never mistake 'matched nothing' for 'successfully forgotten'.",
    )
    message: str = Field(description="Human-readable summary of what was forgotten.")


class MemoryStatsResult(BaseModel):
    chunks: int = Field(description="Total chunks currently in memory.")
    newest_indexed_at: str | None = Field(
        description="ISO-8601 timestamp of the newest chunk, or null if memory is empty."
    )
    stale: bool = Field(description="True when the newest chunk is older than the freshness window.")
    metrics: dict = Field(
        default_factory=dict,
        description="Process metrics since start: counters (searches, abstentions, gap warnings, "
        "verdicts by kind, database reconnects) and latency percentiles. Surfaced here so an "
        "operator can read them without a scrape endpoint.",
    )


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


def index_memory(
    store: PgVectorStore,
    embedder: Embedder,
    path: str,
    on_measured: Callable[[int, int], None] | None = None,
) -> IndexResult:
    """Index a markdown file or folder into memory; return counts + a human message.

    `path` is confined to RECALL_INDEX_ROOT (default: the current working directory) so a client
    cannot read arbitrary files off the server's filesystem. Re-indexing REPLACES each file's
    chunks completely, so a shrunk file leaves no stale chunks behind.

    Before anything is read or embedded, the candidate file set is walked and measured against two
    budget caps — RECALL_INDEX_MAX_FILES and RECALL_INDEX_MAX_BYTES (defaults
    DEFAULT_MAX_INDEX_FILES / DEFAULT_MAX_INDEX_BYTES above) — and the whole request is refused if
    either is exceeded. See SECURITY.md's "Indexing is client-callable" gap for why this exists.

    `on_measured(files, bytes)` is invoked once those per-request caps pass and BEFORE anything is
    embedded, so a caller can meter aggregate spend against the set actually about to be indexed
    (the server debits the tenant's byte quota here). Raising from it aborts the request having
    spent nothing — which is the only reason the hook exists rather than the caller measuring the
    tree itself: a second walk is a second answer, and the one that bills must be the one that
    runs.
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

    max_files = int(os.environ.get("RECALL_INDEX_MAX_FILES", str(DEFAULT_MAX_INDEX_FILES)))
    max_bytes = int(os.environ.get("RECALL_INDEX_MAX_BYTES", str(DEFAULT_MAX_INDEX_BYTES)))
    # Walked ONCE, here, and handed to `index_path` below — measured, not estimated, and the set
    # that is measured is the set that is indexed. Walking again inside `index_path` would ask
    # the filesystem the same question twice: anything landing under the root between the two
    # walks would be embedded without being counted, escaping both the budget check and the
    # tenant's byte quota, and a sync landing there is exactly the deployment shape this serves.
    files = candidate_files(target)
    if len(files) > max_files:
        raise ValueError(
            f"index request for {path!r} exceeds the file-count budget: {len(files)} candidate "
            f"file(s) > limit {max_files}; set RECALL_INDEX_MAX_FILES to raise it."
        )
    total_bytes = sum(f.stat().st_size for f in files)
    if total_bytes > max_bytes:
        raise ValueError(
            f"index request for {path!r} exceeds the byte budget: {total_bytes} candidate "
            f"byte(s) > limit {max_bytes}; set RECALL_INDEX_MAX_BYTES to raise it."
        )
    if on_measured is not None:
        on_measured(len(files), total_bytes)

    stats = Indexer(store, embedder).index_path(target, files=files)
    message = f"Indexed {stats.chunks} chunk(s) from {stats.files} file(s) into memory."
    if stats.skipped:
        message += f" {stats.skipped} file(s) were unchanged and not re-embedded."
    if stats.deleted:
        message += f" Pruned {stats.deleted} source(s) whose files are gone from disk."
    return IndexResult(
        files=stats.files,
        chunks=stats.chunks,
        skipped=stats.skipped,
        deleted=stats.deleted,
        message=message,
    )


def forget_memory(store: PgVectorStore, sources: list[str]) -> ForgetResult:
    """Permanently delete every indexed chunk for the given sources; return what actually went away.

    This is the right-to-erasure path: irreversible and tenant-scoped (only ever touches the
    calling store's own tenant — see `PgVectorStore.delete_sources`). A source that does not
    exist for this tenant is reported in `sources_not_found`, never silently folded into a "0
    removed, success" result — a typo'd source name must be visibly distinguishable from one
    that was actually forgotten.
    """
    if not sources:
        raise ValueError("sources must be a non-empty list")
    requested = list(dict.fromkeys(sources))  # de-dup, preserve order
    existing = store.source_content_hashes()  # {source: content_hash}, this tenant only
    found = [s for s in requested if s in existing]
    not_found = [s for s in requested if s not in existing]
    chunks_removed = store.delete_sources(found) if found else 0
    if found and not_found:
        message = (
            f"Forgot {chunks_removed} chunk(s) from {len(found)} source(s); "
            f"{len(not_found)} source(s) not found: {', '.join(not_found)}."
        )
    elif found:
        message = f"Forgot {chunks_removed} chunk(s) from {len(found)} source(s)."
    else:
        message = f"No matching source(s) found — nothing deleted: {', '.join(not_found)}."
    return ForgetResult(
        chunks_removed=chunks_removed,
        sources_removed=found,
        sources_not_found=not_found,
        message=message,
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
        metrics=METRICS.snapshot(),
    )
