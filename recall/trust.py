"""The trust layer: confidence + provenance + validity over retrieval, with calibrated abstention.

Semantic similarity answers "which memory looks most like the query"; it cannot answer "should
this memory still be believed". This module adds that second judgment as a pure post-processing
step over `HybridRetriever.search()`:

- every hit gets a verdict — ``ok``, ``superseded``, ``expired``, ``not_yet_valid`` or
  ``low_confidence`` — plus a calibrated confidence, provenance, and its validity window;
- a memory that is superseded or outside its validity window loses even with a top cosine:
  valid hits are ordered first (so a retrieved successor outranks the stale memory it replaced),
  and when NO valid hit remains the result is an explicit abstention with a reason;
- the abstention threshold comes from `recall.calibration` when available; otherwise the default
  gap threshold is used and the result is flagged ``calibrated=False``.
"""
from __future__ import annotations

from datetime import datetime, timezone

from recall.calibration import Calibration
from recall.embeddings import Embedder
from recall.frontmatter import validity_bounds
from recall.guards import DEFAULT_GAP_THRESHOLD
from recall.rerank import Reranker
from recall.retriever import HybridRetriever
from recall.store import PgVectorStore
from recall.types import (
    Provenance,
    RetrievalResult,
    ScoredChunk,
    TrustedHit,
    TrustedResult,
    Validity,
)

_UNCALIBRATED = Calibration(embedder="uncalibrated", threshold=DEFAULT_GAP_THRESHOLD)


def resolve_successor(file: str, supersession: dict[str, str]) -> str | None:
    """Terminal successor of `file` in the supersession chain, or None if it has none.

    A cycle (a.md -> b.md -> a.md) cannot loop: the walk stops on the first revisit and the
    cycle member resolves to its direct successor.
    """
    if file not in supersession:
        return None
    seen = {file}
    cur = file
    while cur in supersession:
        nxt = supersession[cur]
        if nxt in seen:
            return supersession[file]
        seen.add(nxt)
        cur = nxt
    return cur


def _verdict(
    hit: ScoredChunk, supersession: dict[str, str], threshold: float, now: datetime
) -> tuple[str, Validity]:
    meta = hit.chunk.metadata
    file = meta.get("file")
    start, end = validity_bounds(meta)
    successor = resolve_successor(file, supersession) if file else None
    validity = Validity(valid_from=start, valid_until=end, superseded_by=successor)
    if successor is not None:
        return "superseded", validity
    if end is not None and now > end:
        return "expired", validity
    if start is not None and now < start:
        return "not_yet_valid", validity
    if hit.score < threshold:
        return "low_confidence", validity
    return "ok", validity


def _abstain_reason(hits: list[TrustedHit]) -> str:
    if not hits:
        return "no memory retrieved at all"
    best = max(hits, key=lambda h: h.cosine)
    if best.verdict == "superseded":
        return (
            f"best candidate ({best.provenance.file}) is superseded by "
            f"{best.validity.superseded_by} — consult the successor, not this memory"
        )
    if best.verdict == "expired":
        return f"best candidate ({best.provenance.file}) is outside its validity window (expired)"
    if best.verdict == "not_yet_valid":
        return f"best candidate ({best.provenance.file}) is not yet valid"
    return "no hit above the calibrated confidence threshold (probable corpus gap)"


def evaluate(
    result: RetrievalResult,
    supersession: dict[str, str],
    calibration: Calibration | None,
    now: datetime,
) -> TrustedResult:
    """Pure trust evaluation of a retrieval result (no DB access, no clock reads).

    Verdict precedence per hit: superseded > expired / not_yet_valid > low_confidence > ok.
    Hits are reordered valid-first so a retrieved successor outranks the superseded memory
    that outscored it semantically. `abstained` is True when no hit earned verdict ``ok``.
    """
    cal = calibration or _UNCALIBRATED
    trusted: list[TrustedHit] = []
    for hit in result.hits:
        verdict, validity = _verdict(hit, supersession, cal.threshold, now)
        meta = hit.chunk.metadata
        trusted.append(
            TrustedHit(
                chunk=hit.chunk,
                cosine=hit.score,
                confidence=cal.confidence(hit.score),
                verdict=verdict,
                provenance=Provenance(
                    source=hit.chunk.source,
                    file=meta.get("file"),
                    ord=meta.get("ord"),
                    indexed_at=hit.indexed_at,
                ),
                validity=validity,
            )
        )
    ok = [h for h in trusted if h.verdict == "ok"]
    rest = [h for h in trusted if h.verdict != "ok"]
    abstained = not ok
    return TrustedResult(
        query=result.query,
        hits=ok + rest,
        abstained=abstained,
        reason=_abstain_reason(rest) if abstained else "",
        calibrated=calibration is not None,
        gap_warning=result.gap_warning,
        staleness=result.staleness,
    )


def trusted_search(
    store: PgVectorStore,
    embedder: Embedder,
    query: str,
    k: int = 5,
    source: str | None = None,
    calibration: Calibration | None = None,
    reranker: Reranker | None = None,
    now: datetime | None = None,
) -> TrustedResult:
    """Hybrid search + trust evaluation in one call — the recommended agent-facing entry point."""
    threshold = calibration.threshold if calibration else DEFAULT_GAP_THRESHOLD
    retriever = HybridRetriever(store, embedder, reranker=reranker, gap_threshold=threshold)
    result = retriever.search(query, k=k, source=source)
    return evaluate(
        result,
        store.supersession_map(),
        calibration,
        now or datetime.now(timezone.utc),
    )
