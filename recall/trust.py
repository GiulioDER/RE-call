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

from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a runtime import cycle: entailment imports trust's abstain wording
    from recall.entailment import EntailmentJudge

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
    Verdict,
)

_UNCALIBRATED = Calibration(embedder="uncalibrated", threshold=DEFAULT_GAP_THRESHOLD)


def resolve_successor(file: str, supersession: dict[str, str]) -> str | None:
    """Terminal successor of `file` in the supersession chain, or None if it has none.

    A cycle (a.md -> b.md -> a.md) cannot loop: the walk stops on the first revisit and the
    cycle member resolves to its direct successor. A self-claim (`supersedes:` the file's own
    name — an authoring mistake) is ignored: a document cannot supersede itself.
    """
    if supersession.get(file) in (None, file):
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
) -> tuple[Verdict, Validity]:
    meta = hit.chunk.metadata
    file = meta.get("file")
    try:
        start, end = validity_bounds(meta)
    except ValueError:
        # Malformed validity metadata (reachable via direct store.upsert, which bypasses the
        # Indexer's fail-fast). Fail CLOSED per hit: an unparseable window must not read as
        # trustworthy, and one bad row must not crash every search that retrieves it.
        return "invalid_metadata", Validity(valid_from=None, valid_until=None, superseded_by=None)
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


def abstain_reason(hits: list[TrustedHit]) -> str:
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
    if best.verdict == "invalid_metadata":
        return f"best candidate ({best.provenance.file}) carries malformed validity metadata"
    return "no hit above the calibrated confidence threshold (probable corpus gap)"


def evaluate(
    result: RetrievalResult,
    supersession: dict[str, str],
    calibration: Calibration | None,
    now: datetime,
) -> TrustedResult:
    """Pure trust evaluation of a retrieval result (no DB access, no clock reads).

    Verdict precedence per hit: invalid_metadata > superseded > expired / not_yet_valid >
    low_confidence > ok.
    Successor promotion: when a superseded hit scored above the threshold (it would have been
    the confident answer), its retrieved successor is promoted from ``low_confidence`` to
    ``ok`` even if its own wording scores lower — the explicit supersession edge transfers the
    topical relevance the stale memory proved. Hits are then reordered valid-first so the
    successor outranks the stale memory. `abstained` is True when no hit earned verdict ``ok``.
    A tz-naive `now` is interpreted as UTC.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
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
    promoted_files = {
        h.validity.superseded_by
        for h in trusted
        if h.verdict == "superseded" and h.cosine >= cal.threshold
    }
    trusted = [
        replace(h, verdict="ok")
        if h.verdict == "low_confidence" and h.provenance.file in promoted_files
        else h
        for h in trusted
    ]
    ok = [h for h in trusted if h.verdict == "ok"]
    rest = [h for h in trusted if h.verdict != "ok"]
    abstained = not ok
    return TrustedResult(
        query=result.query,
        hits=ok + rest,
        abstained=abstained,
        reason=abstain_reason(rest) if abstained else "",
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
    entailment: "EntailmentJudge | None" = None,
) -> TrustedResult:
    """Hybrid search + trust evaluation in one call — the recommended agent-facing entry point.

    `entailment` is OFF by default: when a judge is passed, verdict-ok hits that do not entail
    an answer to the query are demoted to ``not_entailed`` (see `recall.entailment`) — the
    near-miss guard the cosine threshold cannot provide. Costs one judge pass per ok hit.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    # single fallback resolution: the retriever's gap threshold and the verdict threshold must
    # always come from the same calibration (or the same uncalibrated default)
    cal = calibration or _UNCALIBRATED
    retriever = HybridRetriever(store, embedder, reranker=reranker, gap_threshold=cal.threshold)
    result = retriever.search(query, k=k, source=source)
    supersession = store.supersession_map() if result.hits else {}
    trusted = evaluate(
        result,
        supersession,
        calibration,
        now or datetime.now(timezone.utc),
    )
    if entailment is not None:
        from recall.entailment import apply_entailment

        trusted = apply_entailment(trusted, entailment)
    return trusted
