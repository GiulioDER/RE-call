from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

#: Trust verdict for a retrieved hit. Only ``ok`` hits should be relied on.
#: ``not_entailed`` (optional entailment stage): semantically close but does not answer the query.
#: ``ambiguous_supersession``: a supersession edge points at this memory's basename, but the
#: corpus carries that basename in more than one directory. Which document the author meant is
#: unanswerable, so the hit fails closed rather than being served with a guessed successor — or,
#: worse, as healthy because the edge was quietly dropped.
#: `not_yet_known` is the TRANSACTION-time verdict and the only one on that axis. The other
#: temporal verdicts (`expired`, `not_yet_valid`) are VALID-time: they say when a fact was true.
#: `not_yet_known` says the memory had not been written yet, which is a different question and the
#: one a "what did we know on Tuesday" query is actually asking. See `recall.trust.evaluate`.
Verdict = Literal[
    "ok", "superseded", "expired", "not_yet_valid", "not_yet_known", "low_confidence",
    "invalid_metadata", "ambiguous_supersession", "not_entailed",
]


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    #: LAST write. What "how fresh is this corpus" means; staleness reports on this.
    indexed_at: datetime | None = None
    #: FIRST write, preserved across re-indexing. This is the transaction-time axis: `known_as_of`
    #: asks what we HELD at an instant, and a memo edited yesterday was still held last month.
    #: Using `indexed_at` for that made a re-indexed memory read `not_yet_known` for every past
    #: instant. `None` on stores predating the column, and then treated as visible, not hidden.
    first_indexed_at: datetime | None = None


@dataclass(frozen=True)
class StalenessReport:
    stale: bool
    newest_indexed_at: datetime | None
    age: timedelta | None
    max_age: timedelta


@dataclass(frozen=True)
class RetrievalDiagnostics:
    """Non-content metadata for one retrieval execution."""

    embedding_profile: str = "legacy"
    retrieval_profile: str = "legacy"
    index_generation: str = "legacy"
    candidate_pool_size: int = 20
    reranking_ran: bool = False
    stage_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    hits: list[ScoredChunk]
    gap_warning: bool
    staleness: StalenessReport
    diagnostics: RetrievalDiagnostics = field(default_factory=RetrievalDiagnostics)


@dataclass(frozen=True)
class Provenance:
    """Where a memory came from and when it entered the index."""

    source: str
    file: str | None
    ord: int | None
    indexed_at: datetime | None


@dataclass(frozen=True)
class Validity:
    """A memory's validity window and supersession status."""

    valid_from: datetime | None
    valid_until: datetime | None
    superseded_by: str | None  # terminal successor file, when superseded


@dataclass(frozen=True)
class TrustedHit:
    """A retrieved chunk annotated with everything needed to decide whether to trust it."""

    chunk: Chunk
    cosine: float
    confidence: float
    verdict: Verdict
    provenance: Provenance
    validity: Validity


@dataclass(frozen=True)
class TrustedResult:
    """Trust-evaluated retrieval: hits ordered valid-first, plus the abstention decision."""

    query: str
    hits: list[TrustedHit]  # verdict-ok hits first (original order kept within each group)
    abstained: bool
    reason: str  # non-empty only when abstained
    calibrated: bool
    gap_warning: bool
    staleness: StalenessReport
    diagnostics: RetrievalDiagnostics = field(default_factory=RetrievalDiagnostics)
