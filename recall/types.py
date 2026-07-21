from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

#: Trust verdict for a retrieved hit. Only ``ok`` hits should be relied on.
#: ``ambiguous_supersession``: a supersession edge involving this memory cannot be resolved,
#: because one of its endpoints names a basename the corpus carries in more than one directory.
#: Either "is this the superseded one?" or "which document supersedes it?" is unanswerable, so
#: the hit fails closed rather than being served with a guessed successor.
#: ``not_entailed`` (optional entailment stage): semantically close but does not answer the query.
Verdict = Literal[
    "ok", "superseded", "expired", "not_yet_valid", "low_confidence", "invalid_metadata",
    "ambiguous_supersession", "not_entailed",
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
    indexed_at: datetime | None = None


@dataclass(frozen=True)
class StalenessReport:
    stale: bool
    newest_indexed_at: datetime | None
    age: timedelta | None
    max_age: timedelta


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    hits: list[ScoredChunk]
    gap_warning: bool
    staleness: StalenessReport


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
