from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal


def source_content_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

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
#: `unverified` is the DEGRADED-mode verdict and carries no trust claim at all. It is not a
#: weaker `ok`: it says the trust gate never ran, because development mode retrieved without a
#: certified threshold. It exists so a degraded hit cannot be mistaken for a judged one — the
#: alternative, reusing `low_confidence`, would have made "we measured this and it scored badly"
#: indistinguishable from "nobody measured anything". Strict mode never produces it.
Verdict = Literal[
    "ok",
    "superseded",
    "expired",
    "not_yet_valid",
    "not_yet_known",
    "low_confidence",
    "invalid_metadata",
    "ambiguous_supersession",
    "not_entailed",
    "unverified",
]


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AtomicFact:
    """A structured, single claim that a deterministic controller can authorize."""

    namespace: str
    subject: str
    predicate: str
    object: Any
    context: dict[str, Any] = field(default_factory=dict)
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    def __post_init__(self) -> None:
        for name in ("namespace", "subject", "predicate"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())
        if not isinstance(self.context, dict):
            object.__setattr__(self, "context", dict(self.context))
        else:
            object.__setattr__(self, "context", dict(self.context))
        if self.valid_from and self.valid_until and self.valid_from >= self.valid_until:
            raise ValueError("valid_until must be after valid_from")

    def to_payload(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "context": self.context,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "AtomicFact":
        if not isinstance(payload, dict):
            raise ValueError("structured fact must be an object")
        def parse_time(value: object) -> datetime | None:
            if value is None:
                return None
            if not isinstance(value, str):
                raise ValueError("fact validity timestamps must be ISO strings")
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return cls(
            namespace=str(payload.get("namespace", "memory")),
            subject=str(payload["subject"]),
            predicate=str(payload["predicate"]),
            object=payload.get("object"),
            context=dict(payload.get("context") or {}),
            valid_from=parse_time(payload.get("valid_from")),
            valid_until=parse_time(payload.get("valid_until")),
        )


@dataclass(frozen=True)
class EvidenceCard:
    """Immutable server-created evidence projection used by the provenance controller."""

    card_id: str
    chunk_id: str
    source: str
    source_digest: str
    valid_from: datetime | None
    valid_until: datetime | None
    first_indexed_at: datetime | None
    indexed_at: datetime | None
    tenant_id: str
    generation_id: str
    pipeline_fingerprint: str | None
    corpus_fingerprint: str | None
    calibration_id: str | None
    calibration_status: str
    trust_state: str
    verdict: Verdict
    confidence: float
    rank: int
    supersession_links: tuple[str, ...] = ()
    contradiction_links: tuple[str, ...] = ()
    support_refs: tuple[str, ...] = ()
    structured_facts: tuple[AtomicFact, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.chunk_id or not self.source or not self.source_digest:
            raise ValueError("evidence card source identity is incomplete")
        if self.rank < 1:
            raise ValueError("evidence card rank must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("evidence card confidence must be in [0, 1]")
        if self.valid_from and self.valid_until and self.valid_from >= self.valid_until:
            raise ValueError("evidence card validity window is invalid")
        from recall.provenance_card_hash import card_payload, card_id_for_payload
        expected = card_id_for_payload(card_payload(self))
        if not self.card_id:
            object.__setattr__(self, "card_id", expected)
        elif self.card_id != expected:
            raise ValueError("evidence card id does not match its canonical content")

    def to_payload(self) -> dict[str, Any]:
        from recall.provenance_card_hash import card_payload
        payload = card_payload(self)
        payload["card_id"] = self.card_id
        for key in ("valid_from", "valid_until", "first_indexed_at", "indexed_at"):
            value = payload[key]
            payload[key] = value.isoformat() if isinstance(value, datetime) else None
        return payload

    @property
    def calibrated(self) -> bool:
        return (
            self.calibration_status == "certified"
            and self.calibration_id is not None
            and self.trust_state == "trusted"
        )


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
    #: Structural relatedness is not a dense cosine and must not be exposed as one.
    score_kind: Literal["dense_cosine", "structural"] = "dense_cosine"


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
    max_dense_score: float | None = None


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
    first_indexed_at: datetime | None = None


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
    gap_warning: bool
    staleness: StalenessReport
    diagnostics: RetrievalDiagnostics = field(default_factory=RetrievalDiagnostics)
    calibration_id: str | None = None
    calibration_status: str = "missing"
    tenant_id: str | None = None
    generation_id: str | None = None
    pipeline_fingerprint: str | None = None
    corpus_fingerprint: str | None = None
    query_set_digest: str | None = None
    #: `trusted` | `degraded`. Carried IN BAND rather than derived, because every adapter that
    #: rebuilds a result from its parts (LangChain documents, LlamaIndex nodes, the MCP JSON) has
    #: to move it explicitly, and a derived property would silently reset to the safe-looking
    #: value on the far side of any boundary that drops metadata. `refused` never appears here:
    #: a strict refusal raises `TrustRefusal` and produces no result object at all.
    trust_state: str = "trusted"
    #: The stable `TrustFailureCode` value when degraded, else None. See `recall.trust_policy`.
    failure_code: str | None = None

    @property
    def calibrated(self) -> bool:
        """True only for a certified artifact exactly bound to this generation.

        Three independent conditions, all required. `trust_state` is in the conjunction on
        purpose: a degraded result that somehow carried a certified status and an artifact id
        would otherwise report as calibrated, which is precisely the failure this session exists
        to make unrepresentable.
        """
        return (
            self.calibration_status == "certified"
            and self.calibration_id is not None
            and self.trust_state == "trusted"
        )
