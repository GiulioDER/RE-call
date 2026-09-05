"""Deterministic write authorization for structured RE-call facts.

This module is deliberately independent of the model. A caller supplies an atomic claim and
opaque evidence-card identifiers. The controller resolves the cards from a server-owned card
store, checks their current lineage and source digests, verifies deterministic structured support,
and only then asks a fact ledger to append the event.

Arbitrary prose is not an authorization input. A prose passage can be useful evidence for a human
or a proposal engine, but it cannot prove an atomic fact here without a structured source record.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, cast

from recall.types import AtomicFact, EvidenceCard, Verdict


CONTROLLER_SCHEMA_VERSION = 1
CONTROLLER_POLICY_VERSION = "provenance-controller-v1"
_PERMIT_SENTINEL = object()
MAX_PROVENANCE_CARD_IDS = 64
MAX_PROVENANCE_IDENTIFIER_CHARS = 256
MAX_PROVENANCE_REQUEST_BYTES = 64 * 1024


class DecisionCode(StrEnum):
    APPLIED = "APPLIED"
    DUPLICATE = "DUPLICATE"
    CARD_NOT_FOUND = "CARD_NOT_FOUND"
    CARD_TAMPERED = "CARD_TAMPERED"
    SOURCE_CHANGED = "SOURCE_CHANGED"
    VALIDITY_EXPIRED = "VALIDITY_EXPIRED"
    VALIDITY_NOT_STARTED = "VALIDITY_NOT_STARTED"
    GENERATION_MISMATCH = "GENERATION_MISMATCH"
    LINEAGE_MISMATCH = "LINEAGE_MISMATCH"
    TRUST_UNAVAILABLE = "TRUST_UNAVAILABLE"
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
    CONTRADICTION_WITHOUT_SUPERSESSION = "CONTRADICTION_WITHOUT_SUPERSESSION"
    FRESH_SEARCH_UNAVAILABLE = "FRESH_SEARCH_UNAVAILABLE"
    FRESH_SEARCH_INSUFFICIENT = "FRESH_SEARCH_INSUFFICIENT"
    LEDGER_UNAVAILABLE = "LEDGER_UNAVAILABLE"
    MATERIALIZATION_UNAVAILABLE = "MATERIALIZATION_UNAVAILABLE"


def _canonical(value: object) -> object:
    """Return JSON-compatible data with deterministic Unicode and mapping order."""
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, datetime):
        instant = _as_utc(value)
        return instant.isoformat().replace("+00:00", "Z")
    if isinstance(value, Mapping):
        return {
            unicodedata.normalize("NFC", str(key)): _canonical(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: unicodedata.normalize("NFC", str(pair[0])),
            )
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite numbers are not valid provenance data")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported value in canonical provenance data: {type(value).__name__}")


def canonical_json(value: object) -> str:
    return json.dumps(_canonical(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def fact_identity(fact: AtomicFact) -> str:
    digest = hashlib.sha256(canonical_json(fact.to_payload()).encode("utf-8")).hexdigest()
    return f"fact_{digest[:32]}"


def fact_conflict_key(fact: AtomicFact) -> str:
    """Identity used to serialize all competing values of one predicate context."""
    payload = {
        "namespace": fact.namespace,
        "subject": fact.subject,
        "predicate": fact.predicate,
        "context": fact.context,
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"conflict_{digest[:32]}"


def source_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def evidence_card_from_payload(payload: Mapping[str, Any]) -> EvidenceCard:
    """Parse a card produced by ``recall_evidence`` for a local CLI handoff."""
    def parse_time(value: object) -> datetime | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("card timestamps must be ISO strings")
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    return EvidenceCard(
        card_id=str(payload["card_id"]),
        chunk_id=str(payload["chunk_id"]),
        source=str(payload["source"]),
        source_digest=str(payload["source_digest"]),
        valid_from=parse_time(payload.get("valid_from")),
        valid_until=parse_time(payload.get("valid_until")),
        first_indexed_at=parse_time(payload.get("first_indexed_at")),
        indexed_at=parse_time(payload.get("indexed_at")),
        tenant_id=str(payload["tenant_id"]),
        generation_id=str(payload["generation_id"]),
        pipeline_fingerprint=payload.get("pipeline_fingerprint"),
        corpus_fingerprint=payload.get("corpus_fingerprint"),
        calibration_id=payload.get("calibration_id"),
        calibration_status=str(payload.get("calibration_status", "missing")),
        trust_state=str(payload.get("trust_state", "degraded")),
        verdict=cast(Verdict, str(payload.get("verdict", "unverified"))),
        confidence=float(payload.get("confidence", 0.0)),
        rank=int(payload["rank"]),
        supersession_links=tuple(payload.get("supersession_links", ())),
        contradiction_links=tuple(payload.get("contradiction_links", ())),
        support_refs=tuple(payload.get("support_refs", ())),
        structured_facts=tuple(
            AtomicFact.from_payload(item) for item in payload.get("structured_facts", ())
        ),
        schema_version=int(payload.get("schema_version", 1)),
    )


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _valid_now(card: EvidenceCard, now: datetime) -> DecisionCode | None:
    instant = _as_utc(now)
    if card.valid_until is not None and instant >= _as_utc(card.valid_until):
        return DecisionCode.VALIDITY_EXPIRED
    if card.valid_from is not None and instant < _as_utc(card.valid_from):
        return DecisionCode.VALIDITY_NOT_STARTED
    return None


def intervals_overlap(left: AtomicFact, right: AtomicFact) -> bool:
    """Whether two fact validity intervals overlap, treating missing ends as open."""
    left_start = _as_utc(left.valid_from) if left.valid_from else datetime.min.replace(tzinfo=UTC)
    right_start = _as_utc(right.valid_from) if right.valid_from else datetime.min.replace(tzinfo=UTC)
    left_end = _as_utc(left.valid_until) if left.valid_until else datetime.max.replace(tzinfo=UTC)
    right_end = _as_utc(right.valid_until) if right.valid_until else datetime.max.replace(tzinfo=UTC)
    return max(left_start, right_start) < min(left_end, right_end)


def facts_conflict(left: AtomicFact, right: AtomicFact) -> bool:
    return (
        left.namespace == right.namespace
        and left.subject == right.subject
        and left.predicate == right.predicate
        and canonical_json(left.context) == canonical_json(right.context)
        and canonical_json(left.object) != canonical_json(right.object)
        and intervals_overlap(left, right)
    )


class EvidenceCardResolver(Protocol):
    def resolve(self, card_id: str) -> EvidenceCard | None: ...


class EvidenceCardStore:
    """Small server-owned immutable card registry.

    The model receives only identifiers. Cards are inserted by the RE-call evidence path and are
    never accepted from an application request. Production deployments may replace this registry
    with a durable projection, but the controller contract remains the same.
    """

    def __init__(self) -> None:
        self._cards: dict[str, EvidenceCard] = {}
        self._lock = threading.RLock()

    def put(self, cards: Iterable[EvidenceCard]) -> None:
        with self._lock:
            for card in cards:
                existing = self._cards.get(card.card_id)
                if existing is not None and existing != card:
                    raise ValueError(f"evidence card identity collision for {card.card_id}")
                self._cards[card.card_id] = card

    def resolve(self, card_id: str) -> EvidenceCard | None:
        with self._lock:
            return self._cards.get(card_id)


@dataclass(frozen=True)
class FactApplicationRequest:
    claim: AtomicFact
    evidence_card_ids: tuple[str, ...]
    request_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must not be empty")
        if len(self.request_id) > MAX_PROVENANCE_IDENTIFIER_CHARS:
            raise ValueError("request_id is too long")
        if not self.evidence_card_ids or len(self.evidence_card_ids) > MAX_PROVENANCE_CARD_IDS:
            raise ValueError("evidence card count is outside the supported bound")
        if any(
            not isinstance(item, str)
            or not item.strip()
            or len(item) > MAX_PROVENANCE_IDENTIFIER_CHARS
            for item in self.evidence_card_ids
        ):
            raise ValueError("at least one evidence card id is required")
        if len(set(self.evidence_card_ids)) != len(self.evidence_card_ids):
            raise ValueError("evidence card ids must be unique")
        encoded = canonical_json(self.claim.to_payload()).encode("utf-8")
        if len(encoded) > MAX_PROVENANCE_REQUEST_BYTES:
            raise ValueError("structured fact payload is too large")


@dataclass(frozen=True)
class FactEvent:
    event_id: str
    event_type: str
    tenant_id: str
    generation_id: str
    fact: AtomicFact | None
    fact_id: str | None
    evidence_cards: tuple[EvidenceCard, ...]
    supersedes_fact_ids: tuple[str, ...]
    request_id: str
    writer: str
    decision_code: str
    policy_version: str
    controller_version: int
    created_at: datetime
    lease_token: str | None = None


@dataclass(frozen=True)
class LedgerApplyResult:
    event: FactEvent
    duplicate: bool = False


class FactApplicationPermit:
    """Single-use capability minted only after controller validation."""

    def __init__(
        self,
        *,
        tenant_id: str,
        generation_id: str,
        request_id: str,
        fact_id: str,
        card_ids: tuple[str, ...],
        _token: object,
    ) -> None:
        self.tenant_id = tenant_id
        self.generation_id = generation_id
        self.request_id = request_id
        self.fact_id = fact_id
        self.card_ids = card_ids
        self._token = _token
        self._used = False


def _consume_permit(
    permit: FactApplicationPermit | None,
    *,
    tenant_id: str,
    generation_id: str,
    request_id: str,
    fact: AtomicFact,
    cards: tuple[EvidenceCard, ...],
) -> None:
    if (
        permit is None
        or permit._token is not _PERMIT_SENTINEL
        or permit._used
        or permit.tenant_id != tenant_id
        or permit.generation_id != generation_id
        or permit.request_id != request_id
        or permit.fact_id != fact_identity(fact)
        or permit.card_ids != tuple(card.card_id for card in cards)
    ):
        raise PermissionError("ledger assertion requires an unused controller permit")
    permit._used = True


class FactLedger(Protocol):
    @property
    def events(self) -> tuple[FactEvent, ...]: ...

    def current(self, *, tenant_id: str, now: datetime) -> tuple[FactEvent, ...]: ...

    def request_event(self, *, tenant_id: str, request_id: str) -> FactEvent | None: ...

    def apply_assertion(
        self,
        *,
        tenant_id: str,
        generation_id: str,
        fact: AtomicFact,
        cards: tuple[EvidenceCard, ...],
        request_id: str,
        writer: str,
        permit: FactApplicationPermit | None = None,
        supersedes_fact_ids: tuple[str, ...] = (),
        policy_version: str = CONTROLLER_POLICY_VERSION,
        controller_version: int = CONTROLLER_SCHEMA_VERSION,
        now: datetime | None = None,
    ) -> LedgerApplyResult: ...

    def record_decision(
        self,
        *,
        tenant_id: str,
        generation_id: str,
        request_id: str,
        writer: str,
        decision_code: str,
        fact: AtomicFact | None,
        cards: tuple[EvidenceCard, ...] = (),
        now: datetime | None = None,
    ) -> FactEvent: ...


class FactMaterializer(Protocol):
    """Synchronous downstream writer for an already authorized ledger event.

    Implementations must be idempotent by ``event.event_id``. The controller may invoke the
    same event again after a materialization failure so recovery never needs a second fact claim.
    """

    def materialize(self, event: FactEvent) -> None: ...


class FactMaterializationOutbox(Protocol):
    """Durable delivery state for an already appended fact event.

    The event snapshot is enqueued before a downstream materializer is called.  Claiming uses a
    short lease, so a process that dies after claiming can be recovered without creating a second
    fact event.  Implementations must scope every operation to the configured tenant.
    """

    def enqueue(self, event: FactEvent) -> None: ...

    def claim(
        self,
        *,
        tenant_id: str,
        event_id: str | None = None,
        limit: int = 100,
        now: datetime | None = None,
    ) -> tuple[FactEvent, ...]: ...

    def mark_applied(self, *, tenant_id: str, event_id: str, lease_token: str | None = None) -> None: ...

    def mark_failed(self, *, tenant_id: str, event_id: str, error: str, lease_token: str | None = None) -> None: ...


class MaterializationRecovery:
    """Replay pending materialization work without re-evaluating or re-asserting facts."""

    def __init__(
        self,
        *,
        tenant_id: str,
        outbox: FactMaterializationOutbox,
        materializer: FactMaterializer,
    ) -> None:
        self.tenant_id = tenant_id
        self.outbox = outbox
        self.materializer = materializer

    def run_once(self, *, limit: int = 100, now: datetime | None = None) -> int:
        """Deliver at most ``limit`` events and return the number marked applied."""
        if not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        delivered = 0
        for event in self.outbox.claim(tenant_id=self.tenant_id, limit=limit, now=now):
            try:
                self.materializer.materialize(event)
            except Exception as exc:
                self.outbox.mark_failed(
                    tenant_id=self.tenant_id,
                    event_id=event.event_id,
                    error=f"{type(exc).__name__}: {exc}"[:2000],
                    lease_token=event.lease_token,
                )
                continue
            self.outbox.mark_applied(
                tenant_id=self.tenant_id,
                event_id=event.event_id,
                lease_token=event.lease_token,
            )
            delivered += 1
        return delivered

    def reconcile(
        self,
        events: Iterable[FactEvent],
        *,
        limit: int = 100,
        now: datetime | None = None,
    ) -> int:
        """Rebuild missing outbox rows from immutable ledger events, then deliver them.

        This is the recovery path for a crash after the ledger transaction committed but before
        the outbox transaction completed. It never calls the controller and never appends a new
        fact event. Event order is canonicalized so two recovery workers converge on the same
        enqueue order; the outbox claim lease still serializes actual delivery.
        """
        if not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        ordered = sorted(events, key=lambda event: (_as_utc(event.created_at), event.event_id))
        for event in ordered:
            if event.tenant_id != self.tenant_id:
                raise ValueError("ledger event tenant mismatch")
            if event.event_type == "asserted":
                self.outbox.enqueue(event)
        return self.run_once(limit=limit, now=now)


class InMemoryFactLedger:
    """Reference ledger used by local workflows and deterministic tests."""

    def __init__(self) -> None:
        self._events: list[FactEvent] = []
        self._by_request: dict[str, FactEvent] = {}
        self._lock = threading.RLock()

    def _event(
        self,
        *,
        event_type: str,
        tenant_id: str,
        generation_id: str,
        request_id: str,
        writer: str,
        decision_code: str,
        fact: AtomicFact | None,
        cards: tuple[EvidenceCard, ...],
        supersedes_fact_ids: tuple[str, ...] = (),
        now: datetime | None = None,
    ) -> FactEvent:
        moment = _as_utc(now or datetime.now(UTC))
        event_id = "evt_" + hashlib.sha256(
            f"{tenant_id}\0{request_id}\0{event_type}\0{len(self._events)}".encode()
        ).hexdigest()[:32]
        return FactEvent(
            event_id=event_id,
            event_type=event_type,
            tenant_id=tenant_id,
            generation_id=generation_id,
            fact=fact,
            fact_id=fact_identity(fact) if fact else None,
            evidence_cards=cards,
            supersedes_fact_ids=supersedes_fact_ids,
            request_id=request_id,
            writer=writer,
            decision_code=decision_code,
            policy_version=CONTROLLER_POLICY_VERSION,
            controller_version=CONTROLLER_SCHEMA_VERSION,
            created_at=moment,
        )

    def current(self, *, tenant_id: str, now: datetime) -> tuple[FactEvent, ...]:
        with self._lock:
            superseded: set[str] = set()
            for event in self._events:
                if event.tenant_id == tenant_id and event.event_type == "superseded":
                    superseded.update(event.supersedes_fact_ids)
            result: list[FactEvent] = []
            instant = _as_utc(now)
            for event in self._events:
                if event.tenant_id != tenant_id or event.event_type != "asserted":
                    continue
                if event.fact_id in superseded or event.fact is None:
                    continue
                if event.fact.valid_until and instant >= _as_utc(event.fact.valid_until):
                    continue
                if event.fact.valid_from and instant < _as_utc(event.fact.valid_from):
                    continue
                result.append(event)
            return tuple(result)

    def request_event(self, *, tenant_id: str, request_id: str) -> FactEvent | None:
        with self._lock:
            event = self._by_request.get(request_id)
            return event if event is not None and event.tenant_id == tenant_id else None

    def _project_candidates(self, *, tenant_id: str) -> tuple[FactEvent, ...]:
        superseded = {
            fact_id
            for event in self._events
            if event.tenant_id == tenant_id and event.event_type == "superseded"
            for fact_id in event.supersedes_fact_ids
        }
        return tuple(
            event
            for event in self._events
            if event.tenant_id == tenant_id
            and event.event_type == "asserted"
            and event.fact_id not in superseded
            and event.fact is not None
        )

    def apply_assertion(
        self,
        *,
        tenant_id: str,
        generation_id: str,
        fact: AtomicFact,
        cards: tuple[EvidenceCard, ...],
        request_id: str,
        writer: str,
        permit: FactApplicationPermit | None = None,
        supersedes_fact_ids: tuple[str, ...] = (),
        policy_version: str = CONTROLLER_POLICY_VERSION,
        controller_version: int = CONTROLLER_SCHEMA_VERSION,
        now: datetime | None = None,
    ) -> LedgerApplyResult:
        del policy_version, controller_version
        with self._lock:
            _consume_permit(
                permit,
                tenant_id=tenant_id,
                generation_id=generation_id,
                request_id=request_id,
                fact=fact,
                cards=cards,
            )
            prior = self._by_request.get(request_id)
            if prior is not None:
                return LedgerApplyResult(prior, duplicate=True)
            candidates = self._project_candidates(tenant_id=tenant_id)
            identity = fact_identity(fact)
            if any(event.fact_id == identity for event in candidates):
                event = self._event(
                    event_type="asserted",
                    tenant_id=tenant_id,
                    generation_id=generation_id,
                    request_id=request_id,
                    writer=writer,
                    decision_code=DecisionCode.DUPLICATE,
                    fact=fact,
                    cards=cards,
                    now=now,
                )
                self._events.append(event)
                self._by_request[request_id] = event
                return LedgerApplyResult(event, duplicate=True)
            conflicts = tuple(
                event
                for event in candidates
                if event.fact is not None and facts_conflict(event.fact, fact)
            )
            conflict_ids = {event.fact_id for event in conflicts}
            if conflict_ids - set(supersedes_fact_ids):
                raise ValueError(DecisionCode.CONTRADICTION_WITHOUT_SUPERSESSION)
            if supersedes_fact_ids and not conflict_ids.issubset(set(supersedes_fact_ids)):
                raise ValueError(DecisionCode.CONTRADICTION_WITHOUT_SUPERSESSION)
            event = self._event(
                event_type="asserted",
                tenant_id=tenant_id,
                generation_id=generation_id,
                request_id=request_id,
                writer=writer,
                decision_code=DecisionCode.APPLIED,
                fact=fact,
                cards=cards,
                supersedes_fact_ids=supersedes_fact_ids,
                now=now,
            )
            self._events.append(event)
            self._by_request[request_id] = event
            if supersedes_fact_ids:
                replacement = self._event(
                    event_type="superseded",
                    tenant_id=tenant_id,
                    generation_id=generation_id,
                    request_id=request_id + ":supersession",
                    writer=writer,
                    decision_code=DecisionCode.APPLIED,
                    fact=fact,
                    cards=cards,
                    supersedes_fact_ids=tuple(supersedes_fact_ids),
                    now=now,
                )
                self._events.append(replacement)
            return LedgerApplyResult(event)

    def record_decision(
        self,
        *,
        tenant_id: str,
        generation_id: str,
        request_id: str,
        writer: str,
        decision_code: str,
        fact: AtomicFact | None,
        cards: tuple[EvidenceCard, ...] = (),
        now: datetime | None = None,
    ) -> FactEvent:
        with self._lock:
            prior = self._by_request.get(request_id)
            if prior is not None:
                return prior
            event = self._event(
                event_type="rejected" if decision_code != DecisionCode.FRESH_SEARCH_INSUFFICIENT else "abstained",
                tenant_id=tenant_id,
                generation_id=generation_id,
                request_id=request_id,
                writer=writer,
                decision_code=decision_code,
                fact=fact,
                cards=cards,
                now=now,
            )
            self._events.append(event)
            self._by_request[request_id] = event
            return event

    @property
    def events(self) -> tuple[FactEvent, ...]:
        with self._lock:
            return tuple(self._events)


@dataclass(frozen=True)
class ControllerDecision:
    allowed: bool
    code: DecisionCode
    request_id: str
    fact_id: str
    cards: tuple[EvidenceCard, ...] = ()
    event: FactEvent | None = None
    retried: bool = False
    detail: str = ""


class ProvenanceController:
    """Complete mediation for structured fact applications."""

    def __init__(
        self,
        *,
        tenant_id: str,
        generation_id: str,
        cards: EvidenceCardResolver,
        ledger: FactLedger,
        source_digest_for: Callable[[EvidenceCard], str | None] | None = None,
        card_revalidator: Callable[[EvidenceCard], EvidenceCard | None] | None = None,
        materializer: FactMaterializer | None = None,
        materialization_outbox: FactMaterializationOutbox | None = None,
        fresh_search: Callable[[AtomicFact, FactApplicationRequest], Sequence[str]] | None = None,
        now: Callable[[], datetime] | None = None,
        writer: str = "unknown",
    ) -> None:
        if not tenant_id or not generation_id:
            raise ValueError("tenant_id and generation_id must be non-empty")
        self.tenant_id = tenant_id
        self.generation_id = generation_id
        self.cards = cards
        self.ledger = ledger
        self.source_digest_for = source_digest_for
        self.card_revalidator = card_revalidator
        self.materializer = materializer
        if materialization_outbox is not None and materializer is None:
            raise ValueError("materialization_outbox requires a materializer")
        self.materialization_outbox = materialization_outbox
        self.fresh_search = fresh_search
        self.now = now or (lambda: datetime.now(UTC))
        self.writer = writer

    def _resolve(self, request: FactApplicationRequest) -> tuple[EvidenceCard, ...] | DecisionCode:
        resolved: list[EvidenceCard] = []
        for card_id in request.evidence_card_ids:
            try:
                card = self.cards.resolve(card_id)
            except Exception:
                # A malformed or tampered durable payload is a card failure, not a controller
                # crash.  The caller may perform the one bounded fresh-search recovery.
                return DecisionCode.CARD_TAMPERED
            if card is None:
                return DecisionCode.CARD_NOT_FOUND
            if card.card_id != card_id:
                return DecisionCode.CARD_TAMPERED
            if card.tenant_id != self.tenant_id:
                return DecisionCode.LINEAGE_MISMATCH
            if card.generation_id != self.generation_id:
                return DecisionCode.GENERATION_MISMATCH
            if self.card_revalidator is not None:
                try:
                    current_card = self.card_revalidator(card)
                except Exception:
                    return DecisionCode.CARD_TAMPERED
                if current_card is None:
                    return DecisionCode.SOURCE_CHANGED
                if current_card.card_id != card_id:
                    return DecisionCode.SOURCE_CHANGED
                card = current_card
            if card.trust_state != "trusted" or card.verdict != "ok" or not card.calibrated:
                return DecisionCode.TRUST_UNAVAILABLE
            validity = _valid_now(card, self.now())
            if validity is not None:
                return validity
            if self.source_digest_for is not None:
                current_digest = self.source_digest_for(card)
                if current_digest is None:
                    return DecisionCode.SOURCE_CHANGED
                if current_digest != card.source_digest:
                    return DecisionCode.SOURCE_CHANGED
            resolved.append(card)
        return tuple(resolved)

    @staticmethod
    def _supported(fact: AtomicFact, cards: Sequence[EvidenceCard]) -> bool:
        return any(fact_identity(source_fact) == fact_identity(fact) for card in cards for source_fact in card.structured_facts)

    @staticmethod
    def _superseding_fact_ids(fact: AtomicFact, cards: Sequence[EvidenceCard], current: Sequence[FactEvent]) -> tuple[str, ...]:
        old_ids: list[str] = []
        for event in current:
            if event.fact is None or not facts_conflict(event.fact, fact):
                continue
            old_source_ids = {
                value
                for card in event.evidence_cards
                for value in (card.card_id, card.chunk_id, card.source)
            }
            allowed = any(
                event.fact_id in card.supersession_links or old_source_ids.intersection(card.supersession_links)
                for card in cards
            )
            if allowed and event.fact_id is not None:
                old_ids.append(event.fact_id)
        return tuple(old_ids)

    def _decision(
        self,
        request: FactApplicationRequest,
        code: DecisionCode,
        *,
        cards: tuple[EvidenceCard, ...] = (),
        retried: bool = False,
        detail: str = "",
        event: FactEvent | None = None,
    ) -> ControllerDecision:
        return ControllerDecision(
            allowed=code in {DecisionCode.APPLIED, DecisionCode.DUPLICATE},
            code=code,
            request_id=request.request_id,
            fact_id=fact_identity(request.claim),
            cards=cards,
            event=event,
            retried=retried,
            detail=detail,
        )

    def apply_fact(self, request: FactApplicationRequest) -> ControllerDecision:
        """Validate and append one fact, with at most one deterministic fresh-search retry."""
        try:
            prior = self.ledger.request_event(tenant_id=self.tenant_id, request_id=request.request_id)
        except Exception as exc:
            return self._record_refusal(
                request,
                DecisionCode.LEDGER_UNAVAILABLE,
                retried=False,
                cards=(),
                detail=type(exc).__name__,
            )
        if prior is not None and prior.decision_code not in {
            DecisionCode.APPLIED,
            DecisionCode.DUPLICATE,
        }:
            try:
                code = DecisionCode(prior.decision_code)
            except ValueError:
                code = DecisionCode.LEDGER_UNAVAILABLE
            return self._decision(
                request,
                code,
                cards=prior.evidence_cards,
                event=prior,
            )
        resolved = self._resolve(request)
        retried = False
        if isinstance(resolved, DecisionCode) or not self._supported(request.claim, resolved):
            if self.fresh_search is None:
                code = resolved if isinstance(resolved, DecisionCode) else DecisionCode.UNSUPPORTED_CLAIM
                return self._record_refusal(request, code, (), retried=False)
            try:
                fresh_ids = tuple(self.fresh_search(request.claim, request))
            except Exception as exc:
                return self._record_refusal(
                    request, DecisionCode.FRESH_SEARCH_UNAVAILABLE, (), retried=True, detail=type(exc).__name__
                )
            retried = True
            if not fresh_ids:
                return self._record_refusal(request, DecisionCode.FRESH_SEARCH_INSUFFICIENT, (), retried=True)
            refreshed = FactApplicationRequest(request.claim, fresh_ids, request.request_id)
            resolved = self._resolve(refreshed)
            if isinstance(resolved, DecisionCode) or not self._supported(request.claim, resolved):
                code = resolved if isinstance(resolved, DecisionCode) else DecisionCode.FRESH_SEARCH_INSUFFICIENT
                return self._record_refusal(request, code, (), retried=True)
        cards = tuple(resolved)
        try:
            current = self.ledger.current(tenant_id=self.tenant_id, now=self.now())
        except Exception as exc:
            # The read that establishes the conflict set is part of authorization.  Treating a
            # failed read as an empty set would turn an outage into an authorization bypass, so
            # fail closed before minting a permit or attempting any append.
            return self._record_refusal(
                request,
                DecisionCode.LEDGER_UNAVAILABLE,
                cards,
                retried=retried,
                detail=type(exc).__name__,
            )
        supersedes = self._superseding_fact_ids(request.claim, cards, current)
        conflicts = tuple(event for event in current if event.fact and facts_conflict(event.fact, request.claim))
        if conflicts and len(supersedes) != len(conflicts):
            return self._record_refusal(
                request, DecisionCode.CONTRADICTION_WITHOUT_SUPERSESSION, cards, retried=retried
            )
        try:
            permit = FactApplicationPermit(
                tenant_id=self.tenant_id,
                generation_id=self.generation_id,
                request_id=request.request_id,
                fact_id=fact_identity(request.claim),
                card_ids=tuple(card.card_id for card in cards),
                _token=_PERMIT_SENTINEL,
            )
            assertion_kwargs = {
                "tenant_id": self.tenant_id,
                "generation_id": self.generation_id,
                "fact": request.claim,
                "cards": cards,
                "request_id": request.request_id,
                "writer": self.writer,
                "permit": permit,
                "supersedes_fact_ids": supersedes,
                "now": self.now(),
            }
            atomic_apply = getattr(self.ledger, "apply_assertion_with_outbox", None)
            if self.materialization_outbox is not None and callable(atomic_apply):
                result = atomic_apply(
                    materialization_outbox=self.materialization_outbox, **assertion_kwargs
                )
            else:
                result = self.ledger.apply_assertion(
                    tenant_id=self.tenant_id,
                    generation_id=self.generation_id,
                    fact=request.claim,
                    cards=cards,
                    request_id=request.request_id,
                    writer=self.writer,
                    permit=permit,
                    supersedes_fact_ids=supersedes,
                    now=self.now(),
                )
        except ValueError as exc:
            try:
                code = DecisionCode(str(exc))
            except ValueError:
                code = DecisionCode.LEDGER_UNAVAILABLE
            return self._record_refusal(request, code, cards, retried=retried, detail=str(exc))
        except Exception as exc:
            return self._record_refusal(request, DecisionCode.LEDGER_UNAVAILABLE, cards, retried=retried, detail=type(exc).__name__)
        if self.materializer is not None:
            try:
                event = result.event
                if self.materialization_outbox is not None:
                    self.materialization_outbox.enqueue(event)
                    claimed = self.materialization_outbox.claim(
                        tenant_id=self.tenant_id,
                        event_id=event.event_id,
                        limit=1,
                        now=self.now(),
                    )
                    # An already applied outbox entry is a successful idempotent replay.  It is
                    # important not to invoke an external writer again in that case.
                    if not claimed:
                        return self._decision(
                            request,
                            DecisionCode.DUPLICATE if result.duplicate else DecisionCode.APPLIED,
                            cards=cards,
                            retried=retried,
                            event=event,
                        )
                    event = claimed[0]
                self.materializer.materialize(event)
                if self.materialization_outbox is not None:
                        self.materialization_outbox.mark_applied(
                        tenant_id=self.tenant_id, event_id=event.event_id,
                        lease_token=event.lease_token,
                    )
            except Exception as exc:
                # The append is the durable intent. Do not claim the downstream fact store was
                # updated, and leave the outbox event available for an idempotent recovery retry.
                if self.materialization_outbox is not None:
                    try:
                        self.materialization_outbox.mark_failed(
                            tenant_id=self.tenant_id,
                            event_id=result.event.event_id,
                        error=f"{type(exc).__name__}: {exc}"[:2000],
                        lease_token=event.lease_token,
                        )
                    except Exception:
                        # The durable ledger remains the source of intent even if the outbox
                        # dependency is unavailable. The caller still gets a fail-closed result.
                        pass
                return self._decision(
                    request,
                    DecisionCode.MATERIALIZATION_UNAVAILABLE,
                    cards=cards,
                    retried=retried,
                    detail=type(exc).__name__,
                    event=result.event,
                )
        return self._decision(
            request,
            DecisionCode.DUPLICATE if result.duplicate else DecisionCode.APPLIED,
            cards=cards,
            retried=retried,
            event=result.event,
        )

    def _record_refusal(
        self,
        request: FactApplicationRequest,
        code: DecisionCode,
        cards: tuple[EvidenceCard, ...],
        *,
        retried: bool,
        detail: str = "",
    ) -> ControllerDecision:
        try:
            event = self.ledger.record_decision(
                tenant_id=self.tenant_id,
                generation_id=self.generation_id,
                request_id=request.request_id,
                writer=self.writer,
                decision_code=code,
                fact=request.claim,
                cards=cards,
                now=self.now(),
            )
        except Exception:
            event = None
        return self._decision(request, code, cards=cards, retried=retried, detail=detail, event=event)


def cards_from_trusted_result(result: Any, *, selected_only: bool = True) -> tuple[EvidenceCard, ...]:
    """Build immutable cards from a trusted result without changing existing evidence semantics."""
    cards: list[EvidenceCard] = []
    rank = 0
    for hit in result.hits:
        if hit.verdict != "ok":
            continue
        rank += 1
        if selected_only and hasattr(result, "selected_chunk_ids") and hit.chunk.id not in result.selected_chunk_ids:
            continue
        metadata = hit.chunk.metadata or {}
        graph = metadata.get("recall_graph", {})
        if not isinstance(graph, Mapping):
            graph = {}
        facts = tuple(
            AtomicFact.from_payload(item)
            for item in graph.get("facts", metadata.get("facts", []))
            if isinstance(item, Mapping)
        )
        def links(key: str) -> tuple[str, ...]:
            values: list[str] = []
            raw_values = [graph.get(key, metadata.get(key, ()))]
            if key == "authored_supersedes":
                raw_values.append(metadata.get("supersedes"))
            for raw in raw_values:
                if isinstance(raw, str):
                    raw = (raw,)
                if isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
                    values.extend(item for item in raw if isinstance(item, str) and item)
            return tuple(dict.fromkeys(values))
        file_name = hit.provenance.file or hit.chunk.source
        raw_digest = metadata.get("content_hash") or metadata.get("source_digest")
        digest = (
            str(raw_digest)
            if isinstance(raw_digest, str) and raw_digest
            else source_digest(hit.chunk.text)
        )
        cards.append(
            EvidenceCard(
                card_id="",
                chunk_id=hit.chunk.id,
                source=file_name,
                source_digest=digest,
                valid_from=hit.validity.valid_from,
                valid_until=hit.validity.valid_until,
                first_indexed_at=hit.provenance.first_indexed_at or hit.provenance.indexed_at,
                indexed_at=hit.provenance.indexed_at,
                tenant_id=result.tenant_id or "legacy",
                generation_id=result.generation_id or result.diagnostics.index_generation,
                pipeline_fingerprint=result.pipeline_fingerprint,
                corpus_fingerprint=result.corpus_fingerprint,
                calibration_id=result.calibration_id,
                calibration_status=result.calibration_status,
                trust_state=result.trust_state,
                verdict=hit.verdict,
                confidence=hit.confidence,
                rank=rank,
                supersession_links=links("authored_supersedes"),
                contradiction_links=links("authored_contradicts"),
                support_refs=links("support_refs"),
                structured_facts=facts,
            )
        )
    return tuple(cards)
