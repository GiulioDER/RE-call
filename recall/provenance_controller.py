"""Deterministic write authorization for structured RE-call facts.

This module is deliberately independent of the model. A caller supplies an atomic claim and
opaque evidence-card identifiers. The controller resolves the cards from a server-owned card
store, checks their current lineage and source digests, verifies deterministic structured support,
and only then asks a fact ledger to append the event.

The card checks run before the append, outside the ledger's key lock, but an application holds
the ledger's erasure lock shared from its checks to its append and a forget or withdraw holds it
exclusively for its whole erase (`ProvenanceController.apply_fact`,
`recall.fact_ledger.PostgresFactLedger.erasure_lock`), so a forget either sees the assertion in
its plan or runs first and the checks find the memo gone (STAKES-003). A ledger without the lock
(the in-memory reference, `SQLiteFactLedger`) orders nothing.

Arbitrary prose is not an authorization input. A prose passage can be useful evidence for a human
or a proposal engine, but it cannot prove an atomic fact here without a structured source record.
"""
from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Protocol, cast, get_args

from recall import eval_clock
from recall.observability import get_logger
from recall.supersession import reference_path_key, reference_stem_key
from recall.types import (
    AtomicFact,
    EvidenceCard,
    FactCardinality,
    RedactedEvidenceCard,
    Verdict,
    source_content_digest,
)
from recall.fact_events import (  # re-exported: moved verbatim, same objects
    _as_utc as _as_utc,
    _canonical as _canonical,
    canonical_json as canonical_json,
    DecisionCode as DecisionCode,
    fact_identity as fact_identity,
    FactEvent as FactEvent,
)
from recall.fact_projection import (  # re-exported: moved verbatim, same objects
    _EARLIEST as _EARLIEST,
    _FactSuccessorIndex as _FactSuccessorIndex,
    _holds as _holds,
    _LATEST as _LATEST,
    _merge as _merge,
    _span as _span,
    _without as _without,
    applications as applications,
    asserts_fact as asserts_fact,
    fact_conflict_key as fact_conflict_key,
    fact_supersession as fact_supersession,
    fact_window as fact_window,
    FactSupersession as FactSupersession,
    FactWindow as FactWindow,
    Interval as Interval,
    ordered_after as ordered_after,
    reserved_request_id as reserved_request_id,
    retracted_keys as retracted_keys,
    retraction_key as retraction_key,
    RETRACTION_REQUEST_PREFIX as RETRACTION_REQUEST_PREFIX,
    RetractionKey as RetractionKey,
    STATED_REQUEST_PREFIX as STATED_REQUEST_PREFIX,
    SUPERSESSION_REQUEST_SUFFIX as SUPERSESSION_REQUEST_SUFFIX,
    WITHDRAWAL_REQUEST_PREFIX as WITHDRAWAL_REQUEST_PREFIX,
    withdrawn_event_ids as withdrawn_event_ids,
)


CONTROLLER_SCHEMA_VERSION = 1
CONTROLLER_POLICY_VERSION = "provenance-controller-v1"
#: The trust state of a card the server builds from a memo that states its own facts
#: (`recall.stated_facts`). No search produced it, so `trusted_for_application` refuses it and it
#: authorizes `ProvenanceController.apply_stated` only.
STATED_TRUST_STATE = "stated"
#: The writer and policy version of the events the stated path appends.
STATED_WRITER = "recall_stated"
STATED_POLICY_VERSION = "stated-facts-v1"
_PERMIT_SENTINEL = object()
MAX_PROVENANCE_CARD_IDS = 64
MAX_PROVENANCE_IDENTIFIER_CHARS = 256
MAX_PROVENANCE_REQUEST_BYTES = 64 * 1024
#: The most outbox events one `MaterializationRecovery` pass claims.
MAX_OUTBOX_CLAIM_LIMIT = 1000
#: The longest delivery error an outbox row keeps.
MAX_OUTBOX_ERROR_CHARS = 2000


#: Given memo names (an evidence card's `source`), the ones the store serves now.
ServedSources = Callable[[frozenset[str]], Collection[str]]


class ServedState:
    """What a store serves now, per memo name, as the projections ask for it.

    Called like a `ServedSources` (names in, the served ones out). A store that can say more
    answers three further questions: the content hashes it serves each name with
    (`content_hashes`; None only when the store has no such lookup, and then no stated card is
    checked against content; a lookup that FAILED answers an empty mapping, so no stated card
    lives), the names its supersession graph retires (`superseded`), and the names whose memos
    the operator's rule lets state facts now (`stated_admitted`; None means it was not asked, and
    then no stated card is checked against the rule). A fourth hook, `superseded_at`, answers
    the retired names at an instant (the claimant's own validity window, the window rule); when
    given it takes precedence over `superseded`, which then retires at every instant.
    """

    def __init__(
        self,
        served: Callable[[frozenset[str]], Collection[str]],
        *,
        content_hashes: Callable[[frozenset[str]], Mapping[str, frozenset[str]] | None] | None = None,
        superseded: Callable[[frozenset[str]], frozenset[str]] | None = None,
        stated_admitted: Callable[[frozenset[str]], frozenset[str]] | None = None,
        superseded_at: Callable[[frozenset[str], datetime | None], frozenset[str]] | None = None,
    ) -> None:
        self._served = served
        self._content_hashes = content_hashes
        self._superseded = superseded
        self._superseded_at = superseded_at
        self._stated_admitted = stated_admitted

    def __call__(self, names: frozenset[str]) -> Collection[str]:
        return self._served(names)

    def content_hashes(self, names: frozenset[str]) -> Mapping[str, frozenset[str]] | None:
        return None if self._content_hashes is None else self._content_hashes(names)

    def superseded(self, names: frozenset[str], now: datetime | None = None) -> frozenset[str]:
        """The names a claimant memo retires at valid time `now`: `superseded_at` reads each
        claimant's own window (the window rule); a plain `superseded` holds at every instant."""
        if self._superseded_at is not None:
            return self._superseded_at(names, now)
        return frozenset() if self._superseded is None else self._superseded(names)

    def stated_admitted(self, names: frozenset[str]) -> frozenset[str] | None:
        return None if self._stated_admitted is None else self._stated_admitted(names)


def card_liveness(
    served: ServedSources, names: frozenset[str], now: datetime | None = None
) -> Callable[[EvidenceCard], bool]:
    """Whether the memo behind a card still stands behind it, asked once for `names`.

    An applied card lives while its memo's name is served (#932). A stated card lives while its
    memo is served with the content hash the card recorded, so an edit retires what the old text
    stated and a revert restores it, and while the operator's rule still lets that memo state
    facts, so narrowing the rule or turning it off retires what it recorded. Neither lives while
    the store's supersession graph retires its memo: memo-level, the graph search's `superseded`
    verdict comes from, counting only memos the operator lets state facts (`ServedState`).
    """
    present = set(served(names)) if names else set()
    retired: frozenset[str] = frozenset()
    if isinstance(served, ServedState) and names:
        retired = served.superseded(names, now)
    # Asked only once a stated card needs them: on a PostgreSQL tenant holding applied facts
    # alone, the content lookup is a scan of the served generation that nothing would use.
    stated: list[tuple[Mapping[str, frozenset[str]] | None, frozenset[str] | None]] = []

    def stated_view() -> tuple[Mapping[str, frozenset[str]] | None, frozenset[str] | None]:
        if not stated:
            if isinstance(served, ServedState) and names:
                stated.append((served.content_hashes(names), served.stated_admitted(names)))
            else:
                stated.append((None, None))
        return stated[0]

    def live(card: EvidenceCard) -> bool:
        if isinstance(card, RedactedEvidenceCard):
            return False  # its memo was forgotten and its content erased; a namesake is another memo
        if card.source not in present or card.source in retired:
            return False
        if card.trust_state == STATED_TRUST_STATE:
            hashes, admitted = stated_view()
            if admitted is not None and card.source not in admitted:
                return False
            if hashes is not None:
                return card.source_digest in hashes.get(card.source, frozenset())
        return True

    return live


def absent_fact_ids(
    events: Sequence[FactEvent], served: ServedSources | None, now: datetime | None = None
) -> frozenset[str]:
    """Facts none of whose evidence the store serves now: every card names a memo it lacks.

    A memo deleted from disk, pruned, or left out of the promoted generation is a STATE, not a
    decision: a rollback, a restored file or a repaired sync brings it back. So a fact whose every
    card, across all its live applications (DUPLICATE ones included), names a memo the store does
    not serve is hidden at read time and reappears when the memo does, with nothing written.
    A stated card also needs its memo served with the content it recorded, and no card counts
    while the store's supersession graph retires its memo (`card_liveness`).
    `recall_forget` is the decision, and writes a `retracted` event instead. `served` None checks
    nothing (every fact counts as present), which is how a caller without a store reads the
    ledger.
    """
    if served is None:
        return frozenset()
    cards = assertion_cards(events)
    names = frozenset(card.source for group in cards.values() for card in group)
    live = card_liveness(served, names, now)
    return frozenset(
        fact_id for fact_id, group in cards.items()
        if group and not any(live(card) for card in group)
    )


def assertion_cards(
    events: Sequence[FactEvent], *, index: FactSupersession | None = None,
) -> dict[str, tuple[EvidenceCard, ...]]:
    """Per fact id, the cards of every application of it not retracted, DUPLICATE ones included.

    In ledger order, each card once. A fact applied from a second memo is recorded DUPLICATE with
    that memo's card, so the APPLIED event's cards alone name only the first memo: a successor
    linking the second was refused as a contradiction, and the listing cited the first memo after
    the store stopped serving it (2026-10-21 supersession audit). With an index, only applications
    still in force contribute support. DUPLICATE cards belong to the latest unretracted APPLIED
    episode of their fact, rather than creating an independently surviving assertion.
    """
    retracted = retracted_keys(events)
    counted = applications(events) if index is not None else {}
    cards: dict[str, dict[str, EvidenceCard]] = {}
    episode_in_force: dict[str, bool] = {}
    for event in events:
        if event.event_type == "asserted" and event.fact_id is not None and retraction_key(event) not in retracted:
            if index is not None:
                if event.event_id in counted:
                    episode_in_force[event.fact_id] = bool(index.in_force(event))
                if not episode_in_force.get(event.fact_id, False):
                    continue
            group = cards.setdefault(event.fact_id, {})
            for card in event.evidence_cards:
                group.setdefault(card.card_id, card)
    return {fact_id: tuple(group.values()) for fact_id, group in cards.items()}


def live_cards(
    events: Sequence[FactEvent], served: ServedSources | None, now: datetime | None = None
) -> dict[str, tuple[EvidenceCard, ...]]:
    """Per fact id, its live cards across its live applications (`assertion_cards`, `card_liveness`).

    What `recall_current_facts` cites as a fact's evidence. `served` None counts every card.
    """
    cards = assertion_cards(events)
    if served is None:
        return cards
    live = card_liveness(served, frozenset(card.source for group in cards.values() for card in group), now)
    return {fact_id: tuple(card for card in group if live(card)) for fact_id, group in cards.items()}


def live_sources(
    events: Sequence[FactEvent], served: ServedSources | None, now: datetime | None = None
) -> dict[str, tuple[str, ...]]:
    """Per fact id, the sorted memo names of its live cards, across its live applications.

    What `recall_current_facts` shows as a fact's `sources`, so an answer can cite the memo. A
    DUPLICATE application counts, since a second memo stating a fact is a second source. `served`
    None counts every card.
    """
    return {
        fact_id: tuple(sorted({card.source for card in group}))
        for fact_id, group in live_cards(events, served, now).items()
    }


def current_fact_listing(
    ledger: Any, tenant_id: str, now: datetime, served: ServedSources | None
) -> tuple[tuple[FactEvent, ...], dict[str, tuple[EvidenceCard, ...]]]:
    """The facts in force at `now` and, per fact id, the live cards that cite them.

    The one listing rule `recall_current_facts` and `recall provenance current` share, so the two
    surfaces cannot disagree about one ledger; the CLI printed the APPLIED event's own cards and
    left out every memo that restated a fact (2026-10-21 supersession audit). One snapshot serves
    both halves: the materialized current facts and the listed facts' live assertions where the
    ledger has them (`recall.fact_current`), else the history replayed, else what `current`
    returns. `served` None reads the ledger alone and counts every card.
    """
    view = getattr(ledger, "current_view", None)
    if callable(view):
        events, history = view(tenant_id=tenant_id, now=now, served=served)
    elif (recorded := getattr(ledger, "events", None)) is not None:
        history = tuple(event for event in recorded if event.tenant_id == tenant_id)
        events = project_current(history, now, served)
    else:
        events = (
            ledger.current(tenant_id=tenant_id, now=now)
            if served is None
            else ledger.current(tenant_id=tenant_id, now=now, served=served)
        )
        history = events
    return tuple(events), live_cards(history, served, now)


def listing_payload(
    events: Sequence[FactEvent],
    evidence: Mapping[str, tuple[EvidenceCard, ...]],
    permitted: Callable[[str], bool] | None = None,
) -> dict[str, list[dict[str, object]]]:
    """The `facts` and `contested` halves of a current-facts listing, as both surfaces print them.

    A fact lists its memos in `sources` and their cards in `evidence_card_ids`. With `permitted`,
    a memo it refuses is left out of both, and a fact left with no permitted memo is not listed.
    Facts that conflict with no supersession between their memos go under `contested`
    (`split_contested`).
    """
    sources = {
        fact_id: tuple(sorted({card.source for card in group})) for fact_id, group in evidence.items()
    }
    if permitted is not None:
        sources = {fact_id: tuple(name for name in names if permitted(name))
                   for fact_id, names in sources.items()}
        events = tuple(event for event in events if sources.get(event.fact_id or ""))

    def card_ids(event: FactEvent) -> list[str]:
        allowed = sources.get(event.fact_id or "", ())
        return [card.card_id for card in evidence.get(event.fact_id or "", ())
                if permitted is None or card.source in allowed]

    def names(event: FactEvent) -> list[str]:
        return list(sources.get(event.fact_id or "", ()))

    standing, contested = split_contested(events)
    return {
        "facts": [
            {
                "event_id": event.event_id,
                "fact_id": event.fact_id,
                "fact": event.fact.to_payload() if event.fact else None,
                "evidence_card_ids": card_ids(event),
                "generation_id": event.generation_id,
                "writer": event.writer,
                "asserted_at": event.created_at.isoformat(),
                "sources": names(event),
            }
            for event in standing
        ],
        "contested": [
            {
                "namespace": group[0].fact.namespace if group[0].fact else None,
                "subject": group[0].fact.subject if group[0].fact else None,
                "predicate": group[0].fact.predicate if group[0].fact else None,
                "context": group[0].fact.context if group[0].fact else {},
                "values": [
                    {
                        "fact": event.fact.to_payload() if event.fact else None,
                        "fact_id": event.fact_id,
                        "sources": names(event),
                        "evidence_card_ids": card_ids(event),
                    }
                    for event in group
                ],
            }
            for group in contested
        ],
    }


def split_contested(
    current: Sequence[FactEvent],
) -> tuple[tuple[FactEvent, ...], tuple[tuple[FactEvent, ...], ...]]:
    """`current` split into the facts that stand alone and the groups that disagree.

    Two current facts that conflict (`facts_conflict`: same key, another value, overlapping
    windows) are both contested: memos that state different values with no supersession between
    them leave the question open, and neither is listed as the answer. Values in separate windows
    are history, not a dispute. A group is the contested facts of one conflict key, in order.
    """
    by_key: dict[str, list[FactEvent]] = {}
    for event in current:
        if event.fact is not None:
            by_key.setdefault(fact_conflict_key(event.fact), []).append(event)
    contested: set[str] = set()
    for group in by_key.values():
        # Two values of one multi-valued cardinality never conflict (`facts_conflict`), so only a
        # pair with a single-valued side, or of two different cardinalities, is compared. Every
        # pair was: a subject's k emails or visits cost k^2 / 2 comparisons on each current-facts
        # listing, about 53 s at k=1000 (2026-11-04 hunt, PERF-001).
        kinds = [predicate_cardinality(event.fact) for event in group if event.fact is not None]
        if "single" not in kinds and len(set(kinds)) == 1:
            continue
        for index, left in enumerate(group):
            for offset, right in enumerate(group[index + 1:], start=index + 1):
                if kinds[index] == kinds[offset] != "single":
                    continue
                if left.fact and right.fact and facts_conflict(left.fact, right.fact):
                    contested.update((left.event_id, right.event_id))
    standing = tuple(event for event in current if event.event_id not in contested)
    groups = tuple(
        tuple(event for event in group if event.event_id in contested)
        for group in by_key.values()
        if any(event.event_id in contested for event in group)
    )
    return standing, groups


def _link_names(link: str, sources: Sequence[str]) -> bool:
    """Whether a `supersedes:` link designates one of `sources`, by the store's rule.

    A link that names a folder designates the memo at that path (`recall.supersession`,
    `ReferenceIndex.match`), read as root-relative, so `mem/user-drink.md` names a memo stored as
    `/root/mem/user-drink.md`. Its file name alone designated every memo of that name, so
    `supersedes: home/status.md` retired the fact of `work/status.md` (2026-11-04 hunt,
    STAKES-002). A bare name is a stem: `User-Drink.md` designates `mem/user-drink.md`, which a
    lite store, naming memos relative to RECALL_INDEX_ROOT, needs.
    """
    path = reference_path_key(link)
    if not path:
        return False
    if "/" in path:
        return any(
            key == path or key.endswith("/" + path)
            for key in (reference_path_key(source) for source in sources)
        )
    return any(reference_stem_key(source) == path for source in sources)


#: The digest a card's `source_digest` holds; one helper, so cards built here and by
#: `recall.evidence` and the digests the MCP and CLI paths compare cannot drift apart.
source_digest = source_content_digest


def evidence_card_from_payload(payload: Mapping[str, Any]) -> EvidenceCard:
    """Parse a card produced by ``recall_evidence`` for a local CLI handoff.

    A card `recall_forget` erased (`recall.fact_redaction`) parses as a `RedactedEvidenceCard`.
    """
    from recall.provenance_json import decode_jsonb
    payload = decode_jsonb(payload)
    if payload.get("redacted") is True:
        return RedactedEvidenceCard.from_marker(payload)

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
        verdict=stored_verdict(payload.get("verdict", "unverified")),
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


_log = get_logger("provenance")

#: The verdicts this release defines (`recall.types.Verdict`).
KNOWN_VERDICTS: frozenset[str] = frozenset(get_args(Verdict))


def stored_verdict(value: object) -> Verdict:
    """A stored card's verdict, as written.

    A value this release does not define (a newer release's, or a typo) is declared unknown, in
    a warning naming it, and kept, never refused: the card's id covers its verdict, so rewriting
    it would make the stored card read as tampered, and raising lost every card of a ledger read.
    Only `ok` is trusted for application (`EvidenceCard.trusted_for_application`), so an unknown
    verdict never authorizes a fact (2026-11-04 owner decision, CODE-011). The return type is
    `Verdict` because `EvidenceCard.verdict` is; a stored card is the one place it can hold
    another value, which is why this is a cast and not a check (CODE-003).
    """
    verdict = str(value)
    if not verdict_known(verdict):
        _log.warning("a stored evidence card has verdict %r, which this release does not define", verdict)
    return cast(Verdict, verdict)


def verdict_known(verdict: str) -> bool:
    """Whether `verdict` is one this release defines (`KNOWN_VERDICTS`)."""
    return verdict in KNOWN_VERDICTS


def _valid_now(card: EvidenceCard, now: datetime) -> DecisionCode | None:
    instant = _as_utc(now)
    if card.valid_until is not None and instant >= _as_utc(card.valid_until):
        return DecisionCode.VALIDITY_EXPIRED
    if card.valid_from is not None and instant < _as_utc(card.valid_from):
        return DecisionCode.VALIDITY_NOT_STARTED
    return None


def intervals_overlap(left: AtomicFact, right: AtomicFact) -> bool:
    """Whether two fact validity intervals overlap, treating missing ends as open.

    On the same bounds as the supersession index (`_span`), so a refusal and a projection agree
    on what an open end means."""
    left_start, left_end = _span(fact_window(left))
    right_start, right_end = _span(fact_window(right))
    return max(left_start, right_start) < min(left_end, right_end)


PredicateCardinality = FactCardinality

#: The cardinality of a predicate a fact does not declare its own for (matched case-insensitively).
#: Every predicate not named here is single-valued, which is the conservative reading: a second
#: value of a single-valued predicate is refused or contested, never silently added.
DEFAULT_PREDICATE_CARDINALITY: Mapping[str, PredicateCardinality] = MappingProxyType(
    {
        **dict.fromkeys(
            (
                "email", "emails", "email_address", "phone", "phones", "phone_number", "alias",
                "aliases", "nickname", "nicknames", "tag", "tags", "language", "languages",
                "skill", "skills", "member_of", "child", "children", "pet", "pets",
            ),
            "set",
        ),
        **dict.fromkeys(("event", "visited", "attended", "met", "occurred"), "event"),
    }
)


def predicate_cardinality(fact: AtomicFact) -> PredicateCardinality:
    """The fact's own declared cardinality, else its predicate's, else ``single``."""
    if fact.cardinality is not None:
        return fact.cardinality
    return DEFAULT_PREDICATE_CARDINALITY.get(fact.predicate.casefold(), "single")


def facts_conflict(left: AtomicFact, right: AtomicFact) -> bool:
    """Whether two facts disagree: one subject, predicate and context, another value, overlapping
    windows, and not both of one multi-valued cardinality. Two set-valued values (a second email)
    or two events (a second visit) hold at once, so they never contradict each other
    (`predicate_cardinality`). Two that disagree on cardinality do conflict: a fact declaring
    `set` slipped a second value past a single-valued one, both standing and neither contested
    (2026-10-21 audit, STAKES-002). Compared on the canonical (NFC) forms, as the conflict key is."""
    kind = predicate_cardinality(left)
    return (
        left.namespace == right.namespace
        and left.subject == right.subject
        and left.predicate == right.predicate
        and canonical_json(left.context) == canonical_json(right.context)
        and canonical_json(left.object) != canonical_json(right.object)
        and (kind == "single" or kind != predicate_cardinality(right))
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
            raise ValueError(
                "evidence card ids must be non-empty strings of at most "
                f"{MAX_PROVENANCE_IDENTIFIER_CHARS} characters"
            )
        if len(set(self.evidence_card_ids)) != len(self.evidence_card_ids):
            raise ValueError("evidence card ids must be unique")
        encoded = canonical_json(self.claim.to_payload()).encode("utf-8")
        if len(encoded) > MAX_PROVENANCE_REQUEST_BYTES:
            raise ValueError("structured fact payload is too large")


@dataclass(frozen=True)
class LedgerApplyResult:
    event: FactEvent
    duplicate: bool = False


class FactApplicationPermit:
    """Single-use capability minted only after controller validation.

    A guard against accidental direct appends by in-process code, not a security boundary: the
    sentinel it carries is a module global, so code that imports this module can mint one. The
    boundary against such code is the controller's database role and the protected append
    functions (docs/PROVENANCE_CONTROLLER.md).
    """

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


def replays_request(prior: FactEvent, fact: AtomicFact, cards: Sequence[EvidenceCard]) -> bool:
    """Whether applying `fact` on `cards` under `prior`'s request id is that request sent again.

    A request id names one application: one fact on one set of cards. Sent again it is answered
    with the event it recorded (a DUPLICATE). Every ledger answered so whatever the request held,
    so a reused id with another fact returned allowed DUPLICATE with the old event, recorded
    nothing, and named the new fact in the decision (2026-10-21 supersession audit, P1 #7). Every
    ledger refuses that instead (`DecisionCode.REQUEST_ID_REUSED`). Only an assertion replays:
    an id the ledger minted for a `superseded` or `retracted` event is never a client's request.
    """
    return (
        prior.event_type == "asserted"
        and prior.fact_id == fact_identity(fact)
        and {card.card_id for card in prior.evidence_cards} == {card.card_id for card in cards}
    )


class FactLedger(Protocol):
    """An append-only fact ledger, as `ProvenanceController` relies on one.

    - `apply_assertion` consumes the permit (`_consume_permit`) inside its write, and refuses by
      raising ``ValueError(str(DecisionCode))``; the controller reads the code from the message.
    - A request id it already holds is answered with that event (`LedgerApplyResult(prior,
      duplicate=True)`) when `replays_request` holds, else refused REQUEST_ID_REUSED.
    - `record_decision` under a request id it already holds returns that event and writes
      nothing, which is why a REQUEST_ID_REUSED refusal is never recorded.
    - `stated` appends a disagreement with a current fact rather than refusing it; `served`
      names the memos served now, for the absent set of the conflict and duplicate rules.
    - Ids of the forms `reserved_request_id` names are the ledger's own.
    """

    @property
    def events(self) -> tuple[FactEvent, ...]: ...

    def current(
        self, *, tenant_id: str, now: datetime, served: ServedSources | None = None
    ) -> tuple[FactEvent, ...]: ...

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
        served: ServedSources | None = None,
        stated: bool = False,
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


class ConflictHistoryLedger(Protocol):
    """A ledger that answers one conflict key's history (`recall.fact_current`); the controller
    reads it when present (`ProvenanceController._in_force`), else every event."""

    def conflict_history(self, *, tenant_id: str, fact: AtomicFact) -> Sequence[FactEvent]: ...


class ErasureLockLedger(Protocol):
    """A ledger that orders applications against erasures (`PostgresFactLedger.erasure_lock`).

    `apply_fact` holds it shared for each of its passes (the fresh search between them runs
    without it), the stated reconcile holds it shared around its whole write, and a forget or
    withdraw holds it exclusively. `apply_stated` takes none: its caller holds it, and a nested
    PostgreSQL shared hold would queue behind a waiting forget and deadlock (DOC-005)."""

    def erasure_lock(self, *, shared: bool) -> AbstractContextManager[Any]: ...


def erasure_lock_for(ledger: object, *, shared: bool) -> AbstractContextManager[Any]:
    """`ledger`'s erasure lock (`ErasureLockLedger`), or nothing for a ledger without one.

    The one check every holder uses: the controller, the stated reconcile and forget (CODE-002).
    """
    if callable(getattr(ledger, "erasure_lock", None)):
        return cast(ErasureLockLedger, ledger).erasure_lock(shared=shared)
    return nullcontext()


class OutboxAssertionLedger(Protocol):
    """A ledger that appends an assertion and enqueues it for delivery in one transaction; used
    when present and the controller has an outbox, else the two happen in turn."""

    def apply_assertion_with_outbox(
        self, *, materialization_outbox: FactMaterializationOutbox, **kwargs: Any
    ) -> LedgerApplyResult: ...


class FactMaterializer(Protocol):
    """Synchronous downstream writer for an already authorized ledger event.

    Implementations must be idempotent by ``event.event_id``. The controller may invoke the
    same event again after a materialization failure so recovery never needs a second fact claim.
    One may name the event types it takes in a ``materializes`` attribute (`reconcile` reads it);
    without one it takes ``asserted`` events only.
    """

    def materialize(self, event: FactEvent) -> None: ...


#: The event types `MaterializationRecovery.reconcile` enqueues for a materializer that names none.
DEFAULT_MATERIALIZED_TYPES = frozenset({"asserted"})


class FactMaterializationOutbox(Protocol):
    """Durable delivery state for an already appended fact event.

    The event snapshot is enqueued before a downstream materializer is called.  Claiming uses a
    short lease, so a process that dies after claiming can be recovered without creating a second
    fact event.  Implementations must scope every operation to the configured tenant.
    """

    def enqueue(self, event: FactEvent) -> None:
        """Record `event` for delivery, idempotently by `event_id`.

        An event already enqueued keeps its row and status (`reconcile` and every replay enqueue
        again); a different payload under the same id raises.
        """
        ...

    def status(self, event_id: str) -> str | None: ...

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
        if not isinstance(limit, int) or not 1 <= limit <= MAX_OUTBOX_CLAIM_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_OUTBOX_CLAIM_LIMIT}")
        delivered = 0
        for event in self.outbox.claim(tenant_id=self.tenant_id, limit=limit, now=now):
            try:
                self.materializer.materialize(event)
            except Exception as exc:  # BROAD-CATCH: error-translation
                self.outbox.mark_failed(
                    tenant_id=self.tenant_id,
                    event_id=event.event_id,
                    error=f"{type(exc).__name__}: {exc}"[:MAX_OUTBOX_ERROR_CHARS],
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
        if not isinstance(limit, int) or not 1 <= limit <= MAX_OUTBOX_CLAIM_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_OUTBOX_CLAIM_LIMIT}")
        wanted = getattr(self.materializer, "materializes", DEFAULT_MATERIALIZED_TYPES)
        ordered = sorted(events, key=lambda event: (_as_utc(event.created_at), event.event_id))
        for event in ordered:
            if event.tenant_id != self.tenant_id:
                raise ValueError("ledger event tenant mismatch")
            if event.event_type in wanted:
                self.outbox.enqueue(event)
        return self.run_once(limit=limit, now=now)


def project_current(
    events: Sequence[FactEvent], now: datetime, served: ServedSources | None = None
) -> tuple[FactEvent, ...]:
    """The facts in force at `now`: asserted, inside their validity, not retracted, and not hidden
    by a successor valid at `now` (`fact_supersession`).

    With `served`, a fact none of whose memos the store serves now is left out (`absent_fact_ids`).
    A fact in force under two assertions (a re-assertion that adds a supersession is recorded
    anew, `repeats_assertion`) is listed once, under the first.
    """
    index = fact_supersession(events)
    retracted = retracted_keys(events)
    absent = absent_fact_ids(events, served, now)
    counted = applications(events)
    instant = _as_utc(now)
    listed: set[str | None] = set()
    current: list[FactEvent] = []
    for event in events:
        if (
            event.event_id in counted
            and event.fact is not None
            and event.fact_id not in listed
            and retraction_key(event) not in retracted
            and event.fact_id not in absent
            and _holds(fact_window(event.fact), instant)
            and not index.hidden_at(event, instant)
        ):
            listed.add(event.fact_id)
            current.append(event)
    return tuple(current)


def repeats_assertion(
    candidates: Sequence[FactEvent], fact: AtomicFact, supersedes_fact_ids: Sequence[str]
) -> bool:
    """Whether asserting `fact` adds nothing: a candidate asserts it already and supersedes at
    least every fact this application names, so the application is a DUPLICATE.

    One that names more is a new assertion. Paris; Rome for the summer superseding it; Paris
    again superseding Rome: recorded DUPLICATE, the third application's supersession was
    dropped and Rome kept hiding Paris all summer (2026-10-21 audit, STAKES-001).
    """
    identity = fact_identity(fact)
    named = set(supersedes_fact_ids)
    return any(
        event.fact_id == identity and named <= set(event.supersedes_fact_ids) for event in candidates
    )


def unresolved_conflicts(
    history: Sequence[FactEvent],
    fact: AtomicFact,
    supersedes_fact_ids: Sequence[str],
    absent: frozenset[str] = frozenset(),
    *,
    moment: datetime | None = None,
    index: FactSupersession | None = None,
) -> tuple[FactEvent, ...]:
    """The values in force that `fact` contradicts and `supersedes_fact_ids` does not settle.

    A value contradicts `fact` where it is in force inside `fact`'s window, and the check is made
    as if `fact` were already recorded superseding what it names: in A, then B (January to April)
    superseding A, then C (from March) superseding B, C hides A through B wherever C holds
    (`FactSupersession`), so A is no contradiction of C, though C never names it. The one rule
    `ProvenanceController.apply_fact` and every ledger refuse on.
    """
    index = fact_supersession(history) if index is None else index
    named = set(supersedes_fact_ids)
    remaining = [
        event
        for event in project_candidates(history, index=index)
        if event.fact is not None
        and event.fact_id not in absent
        and event.fact_id not in named
        and facts_conflict(event.fact, fact)
        and index.overlaps(event, fact)
    ]
    if not remaining or not named:
        return tuple(remaining)
    pending = FactEvent(
        event_id="\x00pending",
        event_type="asserted",
        tenant_id=remaining[0].tenant_id,
        generation_id=remaining[0].generation_id,
        fact=fact,
        fact_id=fact_identity(fact),
        evidence_cards=(),
        supersedes_fact_ids=tuple(sorted(named)),
        request_id="\x00pending",
        writer="",
        decision_code=DecisionCode.APPLIED,
        policy_version="",
        controller_version=CONTROLLER_SCHEMA_VERSION,
        # After everything recorded for this key, as the ledger will stamp it (`ordered_after`).
        created_at=ordered_after(moment or datetime.now(UTC), history, fact),
    )
    after = fact_supersession([*history, pending])
    return tuple(event for event in remaining if after.overlaps(event, fact))


def project_candidates(
    events: Sequence[FactEvent], *, index: FactSupersession | None = None
) -> tuple[FactEvent, ...]:
    """Asserted, not retracted, and in force at SOME instant: what a new claim may conflict with.

    A value its successors hide over its whole window is no candidate (so asserting it again is
    a new assertion, not a DUPLICATE); one hidden only part of the time still is.
    """
    index = fact_supersession(events) if index is None else index
    retracted = retracted_keys(events)
    # Each as `applications` counts it, so a DUPLICATE standing in for a withdrawn application
    # brings that application's supersession to `repeats_assertion`.
    return tuple(
        event
        for event in applications(events).values()
        if event.fact is not None
        and retraction_key(event) not in retracted
        and index.in_force(event)
    )


class InMemoryFactLedger:
    """Reference ledger used by local workflows and deterministic tests."""

    def __init__(self) -> None:
        self._events: list[FactEvent] = []
        self._by_request: dict[tuple[str, str], FactEvent] = {}
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
        policy_version: str = CONTROLLER_POLICY_VERSION,
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
            policy_version=policy_version,
            controller_version=CONTROLLER_SCHEMA_VERSION,
            created_at=moment,
        )

    def current(
        self, *, tenant_id: str, now: datetime, served: ServedSources | None = None
    ) -> tuple[FactEvent, ...]:
        with self._lock:
            return project_current(
                [event for event in self._events if event.tenant_id == tenant_id], now, served
            )

    def request_event(self, *, tenant_id: str, request_id: str) -> FactEvent | None:
        with self._lock:
            event = self._by_request.get((tenant_id, request_id))
            return event if event is not None and event.tenant_id == tenant_id else None

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
        served: ServedSources | None = None,
        stated: bool = False,
    ) -> LedgerApplyResult:
        del controller_version
        with self._lock:
            _consume_permit(
                permit,
                tenant_id=tenant_id,
                generation_id=generation_id,
                request_id=request_id,
                fact=fact,
                cards=cards,
            )
            prior = self._by_request.get((tenant_id, request_id))
            if prior is not None:
                if not replays_request(prior, fact, cards):
                    raise ValueError(DecisionCode.REQUEST_ID_REUSED)
                return LedgerApplyResult(prior, duplicate=True)
            history = [event for event in self._events if event.tenant_id == tenant_id]
            index = fact_supersession(history)
            candidates = project_candidates(history, index=index)
            now = ordered_after(now or datetime.now(UTC), history, fact)
            if repeats_assertion(candidates, fact, supersedes_fact_ids):
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
                    policy_version=policy_version,
                )
                self._events.append(event)
                self._by_request[(tenant_id, request_id)] = event
                return LedgerApplyResult(event, duplicate=True)
            # A fact whose memos the store no longer serves is a DUPLICATE above (a re-application
            # must not list it twice) but never a conflict: a deleted memo cannot block a new value.
            absent = absent_fact_ids(history, served, now)
            # A stated fact that disagrees is recorded, not refused (`assertion_is_duplicate`).
            if not stated and unresolved_conflicts(
                history, fact, supersedes_fact_ids, absent, moment=now, index=index
            ):
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
                policy_version=policy_version,
            )
            self._events.append(event)
            self._by_request[(tenant_id, request_id)] = event
            if supersedes_fact_ids:
                replacement = self._event(
                    event_type="superseded",
                    tenant_id=tenant_id,
                    generation_id=generation_id,
                    request_id=request_id + SUPERSESSION_REQUEST_SUFFIX,
                    writer=writer,
                    decision_code=DecisionCode.APPLIED,
                    fact=fact,
                    cards=cards,
                    supersedes_fact_ids=tuple(supersedes_fact_ids),
                    now=now,
                    policy_version=policy_version,
                )
                self._events.append(replacement)
                # Registered as every durable ledger holds it (its id is unique per tenant there),
                # so a later request under it is refused here too (`reserved_request_id`).
                self._by_request[(tenant_id, replacement.request_id)] = replacement
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
            prior = self._by_request.get((tenant_id, request_id))
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
            self._by_request[(tenant_id, request_id)] = event
            return event

    @property
    def events(self) -> tuple[FactEvent, ...]:
        with self._lock:
            return tuple(self._events)


@dataclass(frozen=True)
class ControllerDecision:
    """The controller's answer to one application.

    `allowed` holds exactly for APPLIED and DUPLICATE. `event` is the ledger event the answer
    rests on: this request's own, a replayed request's original, the refusal recorded for it, or
    another application's event on REQUEST_ID_REUSED; it is None when nothing was recorded (a
    reserved id, an unrecorded refusal, a ledger that failed to record).
    """

    allowed: bool
    code: DecisionCode
    request_id: str
    fact_id: str
    cards: tuple[EvidenceCard, ...] = ()
    event: FactEvent | None = None
    retried: bool = False
    detail: str = ""


class _FreshSearch:
    """`ProvenanceController._apply_fact`'s answer when the request's own cards failed and the
    fresh search, run outside the erasure lock, is to propose others."""


@dataclass(frozen=True)
class _ConflictSet:
    """A tenant's ledger as `ProvenanceController.apply_fact` judges a claim on it, at one instant."""

    candidates: Sequence[FactEvent]
    index: FactSupersession
    #: The tenant's events, or None for a ledger that exposes only `current`.
    history: Sequence[FactEvent] | None
    absent: frozenset[str]
    instant: datetime


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
        served_sources: ServedSources | None = None,
        card_permitted: Callable[[EvidenceCard], bool] | None = None,
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
        # The evaluation clock (`recall.eval_clock`) when one is installed, else the wall clock.
        self.now = now or (lambda: eval_clock.now() or datetime.now(UTC))
        self.writer = writer
        #: The memo names the store serves now; facts whose every memo it lacks are neither
        #: current nor a conflict (`absent_fact_ids`). None checks nothing.
        self.served_sources = served_sources
        #: Whether the caller may read a card's memo (its source access policy); a card it may
        #: not is refused as not found, so the answer does not say the card exists. None checks
        #: nothing. Without it a principal the policy kept from a memo could apply or supersede a
        #: fact with that memo's card, and learn from the answer what the memo states (2026-11-04
        #: hunt, SEC-002).
        self.card_permitted = card_permitted

    def _resolve(
        self, request: FactApplicationRequest, instant: datetime
    ) -> tuple[EvidenceCard, ...] | DecisionCode:
        resolved: list[EvidenceCard] = []
        for card_id in request.evidence_card_ids:
            try:
                card = self.cards.resolve(card_id)
            except Exception:  # BROAD-CATCH: fail-closed
                # A malformed or tampered durable payload is a card failure, not a controller
                # crash.  The caller may perform the one bounded fresh-search recovery.
                return DecisionCode.CARD_TAMPERED
            if card is None or not self._permitted(card):
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
                except Exception:  # BROAD-CATCH: fail-closed
                    return DecisionCode.CARD_TAMPERED
                if current_card is None:
                    return DecisionCode.SOURCE_CHANGED
                if current_card.card_id != card_id:
                    return DecisionCode.SOURCE_CHANGED
                card = current_card
            if not card.has_usable_support:
                return DecisionCode.UNSUPPORTED_CLAIM
            if not card.trusted_for_application:
                return DecisionCode.TRUST_UNAVAILABLE
            validity = _valid_now(card, instant)
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

    def _permitted(self, card: EvidenceCard) -> bool:
        """Whether the caller may read `card`'s memo (`card_permitted`); an error is a refusal."""
        if self.card_permitted is None:
            return True
        try:
            return bool(self.card_permitted(card))
        except Exception:  # BROAD-CATCH: fail-closed
            return False

    @staticmethod
    def _supported(fact: AtomicFact, cards: Sequence[EvidenceCard]) -> bool:
        target = fact_identity(fact)
        return any(fact_identity(source_fact) == target for card in cards for source_fact in card.structured_facts)

    @staticmethod
    def _superseding_fact_ids(
        fact: AtomicFact,
        cards: Sequence[EvidenceCard],
        current: Sequence[FactEvent],
        support: Mapping[str, Sequence[EvidenceCard]] | None = None,
    ) -> tuple[str, ...]:
        """The facts in `current` that `fact` contradicts and `cards` link to. A link may name any
        card of any live application of the old fact (`support`, `assertion_cards`), not only the
        APPLIED event's: a memo that states the fact through a DUPLICATE is one of its sources."""
        old_ids: list[str] = []
        for event in current:
            if event.fact is None or not facts_conflict(event.fact, fact):
                continue
            if event.fact_id in old_ids:
                continue
            evidence = (support or {}).get(event.fact_id or "") or event.evidence_cards
            old_source_ids = {
                value
                for card in evidence
                for value in (card.card_id, card.chunk_id, card.source)
            }
            old_sources = [card.source for card in evidence if card.source]
            allowed = any(
                event.fact_id in card.supersession_links
                or old_source_ids.intersection(card.supersession_links)
                or any(_link_names(link, old_sources) for link in card.supersession_links)
                for card in cards
            )
            if allowed and event.fact_id is not None:
                old_ids.append(event.fact_id)
        return tuple(old_ids)

    def _current(self, instant: datetime) -> tuple[FactEvent, ...]:
        """The ledger's `current` at `instant`, naming `served` only when the controller has one."""
        if self.served_sources is None:
            return self.ledger.current(tenant_id=self.tenant_id, now=instant)
        return self.ledger.current(tenant_id=self.tenant_id, now=instant, served=self.served_sources)

    def _in_force(self, claim: AtomicFact, instant: datetime) -> _ConflictSet:
        """The tenant's values in force at some instant, with the index that says when.

        From the ledger's own events where it exposes them, as `apply_assertion` reads them: the
        claim's conflict key's history where the ledger offers it (`conflict_history`, answered
        from the materialized current facts, `recall.fact_current`), since nothing outside that
        key bears on the claim, else every event. A ledger that exposes only `current` is read
        through it, so its conflict set is what holds now.

        The first ledger read here is the one authorization fails closed on, so a ledger that
        cannot answer it is refused. With `conflict_history` that is the key read itself, which
        reads exactly what the decision rests on, and the tenant's `current` is not read: it
        decoded every fact current in the tenant only to be discarded (PERF-003, measured
        2026-10-22 at about 0.3 ms per current fact, 98% of one apply at 10k facts on lite and
        91% on PostgreSQL). Without it, `current` is read first (pinned by
        `test_ledger_read_outage_fails_closed_without_asserting` and the provenance evaluation's
        `ledger_outage` case). One clock reading, and the absent set computed once, inside the
        fail-closed read; the second was computed outside it (2026-10-21 audit, CODE-004).
        `instant` is the pass's one reading, which also checked the cards' validity (ENV-004).
        """
        current: tuple[FactEvent, ...] | None = None
        if callable(getattr(self.ledger, "conflict_history", None)):
            history: object = cast(ConflictHistoryLedger, self.ledger).conflict_history(
                tenant_id=self.tenant_id, fact=claim
            )
        else:
            current = self._current(instant)
            history = getattr(self.ledger, "events", None)
        if history is None or not isinstance(history, Sequence):
            current = self._current(instant) if current is None else current
            return _ConflictSet(current, fact_supersession(current), None, frozenset(), instant)
        events = [event for event in history if event.tenant_id == self.tenant_id]
        index = fact_supersession(events)
        absent = absent_fact_ids(events, self.served_sources, instant)
        candidates = [
            event
            for event in project_candidates(events, index=index)
            if event.fact_id not in absent
        ]
        return _ConflictSet(candidates, index, events, absent, instant)

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

    def _resolve_fresh(
        self, request: FactApplicationRequest, fresh_ids: tuple[str, ...], instant: datetime
    ) -> tuple[EvidenceCard, ...] | ControllerDecision:
        """The cards the one bounded fresh search found, or the refusal that ends the retry."""
        refreshed = FactApplicationRequest(request.claim, fresh_ids, request.request_id)
        resolved = self._resolve(refreshed, instant)
        if isinstance(resolved, DecisionCode) or not self._supported(request.claim, resolved):
            # A card failure after the retry keeps its own code; an unsupported claim, from
            # either branch, is the retry finding too little.
            code = (
                resolved
                if isinstance(resolved, DecisionCode) and resolved != DecisionCode.UNSUPPORTED_CLAIM
                else DecisionCode.FRESH_SEARCH_INSUFFICIENT
            )
            return self._record_refusal(request, code, (), retried=True)
        return resolved

    def _erasure_lock(self) -> AbstractContextManager[Any]:
        """The ledger's application side of its erasure lock, or nothing for one without it."""
        return erasure_lock_for(self.ledger, shared=True)

    def apply_fact(self, request: FactApplicationRequest) -> ControllerDecision:
        """Validate and append one fact, with at most one deterministic fresh-search retry.

        Each pass, from the card checks (`_resolve`) to the append, holds the ledger's erasure lock
        shared (`PostgresFactLedger.erasure_lock`), which a forget holds exclusively; the fresh
        search between the two passes runs without it.
        The cards were checked before the append took the ledger's key lock, so a forget of a
        card's memo between the two left an assertion appended after the forget's retraction,
        never retracted and its value never erased (STAKES-003). A ledger without the lock (the
        in-memory reference) orders nothing. A lock that cannot be taken raises: nothing is
        recorded, so the request can be sent again.
        """
        with self._erasure_lock():
            first = self._apply_fact(request, None)
        if not isinstance(first, _FreshSearch):
            return first
        # The fresh search (an embed, a search, a rerank) runs without the lock: it only proposes
        # card ids, which the second pass resolves and checks again under it. Held across the
        # search, the lock made a forget wait for it, and every application queued behind that
        # forget wait too (2026-11-04 audit of the owner-decision change, PERF-001).
        assert self.fresh_search is not None
        try:
            fresh_ids = tuple(self.fresh_search(request.claim, request))
        except Exception as exc:  # BROAD-CATCH: fail-closed
            with self._erasure_lock():
                return self._record_refusal(
                    request, DecisionCode.FRESH_SEARCH_UNAVAILABLE, (), retried=True,
                    detail=type(exc).__name__,
                )
        with self._erasure_lock():
            if not fresh_ids:
                return self._record_refusal(
                    request, DecisionCode.FRESH_SEARCH_INSUFFICIENT, (), retried=True
                )
            decision = self._apply_fact(request, fresh_ids)
        assert not isinstance(decision, _FreshSearch)
        return decision

    def _apply_fact(
        self, request: FactApplicationRequest, fresh_ids: tuple[str, ...] | None
    ) -> ControllerDecision | _FreshSearch:
        """One pass of `apply_fact` under the erasure lock: with `fresh_ids` None the request's
        own cards, else the fresh search's. A first pass whose cards fail asks for the search."""
        if reserved_request_id(request.request_id):
            # Refused before the ledger is read, and not recorded: a refusal under the id would
            # itself take the slot the ledger keeps for its own event.
            return self._decision(
                request, DecisionCode.REQUEST_ID_RESERVED, detail="the ledger mints ids of this form"
            )
        try:
            prior = self.ledger.request_event(tenant_id=self.tenant_id, request_id=request.request_id)
        except Exception as exc:  # BROAD-CATCH: fail-closed
            return self._record_refusal(
                request,
                DecisionCode.LEDGER_UNAVAILABLE,
                retried=False,
                cards=(),
                detail=type(exc).__name__,
            )
        if prior is not None and not all(self._permitted(card) for card in prior.evidence_cards):
            # The request id's recorded event cites a memo this caller may not read: answered as
            # a card the caller cannot see, with no event and nothing recorded, whatever fact
            # the request names. Answered by the branches below, the event's id (a hash of
            # tenant, fact and request id) let a caller who knew a request id test which fact
            # it applied, offline or by which refusal came back, and the refusal replay handed
            # over another principal's refusal event (2026-11-05 deferred pass, SEC-002). A
            # recorded refusal that cites no card passes this check and is replayed to anyone.
            return self._decision(request, DecisionCode.CARD_NOT_FOUND)
        if prior is not None and prior.fact_id != fact_identity(request.claim):
            # Another fact under a used id. Answering with the prior decision named the new fact
            # allowed and recorded nothing; the same fact on other cards is refused by the ledger,
            # which compares the resolved cards (`replays_request`).
            return self._decision(
                request, DecisionCode.REQUEST_ID_REUSED, event=prior,
                detail="request_id already names another fact",
            )
        if prior is not None and prior.decision_code not in {
            DecisionCode.APPLIED,
            DecisionCode.DUPLICATE,
        }:
            return self._decision(
                request,
                _decision_code(prior.decision_code),
                cards=prior.evidence_cards,
                event=prior,
            )
        if (
            prior is not None
            and prior.event_type == "asserted"
            and {card.card_id for card in prior.evidence_cards} == set(request.evidence_card_ids)
        ):
            # The request sent again (`replays_request`): answered with the event it recorded, as
            # every ledger answers it. Validated afresh, a retry after a later value superseded
            # it, its memo changed or its card expired was refused while its APPLIED event stood
            # (2026-11-04 hunt, BUG-002).
            return self._delivered(
                request, LedgerApplyResult(prior, duplicate=True), prior.evidence_cards, retried=False
            )
        # One clock reading for the pass: the cards' validity is checked at the instant the
        # assertion is stamped with. Read per card and again for the stamp, a card checked valid
        # just before its `valid_until` could be stamped at or after it, which a file-backed eval
        # clock makes a step rather than a tick (2026-11-04 owner decision, ENV-004).
        instant = self.now()
        if fresh_ids is None:
            resolved = self._resolve(request, instant)
            retried = False
            if isinstance(resolved, DecisionCode) or not self._supported(request.claim, resolved):
                if self.fresh_search is None:
                    code = resolved if isinstance(resolved, DecisionCode) else DecisionCode.UNSUPPORTED_CLAIM
                    return self._record_refusal(request, code, (), retried=False)
                return _FreshSearch()
        else:
            retry = self._resolve_fresh(request, fresh_ids, instant)
            if isinstance(retry, ControllerDecision):
                return retry
            resolved, retried = retry, True
        cards = tuple(resolved)
        try:
            # The conflict set is the one the ledger refuses on: every value in force at SOME
            # instant of the claim's window, not only the ones current now. Deriving it from
            # `current` let an expired or future value block an authored correction, because the
            # link that names it was never read (2026-10-21 supersession audit, P2 #12).
            in_force = self._in_force(request.claim, instant)
        except Exception as exc:  # BROAD-CATCH: fail-closed
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
        index = in_force.index
        conflicts = tuple(
            event
            for event in in_force.candidates
            if event.fact and facts_conflict(event.fact, request.claim) and index.overlaps(event, request.claim)
        )
        supersedes = self._superseding_fact_ids(
            request.claim, cards, conflicts,
            None
            if in_force.history is None or not conflicts
            else assertion_cards(in_force.history, index=index),
        )
        # Refused on the rule every ledger applies (`unresolved_conflicts`): what the claim
        # contradicts and neither names nor hides through what it names.
        unsettled = (
            unresolved_conflicts(
                in_force.history, request.claim, supersedes, in_force.absent,
                moment=in_force.instant, index=index,
            )
            if in_force.history is not None
            else tuple(event for event in conflicts if event.fact_id not in supersedes)
        )
        if unsettled:
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
            # Keywords, not a dict, so a renamed `apply_assertion` parameter is a type error here
            # rather than a TypeError that the catch below turns into LEDGER_UNAVAILABLE. `served`
            # None is every ledger's default.
            if self.materialization_outbox is not None and callable(
                getattr(self.ledger, "apply_assertion_with_outbox", None)
            ):
                result = cast(OutboxAssertionLedger, self.ledger).apply_assertion_with_outbox(
                    materialization_outbox=self.materialization_outbox,
                    tenant_id=self.tenant_id,
                    generation_id=self.generation_id,
                    fact=request.claim,
                    cards=cards,
                    request_id=request.request_id,
                    writer=self.writer,
                    permit=permit,
                    supersedes_fact_ids=supersedes,
                    now=in_force.instant,
                    served=self.served_sources,
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
                    now=in_force.instant,
                    served=self.served_sources,
                )
        except ValueError as exc:
            return self._record_refusal(
                request, _decision_code(str(exc)), cards, retried=retried, detail=str(exc)
            )
        except Exception as exc:  # BROAD-CATCH: fail-closed
            return self._record_refusal(request, DecisionCode.LEDGER_UNAVAILABLE, cards, retried=retried, detail=type(exc).__name__)
        return self._delivered(request, result, cards, retried=retried)

    def _delivered(
        self,
        request: FactApplicationRequest,
        result: LedgerApplyResult,
        cards: tuple[EvidenceCard, ...],
        *,
        retried: bool,
    ) -> ControllerDecision:
        """The decision for an appended (or replayed) `result`, after its materialization."""
        if self.materializer is not None:
            leased: FactEvent | None = None
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
                        try:
                            delivery_status = self.materialization_outbox.status(event.event_id)
                        except Exception:  # BROAD-CATCH: fail-closed
                            # This read owns no lease. Keep another worker's delivery state
                            # intact rather than entering the materializer failure cleanup.
                            delivery_status = None
                        if delivery_status != "applied":
                            return self._decision(
                                request, DecisionCode.MATERIALIZATION_UNAVAILABLE,
                                cards=cards, retried=retried, event=event,
                            )
                        return self._decision(
                            request,
                            DecisionCode.DUPLICATE if result.duplicate else DecisionCode.APPLIED,
                            cards=cards,
                            retried=retried,
                            event=event,
                        )
                    event = leased = claimed[0]
                self.materializer.materialize(event)
                if self.materialization_outbox is not None:
                    self.materialization_outbox.mark_applied(
                        tenant_id=self.tenant_id, event_id=event.event_id,
                        lease_token=event.lease_token,
                    )
            except Exception as exc:  # BROAD-CATCH: fail-closed
                # The append is the durable intent. Do not claim the downstream fact store was
                # updated, and leave the outbox event available for an idempotent recovery retry.
                # Only a lease this call holds is released: an enqueue or claim that raised took
                # none, and `mark_failed` without a token overwrote the row another worker held
                # or had already applied (2026-11-04 hunt, BUG-003).
                if self.materialization_outbox is not None and leased is not None:
                    try:
                        self.materialization_outbox.mark_failed(
                            tenant_id=self.tenant_id,
                            event_id=result.event.event_id,
                            error=f"{type(exc).__name__}: {exc}"[:MAX_OUTBOX_ERROR_CHARS],
                            lease_token=leased.lease_token,
                        )
                    except Exception:  # BROAD-CATCH: fail-closed
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

    def apply_stated(
        self, fact: AtomicFact, card: EvidenceCard, request_id: str
    ) -> ControllerDecision:
        """Record `fact`, which the memo behind `card` states, with that card as its evidence.

        The path of `recall.stated_facts`: no search ran, so `apply_fact`'s search-trust gates do
        not apply, and the caller has already decided that the memo may state facts. What is
        checked here is that the card is a stated card of this tenant and generation that carries
        exactly this fact. A disagreement with a current fact is appended, never refused: the
        projection lists both as contested. A refusal writes nothing, because a card failing these
        checks is the caller's defect, not a decision about the fact. It does not take the
        ledger's erasure lock: its caller (`recall.stated_facts.reconcile_stated_facts`) holds it
        from its served-memo check to its last write, and a second shared hold queued behind a
        waiting forget would deadlock PostgreSQL.
        """
        request = FactApplicationRequest(fact, (card.card_id,), request_id)
        if (
            card.trust_state != STATED_TRUST_STATE
            or card.tenant_id != self.tenant_id
            or card.generation_id != self.generation_id
            or not self._supported(fact, (card,))
        ):
            return self._decision(request, DecisionCode.UNSUPPORTED_CLAIM, cards=(card,))
        permit = FactApplicationPermit(
            tenant_id=self.tenant_id,
            generation_id=self.generation_id,
            request_id=request_id,
            fact_id=fact_identity(fact),
            card_ids=(card.card_id,),
            _token=_PERMIT_SENTINEL,
        )
        now = self.now()
        # No served lookup: a stated assertion ignores the absence set it would feed, and on
        # PostgreSQL that lookup is a scan of the served generation per fact.
        try:
            result = self.ledger.apply_assertion(
                tenant_id=self.tenant_id,
                generation_id=self.generation_id,
                fact=fact,
                cards=(card,),
                request_id=request_id,
                writer=self.writer,
                permit=permit,
                policy_version=STATED_POLICY_VERSION,
                now=now,
                stated=True,
            )
        except ValueError as exc:
            # A ledger refuses with the decision code as the message, as `apply_fact` reads it.
            return self._decision(request, _decision_code(str(exc)), cards=(card,), detail=str(exc))
        except Exception as exc:  # BROAD-CATCH: fail-closed
            return self._decision(
                request, DecisionCode.LEDGER_UNAVAILABLE, cards=(card,), detail=type(exc).__name__
            )
        return self._decision(
            request,
            DecisionCode.DUPLICATE if result.duplicate else DecisionCode.APPLIED,
            cards=(card,),
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
        except Exception:  # BROAD-CATCH: fail-closed
            event = None
        return self._decision(request, code, cards=cards, retried=retried, detail=detail, event=event)


def _decision_code(value: str) -> DecisionCode:
    """`value` as a decision code: a ledger refuses with the code as the message (`FactLedger`);
    anything else is the ledger failing, LEDGER_UNAVAILABLE."""
    try:
        return DecisionCode(value)
    except ValueError:
        return DecisionCode.LEDGER_UNAVAILABLE


def _evidence_links(
    graph: Mapping[str, object], metadata: Mapping[str, object], key: str
) -> tuple[str, ...]:
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


def cards_from_trusted_result(result: Any, *, selected_only: bool = True) -> tuple[EvidenceCard, ...]:
    """Build write cards from ok hits with complete identity and structured fact support."""
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
        # One malformed authored fact (no subject, a bad date) is skipped, as `recall.evidence`
        # skips it; it raised out of the public `recall.cards_from_trusted_result` and discarded
        # every card of the result (2026-11-04 hunt, CODE-001).
        facts: list[AtomicFact] = []
        raw_facts = graph.get("facts", metadata.get("facts", []))
        for item in raw_facts if isinstance(raw_facts, Sequence) and not isinstance(raw_facts, str) else ():
            if isinstance(item, Mapping):
                try:
                    facts.append(AtomicFact.from_payload(item))
                except (TypeError, ValueError, KeyError):
                    continue
        file_name = hit.provenance.file or hit.chunk.source
        raw_digest = metadata.get("content_hash") or metadata.get("source_digest")
        digest = (
            str(raw_digest)
            if isinstance(raw_digest, str) and raw_digest
            else source_digest(hit.chunk.text)
        )
        card = EvidenceCard(
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
            supersession_links=_evidence_links(graph, metadata, "authored_supersedes"),
            contradiction_links=_evidence_links(graph, metadata, "authored_contradicts"),
            support_refs=_evidence_links(graph, metadata, "support_refs"),
            structured_facts=tuple(facts),
        )
        if card.has_usable_support:
            cards.append(card)
    return tuple(cards)
