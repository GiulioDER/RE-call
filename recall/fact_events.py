"""The fact ledger's event vocabulary: decision codes, canonical identity and the event row.

Moved verbatim out of `recall.provenance_controller`, which re-exports every name, so both
import paths give the same objects. It imports nothing from the controller or the projections.
"""
from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from recall.types import AtomicFact, EvidenceCard


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
    SOURCE_FORGOTTEN = "SOURCE_FORGOTTEN"
    SOURCE_REMOVED = "SOURCE_REMOVED"
    #: A `withdrawn` event: the memo behind an assertion was withdrawn as a mistake
    #: (`recall_withdraw`), so the assertion is no longer listed and supersedes nothing.
    WITHDRAWN = "WITHDRAWN"
    #: Two refusals no ledger ever records: the request id already names another application, or
    #: has a form the ledger mints for its own events (`reserved_request_id`). Recording either
    #: under that id is impossible or would take the server's slot, so neither is written.
    REQUEST_ID_REUSED = "REQUEST_ID_REUSED"
    REQUEST_ID_RESERVED = "REQUEST_ID_RESERVED"


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


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


@dataclass(frozen=True)
class FactEvent:
    """One immutable ledger row.

    `event_type` is one of `asserted` (an application; its `decision_code` is APPLIED, or
    DUPLICATE for a new application of a fact already in force), `superseded` (the receipt of
    a successor retiring `supersedes_fact_ids`), `retracted` (a forget or gc took every memo the
    fact rested on), `withdrawn` (a withdraw gave a superseded value back), `rejected` and
    `abstained` (refusals, coded by `decision_code`). `lease_token` is set only on an event an
    outbox `claim` handed out, and names that lease.
    """

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
    #: Set when `recall_forget` erased this event's fact value (`recall.fact_redaction`): `fact` is
    #: then None and `fact_id` the fresh id the redaction gave it, so its retraction still pairs.
    redaction_id: str | None = None
