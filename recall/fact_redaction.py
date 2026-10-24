"""Erase what a forgotten memo told the fact ledger: its fact values, its evidence cards, and the
replay receipts that echo them.

`recall_forget` retracts a fact whose only evidence was the forgotten memo (`recall.fact_ledger.
plan_retractions`), which hides it; this module then removes the values themselves, in place, from
the append-only tables. Migration 0029 (PostgreSQL) and the lite file's triggers admit exactly
these rewrites and nothing else:

- A CARD is erased when its memo is one this forget erased (or, for `recall provenance erase`,
  always). Every copy of it, in the card table and inside every ledger event, becomes a marker
  that keeps its tenant, generation and memo name (`recall.types.RedactedEvidenceCard`) under a
  fresh random id: the old id is a hash of the content, so keeping it would keep a low-entropy
  value (a four-digit code) recoverable by enumeration.
- A FACT VALUE is erased when at least one of its cards is erased now and every card on every
  event holding the value (asserted, DUPLICATE, rejected, abstained) is erased now or was before.
  A fact with evidence in a memo that survives is not touched, nor is one a memo this forget did
  not name still holds (a memo gc removed: name it in the forget too). Each holding event's
  payload becomes `{fact_id, redacted, redaction_id}` with a fresh fact id, and its event and
  request ids, which hash the value, are replaced too; the `retracted` and `superseded` events
  derived from them, and every `supersedes_fact_ids` naming the fact, follow, so a retraction
  still pairs with its assertion and every projection lists what it listed before.
- RECEIPTS (`recall_idempotency_receipts`) whose result names an erased id or memo are deleted.

What stays is the shape of the history: event types, writers, decision codes, times, the memo
names, and one `recall_fact_redactions` row per redaction. What this cannot reach is disclosed in
`docs/PRODUCTION.md`: PostgreSQL's dead tuples until VACUUM, its WAL, replicas and backups, and the
copies other caches hold.

Ids come from `secrets.token_hex`; `plan_redaction` takes the generator so a test can fix it.
"""
# The SQL identifiers in this module are fixed module constants; values are bound parameters.
# ruff: noqa: S608
from __future__ import annotations

import json
import secrets
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from recall.fact_ledger import FACT_LEDGER_TABLE, decision_event_id
from recall.observability import get_logger
from recall.provenance_controller import fact_conflict_key, fact_identity
from recall.schema import FACT_REDACTION_FUNCTION
from recall.types import AtomicFact, is_redaction_marker

_log = get_logger("fact_redaction")

#: The protected function migration 0029 adds; the controller role is granted EXECUTE on it.
REDACTION_FUNCTION = FACT_REDACTION_FUNCTION
FACT_REDACTIONS_TABLE = "recall_fact_redactions"
EVIDENCE_CARD_TABLE = "recall_evidence_cards"
RECEIPTS_TABLE = "recall_idempotency_receipts"
#: The writer a forget's redaction records; `recall provenance erase` records its own.
REDACTION_WRITER = "recall_forget"
TENANT_ERASE_WRITER = "recall_provenance_erase"
REASONS = frozenset({"forget", "tenant_erase"})
#: The event types that hold a fact value: an application, a refusal, an abstention.
_HOLDERS = frozenset({"asserted", "rejected", "abstained"})
_RETRACTION_PREFIX = "retraction:"
_SUPERSESSION_SUFFIX = "_sup"


def _token() -> str:
    return secrets.token_hex(16)


@dataclass(frozen=True)
class LedgerRow:
    """One ledger event as stored: JSON columns decoded, nothing else interpreted."""

    event_id: str
    event_type: str
    generation_id: str
    fact_id: str | None
    fact: dict[str, Any] | None
    evidence_cards: list[dict[str, Any]]
    supersedes_fact_ids: list[str]
    request_id: str


@dataclass(frozen=True)
class EventRewrite:
    """The new values of one ledger event, keyed by the id it has now."""

    event_id: str
    new_event_id: str
    request_id: str
    fact_id: str | None
    fact: dict[str, Any] | None
    evidence_cards: list[dict[str, Any]]
    supersedes_fact_ids: list[str]

    def payload(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id, "new_event_id": self.new_event_id,
            "request_id": self.request_id, "fact_id": self.fact_id, "fact": self.fact,
            "evidence_cards": self.evidence_cards, "supersedes_fact_ids": self.supersedes_fact_ids,
        }


@dataclass(frozen=True)
class CardRewrite:
    card_id: str
    new_card_id: str
    card: dict[str, Any]

    def payload(self) -> dict[str, Any]:
        return {"card_id": self.card_id, "new_card_id": self.new_card_id, "card": self.card}


@dataclass(frozen=True)
class RedactionPlan:
    """Every rewrite one redaction makes, and what it erases."""

    redaction_id: str
    events: tuple[EventRewrite, ...]
    cards: tuple[CardRewrite, ...]
    #: The distinct fact values erased.
    facts: int
    #: Strings whose presence in a receipt's result deletes the receipt: every replaced id and
    #: the names of the erased memos.
    needles: tuple[str, ...]

    @property
    def empty(self) -> bool:
        return not self.events and not self.cards


@dataclass(frozen=True)
class RedactionReport:
    """What one redaction did. `facts_redacted` is what `recall_forget` reports."""

    facts_redacted: int = 0
    events_rewritten: int = 0
    cards_redacted: int = 0
    receipts_deleted: int = 0
    redaction_id: str | None = None


def redacted_fact(fact_id: str, redaction_id: str) -> dict[str, Any]:
    return {"fact_id": fact_id, "redacted": True, "redaction_id": redaction_id}


def redacted_card(card: Mapping[str, Any], card_id: str, redaction_id: str) -> dict[str, Any]:
    return {
        "card_id": card_id,
        "generation_id": card.get("generation_id"),
        "redacted": True,
        "redaction_id": redaction_id,
        "source": card.get("source"),
        "tenant_id": card.get("tenant_id"),
    }


def _value_id(row: LedgerRow) -> str | None:
    """The id readers give a stored fact (`recall.fact_ledger._decode_fact`), recomputed."""
    if row.fact is None or is_redaction_marker(row.fact):
        return None
    return fact_identity(AtomicFact.from_payload(row.fact))


def plan_redaction(
    tenant_id: str,
    rows: Sequence[LedgerRow],
    stored_cards: Iterable[tuple[str, Mapping[str, Any]]],
    *,
    erased: Callable[[str], bool] | None,
    everything: bool = False,
    token: Callable[[], str] | None = None,
) -> RedactionPlan:
    """The rewrites that erase what `erased` memos supplied (every value, with `everything`).

    `erased(name)` says whether a card naming memo `name` is erased now. `stored_cards` are the
    card table's rows of this tenant that may need it, as `(card_id, card)`. The module docstring
    states the rule. Deterministic for a fixed `token`: maps are built in sorted order.
    """
    token = token or _token  # looked up per call, so a test can fix it
    redaction_id = "red_" + token()

    def erasable(card: Mapping[str, Any]) -> bool:
        if is_redaction_marker(card):
            return False
        return everything or (erased is not None and erased(str(card.get("source", ""))))

    cards_now: set[str] = set()
    names: set[str] = set()
    for row in rows:
        for card in row.evidence_cards:
            if erasable(card):
                cards_now.add(str(card["card_id"]))
                names.add(str(card.get("source", "")))
    table = [(str(card_id), stored) for card_id, stored in stored_cards]
    for card_id, stored in table:
        if erasable(stored):
            cards_now.add(card_id)
            names.add(str(stored.get("source", "")))

    holders: dict[str, list[LedgerRow]] = {}
    for row in rows:
        value = _value_id(row) if row.event_type in _HOLDERS else None
        if value is not None:
            holders.setdefault(value, []).append(row)
    erased_facts: list[str] = []
    for value, group in holders.items():
        cards = [card for row in group for card in row.evidence_cards]
        if everything or (
            any(str(card.get("card_id")) in cards_now for card in cards)
            and all(is_redaction_marker(card) or str(card.get("card_id")) in cards_now for card in cards)
        ):
            erased_facts.append(value)

    fact_map = {value: "fact_" + token() for value in sorted(erased_facts)}
    card_map = {card_id: "card_" + token() for card_id in sorted(cards_now)}
    holder_rows = sorted(
        (row for value in fact_map for row in holders[value]), key=lambda row: row.event_id
    )
    event_map: dict[str, str] = {}
    request_map: dict[str, str] = {}
    holder_value: dict[str, str] = {}
    for row in holder_rows:
        event_map[row.event_id] = "evt_" + token()
        request_map[row.event_id] = "redacted:" + token()
        holder_value[row.event_id] = str(_value_id(row))

    def new_ids(event_id: str, event_type: str, request_id: str) -> tuple[str, str]:
        if event_id in event_map:
            return event_map[event_id], request_map[event_id]
        if event_type == "retracted" and request_id.startswith(_RETRACTION_PREFIX):
            retracted = request_id[len(_RETRACTION_PREFIX):]
            if retracted in event_map:
                new_request = _RETRACTION_PREFIX + event_map[retracted]
                return decision_event_id(tenant_id, new_request, "retracted"), new_request
        if event_type == "superseded" and event_id.endswith(_SUPERSESSION_SUFFIX):
            holder = event_id[: -len(_SUPERSESSION_SUFFIX)]
            if holder in event_map:
                return event_map[holder] + _SUPERSESSION_SUFFIX, request_map[holder] + ":supersession"
        return event_id, request_id

    def new_cards(cards: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [
            redacted_card(card, card_map[str(card["card_id"])], redaction_id)
            if not is_redaction_marker(card) and str(card.get("card_id")) in card_map
            else dict(card)
            for card in cards
        ]

    def new_fact(event_id: str, fact: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if event_id in holder_value:
            return redacted_fact(fact_map[holder_value[event_id]], redaction_id)
        return None if fact is None else dict(fact)

    events: list[EventRewrite] = []
    for row in rows:
        event_id, request_id = new_ids(row.event_id, row.event_type, row.request_id)
        fact = new_fact(row.event_id, row.fact)
        fact_id = fact_map[holder_value[row.event_id]] if row.event_id in holder_value else row.fact_id
        cards = new_cards(row.evidence_cards)
        supersedes = [fact_map.get(value, value) for value in row.supersedes_fact_ids]
        if (event_id, request_id, fact_id, fact, cards, supersedes) != (
            row.event_id, row.request_id, row.fact_id, row.fact, row.evidence_cards,
            row.supersedes_fact_ids,
        ):
            events.append(EventRewrite(
                event_id=row.event_id, new_event_id=event_id, request_id=request_id,
                fact_id=fact_id, fact=fact, evidence_cards=cards, supersedes_fact_ids=supersedes,
            ))

    cards_out = tuple(
        CardRewrite(card_id, card_map[card_id], redacted_card(card, card_map[card_id], redaction_id))
        for card_id, card in sorted(table, key=lambda item: item[0])
        if card_id in card_map and not is_redaction_marker(card)
    )
    replaced = {item.event_id for item in events if item.new_event_id != item.event_id}
    needles = sorted({*fact_map, *card_map, *replaced} | {name for name in names if name})
    return RedactionPlan(
        redaction_id=redaction_id,
        events=tuple(events),
        cards=cards_out,
        facts=len(fact_map),
        needles=tuple(needles),
    )


def conflict_keys(rows: Sequence[LedgerRow], plan: RedactionPlan) -> list[str]:
    """The per-fact advisory lock keys of the values `plan` erases, as `apply_assertion` takes them."""
    erased = {item.event_id for item in plan.events if is_redaction_marker(item.fact)}
    keys = {
        fact_conflict_key(AtomicFact.from_payload(row.fact))
        for row in rows
        if row.event_id in erased and row.fact is not None and not is_redaction_marker(row.fact)
    }
    return sorted(keys)


def _json(value: Any) -> str:
    """The lite file's JSON form (`recall.lite.facts._json`), so an unchanged card reads back equal."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


# --------------------------------------------------------------------------------------- PostgreSQL

_LEDGER_ROW_SQL = (
    "SELECT event_id, event_type, generation_id, fact_id, fact, evidence_cards, "
    f"supersedes_fact_ids, request_id FROM {FACT_LEDGER_TABLE} WHERE tenant_id = %s "
    "ORDER BY created_at, event_id"
)


def _pg_row(row: Sequence[Any]) -> LedgerRow:
    event_id, event_type, generation_id, fact_id, fact, cards, supersedes, request_id = row
    return LedgerRow(
        event_id=str(event_id), event_type=str(event_type), generation_id=str(generation_id),
        fact_id=None if fact_id is None else str(fact_id), fact=fact,
        evidence_cards=list(cards or []), supersedes_fact_ids=[str(v) for v in supersedes or []],
        request_id=str(request_id),
    )


class PostgresFactRedactor:
    """Redaction on PostgreSQL, through migration 0029's protected function.

    The ledger and outbox are read, locked and rewritten on `ledger_dsn`, the controller role's
    (`RECALL_FACT_WRITE_DSN`), which `recall_forget`'s retraction already uses; the card table is
    read on `card_dsn`, the serving store's, since the controller role is granted no card reads.
    """

    def __init__(
        self, ledger_dsn: str, *, tenant_id: str, card_dsn: str | None = None,
        statement_timeout_ms: int = 15000,
    ) -> None:
        if not ledger_dsn or not tenant_id:
            raise ValueError("dsn and tenant_id must be non-empty")
        self.dsn = ledger_dsn
        self.card_dsn = card_dsn or ledger_dsn
        self.tenant_id = tenant_id
        self.statement_timeout_ms = statement_timeout_ms

    def _connect(self, dsn: str) -> Any:
        import psycopg

        conn = psycopg.connect(dsn, autocommit=True, connect_timeout=10)
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (self.tenant_id,))
        conn.execute(f"SET statement_timeout = {int(self.statement_timeout_ms)}")
        return conn

    def card_sources(self) -> set[str]:
        """The memo names the card table's unredacted cards of this tenant cite."""
        with self._connect(self.card_dsn) as conn:
            rows = conn.execute(
                f"SELECT DISTINCT card->>'source' FROM {EVIDENCE_CARD_TABLE} "
                "WHERE tenant_id = %s AND NOT (card ? 'redacted')",
                (self.tenant_id,),
            ).fetchall()
        return {str(name) for (name,) in rows if name}

    def _stored_cards(self, names: Collection[str] | None) -> list[tuple[str, Mapping[str, Any]]]:
        sql = (
            f"SELECT card_id, card FROM {EVIDENCE_CARD_TABLE} "
            "WHERE tenant_id = %s AND NOT (card ? 'redacted')"
        )
        params: list[Any] = [self.tenant_id]
        if names is not None:
            sql += " AND card->>'source' = ANY(%s)"
            params.append(sorted(names))
        with self._connect(self.card_dsn) as conn:
            return [(str(card_id), card) for card_id, card in conn.execute(sql, params).fetchall()]

    def preview(self, *, everything: bool = True) -> RedactionReport:
        """What `redact(everything=...)` would rewrite, writing nothing (`recall provenance erase`)."""
        stored = self._stored_cards(None)
        with self._connect(self.dsn) as conn:
            rows = [_pg_row(row) for row in conn.execute(_LEDGER_ROW_SQL, (self.tenant_id,)).fetchall()]
        return _preview(plan_redaction(self.tenant_id, rows, stored, erased=None, everything=everything))

    def redact(
        self,
        *,
        erased: Callable[[str], bool] | None = None,
        names: Collection[str] | None = None,
        everything: bool = False,
        reason: str = "forget",
        writer: str = REDACTION_WRITER,
        token: Callable[[], str] | None = None,
    ) -> RedactionReport:
        """Plan and apply one redaction in one transaction. `names` bounds the card-table read to
        the memos `erased` may answer for (all of them with `everything`)."""
        from psycopg.types.json import Jsonb

        if reason not in REASONS:
            raise ValueError(f"{reason!r} is not a redaction reason")
        stored = self._stored_cards(None if everything else (names or ()))
        with self._connect(self.dsn) as conn, conn.transaction():
            rows = [_pg_row(row) for row in conn.execute(_LEDGER_ROW_SQL, (self.tenant_id,)).fetchall()]
            # Only to learn which facts to lock: its ids are thrown away, so it draws none.
            first = plan_redaction(
                self.tenant_id, rows, stored, erased=erased, everything=everything, token=lambda: "0" * 32
            )
            if first.empty:
                return RedactionReport()
            for key in conflict_keys(rows, first):
                conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"fact\x1f{self.tenant_id}\x1f{key}",),
                )
            rows = [_pg_row(row) for row in conn.execute(_LEDGER_ROW_SQL, (self.tenant_id,)).fetchall()]
            plan = plan_redaction(self.tenant_id, rows, stored, erased=erased, everything=everything, token=token)
            if plan.empty:
                return RedactionReport()
            # The function checks that this plan is a redaction and nothing else, and rebuilds the
            # outbox snapshots from the rows it rewrites (migration 0029).
            row = conn.execute(
                f"SELECT {REDACTION_FUNCTION}(%s::text, %s::text, %s::text, %s::text, %s::integer, "
                "%s::jsonb, %s::jsonb, %s::jsonb, %s::boolean)",
                (
                    self.tenant_id, plan.redaction_id, reason, writer, plan.facts,
                    Jsonb([item.payload() for item in plan.events]),
                    Jsonb([item.payload() for item in plan.cards]),
                    Jsonb(list(plan.needles)), everything,
                ),
            ).fetchone()
        counts = dict(row[0]) if row and isinstance(row[0], Mapping) else {}
        return RedactionReport(
            facts_redacted=plan.facts,
            events_rewritten=int(counts.get("events_rewritten", len(plan.events))),
            cards_redacted=int(counts.get("cards_redacted", len(plan.cards))),
            receipts_deleted=int(counts.get("receipts_deleted", 0)),
            redaction_id=plan.redaction_id,
        )


# --------------------------------------------------------------------------------------- lite file

#: The lite file's redaction record, PostgreSQL's `recall_fact_redactions` (migration 0029).
SQLITE_FACT_REDACTIONS_DDL = f"""
CREATE TABLE IF NOT EXISTS {FACT_REDACTIONS_TABLE} (
    redaction_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    reason TEXT NOT NULL CHECK (reason IN ('forget', 'tenant_erase')),
    writer TEXT NOT NULL,
    facts_redacted INTEGER NOT NULL,
    events_rewritten INTEGER NOT NULL,
    cards_redacted INTEGER NOT NULL,
    receipts_deleted INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS {FACT_REDACTIONS_TABLE}_no_update BEFORE UPDATE ON {FACT_REDACTIONS_TABLE}
BEGIN SELECT RAISE(ABORT, 'recall fact redactions are append-only'); END;
CREATE TRIGGER IF NOT EXISTS {FACT_REDACTIONS_TABLE}_no_delete BEFORE DELETE ON {FACT_REDACTIONS_TABLE}
BEGIN SELECT RAISE(ABORT, 'recall fact redactions are append-only'); END;
"""


def _lite_row(row: Sequence[Any]) -> LedgerRow:
    event_id, event_type, generation_id, fact_id, fact, cards, supersedes, request_id = row
    return LedgerRow(
        event_id=str(event_id), event_type=str(event_type), generation_id=str(generation_id),
        fact_id=None if fact_id is None else str(fact_id),
        fact=json.loads(fact) if fact else None, evidence_cards=list(json.loads(cards or "[]")),
        supersedes_fact_ids=[str(v) for v in json.loads(supersedes or "[]")],
        request_id=str(request_id),
    )


class LiteFactRedactor:
    """Redaction on a lite file, under its write lane and its triggers' redaction session.

    The caller wraps it in `LiteStore.erasing()` (as `recall_forget` does), which zeroes the bytes
    the rewrite frees and checkpoints the write-ahead log, so the old values leave the file too.
    """

    def __init__(self, store: Any) -> None:
        from recall.lite.facts import ensure_fact_tables

        self._store = store
        self.tenant_id = str(store.tenant)
        ensure_fact_tables(store)

    def card_sources(self) -> set[str]:
        rows = self._store._metadata_values(
            f"SELECT DISTINCT json_extract(card, '$.source') FROM {EVIDENCE_CARD_TABLE} "
            "WHERE tenant_id = ? AND json_extract(card, '$.redacted') IS NOT 1",
            (self.tenant_id,),
        )
        return {str(name) for (name,) in rows if name}

    def _rows(self) -> list[LedgerRow]:
        return [
            _lite_row(row) for row in self._store._metadata_values(
                "SELECT event_id, event_type, generation_id, fact_id, fact, evidence_cards, "
                f"supersedes_fact_ids, request_id FROM {FACT_LEDGER_TABLE} WHERE tenant_id = ? "
                "ORDER BY created_at, event_id",
                (self.tenant_id,),
            )
        ]

    def _cards(self) -> list[tuple[str, Mapping[str, Any]]]:
        return [
            (str(card_id), json.loads(card)) for card_id, card in self._store._metadata_values(
                f"SELECT card_id, card FROM {EVIDENCE_CARD_TABLE} WHERE tenant_id = ?", (self.tenant_id,)
            )
        ]

    def preview(self, *, everything: bool = True) -> RedactionReport:
        """What `redact(everything=...)` would rewrite, writing nothing."""
        return _preview(plan_redaction(self.tenant_id, self._rows(), self._cards(), erased=None, everything=everything))

    def _has_table(self, name: str) -> bool:
        return bool(self._store._metadata_values(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ))

    def redact(
        self,
        *,
        erased: Callable[[str], bool] | None = None,
        names: Collection[str] | None = None,
        everything: bool = False,
        reason: str = "forget",
        writer: str = REDACTION_WRITER,
        token: Callable[[], str] | None = None,
    ) -> RedactionReport:
        """`PostgresFactRedactor.redact` on the file, in one write transaction."""
        from datetime import UTC, datetime

        del names  # one file: the card table is read whole, there is no second connection
        if reason not in REASONS:
            raise ValueError(f"{reason!r} is not a redaction reason")
        conn = self._store._conn
        with self._store._write():
            plan = plan_redaction(
                self.tenant_id, self._rows(), self._cards(), erased=erased, everything=everything, token=token
            )
            if plan.empty:
                return RedactionReport()
            conn.execute("INSERT INTO recall_redaction_session (redaction_id) VALUES (?)", (plan.redaction_id,))
            for item in plan.events:
                changed = conn.execute(
                    f"UPDATE {FACT_LEDGER_TABLE} SET event_id = ?, request_id = ?, fact_id = ?, fact = ?, "
                    "evidence_cards = ?, supersedes_fact_ids = ? WHERE tenant_id = ? AND event_id = ?",
                    (
                        item.new_event_id, item.request_id, item.fact_id,
                        None if item.fact is None else _json(item.fact), _json(item.evidence_cards),
                        _json(item.supersedes_fact_ids), self.tenant_id, item.event_id,
                    ),
                ).rowcount
                if changed != 1:
                    raise RuntimeError(f"fact redaction target event {item.event_id} is not in this tenant")
            for card in plan.cards:
                changed = conn.execute(
                    f"UPDATE {EVIDENCE_CARD_TABLE} SET card_id = ?, chunk_id = 'redacted', "
                    "source_digest = 'redacted', card = ? WHERE tenant_id = ? AND card_id = ?",
                    (card.new_card_id, _json(card.card), self.tenant_id, card.card_id),
                ).rowcount
                if changed != 1:
                    raise RuntimeError(f"fact redaction target card {card.card_id} is not in this tenant")
            receipts = 0
            if self._has_table(RECEIPTS_TABLE):
                if everything:
                    receipts = conn.execute(
                        f"DELETE FROM {RECEIPTS_TABLE} WHERE tenant_id = ?", (self.tenant_id,)
                    ).rowcount
                else:
                    receipts = conn.execute(
                        f"DELETE FROM {RECEIPTS_TABLE} WHERE tenant_id = ? AND EXISTS ("
                        "SELECT 1 FROM json_each(?) AS needle "
                        "WHERE length(needle.value) > 0 AND instr(result, needle.value) > 0)",
                        (self.tenant_id, json.dumps(list(plan.needles))),
                    ).rowcount
            conn.execute(
                f"INSERT INTO {FACT_REDACTIONS_TABLE} (redaction_id, tenant_id, reason, writer, "
                "facts_redacted, events_rewritten, cards_redacted, receipts_deleted, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    plan.redaction_id, self.tenant_id, reason, writer, plan.facts, len(plan.events),
                    len(plan.cards), int(receipts), datetime.now(UTC).isoformat(timespec="microseconds"),
                ),
            )
            conn.execute("DELETE FROM recall_redaction_session")
        return RedactionReport(
            facts_redacted=plan.facts, events_rewritten=len(plan.events),
            cards_redacted=len(plan.cards), receipts_deleted=int(receipts),
            redaction_id=plan.redaction_id,
        )


def _preview(plan: RedactionPlan) -> RedactionReport:
    return RedactionReport(
        facts_redacted=plan.facts, events_rewritten=len(plan.events), cards_redacted=len(plan.cards)
    )


def fact_redactor_for(store: object, env: Mapping[str, str] | None = None) -> Any | None:
    """The redactor for `store`'s fact ledger (`recall.fact_retraction.fact_ledger_for`'s), or None
    for a store that has no ledger."""
    import os

    from recall.fact_retraction import FACT_WRITE_DSN_ENV
    from recall.lite import LiteStore
    from recall.store import PgVectorStore

    if isinstance(store, LiteStore):
        return LiteFactRedactor(store)
    if isinstance(store, PgVectorStore):
        configured = (os.environ if env is None else env).get(FACT_WRITE_DSN_ENV, "").strip()
        return PostgresFactRedactor(configured or store.dsn, tenant_id=store.tenant, card_dsn=store.dsn)
    return None


__all__ = [
    "FACT_REDACTIONS_TABLE",
    "LedgerRow",
    "LiteFactRedactor",
    "PostgresFactRedactor",
    "REDACTION_FUNCTION",
    "RedactionPlan",
    "RedactionReport",
    "SQLITE_FACT_REDACTIONS_DDL",
    "fact_redactor_for",
    "plan_redaction",
]
