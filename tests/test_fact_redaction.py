"""Forgetting a memo erases the fact values it supplied, on PostgreSQL and on a lite store.

`recall_forget` retracted a fact whose only evidence was the forgotten memo, which hid it, but the
value stayed in the append-only fact ledger, in every evidence card citing the memo and in replay
receipts, and the ids that hash the value (fact, card, event) let a low-entropy value be recovered
by enumeration. `recall.fact_redaction` now rewrites all of them in place, through migration 0029's
protected function on PostgreSQL and under the lite file's redaction session.

Invariants and the failure each one catches (every test runs on both backends; `pg` needs the
test database):
- T1 after forgetting the memo that stated `door_code 4512` (and a refused `7777`), no row of the
  ledger, card, outbox or receipt tables contains either value; on a lite store neither does the
  file or its write-ahead log.
- T2 enumerating `0000` to `9999` against every id left in the tenant finds no match, although the
  same enumeration recovers the value before the forget.
- T3 every fact that survives is listed exactly as before (only cards of the forgotten memo are
  replaced), and every forgotten fact stays hidden, including one whose value another memo's
  records still hold.
- T4 a fact a surviving memo also states is neither retracted nor erased.
- T5 the triggers still refuse every DELETE and every ordinary UPDATE, and under the redaction
  flag an UPDATE of a column a redaction never changes.
- T6 a second forget of the same names erases nothing new; an injected failure reports -1 and
  leaves the value, and the same forget re-run erases it.
- T7 both backends write the same events, ids included, for the same sequence.
- T8 (PostgreSQL) a role granted UPDATE on the outbox that sets the redaction flag itself is
  still refused: the triggers honour the flag only for the table's owner.
- T9 (PostgreSQL) the redaction function refuses a plan that renames an event whose value it
  does not erase, and the outbox refuses a snapshot whose fact turns into another value.
- T10 a withdraw erases the values like a forget, and its `withdrawn` event follows the assertion
  the redaction renamed: it still names a ledger event, the value the withdrawn successor hid is
  current again, and no id left recovers the erased value (the withdrawal's id hashes the old
  assertion id, which hashes the value).
- T11 the materialized current facts (migration 0030) are rebuilt in the redaction: they name no
  id the redaction replaced, and the next read is answered from them.

Red proof, 2026-10-19, run on the remote test host, each then restored and green; the node id,
baseline or mutation, production symbol and failure are in each test's docstring.
"""

from __future__ import annotations

import itertools
import json
import re
import sqlite3
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from recall.fact_ledger import (
    PostgresFactLedger,
    PostgresMaterializationOutbox,
    assertion_event_id,
    decision_event_id,
)
from recall.lite import LiteStore
from recall.lite.facts import LiteEvidenceCardStore, LiteFactLedger
from recall.provenance_cards import PostgresEvidenceCardStore
from recall.provenance_controller import (
    DecisionCode,
    FactApplicationRequest,
    WITHDRAWAL_REQUEST_PREFIX,
    ProvenanceController,
    fact_identity,
    withdrawn_event_ids,
)
from recall.store import PgVectorStore
from recall.types import AtomicFact, Chunk
from recall_mcp.lifecycle import forget_memory, withdraw_memory
from tests.conftest import TEST_DSN, requires_db
from tests.fact_fixtures import NOW, evidence_card

DIM = 4
GEN = "generation-redaction"


def _fact(predicate: str, value: str) -> AtomicFact:
    return AtomicFact(namespace="memory", subject="user", predicate=predicate, object=value)


DOOR = _fact("door_code", "4512")
REFUSED = _fact("door_code", "7777")
CITY = _fact("city", "Turin")
PET = _fact("pet", "cat")
PLATE = _fact("plate", "AB123")


@dataclass
class Backend:
    name: str
    store: Any
    tenant: str
    cards: Any

    def ledger(self) -> Any:
        if self.name == "lite":
            return LiteFactLedger(self.store)
        return PostgresFactLedger(TEST_DSN, tenant_id=self.tenant)

    def remember(self, name: str, text: str | None = None) -> None:
        self.store.upsert(
            [Chunk(f"chunk-{name}", f"/notes/{name}", text or f"text of {name}", {"file": name})],
            [[1.0] + [0.0] * (DIM - 1)],
        )

    def apply(self, fact: AtomicFact, name: str, request_id: str, links: tuple[str, ...] = ()) -> str:
        card = evidence_card(
            fact, tenant=self.tenant, generation=GEN, chunk=f"chunk-{name}", source=name, links=links
        )
        self.cards.put((card,))
        decision = ProvenanceController(
            tenant_id=self.tenant, generation_id=GEN, cards=self.cards, ledger=self.ledger(),
            source_digest_for=lambda _card: "digest", now=lambda: NOW, writer="redaction-test",
        ).apply_fact(FactApplicationRequest(fact, (card.card_id,), request_id))
        return str(decision.code)

    def rows(self) -> list[str]:
        """Every value-bearing column of the tenant's fact tables, as text."""
        tables = [
            "SELECT event_id, request_id, fact_id, fact, evidence_cards, supersedes_fact_ids "
            "FROM recall_fact_ledger_events WHERE tenant_id = {p}",
            "SELECT card_id, chunk_id, source_digest, card FROM recall_evidence_cards WHERE tenant_id = {p}",
            "SELECT result FROM recall_idempotency_receipts WHERE tenant_id = {p}",
        ]
        if self.name == "lite":
            present = {name for (name,) in self.store._metadata_values("SELECT name FROM sqlite_master WHERE type = 'table'")}
            return [
                " ".join(str(value) for value in row)
                for sql in tables
                if sql.split(" FROM ")[1].split()[0] in present  # a file makes its receipt table on first use
                for row in self.store._metadata_values(sql.format(p="?"), (self.tenant,))
            ]
        import psycopg

        tables.append("SELECT event_id, event FROM recall_fact_materialization_outbox WHERE tenant_id = {p}")
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (self.tenant,))
            return [
                " ".join(json.dumps(v) if isinstance(v, (dict, list)) else str(v) for v in row)
                for sql in tables
                for row in conn.execute(sql.format(p="%s"), (self.tenant,)).fetchall()
            ]

    def current_fact_ids(self) -> set[str]:
        """Every id the tenant's materialized current-facts tables hold (`recall.fact_current`)."""
        sql = [
            "SELECT event_id, fact_id FROM recall_fact_current_events WHERE tenant_id = {p}",
            "SELECT event_id, fact_id FROM recall_fact_current WHERE tenant_id = {p}",
        ]
        if self.name == "lite":
            rows = [row for q in sql for row in self.store._metadata_values(q.format(p="?"), (self.tenant,))]
        else:
            import psycopg

            with psycopg.connect(TEST_DSN, autocommit=True) as conn:
                conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (self.tenant,))
                rows = [row for q in sql for row in conn.execute(q.format(p="%s"), (self.tenant,)).fetchall()]
        return {str(value) for row in rows for value in row if value is not None}

    def file_bytes(self) -> bytes:
        if self.name != "lite":
            return b""
        path = Path(self.store.path)
        wal = path.with_name(path.name + "-wal")
        return path.read_bytes() + (wal.read_bytes() if wal.exists() else b"")


@pytest.fixture(params=["lite", pytest.param("pg", marks=requires_db)])
def backend(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Backend]:
    tenant = f"redact-{uuid4().hex}"
    if request.param == "lite":
        lite = LiteStore(tmp_path / "memory.db", dim=DIM, tenant=tenant)
        yield Backend("lite", lite, tenant, LiteEvidenceCardStore(lite))
        lite.close()
        return
    make_store: Callable[[int], PgVectorStore] = request.getfixturevalue("make_store")
    table = make_store(DIM).table
    pg = PgVectorStore(TEST_DSN, dim=DIM, table=table, tenant=tenant)
    yield Backend("pg", pg, tenant, PostgresEvidenceCardStore(TEST_DSN, tenant_id=tenant))
    pg.close()


def _holds(value: str, text: str | bytes) -> bool:
    """Whether `text` holds `value` as a value: not inside a longer run of letters and digits.

    A bare substring test also matches the random hex ids a redaction mints, so it failed about
    once in a few hundred runs on an id that happened to contain the four digits of a code (CI
    `floor`, 2026-10-27: `card_c8dda6b905cada350079878f38d45126`). A stored value is bounded by
    quotes, spaces or punctuation, so this keeps every real leak and drops only that collision.
    Red proof for the checks that use it, 2026-10-27 on the remote test host: with
    `FactErasure.redact` (recall/fact_retraction.py) erasing nothing, T1, T6 and T10 fail on their
    value assertions (lite; the PostgreSQL params ran in CI).
    """
    pattern = rf"(?<![0-9A-Za-z]){re.escape(value)}(?![0-9A-Za-z])"
    if isinstance(text, bytes):
        return re.search(pattern.encode(), text) is not None
    return re.search(pattern, text) is not None


def _view(event: Any, forgotten: str) -> tuple[Any, ...]:
    """A listed fact as `recall_current_facts` shows it, with the forgotten memo's card ids out."""
    return (
        event.event_id, event.fact_id, event.fact.to_payload() if event.fact else None,
        event.created_at, len(event.evidence_cards),
        [card.card_id for card in event.evidence_cards if card.source != forgotten],
    )


def _ids(backend: Backend) -> set[str]:
    """Every id the tenant's fact tables still hold, wherever it sits."""
    found: set[str] = set()
    for text in backend.rows():
        for mark in '",[]{}:':
            text = text.replace(mark, " ")
        for token in text.split():
            if token.startswith(("fact_", "card_", "evt_")):
                found.add(token)
    return found


def test_t1_values_are_gone_after_forget(backend: Backend) -> None:
    """Node tests/test_fact_redaction.py::test_t1_values_are_gone_after_forget[lite|pg].

    Red against the base commit `cd93af5fc` (PR 946, before `recall.fact_redaction`), both
    params: fails on "4512 is still in the fact tables" (the asserted payload, its card's
    structured_facts, the retraction's card copy, the receipt). Targets `forget_memory`
    (recall_mcp/lifecycle.py) calling `FactErasure.redact`.
    """
    backend.remember("a.md", "door_code: 4512")
    assert backend.apply(DOOR, "a.md", "r1") == "APPLIED"
    assert backend.apply(REFUSED, "a.md", "r2") == DecisionCode.CONTRADICTION_WITHOUT_SUPERSESSION
    applied = next(e for e in backend.ledger().events if e.event_type == "asserted")
    backend.store.record_operation_receipt(
        "apply-key", json.dumps({"fact_id": applied.fact_id, "fact": DOOR.to_payload()}),
        operation="recall_apply_fact",
    )
    if backend.name == "pg":
        PostgresMaterializationOutbox(TEST_DSN, tenant_id=backend.tenant).enqueue(applied)
    before = " ".join(backend.rows())
    assert _holds("4512", before) and _holds("7777", before), "the scenario must put both values in the tables"

    result = forget_memory(backend.store, ["a.md"])

    after = backend.rows()
    for value in ("4512", "7777"):
        holders = [text for text in after if _holds(value, text)]
        assert not holders, f"{value} is still in the fact tables after forget: {holders[:2]} ({result.message})"
        assert not _holds(value, backend.file_bytes()), f"{value} is still in the lite file's bytes"


def test_t2_low_entropy_values_are_not_recoverable_from_ids(backend: Backend) -> None:
    """Node tests/test_fact_redaction.py::test_t2_low_entropy_values_are_not_recoverable_from_ids[lite|pg].

    Red against the base commit `cd93af5fc`, both params: fails on "recovered 4512 from the ids
    left after forget" (the retained `fact_id`). Also red, mutation of `plan_redaction`
    (recall/fact_redaction.py) keeping each old fact and card id as its new one
    (`fact_map = {value: value ...}`, `card_map = {card_id: card_id ...}`): the same failure.
    """
    backend.remember("a.md", "door_code: 4512")
    assert backend.apply(DOOR, "a.md", "r1") == "APPLIED"

    def recovered() -> list[str]:
        ids = _ids(backend)
        hits = []
        for code in (f"{n:04d}" for n in range(10_000)):
            guess = _fact("door_code", code)
            card = evidence_card(guess, tenant=backend.tenant, generation=GEN, chunk="chunk-a.md", source="a.md")
            candidates = {
                fact_identity(guess), card.card_id, assertion_event_id(backend.tenant, guess, "r1"),
            }
            if candidates & ids:
                hits.append(code)
        return hits

    assert recovered() == ["4512"], "the enumeration must work while the value is held"
    forget_memory(backend.store, ["a.md"])
    hits = recovered()
    assert hits == [], f"recovered {hits} from the ids left after forget"


def test_t3_projections_are_unchanged_for_what_survives(backend: Backend) -> None:
    """Node tests/test_fact_redaction.py::test_t3_projections_are_unchanged_for_what_survives[lite|pg].

    Red, mutation of `plan_redaction` (recall/fact_redaction.py) leaving the card copies of
    `retracted` events as they were (`cards = new_cards(...) if row.event_type != "retracted"
    else list(row.evidence_cards)`), so the assertion's card ids are replaced and its
    retraction's are not: fails on "a forgotten fact is listed again: door_code", the DOOR fact
    whose value the removed memo c.md still holds (retracted, not erased).
    """
    for name in ("a.md", "s.md", "k.md", "c.md"):
        backend.remember(name)
    assert backend.apply(CITY, "k.md", "r-city") == "APPLIED"
    assert backend.apply(PET, "a.md", "r-pet-a") == "APPLIED"
    assert backend.apply(PET, "s.md", "r-pet-s") == "DUPLICATE"
    assert backend.apply(DOOR, "a.md", "r-door-a") == "APPLIED"
    assert backend.apply(DOOR, "c.md", "r-door-c") == "DUPLICATE"
    assert backend.apply(PLATE, "a.md", "r-plate") == "APPLIED"
    # c.md leaves without a forget (a state, as gc or a deleted file): its records keep DOOR's value.
    backend.store.delete_sources(["/notes/c.md"])
    before = {e.fact.predicate: _view(e, "a.md") for e in backend.ledger().current(tenant_id=backend.tenant, now=NOW)}
    assert set(before) == {"city", "pet", "door_code", "plate"}

    result = forget_memory(backend.store, ["a.md"])

    after = {e.fact.predicate: _view(e, "a.md") for e in backend.ledger().current(tenant_id=backend.tenant, now=NOW)}
    for predicate in ("door_code", "plate"):
        assert predicate not in after, f"a forgotten fact is listed again: {predicate} ({result.message})"
    assert after == {k: before[k] for k in ("city", "pet")}, "a surviving fact is listed differently"
    text = " ".join(backend.rows())
    assert not _holds("AB123", text), "the value only the forgotten memo supplied must be erased"
    assert _holds("4512", text), "the value a memo this forget did not name still holds stays"
    assert result.facts_redacted == 1


def test_t4_a_shared_fact_survives(backend: Backend) -> None:
    """Node tests/test_fact_redaction.py::test_t4_a_shared_fact_survives[lite|pg].

    Red, mutation of `plan_redaction` (recall/fact_redaction.py) erasing a value when ANY of its
    cards is erased (`all(...)` replaced by `any(...)` in the fact rule): fails on "the fact a
    surviving memo states is no longer listed".
    """
    backend.remember("a.md")
    backend.remember("s.md")
    assert backend.apply(PET, "a.md", "r1") == "APPLIED"
    assert backend.apply(PET, "s.md", "r2") == "DUPLICATE"

    result = forget_memory(backend.store, ["a.md"])

    listed = [e.fact for e in backend.ledger().current(tenant_id=backend.tenant, now=NOW)]
    assert listed == [PET], f"the fact a surviving memo states is no longer listed: {listed} ({result.message})"
    events = backend.ledger().events
    assert not [e for e in events if e.event_type == "retracted"], "the shared fact was retracted"
    assert all(e.fact == PET for e in events if e.event_type == "asserted"), "the shared fact was erased"
    assert (result.facts_retracted, result.facts_redacted) == (0, 0)


_MARKER_ID = "red_" + "0" * 32


def _refuses(run: Callable[[str], Any], statement: str, errors: tuple[type[BaseException], ...]) -> bool:
    """Whether the database refused `statement`. Only a database error counts: `run` raises an
    AssertionError when the statement matched no row, which would otherwise read as a refusal."""
    try:
        run(statement)
    except errors:
        return True
    return False


def test_t5_the_triggers_still_refuse(backend: Backend) -> None:
    """Node tests/test_fact_redaction.py::test_t5_the_triggers_still_refuse[lite|pg].

    Red, mutation dropping the writer check from the ledger's redaction rule: lite, `AND NEW.writer
    IS OLD.writer` removed from `SQLITE_FACT_LEDGER_DDL` (recall/fact_ledger.py); pg, `AND
    NEW.writer = OLD.writer` removed from `recall_fact_ledger_append_only` in
    0029_fact_value_redaction.sql (checksum recomputed). Fails on "under the redaction flag, the
    ledger admitted: UPDATE ... SET writer".
    """
    backend.remember("a.md")
    assert backend.apply(DOOR, "a.md", "r1") == "APPLIED"
    forget_memory(backend.store, ["a.md"])  # leaves a recall_fact_redactions row
    t = backend.tenant
    plain = [
        f"UPDATE recall_fact_ledger_events SET writer = 'x' WHERE tenant_id = '{t}'",
        f"DELETE FROM recall_fact_ledger_events WHERE tenant_id = '{t}'",
        f"UPDATE recall_evidence_cards SET chunk_id = 'x' WHERE tenant_id = '{t}'",
        f"DELETE FROM recall_evidence_cards WHERE tenant_id = '{t}'",
        f"UPDATE recall_fact_redactions SET writer = 'x' WHERE tenant_id = '{t}'",
        f"DELETE FROM recall_fact_redactions WHERE tenant_id = '{t}'",
    ]
    flagged = [
        f"UPDATE recall_fact_ledger_events SET writer = 'x' WHERE tenant_id = '{t}'",
        f"UPDATE recall_fact_ledger_events SET created_at = created_at || 'x' WHERE tenant_id = '{t}'"
        if backend.name == "lite" else
        f"UPDATE recall_fact_ledger_events SET created_at = created_at + interval '1 second' WHERE tenant_id = '{t}'",
        f"UPDATE recall_evidence_cards SET source_digest = 'x' WHERE tenant_id = '{t}'",
    ]
    errors: tuple[type[BaseException], ...]
    if backend.name == "lite":
        conn = backend.store._conn
        errors = (sqlite3.DatabaseError,)

        def run(statement: str, flag: bool) -> Any:
            conn.execute("BEGIN IMMEDIATE")
            try:
                if flag:
                    conn.execute("INSERT INTO recall_redaction_session VALUES (?)", (_MARKER_ID,))
                cursor = conn.execute(statement)
                if cursor.rowcount < 1:
                    raise AssertionError(f"matched no row, so nothing was tested: {statement}")
            finally:
                conn.execute("ROLLBACK")
    else:
        import psycopg

        errors = (psycopg.Error,)

        def run(statement: str, flag: bool) -> Any:
            with psycopg.connect(TEST_DSN, autocommit=True) as conn:
                conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (t,))
                with conn.transaction(force_rollback=True):
                    if flag:
                        conn.execute("SELECT set_config('recall.redaction', %s, true)", (_MARKER_ID,))
                    cursor = conn.execute(statement)  # type: ignore[call-overload]
                    if cursor.rowcount < 1:
                        raise AssertionError(f"matched no row, so nothing was tested: {statement}")

        plain += [
            f"UPDATE recall_fact_materialization_outbox SET event_id = 'evt_{'1' * 32}' WHERE tenant_id = '{t}'",
            f"DELETE FROM recall_fact_materialization_outbox WHERE tenant_id = '{t}'",
        ]
        applied = next(e for e in backend.ledger().events if e.event_type == "asserted")
        PostgresMaterializationOutbox(TEST_DSN, tenant_id=t).enqueue(applied)

    admitted = [s for s in plain if not _refuses(lambda s: run(s, False), s, errors)]
    assert not admitted, f"without the redaction flag, the tables admitted: {admitted}"
    admitted = [s for s in flagged if not _refuses(lambda s: run(s, True), s, errors)]
    assert not admitted, f"under the redaction flag, the ledger admitted: {admitted}"
    # The positive control: the flag does admit the rewrite a redaction makes.
    run(
        f"UPDATE recall_fact_ledger_events SET request_id = request_id || 'x' WHERE tenant_id = '{t}'",
        True,
    )


def test_t6_idempotent_and_recoverable(backend: Backend, monkeypatch: pytest.MonkeyPatch) -> None:
    """Node tests/test_fact_redaction.py::test_t6_idempotent_and_recoverable[lite|pg].

    Red, mutation of `prepare_fact_erasure` (recall/fact_retraction.py) erasing only memos this
    call deleted (the redaction predicate tests `{n for n, v in resolved.items() if set(v) &
    gone_sources}` instead of `touched_names`): the re-run after the injected failure deletes
    nothing, so it erases nothing; fails on "the re-run did not complete the erasure".
    """
    from recall import fact_redaction

    backend.remember("a.md")
    backend.remember("b.md")
    assert backend.apply(PLATE, "a.md", "r1") == "APPLIED"
    assert backend.apply(DOOR, "b.md", "r2") == "APPLIED"

    first = forget_memory(backend.store, ["a.md"])
    assert first.facts_redacted == 1
    snapshot = sorted(backend.rows())
    again = forget_memory(backend.store, ["a.md"])
    assert again.facts_redacted == 0, again.message
    assert sorted(backend.rows()) == snapshot, "a second forget of the same name rewrote rows"

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("injected redaction failure")

    with monkeypatch.context() as patch:
        patch.setattr(fact_redaction, "plan_redaction", boom)
        failed = forget_memory(backend.store, ["b.md"])
    assert (failed.facts_retracted, failed.facts_redacted) == (1, -1), failed.message
    assert "WARNING" in failed.message and "re-run this forget" in failed.message
    assert _holds("4512", " ".join(backend.rows())), "a failed erasure must leave the value, not half of it"

    rerun = forget_memory(backend.store, ["b.md"])
    assert rerun.facts_redacted == 1, f"the re-run did not complete the erasure: {rerun.message}"
    assert not _holds("4512", " ".join(backend.rows()))


def _scenario(backend: Backend) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for name in ("a.md", "s.md"):
        backend.remember(name)
    out.append(("apply", [
        backend.apply(PET, "a.md", "r1"), backend.apply(PET, "s.md", "r2"), backend.apply(PLATE, "a.md", "r3"),
        backend.apply(_fact("colour", "blue"), "s.md", "r4"),
    ]))
    applied = [e for e in backend.ledger().events if e.event_type == "asserted"]
    backend.store.record_operation_receipt(
        "k1", json.dumps({"fact_id": applied[0].fact_id}), operation="recall_apply_fact"
    )
    backend.store.record_operation_receipt("k2", json.dumps({"note": "unrelated"}), operation="recall_apply_fact")
    first = forget_memory(backend.store, ["a.md"])
    out.append(("forget a", (first.facts_retracted, first.facts_redacted)))
    second = forget_memory(backend.store, ["s.md"])
    out.append(("forget s", (second.facts_retracted, second.facts_redacted)))
    out.append(("events", sorted(
        (e.event_id, e.event_type, e.fact_id, e.redaction_id, e.fact.to_payload() if e.fact else None,
         [c.to_payload() for c in e.evidence_cards], list(e.supersedes_fact_ids), e.request_id,
         e.writer, e.decision_code)
        for e in backend.ledger().events
    )))
    out.append(("receipts", sorted(text for text in backend.rows() if text.startswith("{"))))
    return out


@requires_db
def test_t7_postgres_and_lite_redact_the_same(tmp_path: Path, make_store: Callable[[int], PgVectorStore], monkeypatch: pytest.MonkeyPatch) -> None:
    """Node tests/test_fact_redaction.py::test_t7_postgres_and_lite_redact_the_same.

    Red, mutation of `LiteFactRedactor.redact` (recall/fact_redaction.py) deleting receipts only
    for a tenant erase (`if everything:` without its `else` branch): fails on step "receipts".
    """
    from recall import fact_redaction

    tenant = f"redact-parity-{uuid4().hex}"
    # The same id sequence on both backends, unique to this test: event and card ids are global
    # primary keys, so two tests counting from zero would collide in one database.
    prefix = uuid4().hex[:16]
    results = []
    for name in ("pg", "lite"):
        counter = itertools.count()
        monkeypatch.setattr(fact_redaction, "_token", lambda c=counter: f"{prefix}{next(c):016x}")
        if name == "pg":
            pg = PgVectorStore(TEST_DSN, dim=DIM, table=make_store(DIM).table, tenant=tenant)
            try:
                results.append(_scenario(Backend("pg", pg, tenant, PostgresEvidenceCardStore(TEST_DSN, tenant_id=tenant))))
            finally:
                pg.close()
        else:
            lite = LiteStore(tmp_path / "parity.db", dim=DIM, tenant=tenant)
            try:
                results.append(_scenario(Backend("lite", lite, tenant, LiteEvidenceCardStore(lite))))
            finally:
                lite.close()
    expected, actual = results
    for (label, want), (_, got) in zip(expected, actual, strict=True):
        assert got == want, f"step {label}: postgres {want!r} != lite {got!r}"
    steps = dict(expected)
    assert steps["forget a"] == (1, 1) and steps["forget s"] == (2, 2), steps
    assert steps["receipts"] == ['{"note": "unrelated"}'], steps["receipts"]



@requires_db
def test_t8_a_role_granted_update_cannot_raise_the_flag_itself(make_store: Callable[[int], PgVectorStore]) -> None:
    """Node tests/test_fact_redaction.py::test_t8_a_role_granted_update_cannot_raise_the_flag_itself.

    The outbox grants UPDATE (its delivery columns) to the controller role, and to the serving role
    without `--strict`, and `recall.redaction` is a setting any session can set. Found by an
    automated security review of this change: such a role could set it and rewrite an event
    snapshot. The triggers now honour it only for the table's owner, which `current_user` is only
    inside `recall_redact_fact_values`.

    Red, mutation of 0029_fact_value_redaction.sql dropping the owner check from every trigger
    (`WHEN current_user = ...` made `WHEN true OR current_user = ...`, checksum recomputed): fails
    on "a role granted UPDATE rewrote an outbox snapshot under its own redaction flag".
    """
    import psycopg

    tenant = f"redact-role-{uuid4().hex}"
    store = PgVectorStore(TEST_DSN, dim=DIM, table=make_store(DIM).table, tenant=tenant)
    try:
        backend = Backend("pg", store, tenant, PostgresEvidenceCardStore(TEST_DSN, tenant_id=tenant))
        backend.remember("a.md")
        assert backend.apply(DOOR, "a.md", "r1") == "APPLIED"
        applied = next(e for e in backend.ledger().events if e.event_type == "asserted")
        PostgresMaterializationOutbox(TEST_DSN, tenant_id=tenant).enqueue(applied)
    finally:
        store.close()
    role = f"recall_redact_probe_{uuid4().hex[:12]}"
    refused = None
    with psycopg.connect(TEST_DSN, autocommit=True) as conn, conn.transaction(force_rollback=True):
        conn.execute(f'CREATE ROLE "{role}" NOLOGIN')
        conn.execute(f'GRANT SELECT, UPDATE ON recall_fact_materialization_outbox TO "{role}"')
        conn.execute(f'SET LOCAL ROLE "{role}"')
        conn.execute("SELECT set_config('recall.tenant_id', %s, true)", (tenant,))
        conn.execute("SELECT set_config('recall.redaction', %s, true)", (_MARKER_ID,))
        visible = conn.execute(
            "SELECT count(*) FROM recall_fact_materialization_outbox WHERE tenant_id = %s", (tenant,)
        ).fetchone()
        assert visible == (1,), f"the probe role must see the row it tries to rewrite: {visible}"
        try:
            with conn.transaction():
                conn.execute(
                    "UPDATE recall_fact_materialization_outbox SET event_id = %s, "
                    "event = jsonb_set(event, '{event_id}', to_jsonb(%s::text)) WHERE tenant_id = %s",
                    ("evt_" + "1" * 32, "evt_" + "1" * 32, tenant),
                )
        except psycopg.Error as exc:
            refused = str(exc)
    assert refused and "immutable" in refused, (
        "a role granted UPDATE rewrote an outbox snapshot under its own redaction flag"
    )


@requires_db
def test_t9_the_function_and_the_outbox_refuse_a_rewrite_that_is_not_a_redaction(
    make_store: Callable[[int], PgVectorStore],
) -> None:
    """Node tests/test_fact_redaction.py::test_t9_the_function_and_the_outbox_refuse_a_rewrite_that_is_not_a_redaction.

    `recall_redact_fact_values` runs as the owner, so it must not take the controller's plan on
    trust (found by an automated security review of this change): it refuses a plan that renames
    an event whose value it does not erase, and the outbox trigger refuses a snapshot whose fact
    changes into anything but the redaction marker.

    Red, two mutations of 0029_fact_value_redaction.sql (checksum recomputed), one per refusal:
    the function's `RAISE EXCEPTION 'fact redaction renames event %, ...'` made a `RAISE NOTICE`,
    failing on "the function applied a plan that renames an event it does not erase"; and the
    outbox trigger's `NEW.event->'fact' IS NOT DISTINCT FROM OLD.event->'fact'` made `true`, failing
    on "the outbox took a snapshot whose fact was rewritten to another value".
    """
    import psycopg
    from psycopg.types.json import Jsonb

    tenant = f"redact-forge-{uuid4().hex}"
    store = PgVectorStore(TEST_DSN, dim=DIM, table=make_store(DIM).table, tenant=tenant)
    try:
        backend = Backend("pg", store, tenant, PostgresEvidenceCardStore(TEST_DSN, tenant_id=tenant))
        backend.remember("a.md")
        assert backend.apply(DOOR, "a.md", "r1") == "APPLIED"
        applied = next(e for e in backend.ledger().events if e.event_type == "asserted")
        PostgresMaterializationOutbox(TEST_DSN, tenant_id=tenant).enqueue(applied)
    finally:
        store.close()
    forged = {
        "event_id": applied.event_id, "new_event_id": "evt_" + "2" * 32, "request_id": applied.request_id,
        "fact_id": applied.fact_id, "fact": DOOR.to_payload(),
        "evidence_cards": [card.to_payload() for card in applied.evidence_cards], "supersedes_fact_ids": [],
    }
    function_refused = outbox_refused = None
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        try:
            with conn.transaction(force_rollback=True):
                conn.execute(
                    "SELECT recall_redact_fact_values(%s::text, %s::text, 'forget', 'probe', 0, "
                    "%s::jsonb, '[]'::jsonb, '[]'::jsonb, false)",
                    (tenant, _MARKER_ID, Jsonb([forged])),
                )
        except psycopg.Error as exc:
            function_refused = str(exc)
        try:
            with conn.transaction(force_rollback=True):
                conn.execute("SELECT set_config('recall.redaction', %s, true)", (_MARKER_ID,))
                changed = conn.execute(
                    "UPDATE recall_fact_materialization_outbox "
                    "SET event = jsonb_set(event, '{fact,object}', '\"9999\"') WHERE tenant_id = %s",
                    (tenant,),
                ).rowcount
                assert changed == 1, "the forged snapshot update matched no row, so nothing was tested"
        except psycopg.Error as exc:
            outbox_refused = str(exc)
    assert function_refused and "does not erase" in function_refused, (
        "the function applied a plan that renames an event it does not erase"
    )
    assert outbox_refused and "immutable" in outbox_refused, (
        "the outbox took a snapshot whose fact was rewritten to another value"
    )


OLD_DOOR = _fact("door_code", "1111")


def test_t10_a_withdrawal_follows_the_assertion_it_names(backend: Backend) -> None:
    """Node tests/test_fact_redaction.py::test_t10_a_withdrawal_follows_the_assertion_it_names[lite|pg].

    Red proof 2026-10-24 on the remote test host, each mutation alone, then restored and green.
    Red, mutation of `plan_redaction` (recall/fact_redaction.py) renaming only retractions
    (`_NAMING_PREFIXES` without its "withdrawn" entry): on lite fails on "the withdrawal names an
    assertion the ledger no longer holds" (and the old successor hides 1111 again); on PostgreSQL
    migration 0029's function refuses the plan, failing on "the withdraw did not erase the value".
    Red, mutation of `recall_redact_fact_values` (migration 0029) without its `withdrawn` branch,
    checksum updated: PostgreSQL fails on "the withdraw did not erase the value".
    """
    backend.remember("old.md")
    backend.remember("a.md", "door_code: 4512")
    assert backend.apply(OLD_DOOR, "old.md", "r-old") == "APPLIED"
    assert backend.apply(DOOR, "a.md", "r-new", links=("old.md",)) == "APPLIED"
    ledger = backend.ledger()
    assert [e.fact.object for e in ledger.current(tenant_id=backend.tenant, now=NOW)] == ["4512"]

    result = withdraw_memory(backend.store, ["a.md"])

    assert (result.facts_withdrawn, result.facts_redacted) == (1, 1), (
        f"the withdraw did not erase the value: {result.message}"
    )
    events = backend.ledger().events
    named = withdrawn_event_ids(events)
    assert named and named <= {e.event_id for e in events}, (
        "the withdrawal names an assertion the ledger no longer holds"
    )
    listed = [e.fact.object for e in backend.ledger().current(tenant_id=backend.tenant, now=NOW) if e.fact]
    assert listed == ["1111"], f"withdrawing the erased successor did not give 1111 back: {listed}"
    ids = _ids(backend)
    hits = []
    for code in (f"{n:04d}" for n in range(10_000)):
        old_id = assertion_event_id(backend.tenant, _fact("door_code", code), "r-new")
        withdrawal = decision_event_id(backend.tenant, WITHDRAWAL_REQUEST_PREFIX + old_id, "withdrawn")
        if {old_id, withdrawal} & ids:
            hits.append(code)
    assert hits == [], f"recovered {hits} from the ids left after the withdraw"
    assert not _holds("4512", " ".join(backend.rows()))


def test_t11_the_current_facts_tables_are_rebuilt(backend: Backend) -> None:
    """Node tests/test_fact_redaction.py::test_t11_the_current_facts_tables_are_rebuilt[lite|pg].

    Red proof 2026-10-24 on the remote test host, then restored and green.
    Red, mutation removing the rebuild from both redactors (`_rebuild_pg_current_facts(...)` in
    `PostgresFactRedactor.redact`, `rebuild_tables(...)` in `LiteFactRedactor.redact`,
    recall/fact_redaction.py): fails on "the current-facts tables still name ids the redaction
    replaced", both params.
    """
    backend.remember("a.md")
    backend.remember("k.md")
    assert backend.apply(DOOR, "a.md", "r-door") == "APPLIED"
    assert backend.apply(CITY, "k.md", "r-city") == "APPLIED"
    door = next(e for e in backend.ledger().events if e.event_type == "asserted" and e.fact == DOOR)
    replaced = {door.event_id, str(door.fact_id)}
    assert replaced <= backend.current_fact_ids(), "the tables must file the fact before the forget"

    result = forget_memory(backend.store, ["a.md"])

    assert result.facts_redacted == 1, result.message
    left = replaced & backend.current_fact_ids()
    assert not left, f"the current-facts tables still name ids the redaction replaced: {sorted(left)}"
    ledger = backend.ledger()
    listed = [e.fact.object for e in ledger.current(tenant_id=backend.tenant, now=NOW) if e.fact]
    assert listed == ["Turin"]
    assert ledger.last_read == "table", "the read after the redaction was not answered by the tables"
