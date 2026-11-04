"""Regression tests for the 2026-11-03 CCA hunt of the supersession and fact-ledger family.

One test per confirmed finding that has behaviour to pin; each docstring names the finding, the
production line it guards, and the mutation (or pre-fix code) that turns it red. The PostgreSQL
redaction-lock test (STAKES-001) is in `tests/test_fact_redaction.py` and the stated-facts
forget race (SEC-001) in `tests/test_stated_facts.py`, beside the fixtures they need.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from recall.fact_current import PostgresCurrentFacts, SQLiteCurrentFacts, materialize_events
from recall.fact_redaction import LiteFactRedactor, PostgresFactRedactor
from recall.provenance_controller import FactEvent
from recall.supersession import (
    FORGOTTEN_RECORD,
    WITHDRAWN_RECORD,
    SupersessionRecord,
    resolve_supersession_graph,
)
from recall.supersession_arbiter import Note, declared_pair
from tests.test_fact_current import PARIS, TENANT, apply, city, lite

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _row(file: str, supersedes: str | None, root: str = "/r/") -> tuple[Any, ...]:
    return (file, supersedes, T0, root + file, None, None)


# ------------------------------------------------------------------------------- supersession.py


def test_bug001_a_legacy_flow_sequence_claim_supersedes_each_target() -> None:
    """BUG-001. A chunk indexed before `supersedes` held a list stores "[a.md, b.md]" as one string,
    which the PostgreSQL scan returns as one row; the resolver must read it as two claims, as
    `chunk_supersedes_targets` does on every other store.

    Red: pre-fix `_resolve_rows` (recall/supersession.py, `parsed = [_row_parts(row) ...]` without
    `_claim_rows`) keys one dangling edge "[a.md, b.md]" and fails on the edges assertion.
    """
    graph = resolve_supersession_graph(
        [_row("a.md", None), _row("b.md", None), _row("c.md", "[a.md, b.md]")]
    )
    assert graph.edges == {"a.md": "c.md", "b.md": "c.md"}, f"legacy list read as one claim: {graph.edges}"


def test_bug002_a_forgotten_memo_indexed_again_in_nfd_lets_its_live_claims_decide() -> None:
    """BUG-002. A forgotten memo indexed again without its claim must void its record, even when
    the file system spells its name in NFD (a macOS checkout) where the record holds NFC.

    Red: pre-fix `_standing_records` (recall/supersession.py) compares raw strings, keeps the
    ghost standing and fails on "old.md is still hidden".
    """
    nfc = unicodedata.normalize("NFC", "café.md")
    record = SupersessionRecord(FORGOTTEN_RECORD, nfc, "/r/" + nfc, (("old.md", T0, None, None),))
    again = unicodedata.normalize("NFD", nfc)
    graph = resolve_supersession_graph([_row("old.md", None), _row(again, None)], [record])
    assert graph.edges == {}, f"old.md is still hidden by the forgotten memo's ghost: {graph.edges}"


def test_bug002_control_a_memo_differing_only_in_case_does_not_void_a_forget() -> None:
    """Control for BUG-002: on a case-sensitive file system `Café.md` is another memo, so the
    forgotten `café.md`'s ghost keeps old.md retired (forget keeps retired, owner 2026-10-22).

    Red by mutation: `_nfc` in `_standing_records` replaced by `_fold` (NFC plus casefold).
    """
    nfc = unicodedata.normalize("NFC", "café.md")
    record = SupersessionRecord(FORGOTTEN_RECORD, nfc, "/r/" + nfc, (("old.md", T0, None, None),))
    graph = resolve_supersession_graph([_row("old.md", None), _row("Café.md", None)], [record])
    assert graph.edges == {"old.md": nfc}, f"a different memo voided the forget: {graph.edges}"


def test_bug003_withdrawing_an_ambiguous_successor_hands_its_ambiguity_to_its_claimant() -> None:
    """BUG-003. b.md ambiguously claims two notes files and c.md claims b.md; withdrawing b.md
    splices c.md onto what b.md claimed, so both notes stay unresolved (the read fails closed).

    Red: pre-fix `_resolve_rows` drops a withdrawn memo's ambiguous claim (`if kind ==
    "resolved"` only) and fails on the unresolved assertion with an empty set.
    """
    rows = [_row("a/notes.md", None), _row("b/notes.md", None), _row("c.md", "b")]
    record = SupersessionRecord(WITHDRAWN_RECORD, "b.md", "/r/b.md", (("notes", T0, None, None),))
    graph = resolve_supersession_graph(rows, [record])
    assert graph.unresolved == {"a/notes.md", "b/notes.md"}, f"ambiguity dropped: {sorted(graph.unresolved)}"
    assert graph.ambiguous_claims == {"c.md": ("notes",)}, graph.ambiguous_claims


def test_bug003_withdrawing_a_successor_with_a_dangling_claim_hands_it_to_its_claimant() -> None:
    """BUG-003 (STAKES-003's dangling half). b.md names notes.md, which is not indexed; c.md
    claims b.md; withdraw b.md and c.md must claim notes.md, so it is hidden once indexed.

    Red: pre-fix `_resolve_rows` drops the dangling claim of a withdrawn memo and fails on the
    edges assertion with {}.
    """
    record = SupersessionRecord(WITHDRAWN_RECORD, "b.md", "/r/b.md", (("notes.md", T0, None, None),))
    graph = resolve_supersession_graph([_row("c.md", "b")], [record])
    assert graph.edges == {"notes.md": "c.md"}, f"dangling claim dropped: {graph.edges}"


def test_bug003_control_an_unclaimed_withdrawn_memo_leaves_nothing_unresolved() -> None:
    """Control for BUG-003: with no claimant, withdrawing b.md gives its targets back clean."""
    record = SupersessionRecord(WITHDRAWN_RECORD, "b.md", "/r/b.md", (("notes", T0, None, None),))
    graph = resolve_supersession_graph([_row("a/notes.md", None), _row("b/notes.md", None)], [record])
    assert (graph.edges, set(graph.unresolved)) == ({}, set())


# ---------------------------------------------------------------------- supersession_arbiter.py


@pytest.mark.parametrize(("declared", "name"), [("[[Notes]]", "notes.md"), ("x", "x.markdown")])
def test_code006_a_pair_the_resolver_links_is_already_declared(declared: str, name: str) -> None:
    """CODE-006. `declared_pair` must agree with the resolver's stem rule (case, NFC, extension).

    Red: pre-fix `declared_pair` (recall/supersession_arbiter.py) compares `supersedes_key`
    stems and returns False, so the pair is sent to the paid model.
    """
    newer = Note(name="new.md", body="", modified=None, day=None, declares=(declared,))
    older = Note(name=name, body="", modified=None, day=None, declares=())
    assert declared_pair(newer, older), f"{declared!r} -> {name!r} not seen as declared"


# ------------------------------------------------------------------------------ fact_current.py


class _RewrittenUnderTheLock:
    """A `Tables` whose ledger loses `rewritten` when the tenant lock is taken: a redaction that
    committed between `materialize_events`'s first read and its locks."""

    tenant_id = TENANT

    def __init__(self, events: Sequence[FactEvent], rewritten: str) -> None:
        self.ledger = {event.event_id: event for event in events}
        self.rewritten = rewritten
        self.filed_rows: list[tuple[str, str | None, str | None, str]] = []

    def filed(self, event_ids: Sequence[str]) -> set[str]:
        return {row[0] for row in self.filed_rows} & set(event_ids)

    def events(self, event_ids: Sequence[str] | None) -> list[FactEvent]:
        ids = list(self.ledger) if event_ids is None else event_ids
        return [self.ledger[event_id] for event_id in ids if event_id in self.ledger]

    def named_key(self, fact_id: str) -> tuple[bool, str | None]:
        return False, None

    def lock_keys(self, keys: Sequence[str]) -> None:
        pass

    def lock_tenant(self, *, exclusive: bool) -> None:
        self.ledger.pop(self.rewritten, None)

    def file(self, rows: Sequence[tuple[str, str | None, str | None, str]]) -> None:
        self.filed_rows.extend(rows)

    def key_events(self, key: str) -> list[FactEvent] | None:
        return [event for event in self.ledger.values() if event.event_id in {r[0] for r in self.filed_rows}]

    def write_key(self, key: str, projection: object) -> None:
        pass

    def set_replay(self, reason: str) -> None:
        pass


def test_dat002_a_refusal_rewritten_by_a_redaction_mid_materialization_is_not_filed(tmp_path: Path) -> None:
    """DAT-002. A refused claim's ids hash its value; a redaction that rewrites them while it is
    being materialized must not leave the old ids in the current-facts tables.

    Red: pre-fix `materialize_events` (recall/fact_current.py) files what it read before taking
    its locks; `pytest.raises` fails, and the stale id is in `filed_rows`.
    """
    from recall.lite.facts import LiteEvidenceCardStore
    from recall.provenance_controller import FactApplicationRequest, ProvenanceController
    from tests.fact_fixtures import NOW
    from tests.test_fact_current import GEN, card

    with lite(tmp_path / "f.db") as (store, ledger):
        cards = LiteEvidenceCardStore(store)
        paris_card, rome_card = card(PARIS, "paris.md"), card(city("rome"), "rome.md")
        cards.put((paris_card, rome_card))
        controller = ProvenanceController(
            tenant_id=TENANT, generation_id=GEN, cards=cards, ledger=ledger,
            source_digest_for=lambda _c: "digest", now=lambda: NOW, writer="t",
        )
        controller.apply_fact(FactApplicationRequest(PARIS, (paris_card.card_id,), "r1"))
        controller.apply_fact(FactApplicationRequest(city("rome"), (rome_card.card_id,), "r2"))
        (refusal,) = [event for event in ledger.events if event.event_type == "rejected"]
    tables = _RewrittenUnderTheLock([refusal], refusal.event_id)
    with pytest.raises(LookupError):
        materialize_events(tables, [refusal.event_id])  # type: ignore[arg-type]
    assert not tables.filed_rows, f"filed ids a redaction rewrote: {tables.filed_rows}"


def test_code002_reconcile_files_a_withdrawal_with_its_redacted_successor(tmp_path: Path) -> None:
    """CODE-002. Paris, then Rome superseding it; Rome's memo forgotten (its value erased), then
    withdrawn. Reconciling every event in one batch must file the tables as a rebuild does.

    Red: pre-fix `materialize_events` (recall/fact_current.py) resolves keys through
    `tables.named_key` alone, files the withdrawal under no key, and `verify().matches` is False.
    """
    rome = city("rome")
    with lite(tmp_path / "f.db") as (store, ledger):
        apply(ledger, PARIS, "r1")
        apply(ledger, rome, "r2", supersedes=(PARIS,))
        LiteFactRedactor(store).redact(erased=lambda name: name == "r2.md", names=["r2.md"])
        ledger.withdraw_forgotten(tenant_id=TENANT, touched=lambda c: c.source == "r2.md", gone=lambda _c: True)
        facts = SQLiteCurrentFacts(store._conn, tenant_id=TENANT, write=store._write)
        assert facts.verify().matches, "control: the incremental tables already disagree"
        facts.purge()
        facts.reconcile()
        verdict = facts.verify()
        assert verdict.matches, f"one-batch reconcile disagrees with a rebuild: {verdict.to_payload()}"


# ----------------------------------------------------------------------------- fact_redaction.py


def test_env001_the_redaction_gets_the_timeout_of_the_rebuild_it_runs() -> None:
    """ENV-001. A forget's redaction runs `rebuild_tables`, which `PostgresCurrentFacts` gives 60 s;
    a shorter limit rolls the erasure back after the retraction committed.

    Red: pre-fix `PostgresFactRedactor.__init__` (recall/fact_redaction.py) defaults to 15000.
    """
    redactor = PostgresFactRedactor("postgresql://unused", tenant_id="t")
    facts = PostgresCurrentFacts("postgresql://unused", tenant_id="t")
    assert redactor.statement_timeout_ms >= facts.statement_timeout_ms, redactor.statement_timeout_ms


# ------------------------------------------------------------------------------ stated_facts.py


def test_perf003_a_store_whose_memos_state_nothing_never_reads_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PERF-003. With no admitted memo stating a fact, reconcile must not decode the tenant's ledger.

    Red: pre-fix `_reconcile` (recall/stated_facts.py) opens the fact stores and reads
    `ledger.events` before checking `pending`; the stub raises and the report is a failure.
    """
    import recall.stated_facts as stated
    from recall.lite import LiteStore
    from recall.stated_facts import StatingRule

    def never(*_args: object, **_kwargs: object) -> tuple[Any, Any]:
        raise AssertionError("the ledger was read")

    monkeypatch.setattr(stated, "_fact_stores", never)
    with LiteStore(tmp_path / "m.db", dim=4) as store:
        report = stated._reconcile(store, StatingRule(prefixes=(), tenant=store.tenant))
    assert report.facts_stated == 0 and "no admitted memo" in report.reason, report


# ------------------------------------------------- 2026-11-04 deferred-pool hunt, same family


def test_arch1_a_chain_of_withdrawn_claimants_hands_the_ambiguity_to_the_live_one() -> None:
    """ARCH-1. Withdrawn a.md ambiguously claims two notes files, withdrawn c.md claims a.md and
    live d.md claims c.md: d.md inherits a.md's ambiguity, so both notes stay unresolved.

    Red: pre-fix `_resolve_rows` (recall/supersession.py) reads the inheritors from `order[a.md]`
    alone, which a withdrawn claimant never enters, and fails on the unresolved assertion with an
    empty set.
    """
    rows = [_row("x/notes.md", None), _row("y/notes.md", None), _row("d.md", "c")]
    records = [
        SupersessionRecord(WITHDRAWN_RECORD, "a.md", "/r/a.md", (("notes", T0, None, None),)),
        SupersessionRecord(WITHDRAWN_RECORD, "c.md", "/r/c.md", (("a", T0, None, None),)),
    ]
    graph = resolve_supersession_graph(rows, records)
    assert graph.unresolved == {"x/notes.md", "y/notes.md"}, f"ambiguity dropped: {sorted(graph.unresolved)}"
    assert graph.ambiguous_claims == {"d.md": ("notes",)}, graph.ambiguous_claims


def test_arch1_control_an_unclaimed_chain_of_withdrawn_memos_drops_the_claim() -> None:
    """Control for ARCH-1: with nothing live at the end of the chain, the claim simply goes."""
    records = [
        SupersessionRecord(WITHDRAWN_RECORD, "a.md", "/r/a.md", (("notes", T0, None, None),)),
        SupersessionRecord(WITHDRAWN_RECORD, "c.md", "/r/c.md", (("a", T0, None, None),)),
    ]
    graph = resolve_supersession_graph([_row("x/notes.md", None), _row("y/notes.md", None)], records)
    assert (graph.edges, set(graph.unresolved), graph.ambiguous_claims) == ({}, set(), {})


def test_bug006_a_failed_sqlite_commit_keeps_its_error_and_serves_no_unpersisted_event(tmp_path: Path) -> None:
    """BUG-006. When SQLite ends the transaction itself (SQLITE_FULL at COMMIT), the original error
    must propagate and the in-memory ledger must not keep the event that was never persisted.

    Red: pre-fix `SQLiteFactLedger.apply_assertion` (recall/fact_ledger.py) runs a bare ROLLBACK,
    which raises "cannot rollback - no transaction is active" and skips the restore, so the
    message assertion fails (and the failed event stays in `ledger.events`).
    """
    import sqlite3

    from recall.fact_ledger import SQLiteFactLedger

    class _FullAtCommit:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self._real = conn

        def execute(self, sql: str, *args: Any) -> Any:
            if sql == "COMMIT":
                self._real.execute("ROLLBACK")  # what SQLite does on SQLITE_FULL
                raise sqlite3.OperationalError("disk full")
            return self._real.execute(sql, *args)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real, name)

    with SQLiteFactLedger(str(tmp_path / "ledger.db"), tenant_id=TENANT) as ledger:
        real = ledger._conn
        ledger._conn = _FullAtCommit(real)  # type: ignore[assignment]
        try:
            with pytest.raises(sqlite3.OperationalError) as raised:
                apply(ledger, PARIS, "r1")
        finally:
            ledger._conn = real
        assert str(raised.value) == "disk full", f"the original error was masked: {raised.value}"
        assert [event for event in ledger.events if event.request_id == "r1"] == [], ledger.events


def test_env007_a_cache_path_with_a_stray_space_still_expands_home() -> None:
    """ENV-007. `RECALL_ARBITER_CACHE` with a leading space must name the same absolute file.

    Red: pre-fix `default_cache_path` (recall/supersession_arbiter.py) builds the Path from the
    unstripped value, so `expanduser` sees " ~" and the result is relative.
    """
    from recall.supersession_arbiter import default_cache_path

    spaced = default_cache_path({"RECALL_ARBITER_CACHE": " ~/c/a.sqlite3"})
    plain = default_cache_path({"RECALL_ARBITER_CACHE": "~/c/a.sqlite3"})
    assert spaced is not None and spaced.is_absolute(), spaced
    assert spaced == plain, (spaced, plain)


def test_code007_the_prompts_state_the_quote_rule_the_code_enforces() -> None:
    """CODE-007, a nonbehavioural pin: the prompts' quote length and probability gate must match
    `MIN_QUOTE_CHARS` and `QUOTE_REQUIRED_AT` in recall/supersession_arbiter.py, because the
    prompt text cannot be templated (it is hashed into the cache key and must stay verbatim).

    Red by mutation: `MIN_QUOTE_CHARS = 30` fails the first assertion.
    """
    from recall.supersession_arbiter import DIRECTION_PROMPT, MIN_QUOTE_CHARS, QUOTE_REQUIRED_AT, SYSTEM_PROMPT

    length = f"at least {MIN_QUOTE_CHARS} characters"
    assert length in SYSTEM_PROMPT and length in DIRECTION_PROMPT, length
    assert f"{int(QUOTE_REQUIRED_AT * 100)} or more" in SYSTEM_PROMPT


def test_perf004_admitted_memos_reads_the_text_free_scan_when_the_store_has_one() -> None:
    """PERF-004. A store with `_iter_chunks_with_times` must be read through it with no text.

    Red: pre-fix `_admitted_memos` (recall/stated_facts.py) calls `store.iter_chunks()`, which
    this fake makes raise, so the call fails instead of returning the memo.
    """
    from recall.frontmatter import GRAPH_KEY
    from recall.stated_facts import StatingRule, _admitted_memos
    from recall.types import Chunk

    memo = Chunk("c1", "/r/memos/a.md", "", {GRAPH_KEY: {"facts": []}, "file": "memos/a.md", "ord": 0})

    class _Store:
        def iter_chunks(self) -> Any:
            raise AssertionError("the full-text scan was used")

        def _iter_chunks_with_times(self, batch_size: int, *, include_text: bool) -> Any:
            assert not include_text
            yield memo, None

    found = _admitted_memos(_Store(), StatingRule(prefixes=("memos",)))
    assert [(name, chunk.id) for name, chunk in found] == [("memos/a.md", "c1")]
