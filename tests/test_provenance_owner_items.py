"""The owner's 2026-11-04 decisions on the provenance_controller hunt's open items.

Hunt 20261104T131626Z (PR 1175) and its deferred pass (PR 1181) left these to the owner, who
decided: normalise fact identity (NUM-002, NUM-003) together with the re-key migration after the
AML window, shipping the inventory check now; keep unknown card verdicts readable (CODE-011); type
the cardinality (CODE-010); and fix the forget-versus-apply race (STAKES-003), the erased-successor
ordering (NUM-001) and the card access check (SEC-002). The DEEP audit of that change (run
20261104T185256Z) found that normalising on read broke stored data (STAKES-001, DAT-002), so the
owner held normalisation for the re-key. Each behaviour test's docstring says how its red state
was proved.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from recall.provenance_controller import (
    DecisionCode,
    EvidenceCardStore,
    FactApplicationRequest,
    InMemoryFactLedger,
    ProvenanceController,
    evidence_card_from_payload,
)
from recall.types import AtomicFact
from recall_mcp.lifecycle import forget_memory
from tests.conftest import requires_db
from tests.fact_fixtures import NOW, evidence_card
from tests.test_fact_redaction import GEN, Backend, _holds, backend  # noqa: F401  (fixture)
from tests.test_lite_writer_sites import lite_env  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[1]


def _fact(value: object, **window: datetime) -> AtomicFact:
    return AtomicFact(namespace="memory", subject="user", predicate="setpoint", object=value, **window)


# STAKES-001, DAT-002 (audit of this change): a fact stored in the old spelling stays readable.


def _stored_spelling(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """A card payload and fact id exactly as the code before this change stored a fact whose
    object is the float `20.0` and whose bound carries `+01:00`, hashed without normalising."""
    import hashlib

    from recall.provenance_card_hash import card_id_for_payload
    from recall.provenance_controller import canonical_json

    fact = {**payload["structured_facts"][0], "object": 20.0, "valid_from": "2026-01-01T01:00:00+01:00"}
    card = {**payload, "structured_facts": [fact]}
    body = {key: value for key, value in card.items() if key != "card_id"}
    for key in ("valid_from", "valid_until", "first_indexed_at", "indexed_at"):
        if isinstance(body.get(key), str):
            body[key] = datetime.fromisoformat(body[key])
    card["card_id"] = card_id_for_payload(body)
    fact_id = "fact_" + hashlib.sha256(canonical_json(fact).encode("utf-8")).hexdigest()[:32]
    return card, fact_id


def test_stakes001_a_card_stored_with_a_whole_float_and_an_offset_still_reads():
    """STAKES-001 (P1): normalising `AtomicFact` on construction broke the id check of every
    stored card whose fact was spelled `20.0` or carried an offset, and every ledger read decodes
    every card, so one such row made the tenant's ledger unreadable and its forgets impossible.
    Facts are read as stored until the post-AML re-key normalises stored and new facts together.

    Red proof: against the normalising tree (snapshot of run 20261104T185256Z) the decode raised
    "evidence card id does not match its canonical content".
    """
    claim = AtomicFact(namespace="memory", subject="user", predicate="setpoint", object=21.5)
    payload = evidence_card(claim, tenant="tenant-a", generation="gen-a", chunk="c", source="a.md").to_payload()
    stored, _fact_id = _stored_spelling(json.loads(json.dumps(payload)))
    card = evidence_card_from_payload(json.loads(json.dumps(stored)))
    assert card.card_id == stored["card_id"]
    assert card.structured_facts[0].object == 20.0


def test_dat002_a_fact_stored_with_a_whole_float_keeps_the_id_stored_links_name():
    """DAT-002 (P1): the ledger recomputes a stored fact's id from its decoded payload, and the
    links that retire or retract it (`supersedes_fact_ids`) hold the id it was stored under, so a
    decode that normalised `20.0` or `+01:00` gave the fact a new id and every such link stopped
    naming it: a superseded value read as current again.

    Red proof: against the normalising tree the recomputed id differed from the stored one.
    """
    from recall.provenance_controller import fact_identity

    claim = AtomicFact(namespace="memory", subject="user", predicate="setpoint", object=21.5)
    payload = evidence_card(claim, tenant="tenant-a", generation="gen-a", chunk="c", source="a.md").to_payload()
    stored, fact_id = _stored_spelling(json.loads(json.dumps(payload)))
    assert fact_identity(AtomicFact.from_payload(stored["structured_facts"][0])) == fact_id


def _ledger_file(path: Path, facts: list[dict], cards: list[dict]) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE recall_fact_ledger_events (fact TEXT, evidence_cards TEXT)")
    conn.execute("CREATE TABLE recall_evidence_cards (card TEXT)")
    for fact in facts:
        conn.execute("INSERT INTO recall_fact_ledger_events VALUES (?, '[]')", (json.dumps(fact),))
    for card in cards:
        conn.execute("INSERT INTO recall_evidence_cards VALUES (?)", (json.dumps(card),))
    conn.commit()
    conn.close()


def _normcheck() -> Any:
    """The check script loaded as a module, for tests that call its functions directly."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("normcheck", ROOT / "scripts/fact_identity_normalisation_check.py")
    assert spec is not None and spec.loader is not None
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    return check


def _check(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/fact_identity_normalisation_check.py"), *args],
        capture_output=True, text=True, cwd=ROOT, env=env,
    )


def test_normalisation_check_counts_the_rows_whose_identity_changes(tmp_path: Path):
    """The pre-deploy check (POST_AML_DEPLOY.md, step 1) finds a stored offset, a stored whole
    float, and a card holding one, and exits 1. Its control: a clean file exits 0.

    Red proof: with `fact_changes` returning False (mutation of the script) the first run printed
    0 rows and exited 0, failing the first assertion.
    """
    plain = {"subject": "user", "predicate": "p", "object": "x", "valid_from": "2026-01-01T00:00:00+00:00"}
    offset = {**plain, "valid_from": "2026-01-01T01:00:00+01:00"}
    whole = {**plain, "object": 20.0}
    dirty = tmp_path / "dirty.db"
    _ledger_file(dirty, [plain, offset, whole], [{"structured_facts": [whole]}, {"structured_facts": [plain]}])
    run = _check("--sqlite", str(dirty))
    assert run.returncode == 1, run.stdout + run.stderr
    assert "recall_fact_ledger_events.fact: 2" in run.stdout
    assert "recall_evidence_cards.card: 1" in run.stdout
    clean = tmp_path / "clean.db"
    _ledger_file(clean, [plain, {"redacted": True, "fact_id": "f", "redaction_id": "r"}], [{"structured_facts": [plain]}])
    assert _check("--sqlite", str(clean)).returncode == 0


def test_deploy001_the_check_reads_a_number_as_the_ledger_decodes_it():
    """DEPLOY-001: on PostgreSQL jsonb expands an exponent float (`1e20` reads back as an int);
    the original spelling is in the `__recall_json` companion the ledger decodes. The check read
    the expanded value and counted the fact unchanged.

    Red proof: against the pre-fix script (snapshot of run 20261104T185256Z) `count` returned 0.
    """
    from recall.provenance_json import encode_jsonb

    check = _normcheck()
    fact = {"subject": "user", "predicate": "p", "namespace": "memory", "object": 1e20, "context": {}}
    stored = {**encode_jsonb(fact), "object": 10**20}  # what jsonb hands back
    assert check.count("fact", [stored]) == 1
    assert check.count("cards", [[{**encode_jsonb({"structured_facts": [fact]}), "structured_facts": [{**fact, "object": 10**20}]}]]) == 1


def test_env002_the_check_takes_the_dsn_from_the_environment(monkeypatch: pytest.MonkeyPatch):
    """ENV-002: the check took the DSN, password included, only from argv.

    Red proof: against the pre-fix script `main(["--tenant", "memory"])` exited with a usage
    error ("one of the arguments --sqlite --dsn is required") instead of reading the variable.
    """
    check = _normcheck()
    seen: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(check, "_postgres_counts", lambda dsn, tenants: seen.append((dsn, tenants)) or {})
    monkeypatch.setenv("RECALL_FACT_WRITE_DSN", "postgresql://from-env/db")
    assert check.main(["--tenant", "memory"]) == 0
    assert seen == [("postgresql://from-env/db", ["memory"])]


# CODE-010, CODE-011.


def test_code010_cardinality_is_checked_as_a_fact_cardinality():
    """Nonbehavioural pin (CODE-010 is a typing change): a payload's cardinality is still
    refused unless it is one of the three, exactly as before; no red proof applies."""
    assert AtomicFact.from_payload({"subject": "u", "predicate": "p", "cardinality": "set"}).cardinality == "set"
    for bad in ("SET", 1, ""):
        with pytest.raises(ValueError, match="cardinality must be one of"):
            AtomicFact.from_payload({"subject": "u", "predicate": "p", "cardinality": bad})


def test_code011_an_unknown_verdict_is_read_declared_and_never_trusted():
    """CODE-011 (owner: do not fail, declare it unknown): a stored card whose verdict no release
    defines reads back as written, `verdict_known` says it is unknown, and it authorizes nothing.

    Red proof: against the pre-fix controller the import of `verdict_known` below fails; that is
    not behavioural proof, so the behavioural half (reads, refuses) is a pin of behaviour that did
    not change: the owner decided the reading should stay as it was.
    """
    claim = _fact("20")
    written = replace(
        evidence_card(claim, tenant="tenant-a", generation="gen-a", chunk="c", source="a.md"),
        card_id="", verdict="OK",
    )
    card = evidence_card_from_payload(json.loads(json.dumps(written.to_payload())))
    assert card.verdict == "OK"
    from recall.provenance_controller import verdict_known

    assert not verdict_known(card.verdict) and verdict_known("ok")
    store = EvidenceCardStore()
    store.put([card])
    decision = ProvenanceController(
        tenant_id="tenant-a", generation_id="gen-a", cards=store, ledger=InMemoryFactLedger(),
        source_digest_for=lambda _card: "digest", now=lambda: NOW, writer="test",
    ).apply_fact(FactApplicationRequest(claim, (card.card_id,), "r-1"))
    assert decision.code == DecisionCode.TRUST_UNAVAILABLE and not decision.allowed


# SEC-002: the caller's source access policy decides which cards it may cite.


def test_sec002_a_card_the_caller_may_not_read_is_not_found():
    """SEC-002: a principal the source policy kept from a memo applied a fact with that memo's
    card and learned from the answer what the memo states.

    Red proof: with `ProvenanceController._permitted` mutated to return True the first apply was
    APPLIED, failing the first assertion.
    """
    claim = _fact("20")
    card = evidence_card(claim, tenant="tenant-a", generation="gen-a", chunk="c", source="secret.md")
    store = EvidenceCardStore()
    store.put([card])
    ledger = InMemoryFactLedger()

    def controller(permitted):
        return ProvenanceController(
            tenant_id="tenant-a", generation_id="gen-a", cards=store, ledger=ledger,
            source_digest_for=lambda _card: "digest", now=lambda: NOW, writer="test",
            card_permitted=permitted,
        )

    denied = controller(lambda c: c.source != "secret.md").apply_fact(
        FactApplicationRequest(claim, (card.card_id,), "r-denied")
    )
    assert denied.code == DecisionCode.CARD_NOT_FOUND and not denied.allowed
    # Another principal applies it; the denied one cannot learn that through a replay either.
    assert controller(lambda c: True).apply_fact(FactApplicationRequest(claim, (card.card_id,), "r-ok")).allowed
    replay = controller(lambda c: c.source != "secret.md").apply_fact(
        FactApplicationRequest(claim, (card.card_id,), "r-ok")
    )
    assert replay.code == DecisionCode.CARD_NOT_FOUND and not replay.allowed
    broken = controller(lambda c: 1 / 0).apply_fact(FactApplicationRequest(claim, (card.card_id,), "r-err"))
    assert broken.code == DecisionCode.CARD_NOT_FOUND


# NUM-001 and STAKES-003, on lite and PostgreSQL.


CITY = AtomicFact(namespace="memory", subject="user", predicate="city", object="Turin")
ROME = AtomicFact(namespace="memory", subject="user", predicate="city", object="Rome")


def _current_objects(target: Backend) -> list[object]:
    events = target.ledger().current(tenant_id=target.tenant, now=NOW + timedelta(seconds=1))
    return sorted(str(event.fact.object) for event in events if event.fact is not None)


def test_num001_a_value_asserted_again_after_its_forgotten_successor_is_current(backend: Backend):  # noqa: F811
    """NUM-001: under a fixed clock, Turin, then Rome superseding it, then Rome's memo forgotten
    (its receipt kept, with no fact) and Turin asserted again: the new Turin took the receipt's
    own time, which hides what it names, and stayed hidden although the ledger answered APPLIED.

    Red proof: against the pre-fix `ordered_after` (recall/provenance_controller.py snapshot
    2026-11-04) the last assertion found no current value, on lite and PostgreSQL.
    """
    for name in ("a.md", "b.md", "c.md"):
        backend.remember(name)
    assert backend.apply(CITY, "a.md", "r-a") == DecisionCode.APPLIED
    assert backend.apply(ROME, "b.md", "r-b", links=("a.md",)) == DecisionCode.APPLIED
    assert _current_objects(backend) == ["Rome"]
    forget_memory(backend.store, ["b.md"])
    assert backend.apply(CITY, "c.md", "r-c") == DecisionCode.APPLIED
    assert _current_objects(backend) == ["Turin"]


def test_stakes003_an_application_racing_a_forget_is_retracted_and_erased(
    backend: Backend, monkeypatch: pytest.MonkeyPatch  # noqa: F811
):
    """STAKES-003: a forget of a card's memo landing between the controller's card check and the
    append left the assertion appended after the forget's retraction: current, never retracted,
    its value never erased. The forget now waits for the application (its erasure lock), then
    retracts and erases what it appended.

    The forget runs in a thread started from inside the card check. The check waits briefly for
    the forget to get past its locks (into `_forget_memory_unlocked`); if it does, nothing holds a
    lock against it, and the check waits for the whole forget, so the forget completes before the
    append however slow it is. If it does not, the application goes on and the forget follows it.
    It used to wait for the whole forget for a flat two seconds, paid on every green run, and a
    slow runner could let a regression pass (2026-11-05 deferred pass, CODE-006).

    Red proof: against the pre-fix controller, lifecycle and ledgers (snapshot 2026-11-04), and
    with the erasure lock mutated to `nullcontext()` (2026-11-05: `recall_mcp.lifecycle._erasure_lock`
    for PostgreSQL, `LiteFactLedger.erasure_lock` for lite), the forget ran first, the racer and
    APPLIED assertions passed, and `assert _current_objects(backend) == []` failed with
    `['Turin'] == []` (DOC-007).
    """
    import recall_mcp.lifecycle as lifecycle

    backend.remember("a.md")
    card = evidence_card(CITY, tenant=backend.tenant, generation=GEN, chunk="chunk-a.md", source="a.md")
    backend.cards.put((card,))
    errors: list[BaseException] = []
    started, done = threading.Event(), threading.Event()
    real_unlocked = lifecycle._forget_memory_unlocked

    def observed(*args, **kwargs):
        started.set()
        return real_unlocked(*args, **kwargs)

    monkeypatch.setattr(lifecycle, "_forget_memory_unlocked", observed)

    def forget() -> None:
        try:
            forget_memory(backend.store, ["a.md"])
        except BaseException as exc:  # recorded and re-raised below
            errors.append(exc)
        finally:
            done.set()

    racer = threading.Thread(target=forget)

    def digest(_card) -> str:
        if racer.ident is None:
            racer.start()
            if started.wait(0.5):
                done.wait(60)
        return "digest"

    decision = ProvenanceController(
        tenant_id=backend.tenant, generation_id=GEN, cards=backend.cards, ledger=backend.ledger(),
        source_digest_for=digest, now=lambda: NOW, writer="race-test",
    ).apply_fact(FactApplicationRequest(CITY, (card.card_id,), "r-race"))
    racer.join(timeout=60)
    assert not racer.is_alive() and not errors, errors
    assert decision.code == DecisionCode.APPLIED
    assert _current_objects(backend) == []
    assert not any(_holds("Turin", row) for row in backend.rows())


# The audit of this change (run 20261104T185256Z): PERF-001, DAT-004, ENV-001, DEPLOY-002.


class _LockRecordingLedger(InMemoryFactLedger):
    """An in-memory ledger with an erasure lock that only records whether it is held."""

    def __init__(self) -> None:
        super().__init__()
        self.held = False

    @contextmanager
    def erasure_lock(self, *, shared: bool):
        self.held = True
        try:
            yield
        finally:
            self.held = False


def test_perf001_the_fresh_search_runs_without_the_erasure_lock():
    """PERF-001: `apply_fact` held the shared erasure lock through its fresh search (an embed, a
    search, a rerank), so a forget waited for the search and every application queued behind the
    forget waited too. Both passes still hold it; the search between them does not.

    Red proof: against the pre-fix controller (snapshot of run 20261104T185256Z) the search saw
    the lock held and the first assertion failed.
    """
    claim = _fact("20")
    card = evidence_card(claim, tenant="tenant-a", generation="gen-a", chunk="c", source="a.md")
    store = EvidenceCardStore()
    store.put([card])
    ledger = _LockRecordingLedger()
    during_search: list[bool] = []

    def fresh_search(_claim, _request):
        during_search.append(ledger.held)
        return (card.card_id,)

    decision = ProvenanceController(
        tenant_id="tenant-a", generation_id="gen-a", cards=store, ledger=ledger,
        source_digest_for=lambda _card: "digest", fresh_search=fresh_search, now=lambda: NOW,
        writer="test",
    ).apply_fact(FactApplicationRequest(claim, ("card_missing",), "r-1"))
    assert during_search == [False]
    assert decision.code == DecisionCode.APPLIED and decision.retried


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from tests.test_lite_calibration import _write_memos

    path = tmp_path / "root"
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(path))
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    monkeypatch.delenv("RECALL_STATED_FACT_SOURCES", raising=False)
    path.mkdir(parents=True)
    _write_memos(path / "memory", 4)
    return path


def test_dat004_a_reconcile_keeps_the_generation_its_cards_were_built_with(
    root: Path, monkeypatch: pytest.MonkeyPatch
):
    """DAT-004: the stated reconcile read the store's generation again under the erasure lock,
    after building its cards with the first reading; on lite a forget that lock waited for moves
    the generation (it is the fingerprint), and a controller on the new one refused every card.

    Red proof: against the pre-fix stated_facts (snapshot of run 20261104T185256Z) the report was
    facts_stated=-1 with UNSUPPORTED_CLAIM, failing the assertion.
    """
    import recall.stated_facts as stated
    from recall.lite import LiteStore
    from recall.lite.store import generation_id_for
    from tests.test_stated_facts import DIM, memo, refreshed

    real_reconcile = stated.reconcile_stated_facts
    with LiteStore(root.parent / ".recall" / "memory.db", dim=DIM) as store:
        memo(root / "memory", "door.md", facts=[{"subject": "user", "predicate": "door_code", "object": "4512"}])
        monkeypatch.setattr(stated, "reconcile_stated_facts", lambda *a, **k: stated.StatedFactsReport(True, ""))
        refreshed(store, root, key_root=root)
        monkeypatch.setattr(stated, "reconcile_stated_facts", real_reconcile)
        reads = {"n": 0}

        def generation(self: LiteStore) -> str:
            reads["n"] += 1
            real = generation_id_for(self._matrix.fingerprint)
            return real if reads["n"] == 1 else real + "-moved"

        monkeypatch.setattr(LiteStore, "generation_id", property(generation))
        report = real_reconcile(store, roots=[root])
    assert report.facts_stated == 1, report


@requires_db
def test_env001_the_erasure_lock_survives_an_idle_session_timeout():
    """ENV-001: the PostgreSQL erasure lock sits on an idle connection; a role or server with an
    idle_session_timeout ended that session mid-forget and released the lock unseen. The lock's
    session turns the timeout off, as `recall.index_lock` does.

    Red proof: against the pre-fix fact_ledger (snapshot of run 20261104T185256Z) a second session
    took the lock 2.5 s into a hold under a role with a 1 s idle_session_timeout.
    """
    import time
    from uuid import uuid4

    import psycopg
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    from recall.fact_ledger import PostgresFactLedger, fact_erasure_lock
    from tests.conftest import TEST_DSN

    role = f"recall_idle_probe_{uuid4().hex[:12]}"
    key = fact_erasure_lock("idle-probe")
    with psycopg.connect(TEST_DSN, autocommit=True) as admin:
        admin.execute(f"CREATE ROLE {role} LOGIN PASSWORD 'probe'")
        admin.execute(f"ALTER ROLE {role} SET idle_session_timeout = '1s'")
    try:
        params = {**conninfo_to_dict(TEST_DSN), "user": role, "password": "probe"}
        ledger = PostgresFactLedger(make_conninfo(**params), tenant_id="idle-probe")
        with ledger.erasure_lock(shared=False):
            time.sleep(2.5)
            with psycopg.connect(TEST_DSN, autocommit=True) as other:
                row = other.execute("SELECT pg_try_advisory_lock(hashtextextended(%s, 0))", (key,)).fetchone()
                taken = bool(row and row[0])
                if taken:
                    other.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (key,))
            assert not taken, "the erasure lock was released while its holder was still erasing"
    finally:
        with psycopg.connect(TEST_DSN, autocommit=True) as admin:
            admin.execute(f"DROP ROLE IF EXISTS {role}")


def test_deploy002_an_apply_that_cannot_take_the_lock_releases_its_reservation(
    lite_env: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch  # noqa: F811
):
    """DEPLOY-002: `recall_apply_fact` can now fail on the erasure lock before anything is written
    (a busy lite lane here), and kept its idempotency reservation, so the retry the controller's
    docstring promises was refused under a limiter.

    Red proof: against the pre-fix server (snapshot of run 20261104T185256Z) `released` stayed
    empty and the assertion failed.
    """
    from recall_mcp import server as srv
    from tests.test_lite_mcp import _serve
    from tests.test_lite_writer_sites import _db, _hold

    released: list[str | None] = []
    real = srv._release_mutation_reservation

    async def record(state: dict[str, object], tenant: str, idempotency_key: str | None) -> None:
        released.append(idempotency_key)
        await real(state, tenant, idempotency_key)

    monkeypatch.setattr(srv, "_release_mutation_reservation", record)
    monkeypatch.setenv("RECALL_LITE_WRITE_WAIT", "0")
    holder = _hold(_db(lite_env), tmp_path)
    try:
        async def body(state: dict[str, Any], call: Any) -> None:
            await call(
                "recall_apply_fact", namespace="memory", subject="user", predicate="door_code",
                object="4512", evidence_card_ids=["card_" + "0" * 32], request_id="r-busy",
            )

        _serve(lite_env, body)
    finally:
        holder.kill()
        holder.wait(10)
    assert "r-busy" in released, f"a busy apply kept its reservation (released: {released})"


def test_env004_a_card_is_checked_at_the_instant_its_assertion_is_stamped():
    """ENV-004 (owner decision 2026-11-04, option a): one clock reading per application pass.
    The controller read the clock per card in `_resolve` and again in `_in_force`, whose reading
    stamps the assertion, so a card checked valid just before its `valid_until` was stamped at or
    after it. A file-backed eval clock (lite MCP) turns that gap from a tick into a step, which
    this clock models: the first reading is inside the card's validity, every later one past it.

    Red proof: against the per-card reads (commit 866380d4e) the assertion was stamped at
    NOW + 2h, after the card's valid_until of NOW + 1h, and the stamp assertion failed.
    """
    claim = _fact("20")
    card = replace(
        evidence_card(claim, tenant="tenant-a", generation="gen-a", chunk="c", source="a.md"),
        card_id="", valid_until=NOW + timedelta(hours=1),
    )
    store = EvidenceCardStore()
    store.put([card])
    readings = iter([NOW])

    def clock() -> datetime:
        return next(readings, NOW + timedelta(hours=2))

    decision = ProvenanceController(
        tenant_id="tenant-a", generation_id="gen-a", cards=store, ledger=InMemoryFactLedger(),
        source_digest_for=lambda _card: "digest", now=clock, writer="test",
    ).apply_fact(FactApplicationRequest(claim, (card.card_id,), "r-1"))
    assert decision.event is not None
    assert decision.event.created_at < card.valid_until, (
        f"stamped at {decision.event.created_at}, after the card's validity ended "
        f"at {card.valid_until} (decision {decision.code})"
    )
    assert decision.code == DecisionCode.APPLIED


# The deferred pass over run 20261104T185256Z (run 20261105T095124Z).


def test_bug003_a_reconcile_with_nothing_to_record_takes_no_erasure_lock(
    root: Path, monkeypatch: pytest.MonkeyPatch
):
    """BUG-003: the stated reconcile took the erasure lock before finding it had nothing to
    record, so every no-op reconcile opened a PostgreSQL connection and one waiting behind a
    forget reported the reconcile failed.

    Red proof: against the pre-fix stated_facts (snapshot of run 20261105T095124Z) the second
    reconcile took the lock once and `assert taken == []` failed with `[True] == []`.
    """
    import recall.stated_facts as stated
    from recall.lite import LiteStore
    from recall.lite.facts import LiteFactLedger
    from tests.test_stated_facts import DIM, memo, refreshed

    with LiteStore(root.parent / ".recall" / "memory.db", dim=DIM) as store:
        memo(root / "memory", "door.md", facts=[{"subject": "user", "predicate": "door_code", "object": "4512"}])
        refreshed(store, root, key_root=root)
        taken: list[bool] = []
        real_lock = LiteFactLedger.erasure_lock

        def recording(self, *, shared: bool):
            taken.append(shared)
            return real_lock(self, shared=shared)

        monkeypatch.setattr(LiteFactLedger, "erasure_lock", recording)
        report = stated.reconcile_stated_facts(store, roots=[root])
    assert report.facts_stated == 0 and "already recorded" in report.reason, report
    assert taken == []


def test_code004_an_unknown_verdict_is_declared_in_a_warning(monkeypatch: pytest.MonkeyPatch):
    """CODE-004 (owner, CODE-011: declare an unknown verdict, do not fail): `verdict_known` had no
    caller, so an unknown verdict was read and declared nowhere. Reading such a card now warns,
    naming the value; a known verdict does not.

    Red proof: against the pre-fix controller (snapshot of run 20261105T095124Z) nothing was
    logged and `assert warned == ["OK"]` failed with `[] == ['OK']`.
    """
    import recall.provenance_controller as controller

    warned: list[object] = []

    class Recorder:
        def warning(self, _message: str, *args: object) -> None:
            warned.extend(args)

    monkeypatch.setattr(controller, "_log", Recorder(), raising=False)
    claim = _fact("20")
    for verdict in ("OK", "ok"):
        written = replace(
            evidence_card(claim, tenant="tenant-a", generation="gen-a", chunk="c", source="a.md"),
            card_id="", verdict=verdict,
        )
        evidence_card_from_payload(json.loads(json.dumps(written.to_payload())))
    assert warned == ["OK"]


def test_code005_a_fact_call_for_another_tenant_is_refused_before_any_read(monkeypatch: pytest.MonkeyPatch):
    """CODE-005: under a source policy `recall_apply_fact` with an access context for another
    tenant read the ledger and answered CARD_NOT_FOUND, while `recall_current_facts` raised
    `PermissionError`; one rule now refuses both before anything is read.

    Red proof: against the pre-fix recall_mcp/provenance.py (snapshot of run 20261105T095124Z) the
    call reached the ledger and `assert outcome == "refused"` failed with `'read' == 'refused'`.
    """
    from types import SimpleNamespace

    import recall_mcp.provenance as provenance

    def ledger_read(_store):
        raise RuntimeError("the ledger was read")

    monkeypatch.setattr(provenance, "_fact_backends", ledger_read)
    try:
        provenance.apply_fact_memory(
            SimpleNamespace(tenant="acme"),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            claim={"subject": "user", "predicate": "city", "object": "Turin"},
            evidence_card_ids=["card_" + "0" * 32],
            request_id="r-1",
            writer="test",
            security_policy=object(),  # type: ignore[arg-type]
            access_context=SimpleNamespace(tenant="other"),  # type: ignore[arg-type]
        )
        outcome = "returned"
    except PermissionError:
        outcome = "refused"
    except RuntimeError:
        outcome = "read"
    assert outcome == "refused"


def test_sec002_a_resend_the_caller_may_not_read_returns_no_event_and_records_nothing():
    """SEC-002 (deferred pass): a principal kept from a memo resent another principal's request
    id and got CARD_NOT_FOUND carrying the asserted event's id, which names the tenant, the fact
    and the request id, so it confirmed that the request had applied that fact. It also recorded
    a refusal under the id.

    Red proof: against the pre-fix controller (snapshot of run 20261105T095124Z) the answer
    carried the asserted event and `assert replay.event is None` failed.
    """
    claim = _fact("20")
    card = evidence_card(claim, tenant="tenant-a", generation="gen-a", chunk="c", source="secret.md")
    store = EvidenceCardStore()
    store.put([card])
    ledger = InMemoryFactLedger()

    def controller(permitted):
        return ProvenanceController(
            tenant_id="tenant-a", generation_id="gen-a", cards=store, ledger=ledger,
            source_digest_for=lambda _card: "digest", now=lambda: NOW, writer="test",
            card_permitted=permitted,
        )

    assert controller(lambda c: True).apply_fact(FactApplicationRequest(claim, (card.card_id,), "r-ok")).allowed
    recorded = len(ledger.events)
    replay = controller(lambda c: c.source != "secret.md").apply_fact(
        FactApplicationRequest(claim, (card.card_id,), "r-ok")
    )
    assert replay.event is None
    assert replay.code == DecisionCode.CARD_NOT_FOUND and not replay.allowed and replay.cards == ()
    assert len(ledger.events) == recorded


def test_sec003_a_forget_the_caller_may_not_make_never_takes_the_erasure_lock(
    root: Path, monkeypatch: pytest.MonkeyPatch
):
    """SEC-003: a forget of a memo the caller's policy does not allow took the exclusive erasure
    lock, and so queued every fact application behind it, before it was refused. It is now
    refused before the locks; the control, an allowed forget, still takes it.

    Red proof: against the pre-fix recall_mcp/lifecycle.py (snapshot of run 20261105T095124Z) the
    refused forget took the lock and `assert taken == []` failed with `[1] == []`.
    """
    from contextlib import nullcontext
    from types import SimpleNamespace

    import recall_mcp.lifecycle as lifecycle
    from recall.lite import LiteStore
    from tests.test_stated_facts import DIM

    taken: list[int] = []
    monkeypatch.setattr(lifecycle, "_erasure_lock", lambda _store: taken.append(1) or nullcontext())

    class Policy:
        def __init__(self, allowed: bool) -> None:
            self.allowed = allowed

        def decide(self, _name: str, _context: object) -> SimpleNamespace:
            return SimpleNamespace(allowed=self.allowed, redactions=())

    with LiteStore(root.parent / ".recall" / "memory.db", dim=DIM) as store:
        context = SimpleNamespace(tenant=store.tenant)
        with pytest.raises(lifecycle.ErasureNotAuthorized):
            forget_memory(store, ["a.md"], security_policy=Policy(False), security_context=context)  # type: ignore[arg-type]
        assert taken == []
        forget_memory(store, ["a.md"], security_policy=Policy(True), security_context=context)  # type: ignore[arg-type]
    assert taken == [1]


def test_env004_a_file_with_no_fact_tables_is_not_a_clean_inventory(tmp_path: Path):
    """ENV-004: a SQLite file holding none of the fact tables (the wrong file) printed 0 and
    exited 0, which reads as nothing to re-key. It now exits 2 and names the missing tables; the
    control, an empty but real ledger, is still clean.

    Red proof: against the pre-fix script (snapshot of run 20261105T095124Z) the run exited 0 and
    `assert wrong.returncode == 2` failed with `0 == 2`.
    """
    stranger = tmp_path / "notes.db"
    conn = sqlite3.connect(stranger)
    conn.execute("CREATE TABLE notes (body TEXT)")
    conn.commit()
    conn.close()
    wrong = _check("--sqlite", str(stranger))
    assert wrong.returncode == 2, (wrong.returncode, wrong.stdout, wrong.stderr)
    assert "not a fact ledger" in wrong.stderr
    ledger = tmp_path / "ledger.db"
    conn = sqlite3.connect(ledger)
    conn.execute("CREATE TABLE recall_fact_ledger_events (fact TEXT, evidence_cards TEXT)")
    conn.commit()
    conn.close()
    assert _check("--sqlite", str(ledger)).returncode == 0


def test_sec002_another_fact_under_a_request_id_the_caller_may_not_read_learns_nothing():
    """SEC-002 (architect gate of run 20261105T095124Z): the REQUEST_ID_REUSED answer ran before
    the access check and returned the recorded event, whose id hashes (tenant, fact, request id),
    so a caller kept from the memo could send any fact under a known request id and test guesses
    against the event offline, or by which code came back. It now gets CARD_NOT_FOUND with no
    event either way; a permitted caller still gets REQUEST_ID_REUSED.

    Red proof: with the access check after the REQUEST_ID_REUSED branch (the order before the
    gate) the guess got REQUEST_ID_REUSED and `assert guess.code == DecisionCode.CARD_NOT_FOUND`
    failed.
    """
    claim = _fact("20")
    card = evidence_card(claim, tenant="tenant-a", generation="gen-a", chunk="c", source="secret.md")
    store = EvidenceCardStore()
    store.put([card])
    ledger = InMemoryFactLedger()

    def controller(permitted):
        return ProvenanceController(
            tenant_id="tenant-a", generation_id="gen-a", cards=store, ledger=ledger,
            source_digest_for=lambda _card: "digest", now=lambda: NOW, writer="test",
            card_permitted=permitted,
        )

    assert controller(lambda c: True).apply_fact(FactApplicationRequest(claim, (card.card_id,), "r-ok")).allowed
    guess = controller(lambda c: c.source != "secret.md").apply_fact(
        FactApplicationRequest(_fact("21"), (card.card_id,), "r-ok")
    )
    assert guess.code == DecisionCode.CARD_NOT_FOUND
    assert guess.event is None and guess.cards == ()
    permitted = controller(lambda c: True).apply_fact(FactApplicationRequest(_fact("21"), (card.card_id,), "r-ok"))
    assert permitted.code == DecisionCode.REQUEST_ID_REUSED


def test_env003_the_postgres_check_names_the_tenants_it_read(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """ENV-003 (owner decision 2026-11-05): on PostgreSQL the check reads only the tenants named,
    since row-level security hides the rest, and its clean exit said nothing of the ones it never
    read, so a tenant left off the command line passed the deploy gate unread. It now names the
    tenants it read, once each.

    Red proof: against the pre-fix script (master 03ba03586 for this file) the output held only
    the counts, and the assertion on the "tenants read" line failed.
    """
    check = _normcheck()
    monkeypatch.setattr(check, "_postgres_counts", lambda dsn, tenants: {f"{t}:recall_evidence_cards.card": 0 for t in tenants})
    monkeypatch.setenv("RECALL_FACT_WRITE_DSN", "postgresql://from-env/db")
    assert check.main(["--tenant", "memory", "--tenant", "re-call-docs", "--tenant", "memory"]) == 0
    out = capsys.readouterr().out
    assert "tenants read: memory, re-call-docs; a tenant not named here was not read" in out, out


@requires_db
def test_sec001_the_check_refuses_a_role_that_bypasses_row_level_security(make_store, unprivileged_dsn):
    """2026-11-05 audit SEC-001: the check reads each tenant through row-level security alone,
    and a superuser or BYPASSRLS DSN counted every tenant's rows under each name given while the
    output said a tenant not named was not read. Such a role is now refused with exit 2; the
    control, a role bound by row-level security (`unprivileged_dsn`), is still read.

    Red proof: against the pre-fix script (snapshot of run 20261105T194718Z) the privileged run
    read the tables, exited 0 and `assert run.returncode == 2` failed with `0 == 2`.
    """
    import os

    from tests.conftest import TEST_DSN, role_is_unprivileged

    if role_is_unprivileged(TEST_DSN):
        pytest.skip("RECALL_TEST_DSN's role is bound by row-level security; no privileged role to refuse")
    make_store(8)  # the schema the check reads
    run = _check("--tenant", "memory", env={**os.environ, "RECALL_FACT_WRITE_DSN": TEST_DSN})
    assert run.returncode == 2, (run.returncode, run.stdout, run.stderr)
    assert "bypasses row-level security" in run.stderr
    assert "tenants read" not in run.stdout
    control = _check("--tenant", "memory", env={**os.environ, "RECALL_FACT_WRITE_DSN": unprivileged_dsn})
    assert control.returncode == 0, (control.returncode, control.stdout, control.stderr)
    assert "tenants read: memory" in control.stdout
