"""The materialized current facts on PostgreSQL: the property, crash recovery, rebuild and grants.

The property is `tests/test_fact_current.py`'s, with the materializing ledger on PostgreSQL and
the replaying one a lite file of the same tenant, so the comparison also spans the backends
(their event ids and stamps are equal by design, `tests/test_lite_fact_parity.py`).

Red proof, 2026-10-22, each mutation alone on the remote test host, then restored and green; the
mutation and the failing assertion are named in each test's docstring (nineteen in all, with
`tests/test_fact_current.py`'s).
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from psycopg import sql

from recall.fact_current import CURRENT_TABLE, PostgresCurrentFacts
from recall.fact_ledger import PostgresFactLedger, PostgresMaterializationOutbox
from recall.lite import LiteStore
from recall.lite.facts import LiteFactLedger
from recall.provenance_controller import MaterializationRecovery, project_current
from recall.schema import controller_grants, serving_grants
from tests.conftest import TEST_DSN, requires_db
from tests.fact_fixtures import NOW
from tests.test_fact_current import CASES, DAY, OPS, PARIS, Op, apply, check_equivalence, city

pytestmark = requires_db


def tenant() -> str:
    return f"current-{uuid4().hex}"


@settings(
    max_examples=15, deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(st.lists(OPS, max_size=10))
@example(CASES[0])
@example(CASES[1])
@example(CASES[2])
@example(CASES[3])
@example(CASES[4])
@example(CASES[5])
def test_postgres_tables_answer_as_the_replay_on_any_ledger(ops: list[Op]) -> None:
    """Random ledgers on PostgreSQL: writes decide as on a replaying lite file, and every table
    read equals the replay.

    Red: `PostgresTables.intervals_at` comparing `valid_until >= %s` (closed at the end, the
    window rule's half-open bound lost) also lists Paris at the instant Rome starts, where Paris's
    interval ends, and "current == project_current" fails on the first explicit example and three
    others. The `project_key` own-window mutation fails it the same way.
    """
    name = tenant()
    with tempfile.TemporaryDirectory() as tmp:
        replay_store = LiteStore(Path(tmp) / "replay.db", dim=4, tenant=name)
        try:
            check_equivalence(
                PostgresFactLedger(TEST_DSN, tenant_id=name), LiteFactLedger(replay_store, materialize=False),
                ops, tenant=name, verify=PostgresCurrentFacts(TEST_DSN, tenant_id=name).verify,
            )
        finally:
            replay_store.close()


def test_reconcile_through_the_outbox_files_what_an_older_writer_left() -> None:
    """Events a ledger appended without materializing are filed by `MaterializationRecovery`.

    Crash recovery as the outbox defines it: the events the tables lack get outbox rows and are
    delivered through `PostgresCurrentFacts.materialize`, every type of them, not only assertions.

    Red: `PostgresCurrentFacts.materializes` removed (so `reconcile` enqueues assertions only, its
    default) leaves the `superseded` event out of the outbox: `reconcile` delivers 2, not 3, and
    that assertion fails.
    """
    name = tenant()
    old = PostgresFactLedger(TEST_DSN, tenant_id=name, materialize=False)
    apply(old, PARIS, "r1", tenant=name)
    apply(old, city("rome", NOW + 30 * DAY), "r2", supersedes=(PARIS,), tenant=name)
    facts = PostgresCurrentFacts(TEST_DSN, tenant_id=name)
    assert not facts.status().current
    outbox = PostgresMaterializationOutbox(TEST_DSN, tenant_id=name)
    recovery = MaterializationRecovery(tenant_id=name, outbox=outbox, materializer=facts)
    assert recovery.reconcile(facts.unfiled_events(), now=NOW) == 3
    status = facts.status()
    assert status.current, status
    assert {outbox.status(event.event_id) for event in old.events} == {"applied"}
    ledger = PostgresFactLedger(TEST_DSN, tenant_id=name)
    for at in (NOW, NOW + 40 * DAY):
        assert ledger.current(tenant_id=name, now=at) == project_current(ledger.events, at)
        assert ledger.last_read == "table"


def test_verify_on_postgres_reads_the_bounds_back_and_finds_an_edited_interval() -> None:
    """`verify` on PostgreSQL: a clean table matches, open bounds included; a moved end does not.

    Red: `PostgresTables.stored` returning the bounds without their zone (`.replace(tzinfo=UTC)`
    dropped on both) compares naive with aware datetimes, so the clean table reports every
    interval missing and the first `matches` assertion fails.
    """
    name = tenant()
    ledger = PostgresFactLedger(TEST_DSN, tenant_id=name)
    apply(ledger, PARIS, "r1", tenant=name)
    apply(ledger, city("oslo", NOW - 400 * DAY, NOW - 30 * DAY), "r2", supersedes=(PARIS,), tenant=name)
    facts = PostgresCurrentFacts(TEST_DSN, tenant_id=name)
    clean = facts.verify()
    assert clean.matches, clean.to_payload()
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (name,))
        conn.execute(
            f"UPDATE {CURRENT_TABLE} SET valid_until = valid_until + interval '1 day' "
            "WHERE tenant_id = %s AND valid_until < '9000-01-01'",
            (name,),
        )
    assert facts.status().current  # the counts cannot see it
    edited = facts.verify()
    assert not edited.matches and edited.intervals_missing and edited.intervals_extra


def test_purge_sends_reads_to_replay_and_rebuild_restores_the_tables() -> None:
    """The erasure hook: a purge leaves no row for the tenant, and reads replay until a rebuild.

    Red: `PostgresCurrentFacts.purge_in` not clearing `recall_fact_current_events` (mutation:
    `tables.clear()` to deleting `CURRENT_TABLE` alone) keeps the counts equal, so the read after
    the purge answers from the emptied table and the value assertion fails.
    """
    name = tenant()
    ledger = PostgresFactLedger(TEST_DSN, tenant_id=name)
    apply(ledger, PARIS, "r1", tenant=name)
    facts = PostgresCurrentFacts(TEST_DSN, tenant_id=name)
    facts.purge()
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (name,))
        assert conn.execute(f"SELECT count(*) FROM {CURRENT_TABLE} WHERE tenant_id = %s", (name,)).fetchone() == (0,)
    assert [e.fact.object for e in ledger.current(tenant_id=name, now=NOW) if e.fact] == ["paris"]
    assert ledger.last_read == "replay"
    assert facts.rebuild().current
    assert ledger.current(tenant_id=name, now=NOW) == project_current(ledger.events, NOW)
    assert ledger.last_read == "table"


@contextmanager
def role(grants: Callable[[str], Sequence[str]]) -> Iterator[str]:
    """A login role given exactly `grants(its name)`; yields its DSN."""
    with psycopg.connect(TEST_DSN, autocommit=True) as probe:
        allowed = probe.execute(
            "SELECT rolsuper OR rolcreaterole FROM pg_roles WHERE rolname = current_user"
        ).fetchone()
    if not allowed or not allowed[0]:
        pytest.skip("RECALL_TEST_DSN's role cannot create the role a privilege test needs")
    name, password = f"fact_current_{uuid4().hex[:12]}", uuid4().hex
    try:
        with psycopg.connect(TEST_DSN, autocommit=True) as admin:
            admin.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(sql.Identifier(name), sql.Literal(password)))
            for statement in grants(name):
                admin.execute(statement)
        yield TEST_DSN.replace("recall:recall@", f"{name}:{password}@")
    finally:
        with psycopg.connect(TEST_DSN, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(name)))
            admin.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(name)))


def test_a_strict_serving_role_reads_the_tables_and_cannot_write_them() -> None:
    """`serving_grants(strict=True)`: the tables answer its reads; it can write none of them.

    Red: `serving_grants` without its `GRANT SELECT ON ... FACT_CURRENT_TABLES` line makes the
    read degrade (InsufficientPrivilege, replayed), and `last_read` reads "replay".
    """
    name = tenant()
    apply(PostgresFactLedger(TEST_DSN, tenant_id=name), PARIS, "r1", tenant=name)
    with role(lambda grantee: serving_grants(grantee, strict=True)) as dsn:
        reader = PostgresFactLedger(dsn, tenant_id=name)
        assert [e.fact.object for e in reader.current(tenant_id=name, now=NOW) if e.fact] == ["paris"]
        assert reader.last_read == "table"
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (name,))
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute(f"DELETE FROM {CURRENT_TABLE} WHERE tenant_id = %s", (name,))


def test_the_controller_role_maintains_the_tables() -> None:
    """`controller_grants`: a ledger appending as the controller role keeps the tables current.

    Red: `controller_grants` without its DML grant on `FACT_CURRENT_TABLES` makes the in-append
    materialization fail (rolled back to its savepoint, the append kept), so the status is not
    current and the assertion fails.
    """
    name = tenant()
    with role(controller_grants) as dsn:
        ledger = PostgresFactLedger(dsn, tenant_id=name)
        apply(ledger, PARIS, "r1", tenant=name)
        apply(ledger, city("rome", NOW + 30 * DAY), "r2", supersedes=(PARIS,), tenant=name)
        assert PostgresCurrentFacts(TEST_DSN, tenant_id=name).status().current
        assert ledger.current(tenant_id=name, now=NOW + 40 * DAY) == project_current(ledger.events, NOW + 40 * DAY)
        assert ledger.last_read == "table"
