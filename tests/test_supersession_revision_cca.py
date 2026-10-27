"""Public PostgreSQL cache work and freshness controls for PR987 FIX-004."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.types.json import Jsonb
import pytest

from recall.db_constants import TENANT_GUC
from recall.schema import serving_grants
from tests.conftest import TEST_DSN, requires_db
from tests.test_supersession_records import Corpus

pytestmark = requires_db


class _ObservedCursor:
    """Observe SQL without replacing cursor results or database behavior."""

    def __init__(self, cursor: Any, record: Callable[[Any], None]) -> None:
        self._cursor, self._record = cursor, record

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._cursor)

    def __next__(self) -> Any:
        return next(self._cursor)

    def __enter__(self) -> _ObservedCursor:
        self._cursor.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        return self._cursor.__exit__(*args)

    def execute(self, query: Any, *args: Any, **kwargs: Any) -> _ObservedCursor:
        self._record(query)
        self._cursor.execute(query, *args, **kwargs)
        return self

    def executemany(self, query: Any, *args: Any, **kwargs: Any) -> Any:
        self._record(query)
        return self._cursor.executemany(query, *args, **kwargs)


class _ObservedConnection:
    """Keep the real connection, including its transaction and tenant settings."""

    def __init__(self, connection: Any, statements: list[str]) -> None:
        self._connection, self._statements = connection, statements

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)

    def _record(self, query: Any) -> None:
        if isinstance(query, bytes):
            text = query.decode()
        elif isinstance(query, str):
            text = query
        else:
            text = query.as_string(self._connection)
        self._statements.append(text)

    def execute(self, query: Any, *args: Any, **kwargs: Any) -> Any:
        self._record(query)
        return self._connection.execute(query, *args, **kwargs)

    def cursor(self, *args: Any, **kwargs: Any) -> _ObservedCursor:
        return _ObservedCursor(self._connection.cursor(*args, **kwargs), self._record)


@pytest.fixture(params=["postgres", "generation"])
def revision_corpus(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Corpus]:
    """Use existing isolated-table and isolated-tenant generation fixtures."""
    corpus = Corpus(request.param, tmp_path)
    try:
        corpus.index({"a.md": {}, "b.md": {"supersedes": "a.md"}})
        corpus.forget("b.md")
        yield corpus
    finally:
        corpus.close()


def test_warm_graph_cache_reads_no_supersession_audit_history(
    revision_corpus: Corpus, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FIX-004: cache-hit SQL work must not inspect the append-only event history.

    Node: tests/test_supersession_revision_cca.py::test_warm_graph_cache_reads_no_supersession_audit_history
    Parameters: postgres and generation.
    Original HEAD: ab1a48ba427599bd174e9d99b5280182f37b2aea.
    Target: public supersession_graph cache-hit path on each PostgreSQL store class.
    Predicted RED: "a warm graph cache read supersession audit history" because the
    original fingerprint executes COUNT(*) over recall_audit_events before its cache hit.
    Both store classes failed the original history-read assertion and passed after repair.

    History is seeded through public record-only withdrawals, not fabricated database rows.
    The observer delegates all SQL and results. No latency or benchmark claim is made;
    querying a bounded revision table is allowed, recounting audit history is not.
    """
    corpus = revision_corpus
    sources = corpus.store.recorded_sources_for_identifiers(["b.md"])["b.md"]
    for _ in range(24):
        assert corpus.store.delete_sources(sources, withdraw=True) == 0
    reader = corpus.second_store()
    try:
        expected = reader.supersession_graph().parts()
        statements: list[str] = []
        retry = reader._with_retry

        def observe(operation: Callable[[Any], Any]) -> Any:
            return retry(lambda connection: operation(_ObservedConnection(connection, statements)))

        monkeypatch.setattr(reader, "_with_retry", observe)
        for _ in range(3):
            assert reader.supersession_graph().parts() == expected

        assert statements, "the observer did not see the cache's real SQL"
        history_reads = [
            statement for statement in statements
            if "recall_audit_events" in statement.casefold()
        ]
        assert not history_reads, f"a warm graph cache read supersession audit history: {history_reads}"
    finally:
        reader.close()


def test_serving_role_revision_is_transactional_replay_safe_and_tenant_confined(
    make_store: Callable[[int], Any],
) -> None:
    """FIX-004: an ordinary serving role can advance only its tenant's committed revision.

    Node: tests/test_supersession_revision_cca.py::test_serving_role_revision_is_transactional_replay_safe_and_tenant_confined
    Entry point: a real audit INSERT under exactly the generated serving grants, on the
    owned testbench database. It exercises the migration0034 trigger, not a private helper.
    The same transaction must see its increment, rollback must remove both writes, commit
    must retain both, and replaying an existing event ID must not increment again.

    Original HEAD ab1a48ba427599bd174e9d99b5280182f37b2aea lacks the revision table;
    that missing-table error is NOT valid RED proof. The verified mutation disables the actual
    AFTER INSERT trigger and fails "audit INSERT did not advance its revision in-transaction".
    Restoring the trigger passes the same test, including rollback, commit, replay, unrelated
    events, generated grants and tenant confinement. Exact RED/GREEN receipts accompany the audit.
    """
    store = make_store(8)
    role = "cca_revision_" + uuid4().hex[:12]
    tenant, other_tenant, forbidden_tenant = ("testbench-" + uuid4().hex for _ in range(3))
    event_ids = ["evt_" + uuid4().hex for _ in range(3)]
    insert = (
        "INSERT INTO recall_audit_events "
        "(tenant_id, event_id, event_type, actor, source_uri, payload) "
        "VALUES (%s, %s, %s, 'testbench', '/testbench/b.md', %s) "
        "ON CONFLICT (tenant_id, event_id) DO NOTHING"
    )
    payload = Jsonb({"scope": store.table, "file": "b.md", "claims": []})

    class RollbackProbe(Exception):
        pass

    with psycopg.connect(TEST_DSN, autocommit=True) as admin:
        may_create = admin.execute(
            "SELECT rolsuper OR rolcreaterole FROM pg_roles WHERE rolname = current_user"
        ).fetchone()
        if not (may_create and may_create[0]):
            pytest.skip("testbench database role cannot CREATE ROLE for the privilege control")
        database = admin.execute("SELECT current_database()").fetchone()[0]
        created = False
        try:
            admin.execute(sql.SQL(
                "CREATE ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOBYPASSRLS"
            ).format(sql.Identifier(role), sql.Literal("testbench-password")))
            created = True
            admin.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                sql.Identifier(database), sql.Identifier(role),
            ))
            admin.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(role)))
            for statement in serving_grants(role, table=store.table, strict=True):
                admin.execute(statement)

            # A real second tenant row makes the visibility assertion non-vacuous.
            admin.execute("SELECT set_config(%s, %s, false)", (TENANT_GUC, other_tenant))
            admin.execute(
                "INSERT INTO recall_supersession_revisions (tenant_id, revision) VALUES (%s, 1)",
                (other_tenant,),
            )
            assert admin.execute(
                "SELECT revision FROM recall_supersession_revisions WHERE tenant_id = %s",
                (other_tenant,),
            ).fetchone() == (1,), "the second tenant's visibility control was not seeded"

            as_role = make_conninfo(TEST_DSN, user=role, password="testbench-password")
            with psycopg.connect(as_role, autocommit=True) as serving:
                serving.execute("SELECT set_config(%s, %s, false)", (TENANT_GUC, tenant))
                assert serving.execute("SELECT current_user").fetchone()[0] == role
                flags = serving.execute(
                    "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
                ).fetchone()
                assert flags == (False, False), "the privilege control bypassed ordinary role checks"

                def revision() -> int:
                    row = serving.execute(
                        "SELECT revision FROM recall_supersession_revisions WHERE tenant_id = %s",
                        (tenant,),
                    ).fetchone()
                    return int(row[0]) if row else 0

                initial = revision()
                with pytest.raises(RollbackProbe), serving.transaction():
                    serving.execute(insert, (tenant, event_ids[0], "supersession_forgotten", payload))
                    assert revision() == initial + 1, (
                        "audit INSERT did not advance its revision in-transaction"
                    )
                    raise RollbackProbe
                assert revision() == initial, "rollback retained a revision increment"
                assert not serving.execute(
                    "SELECT EXISTS (SELECT 1 FROM recall_audit_events "
                    "WHERE tenant_id = %s AND event_id = %s)",
                    (tenant, event_ids[0]),
                ).fetchone()[0], "rollback retained the audit event"

                with serving.transaction():
                    serving.execute(insert, (tenant, event_ids[1], "supersession_withdrawn", payload))
                    assert revision() == initial + 1, "withdrawal INSERT did not advance its revision"
                committed = revision()
                assert committed == initial + 1, "commit lost the revision increment"
                assert serving.execute(
                    "SELECT EXISTS (SELECT 1 FROM recall_audit_events "
                    "WHERE tenant_id = %s AND event_id = %s)",
                    (tenant, event_ids[1]),
                ).fetchone()[0], "commit lost the audit event"

                serving.execute(insert, (tenant, event_ids[1], "supersession_withdrawn", payload))
                assert revision() == committed, "replaying an existing event ID advanced the revision"
                serving.execute(insert, (tenant, event_ids[2], "testbench_unrelated", payload))
                assert revision() == committed, "an unrelated audit event advanced the revision"

                assert serving.execute(
                    "SELECT revision FROM recall_supersession_revisions WHERE tenant_id = %s",
                    (other_tenant,),
                ).fetchall() == [], "the serving role read another tenant's revision"
                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    serving.execute(
                        "INSERT INTO recall_supersession_revisions (tenant_id, revision) VALUES (%s, 1)",
                        (forbidden_tenant,),
                    )
        finally:
            # Every cleanup uses the same owned testbench DSN, even after a failed assertion.
            failures: list[str] = []
            for owned_tenant in (tenant, other_tenant, forbidden_tenant):
                try:
                    admin.execute("SELECT set_config(%s, %s, false)", (TENANT_GUC, owned_tenant))
                    admin.execute("DELETE FROM recall_audit_events WHERE tenant_id = %s", (owned_tenant,))
                    admin.execute(
                        "DELETE FROM recall_supersession_revisions WHERE tenant_id = %s", (owned_tenant,),
                    )
                except psycopg.Error as error:
                    failures.append(type(error).__name__)
            if created:
                for statement in (
                    sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)),
                    sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)),
                ):
                    try:
                        admin.execute(statement)
                    except psycopg.Error as error:
                        failures.append(type(error).__name__)
                leaked = admin.execute(
                    "SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = %s)", (role,),
                ).fetchone()[0]
                assert not leaked, f"testbench privilege control leaked its temporary role: {failures}"


def test_record_only_withdrawal_refreshes_another_handles_warm_graph(
    revision_corpus: Corpus,
) -> None:
    """FIX-004 control: bounded freshness cannot become a permanently cached revision.

    Node: tests/test_supersession_revision_cca.py::test_record_only_withdrawal_refreshes_another_handles_warm_graph
    Parameters: postgres and generation.
    Original HEAD: ab1a48ba427599bd174e9d99b5280182f37b2aea already preserves freshness
    through its history count, so baseline green alone is not regression proof here.
    Verified post-fix mutation: freeze the new revision read at the reader's warmed value.
    Predicted RED: "a record-only withdrawal left another handle's graph stale" after a
    committed withdrawal removes zero chunks and leaves the served corpus unchanged.
    Both mutated store classes failed the stale-graph assertion, then passed after restoration.
    """
    corpus = revision_corpus
    reader = corpus.second_store()
    try:
        assert reader.supersession_graph().successor("a.md") == "b.md"
        before = reader.source_content_hashes()
        sources = corpus.store.recorded_sources_for_identifiers(["b.md"])["b.md"]
        assert corpus.store.delete_sources(sources, withdraw=True) == 0
        assert reader.source_content_hashes() == before, "the control changed served chunks"
        assert reader.supersession_graph().successor("a.md") is None, (
            "a record-only withdrawal left another handle's graph stale"
        )
    finally:
        reader.close()
