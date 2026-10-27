"""Regression proof observed 2026-10-26 on the isolated testbench.
Baseline: 51c400cc96fef80c5354c632edb6fd0ae44a2c97. The unchanged cases listed
below reached their intended assertion failures; setup errors and skips were not RED.
All 17 cases passed after repair and again after hash-verified mutation restoration.
No timing result is asserted.
test_fact_rebuild_writer_cca_981.py::test_mutating_cli_uses_writer_and_keeps_checks_on_reader[rebuild]: baseline assertion RED, repaired GREEN.
test_fact_rebuild_writer_cca_981.py::test_mutating_cli_uses_writer_and_keeps_checks_on_reader[reconcile]: baseline assertion RED, repaired GREEN.
Control mutation: provenance_cmd._rebuild_postgres: writer DSN also selected for check.
Both controls failed their intended assertion and passed after restoration."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from recall.cli import main
from recall.fact_ledger import PostgresFactLedger
from recall.provenance_cards import PostgresEvidenceCardStore
from recall.provenance_controller import DecisionCode, FactApplicationRequest, ProvenanceController
from recall.schema import controller_grants, serving_grants
from recall.types import AtomicFact
from tests.conftest import TEST_DSN, requires_db
from tests.fact_fixtures import NOW, evidence_card

pytestmark = requires_db


@dataclass(frozen=True)
class DatabaseRole:
    name: str
    dsn: str


@dataclass(frozen=True)
class WriterCase:
    tenant: str
    reader: DatabaseRole
    writer: DatabaseRole


@contextmanager
def database_role(grants: Callable[[str], Sequence[str]]) -> Iterator[DatabaseRole]:
    name, password = f"rebuild_role_{uuid4().hex[:12]}", uuid4().hex
    with psycopg.connect(TEST_DSN, autocommit=True) as admin:
        admin.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE").format(
                sql.Identifier(name), sql.Literal(password),
            )
        )
    try:
        with psycopg.connect(TEST_DSN, autocommit=True) as admin:
            for statement in grants(name):
                admin.execute(statement)
        yield DatabaseRole(name, make_conninfo(TEST_DSN, user=name, password=password))
    finally:
        with psycopg.connect(TEST_DSN, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(name)))
            admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(name)))


@pytest.fixture
def writer_case() -> Iterator[WriterCase]:
    with psycopg.connect(TEST_DSN, autocommit=True) as admin:
        allowed = admin.execute(
            "SELECT rolsuper OR rolcreaterole FROM pg_roles WHERE rolname = current_user"
        ).fetchone()
    if not allowed or not allowed[0]:
        pytest.skip("the test database role cannot create the roles this permission regression needs")

    with (
        database_role(lambda name: serving_grants(name, strict=True)) as reader,
        database_role(controller_grants) as writer,
    ):
        tenant = f"rebuild-writer-{uuid4().hex}"
        generation = "generation-rebuild-writer"
        fact = AtomicFact(namespace="memory", subject="service:api", predicate="owner", object="team:platform")
        card = evidence_card(fact, tenant=tenant, generation=generation, chunk="chunk-owner", source="owner.md")
        cards = PostgresEvidenceCardStore(TEST_DSN, tenant_id=tenant)
        cards.put((card,))
        decision = ProvenanceController(
            tenant_id=tenant, generation_id=generation, cards=cards,
            ledger=PostgresFactLedger(TEST_DSN, tenant_id=tenant, materialize=False),
            source_digest_for=lambda _card: "digest", now=lambda: NOW, writer="writer-route-test",
        ).apply_fact(FactApplicationRequest(fact, (card.card_id,), "seed"))
        assert decision.code == DecisionCode.APPLIED and decision.event is not None
        yield WriterCase(tenant, reader, writer)


def run_cli(case: WriterCase, flags: list[str], capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    try:
        main([
            "--serving-dsn", case.reader.dsn, "--migration-dsn", TEST_DSN, "--tenant", case.tenant,
            "provenance", "rebuild", *flags,
        ])
    except SystemExit as exc:
        # A check reports a real mismatch with exit 1; that is the seeded state.
        if flags != ["--check", "--verify"] or exc.code != 1:
            raise
    return json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("action", ("rebuild", "reconcile"))
def test_mutating_cli_uses_writer_and_keeps_checks_on_reader(
    writer_case: WriterCase, action: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Invariant: configured writer credentials repair rows while check/verify uses the reader.

    Historical baseline: 51c400cc96fef80c5354c632edb6fd0ae44a2c97. Target symbol:
    recall.cli_commands.provenance_cmd._rebuild_postgres. Both parameters must fail at the
    explicit 'mutation ignored configured fact writer DSN' assertion, after the serving role
    raises InsufficientPrivilege during the mutation. Only that mutation error is translated
    into the expected assertion; fixture and connection failures propagate normally.
    Execution and assertion red proof: observed in the module proof receipt.

    Read-routing mutation plan: resolve the writer DSN for action='check' as well as writes.
    The delegated connection observation must then fail 'check/verify used the writer connection'.
    This additional mutation proof is also observed in the module proof receipt; no connection or SQL result is simulated.
    """
    case = writer_case
    monkeypatch.setenv("RECALL_FACT_WRITE_DSN", case.writer.dsn)
    monkeypatch.setenv("RECALL_ALLOW_INSECURE_DSN", "1")
    monkeypatch.setenv("RECALL_IGNORE_BROKEN_DOTENV", "1")
    principals: list[str] = []
    connect = psycopg.connect

    def observed_connect(*args: Any, **kwargs: Any) -> psycopg.Connection[Any]:
        connection = connect(*args, **kwargs)
        principals.append(connection.info.user)
        return connection

    monkeypatch.setattr(psycopg, "connect", observed_connect)
    before = run_cli(case, ["--check", "--verify"], capsys)
    assert principals and set(principals) == {case.reader.name}, "check/verify used the writer connection"
    assert before["ledger_events"] == 1 and before["filed_events"] == 0
    assert before["current"] is False and before["verify"]["matches"] is False

    principals.clear()
    try:
        repaired = run_cli(case, ["--reconcile"] if action == "reconcile" else [], capsys)
    except psycopg.errors.InsufficientPrivilege:
        pytest.fail(f"{action} mutation ignored configured fact writer DSN", pytrace=False)
    assert case.writer.name in principals, "mutation did not open the configured writer connection"
    assert repaired["current"] is True and repaired["filed_events"] == 1

    principals.clear()
    after = run_cli(case, ["--check", "--verify"], capsys)
    assert principals and set(principals) == {case.reader.name}, "check/verify used the writer connection"
    assert after["current"] is True and after["verify"]["matches"] is True
    assert after["ledger_events"] == before["ledger_events"], "repair appended an unrelated fact event"
