"""PostgreSQL integration coverage for the provenance controller boundary."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import pytest
import psycopg
from psycopg import sql

from recall.fact_ledger import PostgresFactLedger, PostgresMaterializationOutbox
from recall.provenance_cards import PostgresEvidenceCardStore
from recall.provenance_controller import (
    DecisionCode,
    FactApplicationRequest,
    MaterializationRecovery,
    ProvenanceController,
)
from recall.types import AtomicFact, EvidenceCard
from recall.schema import serving_grants
from recall.wizard.headless import HeadlessConfig, _RealServices
from tests.conftest import TEST_DSN, requires_db


pytestmark = requires_db
NOW = datetime(2026, 9, 4, 12, tzinfo=UTC)


def _fixture():
    tenant = f"provenance-{uuid4().hex}"
    generation = f"generation-{uuid4().hex}"
    claim = AtomicFact(
        namespace="memory",
        subject="service:api",
        predicate="owner",
        object="team:platform",
    )
    card = EvidenceCard(
        card_id="",
        chunk_id=f"chunk-{uuid4().hex}",
        source="owner.md",
        source_digest="digest",
        valid_from=None,
        valid_until=None,
        first_indexed_at=NOW - timedelta(days=1),
        indexed_at=NOW,
        tenant_id=tenant,
        generation_id=generation,
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
        calibration_id="calibration",
        calibration_status="certified",
        trust_state="trusted",
        verdict="ok",
        confidence=0.99,
        rank=1,
        structured_facts=(claim,),
    )
    return tenant, generation, claim, card


def _controller(tenant, generation, claim, card, request_id):
    cards = PostgresEvidenceCardStore(TEST_DSN, tenant_id=tenant)
    cards.put((card,))
    return ProvenanceController(
        tenant_id=tenant,
        generation_id=generation,
        cards=cards,
        ledger=PostgresFactLedger(TEST_DSN, tenant_id=tenant),
        source_digest_for=lambda _card: "digest",
        now=lambda: NOW,
        writer="postgres-test",
    ), FactApplicationRequest(claim, (card.card_id,), request_id)


def test_postgres_cards_and_ledger_are_tenant_scoped_and_direct_assertion_is_refused():
    tenant, generation, claim, card = _fixture()
    controller, request = _controller(tenant, generation, claim, card, f"request-{uuid4().hex}")

    decision = controller.apply_fact(request)

    assert decision.allowed and decision.code == DecisionCode.APPLIED
    with pytest.raises(PermissionError, match="controller permit"):
        PostgresFactLedger(TEST_DSN, tenant_id=tenant).apply_assertion(
            tenant_id=tenant,
            generation_id=generation,
            fact=claim,
            cards=(card,),
            request_id=f"direct-{uuid4().hex}",
            writer="bypass",
            now=NOW,
        )

    other_tenant = PostgresEvidenceCardStore(TEST_DSN, tenant_id=f"other-{uuid4().hex}")
    assert other_tenant.resolve(card.card_id) is None


def test_postgres_conflicting_applications_serialize_at_the_ledger():
    tenant, generation, claim, card = _fixture()
    cards = PostgresEvidenceCardStore(TEST_DSN, tenant_id=tenant)
    cards.put((card,))
    conflicting_claim = AtomicFact(
        namespace=claim.namespace,
        subject=claim.subject,
        predicate=claim.predicate,
        object="team:security",
    )
    second_card = EvidenceCard(
        **{
            **card.__dict__,
            "card_id": "",
            "chunk_id": f"chunk-{uuid4().hex}",
            "structured_facts": (conflicting_claim,),
        }
    )
    cards.put((second_card,))

    def apply(fact, evidence, request_id):
        return ProvenanceController(
            tenant_id=tenant,
            generation_id=generation,
            cards=cards,
            ledger=PostgresFactLedger(TEST_DSN, tenant_id=tenant),
            source_digest_for=lambda _card: "digest",
            now=lambda: NOW,
            writer="postgres-concurrency-test",
        ).apply_fact(FactApplicationRequest(fact, (evidence.card_id,), request_id))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda args: apply(*args),
                (
                    (claim, card, f"request-{uuid4().hex}"),
                    (conflicting_claim, second_card, f"request-{uuid4().hex}"),
                ),
            )
        )

    assert sorted(result.allowed for result in results) == [False, True]
    assert any(
        result.code == DecisionCode.CONTRADICTION_WITHOUT_SUPERSESSION for result in results
    )


def test_postgres_materialization_outbox_recovers_after_downstream_failure():
    tenant, generation, claim, card = _fixture()
    cards = PostgresEvidenceCardStore(TEST_DSN, tenant_id=tenant)
    cards.put((card,))
    ledger = PostgresFactLedger(TEST_DSN, tenant_id=tenant)
    outbox = PostgresMaterializationOutbox(TEST_DSN, tenant_id=tenant)

    class Materializer:
        calls = 0

        def materialize(self, _event):
            self.calls += 1
            if self.calls == 1:
                raise OSError("downstream unavailable")

    materializer = Materializer()
    request = FactApplicationRequest(claim, (card.card_id,), f"request-{uuid4().hex}")
    first = ProvenanceController(
        tenant_id=tenant,
        generation_id=generation,
        cards=cards,
        ledger=ledger,
        source_digest_for=lambda _card: "digest",
        materializer=materializer,
        materialization_outbox=outbox,
        now=lambda: NOW,
        writer="postgres-outbox-test",
    ).apply_fact(request)

    assert first.code == DecisionCode.MATERIALIZATION_UNAVAILABLE
    assert first.event is not None
    assert outbox.status(first.event.event_id) == "failed"

    delivered = MaterializationRecovery(
        tenant_id=tenant,
        outbox=outbox,
        materializer=materializer,
    ).run_once(now=NOW)
    assert delivered == 1
    assert outbox.status(first.event.event_id) == "applied"
    assert materializer.calls == 2


def test_postgres_materialization_reconcile_recovers_after_enqueue_gap():
    tenant, generation, claim, card = _fixture()
    cards = PostgresEvidenceCardStore(TEST_DSN, tenant_id=tenant)
    cards.put((card,))
    ledger = PostgresFactLedger(TEST_DSN, tenant_id=tenant)
    applied = ProvenanceController(
        tenant_id=tenant,
        generation_id=generation,
        cards=cards,
        ledger=ledger,
        source_digest_for=lambda _card: "digest",
        now=lambda: NOW,
        writer="postgres-reconcile-test",
    ).apply_fact(FactApplicationRequest(claim, (card.card_id,), f"request-{uuid4().hex}"))
    assert applied.allowed and applied.event is not None

    outbox = PostgresMaterializationOutbox(TEST_DSN, tenant_id=tenant)
    delivered = MaterializationRecovery(
        tenant_id=tenant,
        outbox=outbox,
        materializer=type("Working", (), {"materialize": lambda _self, _event: None})(),
    ).reconcile(ledger.events, now=NOW)
    assert delivered == 1
    assert outbox.status(applied.event.event_id) == "applied"


def test_postgres_atomic_outbox_failure_rolls_back_assertion():
    tenant, generation, claim, card = _fixture()
    cards = PostgresEvidenceCardStore(TEST_DSN, tenant_id=tenant)
    cards.put((card,))

    class BrokenOutbox:
        def _enqueue_in_transaction(self, _conn, _event):
            raise OSError("outbox unavailable")

    request = FactApplicationRequest(claim, (card.card_id,), f"request-{uuid4().hex}")
    decision = ProvenanceController(
        tenant_id=tenant,
        generation_id=generation,
        cards=cards,
        ledger=PostgresFactLedger(TEST_DSN, tenant_id=tenant),
        source_digest_for=lambda _card: "digest",
        materializer=type("Unused", (), {"materialize": lambda _self, _event: None})(),
        materialization_outbox=BrokenOutbox(),
        now=lambda: NOW,
        writer="postgres-atomic-rollback-test",
    ).apply_fact(request)

    assert not decision.allowed
    assert decision.code == DecisionCode.LEDGER_UNAVAILABLE
    events = PostgresFactLedger(TEST_DSN, tenant_id=tenant).events
    assert not [event for event in events if event.event_type == "asserted"]


def test_postgres_serving_role_is_tenant_scoped_and_cannot_mutate_snapshots():
    tenant, generation, claim, card = _fixture()
    cards = PostgresEvidenceCardStore(TEST_DSN, tenant_id=tenant)
    cards.put((card,))
    applied = ProvenanceController(
        tenant_id=tenant,
        generation_id=generation,
        cards=cards,
        ledger=PostgresFactLedger(TEST_DSN, tenant_id=tenant),
        source_digest_for=lambda _card: "digest",
        now=lambda: NOW,
        writer="postgres-permission-test",
    ).apply_fact(FactApplicationRequest(claim, (card.card_id,), f"request-{uuid4().hex}"))
    assert applied.event is not None

    role = f"prov_serving_{uuid4().hex}"
    password = uuid4().hex
    role_dsn = TEST_DSN.replace("recall:recall@", f"{role}:{password}@")
    try:
        with psycopg.connect(TEST_DSN, autocommit=True) as admin:
            admin.execute(
                sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                    sql.Identifier(role), sql.Literal(password)
                )
            )
            for statement in serving_grants(role, strict=True):
                admin.execute(statement)

        with psycopg.connect(role_dsn, autocommit=True) as serving:
            serving.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
            assert serving.execute(
                "SELECT count(*) FROM recall_evidence_cards WHERE card_id = %s", (card.card_id,)
            ).fetchone()[0] == 1
            assert serving.execute(
                "SELECT count(*) FROM recall_fact_ledger_events WHERE event_id = %s",
                (applied.event.event_id,),
            ).fetchone()[0] == 1
            serving.execute("SELECT set_config('recall.tenant_id', %s, false)", (f"other-{tenant}",))
            assert serving.execute(
                "SELECT count(*) FROM recall_evidence_cards WHERE card_id = %s", (card.card_id,)
            ).fetchone()[0] == 0
            serving.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                serving.execute(
                    "UPDATE recall_evidence_cards SET source_digest = source_digest WHERE card_id = %s",
                    (card.card_id,),
                )
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                serving.execute(
                    "DELETE FROM recall_fact_ledger_events WHERE event_id = %s",
                    (applied.event.event_id,),
                )
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                serving.execute(
                    "INSERT INTO recall_fact_ledger_events "
                    "(event_id, tenant_id, generation_id, event_type, fact, request_id, writer, "
                    "decision_code, policy_version, controller_version) "
                    "VALUES (%s, %s, %s, 'asserted', '{}'::jsonb, %s, 'bypass', 'APPLIED', 'forged', 1)",
                    (f"forged-{uuid4().hex}", tenant, generation, f"forged-{uuid4().hex}"),
                )
    finally:
        with psycopg.connect(TEST_DSN, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP OWNED BY {} ").format(sql.Identifier(role)))
            admin.execute(sql.SQL("DROP ROLE IF EXISTS {} ").format(sql.Identifier(role)))


def test_wizard_configures_the_strict_fact_boundary_for_distinct_roles():
    serving_role = f"prov_wizard_serving_{uuid4().hex}"
    controller_role = f"prov_wizard_controller_{uuid4().hex}"
    serving_password = uuid4().hex
    controller_password = "controller@password:42"
    controller_dsn = TEST_DSN.replace(
        "recall:recall@", f"{controller_role}:{quote(controller_password, safe='')}@"
    )
    try:
        with psycopg.connect(TEST_DSN, autocommit=True) as admin:
            admin.execute(
                sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                    sql.Identifier(serving_role), sql.Literal(serving_password)
                )
            )
        config = HeadlessConfig(
            embedder="hashing",
            corpus_version="2026-09-05",
            docs_root=Path("."),
            code_root=Path("."),
            memory_root=Path("."),
            dsn=TEST_DSN,
        )
        _RealServices(config).configure_fact_boundary(
            TEST_DSN,
            serving_role=serving_role,
            controller_dsn=controller_dsn,
        )

        serving_dsn = TEST_DSN.replace("recall:recall@", f"{serving_role}:{serving_password}@")
        with psycopg.connect(serving_dsn, autocommit=True) as serving:
            assert serving.execute(
                "SELECT has_table_privilege(current_user, %s, 'SELECT')",
                ("recall_fact_ledger_events",),
            ).fetchone()[0]
            assert not serving.execute(
                "SELECT has_table_privilege(current_user, %s, 'INSERT')",
                ("recall_fact_ledger_events",),
            ).fetchone()[0]
        with psycopg.connect(controller_dsn, autocommit=True) as controller:
            assert not controller.execute(
                "SELECT has_table_privilege(current_user, %s, 'INSERT')",
                ("recall_fact_ledger_events",),
            ).fetchone()[0]
            assert controller.execute(
                "SELECT has_function_privilege(current_user, %s::regprocedure, 'EXECUTE')",
                ("recall_append_fact_ledger_event(text,text,text,text,text,jsonb,jsonb,jsonb,text,text,text,text,integer,timestamptz)",),
            ).fetchone()[0]
            assert controller.execute(
                "SELECT has_function_privilege(current_user, %s::regprocedure, 'EXECUTE')",
                ("recall_append_fact_materialization(text,text,jsonb)",),
            ).fetchone()[0]
            controller.execute("SELECT set_config('recall.tenant_id', %s, false)", ("wizard-live",))
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                controller.execute(
                    "INSERT INTO recall_fact_ledger_events "
                    "(event_id, tenant_id, generation_id, event_type, fact, request_id, writer, "
                    "decision_code, policy_version, controller_version) "
                    "VALUES (%s, 'wizard-live', 'generation', 'asserted', '{}'::jsonb, %s, "
                    "'bypass', 'APPLIED', 'forged', 1)",
                    (f"controller-forged-{uuid4().hex}", f"controller-forged-{uuid4().hex}"),
                )
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                controller.execute(
                    "INSERT INTO recall_fact_materialization_outbox "
                    "(event_id, tenant_id, event, status) VALUES (%s, 'wizard-live', '{}'::jsonb, 'pending')",
                    (f"outbox-forged-{uuid4().hex}",),
                )

        fact = AtomicFact(
            namespace="memory",
            subject=f"service:{uuid4().hex}",
            predicate="owner",
            object="team:platform",
        )
        controller_ledger = PostgresFactLedger(controller_dsn, tenant_id="wizard-live")
        controller_outbox = PostgresMaterializationOutbox(controller_dsn, tenant_id="wizard-live")
        card = EvidenceCard(
            card_id="",
            chunk_id=f"wizard-chunk-{uuid4().hex}",
            source="wizard.md",
            source_digest="d" * 64,
            valid_from=None,
            valid_until=None,
            first_indexed_at=NOW - timedelta(days=1),
            indexed_at=NOW,
            tenant_id="wizard-live",
            generation_id="generation",
            pipeline_fingerprint="p" * 64,
            corpus_fingerprint="c" * 64,
            calibration_id="calibration",
            calibration_status="certified",
            trust_state="trusted",
            verdict="ok",
            confidence=0.99,
            rank=1,
            structured_facts=(fact,),
        )
        card_store = PostgresEvidenceCardStore(TEST_DSN, tenant_id="wizard-live")
        card_store.put((card,))
        controller = ProvenanceController(
            tenant_id="wizard-live",
            generation_id="generation",
            cards=card_store,
            ledger=controller_ledger,
            now=lambda: NOW,
            writer="controller-test",
        )
        decision = controller.apply_fact(
            FactApplicationRequest(
                claim=fact,
                evidence_card_ids=(card.card_id,),
                request_id=f"controller-apply-{uuid4().hex}",
            )
        )
        assert decision.allowed and decision.event is not None
        assert controller_ledger.current(tenant_id="wizard-live", now=NOW)
        controller_outbox.enqueue(decision.event)
        assert controller_outbox.status(decision.event.event_id) == "pending"
    finally:
        with psycopg.connect(TEST_DSN, autocommit=True) as admin:
            for role in (serving_role, controller_role):
                if admin.execute(
                    "SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)
                ).fetchone():
                    admin.execute(sql.SQL("DROP OWNED BY {} ").format(sql.Identifier(role)))
                    admin.execute(sql.SQL("DROP ROLE IF EXISTS {} ").format(sql.Identifier(role)))
