"""Surviving regressions from PRs 580 through 589.

Red proof: run this file on baseline 9807af5f9 before the corresponding fixes.
Each test names its invariant and fails a behavioral assertion on that baseline.
The same nodes must pass after repair; setup errors and skips are not proof.
"""
# Imported pytest fixtures are intentionally shadowed by their injected values.
# ruff: noqa: F811
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4
import json

import psycopg
import pytest

from recall import index
from recall.embeddings import HashingEmbedder
from recall.fact_ledger import (
    InMemoryMaterializationOutbox, SQLiteMaterializationOutbox,
    PostgresFactLedger, PostgresMaterializationOutbox,
)
from recall.generation_store import GenerationStore
from recall.lite import LiteStore
from recall.provenance_cards import PostgresEvidenceCardStore, SQLiteEvidenceCardStore
from recall.provenance_controller import (
    DecisionCode, EvidenceCardStore, FactApplicationRequest, InMemoryFactLedger,
    ProvenanceController, fact_identity,
)
from recall_hooks import screening
from tests.conftest import TEST_DSN, requires_db
from tests.test_provenance_controller import NOW, card, controller, fact
from tests.test_provenance_postgres import _fixture
from tests.test_generations import manager, _manifest, _pipeline, _reader, _Embedder  # noqa: F401
from tests.test_calibration_v2 import (
    calibration_tenant, _CalibrationEmbedder, _ready, _labels, _production,  # noqa: F401
    CalibrationRepository,
)


def test_google_trailing_hyphen_is_withheld(tmp_path):
    """SEC-001: screen must withhold credentials ending in a permitted hyphen."""
    key = "AI" + "za" + "B7kQ2d3F9m" * 3 + "Q7z4-"
    memo = tmp_path / "private.md"
    memo.write_text(f'key = "{key}"\n', encoding="utf-8")
    allowed, withheld = screening.screen([SimpleNamespace(name=memo.name, path=memo)])
    assert not allowed, "credential-bearing memo must not reach hosted upload candidates"
    assert list(withheld) == [memo.name]
    assert key not in screening.summarise(withheld)


@pytest.mark.parametrize("refused", [False, True])
def test_shared_request_id_is_tenant_scoped(refused):
    """BUG-003: a second tenant must never receive the first tenant's receipt."""
    ledger = InMemoryFactLedger()
    decisions = []
    for tenant in ("tenant-a", "tenant-b"):
        claim = fact("team:" + tenant)
        evidence = replace(card(claim), card_id="", tenant_id=tenant)
        cards = EvidenceCardStore()
        cards.put((evidence,))
        ctl = ProvenanceController(
            tenant_id=tenant, generation_id="gen-a", cards=cards, ledger=ledger,
            source_digest_for=lambda _card: "digest", now=lambda: NOW, writer="test",
        )
        ids = ("missing-card",) if refused else (evidence.card_id,)
        request = FactApplicationRequest(claim, ids, "shared-request")
        decision = ctl.apply_fact(request)
        assert decision.event is not None
        assert decision.event.tenant_id == tenant, "request receipts must belong to the requesting tenant"
        assert decision.allowed is (not refused)
        assert ctl.apply_fact(request).event.tenant_id == tenant
        decisions.append(decision)
    assert decisions[0].event.event_id != decisions[1].event.event_id


@pytest.mark.parametrize("durable", [False, True])
def test_processing_lease_is_not_materialization_success(tmp_path, durable):
    """BUG-001: only an applied row can turn an empty claim into replay success."""
    claim = fact("team:platform")
    evidence = card(claim)
    ledger = InMemoryFactLedger()
    ctl = controller((evidence,), ledger)
    request = FactApplicationRequest(claim, (evidence.card_id,), "held-lease")
    event = ctl.apply_fact(request).event
    path = str(tmp_path / "outbox.db")
    queue = SQLiteMaterializationOutbox(path, tenant_id="tenant-a") if durable else InMemoryMaterializationOutbox()
    worker = None
    try:
        queue.enqueue(event)
        worker = SQLiteMaterializationOutbox(path, tenant_id="tenant-a") if durable else queue
        claimed = worker.claim(tenant_id="tenant-a", now=NOW)[0]
        ctl.materializer = SimpleNamespace(materialize=lambda _event: pytest.fail("held lease was delivered twice"))
        ctl.materialization_outbox = queue
        replay = ctl.apply_fact(request)
        assert not replay.allowed, "a processing lease is not completed materialization"
        assert replay.code == DecisionCode.MATERIALIZATION_UNAVAILABLE
        worker.mark_applied(tenant_id="tenant-a", event_id=event.event_id, lease_token=claimed.lease_token)
        assert ctl.apply_fact(request).allowed, "another worker's completed delivery must be read authoritatively"
    finally:
        if durable:
            if worker is not None:
                worker.close()
            queue.close()


@pytest.mark.parametrize("durable", [False, True])
def test_status_read_failure_preserves_another_workers_lease(tmp_path, monkeypatch, durable):
    """BUG-001 repair proof: a failed status read must not release an unowned lease.

    Red baseline is the first empty-claim status check, whose outer error cleanup
    called mark_failed with no ownership token after the injected read failure.
    """
    claim = fact("team:platform")
    evidence = card(claim)
    ledger = InMemoryFactLedger()
    ctl = controller((evidence,), ledger)
    request = FactApplicationRequest(claim, (evidence.card_id,), "held-lease-read-error")
    event = ctl.apply_fact(request).event
    path = str(tmp_path / "outbox.db")
    queue = SQLiteMaterializationOutbox(path, tenant_id="tenant-a") if durable else InMemoryMaterializationOutbox()
    worker = None

    def unavailable(_event_id):
        raise RuntimeError("status read unavailable")

    try:
        queue.enqueue(event)
        worker = SQLiteMaterializationOutbox(path, tenant_id="tenant-a") if durable else queue
        worker.claim(tenant_id="tenant-a", now=NOW)
        ctl.materializer = SimpleNamespace(materialize=lambda _event: pytest.fail("held lease was delivered twice"))
        ctl.materialization_outbox = queue
        with monkeypatch.context() as patch:
            patch.setattr(queue, "status", unavailable)
            replay = ctl.apply_fact(request)
        assert not replay.allowed and replay.code == DecisionCode.MATERIALIZATION_UNAVAILABLE
        assert worker.status(event.event_id) == "processing", "a status read failure must not release another worker's lease"
    finally:
        if durable:
            if worker is not None:
                worker.close()
            queue.close()


def test_indexing_does_not_revisit_an_exception(tmp_path, monkeypatch):
    """BUG-002: public indexing must propagate the backend failure despite a cause cycle."""
    class Revisited(Exception):
        pass

    class BackendFailure(RuntimeError):
        visits = 0

        def __str__(self):
            self.visits += 1
            if self.visits > 32:
                raise Revisited
            return "backend request failed"

    first = BackendFailure("first")
    second = BackendFailure("second")
    first.__cause__ = second
    second.__cause__ = first

    def fail(*_args, **_kwargs):
        raise first

    monkeypatch.setattr(index, "embed_with_cache", fail)
    document = tmp_path / "notes.md"
    document.write_text("A small offline indexing source.", encoding="utf-8")
    store = LiteStore(tmp_path / "memory.db", dim=64)
    observed = None
    try:
        try:
            index.Indexer(store, HashingEmbedder(dim=64), cache=None).index_path(document)
        except BackendFailure as error:
            observed = error
        except Revisited:
            pass
    finally:
        store.close()
    assert observed is first, "classification must terminate without revisiting the exception cycle"


@pytest.mark.parametrize("transport", ["sse", "streamable-http"])
def test_codex_command_dispatches_stdio(tmp_path, monkeypatch, transport):
    """ENV-001: a command MCP launcher must override inherited HTTP transport."""
    from recall_mcp import codex_server
    from recall_mcp.settings import Settings

    config = tmp_path / "hook.json"
    config.write_text(json.dumps({"dsn": "postgresql://installed/recall", "embedder": "hashing"}), encoding="utf-8")
    monkeypatch.setenv("RECALL_CODEX_CONFIG", str(config))
    monkeypatch.setenv("RECALL_TRANSPORT", transport)
    captured = []
    monkeypatch.setattr("recall_mcp.server.main", lambda: captured.append(Settings.from_env().transport))
    assert codex_server.main() == 0
    assert captured == ["stdio"], "the installed command must dispatch stdio"
    assert codex_server.os.environ["RECALL_TRANSPORT"] == transport


def test_integer_confidence_roundtrip(tmp_path):
    """DAT-003: accepted confidence=1 must keep its identity across the durable codec."""
    evidence = replace(card(fact("team:platform")), card_id="", confidence=1)
    failure = None
    resolved = None
    with SQLiteEvidenceCardStore(str(tmp_path / "cards.db"), tenant_id=evidence.tenant_id) as store:
        try:
            store.put((evidence,))
            resolved = store.resolve(evidence.card_id)
        except ValueError as error:
            failure = str(error)
    assert failure is None, f"accepted confidence must roundtrip: {failure}"
    assert resolved == evidence


@requires_db
@pytest.mark.parametrize("boundary", ["cards", "ledger", "outbox"])
def test_jsonb_numeric_roundtrip(boundary):
    """DAT-004: JSONB must preserve immutable identities for accepted exponent floats."""
    tenant, generation, claim, evidence = _fixture()
    claim = replace(claim, object=1e20)
    evidence = replace(evidence, card_id="", structured_facts=(claim,))
    failure = None
    resolved = None
    try:
        if boundary == "cards":
            store = PostgresEvidenceCardStore(TEST_DSN, tenant_id=tenant)
            store.put((evidence,))
            store.put((evidence,))
            resolved = store.resolve(evidence.card_id)
        else:
            cards = EvidenceCardStore()
            cards.put((evidence,))
            ledger = PostgresFactLedger(TEST_DSN, tenant_id=tenant) if boundary == "ledger" else InMemoryFactLedger()
            ctl = ProvenanceController(
                tenant_id=tenant, generation_id=generation, cards=cards, ledger=ledger,
                source_digest_for=lambda _card: "digest", now=lambda: NOW, writer="test",
            )
            event = ctl.apply_fact(FactApplicationRequest(claim, (evidence.card_id,), "numeric")).event
            if boundary == "ledger":
                event = ledger.events[0]
            else:
                queue = PostgresMaterializationOutbox(TEST_DSN, tenant_id=tenant)
                queue.enqueue(event)
                queue.enqueue(event)
                event = queue.claim(tenant_id=tenant, now=NOW)[0]
            resolved = event.evidence_cards[0]
            assert event.fact_id == fact_identity(claim)
    except ValueError as error:
        failure = str(error)
    assert failure is None, f"accepted JSON number must preserve identity: {failure}"
    assert resolved == evidence


@requires_db
def test_jsonb_numeric_mixed_mapping_keys():
    """DAT-004 repair proof: raw sort_keys raised TypeError before JSON key coercion.

    Red baseline is the first codec implementation, before JSON normalization.
    The public store must preserve card identity and JSON's string key representation.
    """
    tenant, _generation, claim, evidence = _fixture()
    claim = replace(claim, object={"amount": 1e20, 1: "observed"})
    evidence = replace(evidence, card_id="", structured_facts=(claim,))
    store = PostgresEvidenceCardStore(TEST_DSN, tenant_id=tenant)
    failure = None
    resolved = None
    try:
        store.put((evidence,))
        resolved = store.resolve(evidence.card_id)
    except (TypeError, ValueError) as error:
        failure = str(error)
    assert failure is None, f"accepted numeric mapping must roundtrip: {failure}"
    assert resolved.card_id == evidence.card_id
    assert resolved.structured_facts[0].object == {"amount": 1e20, "1": "observed"}


@pytest.mark.parametrize("elapsed,offset,expected", [(1, 120, 0), (301, -120, 1)])
def test_sqlite_lease_uses_instants(tmp_path, elapsed, offset, expected):
    """NUM-001: offset spelling must not change whether a 300-second lease expired."""
    claim = fact("team:platform")
    evidence = card(claim)
    event = controller((evidence,)).apply_fact(FactApplicationRequest(claim, (evidence.card_id,), "lease-probe")).event
    with SQLiteMaterializationOutbox(str(tmp_path / "outbox.db"), tenant_id=evidence.tenant_id) as queue:
        queue.enqueue(event)
        assert len(queue.claim(tenant_id=evidence.tenant_id, now=NOW)) == 1
        later = (NOW + timedelta(seconds=elapsed)).astimezone(timezone(timedelta(minutes=offset)))
        assert len(queue.claim(tenant_id=evidence.tenant_id, now=later)) == expected


@pytest.mark.parametrize("backend", ["memory", "sqlite", pytest.param("postgres", marks=requires_db)])
def test_outbox_lease_duration_across_daylight_saving_fallback(tmp_path, backend):
    """NUM-001 repair proof: aware local arithmetic made a 300-second lease last 65 minutes.

    Red baseline is the parsed-expiry repair before UTC clock normalization.
    Elapsed time is advanced in UTC, independent of the local clock's repeated hour.
    """
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    try:
        zone = ZoneInfo("Europe/Rome")
    except ZoneInfoNotFoundError:
        pytest.skip("requires system IANA timezone data")
    initial = datetime(2026, 10, 25, 2, 58, tzinfo=zone, fold=0)
    later = (initial.astimezone(UTC) + timedelta(seconds=301)).astimezone(zone)
    claim = fact("team:platform")
    evidence = card(claim)
    event = controller((evidence,)).apply_fact(FactApplicationRequest(claim, (evidence.card_id,), "fallback-lease")).event
    if backend == "sqlite":
        queue = SQLiteMaterializationOutbox(str(tmp_path / "fallback.db"), tenant_id=evidence.tenant_id)
    elif backend == "postgres":
        queue = PostgresMaterializationOutbox(TEST_DSN, tenant_id=evidence.tenant_id)
    else:
        queue = InMemoryMaterializationOutbox()
    try:
        queue.enqueue(event)
        assert len(queue.claim(tenant_id=evidence.tenant_id, now=initial)) == 1
        assert len(queue.claim(tenant_id=evidence.tenant_id, now=later)) == 1, "lease must expire after 300 real seconds across a repeated local hour"
    finally:
        if backend == "sqlite":
            queue.close()


@requires_db
@pytest.mark.parametrize("stage", ["build", "validate", "rebuild"])
def test_generation_dependency_projection_is_ready(manager, stage):
    """DAT-001: every generation lifecycle graph writer persists dependency readiness."""
    data = b"---\ndepends_on: [missing.md]\n---\nA dependent observation.\n"
    manifest = _manifest(manager.tenant_id, data)
    generation = manager.create(manifest, _pipeline("model-a"))
    gid = generation.generation_id
    manager.build(gid, _reader(manifest, data), _Embedder(1), lambda text: [text])
    if stage != "build":
        with manager._connect() as conn:
            conn.execute("UPDATE recall_generations SET validation_summary = validation_summary - 'dependency_projection' WHERE tenant_id=%s AND generation_id=%s", (manager.tenant_id, gid))
        manager.validate(gid)
    if stage == "rebuild":
        with manager._connect() as conn:
            conn.execute("UPDATE recall_generations SET validation_summary = validation_summary - 'dependency_projection' WHERE tenant_id=%s AND generation_id=%s", (manager.tenant_id, gid))
        manager.rebuild_graph(gid)
    with manager._connect() as conn:
        summary = conn.execute("SELECT validation_summary FROM recall_generations WHERE tenant_id=%s AND generation_id=%s", (manager.tenant_id, gid)).fetchone()[0]
    marker = summary.get("dependency_projection")
    assert marker and marker.get("ready") is True, "generation must persist its dependency readiness"
    if stage != "build":
        manager.promote(gid, unsafe_development=True)
        with GenerationStore(TEST_DSN, 64, tenant=manager.tenant_id, dependency_mode="enforce") as store:
            assert store.dependency_projection() is not None, "enforce mode must resolve the persisted projection"


@requires_db
def test_dependency_projection_generated_runtime_role(manager):
    """DAT-001 repair proof: generated grants omitted both dependency tables.

    Red baseline is the graph repair with the original schema table tuple.
    A real nonowner role must resolve the projection through GenerationStore.
    """
    from psycopg.conninfo import make_conninfo
    from recall.schema import serving_grants

    data = b"---\ndepends_on: [missing.md]\n---\nA dependent observation.\n"
    manifest = _manifest(manager.tenant_id, data)
    generation = manager.create(manifest, _pipeline("model-a"))
    gid = generation.generation_id
    manager.build(gid, _reader(manifest, data), _Embedder(1), lambda text: [text])
    manager.validate(gid)
    manager.promote(gid, unsafe_development=True)
    role = "dependency_runtime_" + uuid4().hex[:12]
    password = uuid4().hex
    identity = psycopg.sql.Identifier(role)
    failure = None
    projection = None
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute(psycopg.sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(identity, psycopg.sql.Literal(password)))
        try:
            for statement in serving_grants(role):
                conn.execute(statement)
            try:
                with GenerationStore(make_conninfo(TEST_DSN, user=role, password=password), 64, tenant=manager.tenant_id, dependency_mode="enforce") as store:
                    projection = store.dependency_projection()
            except psycopg.errors.InsufficientPrivilege as error:
                failure = str(error)
        finally:
            conn.execute(psycopg.sql.SQL("DROP OWNED BY {}").format(identity))
            conn.execute(psycopg.sql.SQL("DROP ROLE {}").format(identity))
    assert failure is None, f"generated runtime role must read its dependency projection: {failure}"
    assert projection is not None


@requires_db
def test_certified_promotion_audit_type(calibration_tenant):
    """DAT-002: the production audit must distinguish certified from unsafe promotion."""
    tenant, manager = calibration_tenant
    embedder = _CalibrationEmbedder()
    gid = _ready(manager, embedder, b"answer corpus", "v1")
    repository = CalibrationRepository(TEST_DSN, tenant, actor="pytest")
    artifact = repository.calibrate(gid, _labels(), embedder)
    repository.publish(artifact.calibration_id)
    _production(manager).promote(gid)
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        rows = conn.execute("SELECT event_type FROM recall_audit_events WHERE tenant_id=%s AND generation_id=%s AND event_type LIKE %s", (tenant, gid, "generation_promoted_%")).fetchall()
    assert rows == [("generation_promoted_certified",)], "certified promotion must retain its audit meaning"
