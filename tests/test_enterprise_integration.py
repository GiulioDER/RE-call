"""The enterprise control plane against a real PostgreSQL, as an unprivileged role.

Everything here needs a database, and most of it needs a database role that row level security
actually constrains. That second requirement is why this file exists separately from
`test_enterprise_control_plane.py`: that one runs on whatever `RECALL_TEST_DSN` points at, and the
shipped `docker-compose.yml` points it at the cluster superuser, for whom every RLS assertion in
this repository passes vacuously.

Three areas had **no test reference anywhere** in `tests/` before this file: route notification and
polling, active/shadow dual writes, and atomic cross-generation forget. `replay_pending` had one
reference in the entire repository, its own definition, so a crash mid-migration left an outbox
that permanently refused `cutover` and had no shipped drain.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path

import psycopg
import pytest

from recall.control_plane import ControlPlane, validate_table_name
from recall.embeddings import HashingEmbedder
from recall.index import Indexer, ShadowIndexTarget, candidate_files
from recall.migration import validate_generation_parity
from recall.schema import apply_migrations
from recall.store import PgVectorStore
from recall_mcp.service import forget_memory
from recall_mcp.stores import StoreRegistry
from tests.conftest import TEST_DSN, requires_db, role_is_unprivileged

pytestmark = requires_db

DIM = 32


# --------------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------------


@pytest.fixture
def suffix() -> str:
    return uuid.uuid4().hex[:10]


class _Fixture:
    """One tenant, two generations, two physical tables, all torn down together."""

    def __init__(self, dsn: str, suffix: str) -> None:
        # Re-established per fixture, not once per session, and not delegated to the fixture's
        # name. Every isolation assertion reached through this object depends on the role being
        # constrained by row level security, so the premise travels with the object rather than
        # sitting in a docstring three files away.
        if not role_is_unprivileged(dsn):
            raise AssertionError(
                "the enterprise fixture was handed a superuser or BYPASSRLS role; every row "
                "level security assertion below it would pass vacuously"
            )
        self.dsn = dsn
        self.tenant = f"t_{suffix}"
        self.active_id = f"g_active_{suffix}"
        self.shadow_id = f"g_shadow_{suffix}"
        self.active_table = f"c_active_{suffix}"
        self.shadow_table = f"c_shadow_{suffix}"
        self.control = ControlPlane(dsn)
        self.control.apply_migrations()
        for table in (self.active_table, self.shadow_table):
            apply_migrations(dsn, table=table, dim=DIM)
        self.control.register_generation(self.active_id, self.active_table, "profile-a", DIM)
        self.control.register_generation(self.shadow_id, self.shadow_table, "profile-a", DIM)
        self.control.set_generation_state(self.active_id, "ready", chunk_count=0, source_count=0)

    def store(self, generation_id: str, table: str) -> PgVectorStore:
        return PgVectorStore(
            self.dsn, dim=DIM, table=table, tenant=self.tenant, generation_id=generation_id
        )

    def close(self) -> None:
        """Teardown that works WITHOUT superuser, which the pre-existing teardown did not.

        `recall_tenant_routes` and `recall_migration_events` carry FORCE row level security keyed
        on the `recall.tenant_id` GUC. A cleanup connection that never sets it deletes zero rows
        and reports success, and the route then blocks the generation delete on a foreign key.
        On a superuser DSN that is invisible, because RLS is bypassed; this is the first thing
        that broke when the suite was pointed at an unprivileged role.
        """
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (self.tenant,))
            conn.execute("DELETE FROM recall_migration_events WHERE tenant_id = %s", (self.tenant,))
            conn.execute("DELETE FROM recall_tenant_routes WHERE tenant_id = %s", (self.tenant,))
            conn.execute(
                "DELETE FROM recall_index_generations WHERE generation_id = ANY(%s)",
                ([self.active_id, self.shadow_id],),
            )
            for table in (self.active_table, self.shadow_table):
                conn.execute(f"DROP TABLE IF EXISTS {table}")
                conn.execute(
                    "DELETE FROM recall_schema_migrations WHERE target_table = %s", (table,)
                )


@pytest.fixture
def enterprise(unprivileged_dsn: str, suffix: str):
    fixture = _Fixture(unprivileged_dsn, suffix)
    try:
        yield fixture
    finally:
        fixture.close()


def _chunks(sources: list[str], embedder: HashingEmbedder):
    from recall.types import Chunk

    chunks = [
        Chunk(id=f"{s}#0", source=s, text=f"contents of {s}", metadata={"file": Path(s).name})
        for s in sources
    ]
    return chunks, [embedder.embed([c.text])[0] for c in chunks]


# --------------------------------------------------------------------------------------------
# 1. The role itself. Every RLS assertion below rests on this one.
# --------------------------------------------------------------------------------------------


def test_the_test_role_is_neither_superuser_nor_bypassrls(unprivileged_dsn: str) -> None:
    """Prove the premise INSIDE the test, not in the fixture's name.

    A skipped or vacuous version of this file is worse than none: it would report a green tenant
    isolation suite while every policy below it was inert. So the property is re-established here
    from the catalog, and `PgVectorStore.check_rls_effective` (the function the MCP server uses
    to decide whether to warn an operator) is required to agree.
    """
    with psycopg.connect(unprivileged_dsn, autocommit=True) as conn:
        role, superuser, bypass = conn.execute(
            "SELECT current_user, rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
        ).fetchone()
    assert superuser is False, f"{role!r} is a superuser; every RLS assertion here would be vacuous"
    assert bypass is False, f"{role!r} has BYPASSRLS; every RLS assertion here would be vacuous"

    store = PgVectorStore(unprivileged_dsn, dim=DIM, table="chunks", tenant="default")
    try:
        assert store.check_rls_effective() is True
    finally:
        store.close()


# --------------------------------------------------------------------------------------------
# 2. Row level security: chunk tables and the control plane
# --------------------------------------------------------------------------------------------


def test_chunk_table_rls_hides_another_tenants_rows(enterprise) -> None:
    """Two layers, asserted separately, because only one of them is row level security.

    `PgVectorStore.count()` carries `WHERE tenant_id = %s` (`recall/store.py:1776`), so it
    answers zero for the other tenant whether or not a single policy exists. Asserting through
    the store alone would therefore have tested the query PREDICATE while claiming to test the
    POLICY, and a `NO FORCE ROW LEVEL SECURITY` mutation left it green.

    The load-bearing assertion is the raw read below: no tenant predicate in the SQL at all, so
    the only thing that can return zero rows is the policy.
    """
    embedder = HashingEmbedder(dim=DIM)
    chunks, vectors = _chunks(["/corpus/a.md"], embedder)
    mine = enterprise.store(enterprise.active_id, enterprise.active_table)
    other_tenant = f"{enterprise.tenant}_other"
    theirs = PgVectorStore(
        enterprise.dsn, dim=DIM, table=enterprise.active_table, tenant=other_tenant
    )
    try:
        mine.upsert(chunks, vectors)
        assert mine.count() == 1
        # Layer one: the application predicate.
        assert theirs.count() == 0
        assert theirs.delete_sources(["/corpus/a.md"]) == 0
        assert mine.count() == 1, "another tenant's DELETE reached across the policy"
    finally:
        mine.close()
        theirs.close()

    # Layer two: the policy, with nothing else in the way.
    with psycopg.connect(enterprise.dsn, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (other_tenant,))
        visible = conn.execute(
            f"SELECT count(*) FROM {enterprise.active_table}"
        ).fetchone()[0]
        assert visible == 0, (
            "an unpredicated SELECT under another tenant's GUC returned rows: row level security "
            "is not constraining this role"
        )
        text_reachable = conn.execute(
            f"SELECT count(*) FROM {enterprise.active_table} WHERE text LIKE %s", ("%contents%",)
        ).fetchone()[0]
        assert text_reachable == 0, "another tenant reached the corpus text through a raw filter"

        # And the same connection DOES see the rows once it names the owning tenant, which rules
        # out the boring explanation that the table is simply empty.
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (enterprise.tenant,))
        assert conn.execute(
            f"SELECT count(*) FROM {enterprise.active_table}"
        ).fetchone()[0] == 1


def test_control_plane_rls_hides_another_tenants_route_and_outbox(enterprise) -> None:
    enterprise.control.set_route(enterprise.tenant, enterprise.active_id, enterprise.shadow_id)
    enterprise.control.append_event(
        enterprise.tenant, f"op-{enterprise.tenant}", "index", {"sources": ["/corpus/a.md"]}
    )
    assert enterprise.control.route(enterprise.tenant) is not None
    assert len(enterprise.control.pending_events(enterprise.tenant)) == 1

    # A connection that never establishes the GUC sees nothing, rather than everything.
    with psycopg.connect(enterprise.dsn, autocommit=True) as conn:
        routes = conn.execute(
            "SELECT count(*) FROM recall_tenant_routes WHERE tenant_id = %s", (enterprise.tenant,)
        ).fetchone()[0]
        events = conn.execute(
            "SELECT count(*) FROM recall_migration_events WHERE tenant_id = %s",
            (enterprise.tenant,),
        ).fetchone()[0]
    assert routes == 0, "recall_tenant_routes is readable without a tenant GUC"
    assert events == 0, "recall_migration_events is readable without a tenant GUC"

    # And a DIFFERENT tenant's GUC does not reveal them either. Reading zero with no GUC could be
    # explained by `current_setting(..., true)` returning NULL; this rules that explanation out.
    with psycopg.connect(enterprise.dsn, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", ("someone-else",))
        assert conn.execute(
            "SELECT count(*) FROM recall_tenant_routes WHERE tenant_id = %s", (enterprise.tenant,)
        ).fetchone()[0] == 0


def test_forced_rls_binds_the_table_owner_too(enterprise) -> None:
    """`relforcerowsecurity`, not just `relrowsecurity`.

    Without FORCE, the owner of a table bypasses its policies, and the migration role owns these
    tables. `readiness_facts` reports the conjunction, so this asserts what readiness reads.
    """
    store = enterprise.store(enterprise.active_id, enterprise.active_table)
    try:
        facts = store.readiness_facts()
    finally:
        store.close()
    assert facts["rls_enabled"] is True
    with psycopg.connect(enterprise.dsn, autocommit=True) as conn:
        enabled, forced = conn.execute(
            "SELECT relrowsecurity, relforcerowsecurity FROM pg_class WHERE oid = %s::regclass",
            (enterprise.active_table,),
        ).fetchone()
    assert (enabled, forced) == (True, True)


# --------------------------------------------------------------------------------------------
# 3. Index validity: HNSW and full text
# --------------------------------------------------------------------------------------------


def test_hnsw_and_full_text_indexes_are_present_and_valid(enterprise) -> None:
    store = enterprise.store(enterprise.active_id, enterprise.active_table)
    try:
        assert store.readiness_facts()["indexes_valid"] is True
    finally:
        store.close()
    with psycopg.connect(enterprise.dsn, autocommit=True) as conn:
        rows = dict(
            conn.execute(
                "SELECT c.relname, am.amname FROM pg_index i "
                "JOIN pg_class c ON c.oid = i.indexrelid "
                "JOIN pg_am am ON am.oid = c.relam "
                "WHERE i.indrelid = %s::regclass AND i.indisvalid",
                (enterprise.active_table,),
            ).fetchall()
        )
    assert rows.get(f"{enterprise.active_table}_emb_idx") == "hnsw"
    assert rows.get(f"{enterprise.active_table}_tsv_idx") == "gin"


def test_readiness_facts_report_a_present_but_invalid_index_as_invalid(enterprise) -> None:
    """The state a name-only check misses: the index EXISTS and `indisvalid` is false.

    That is what a failed or interrupted `CREATE INDEX CONCURRENTLY` leaves behind, and it is the
    difference between checking `pg_index.indisvalid` and checking that a name is present. It is
    produced here the way PostgreSQL really produces it (a concurrent unique build that fails on
    duplicate rows) rather than by writing to `pg_index`, which needs a superuser and would have
    tested a role this suite deliberately does not use.

    Both directions are asserted. A check that reports invalid forever would pass a one-sided
    version of this test while making every deployment permanently unready.
    """
    embedder = HashingEmbedder(dim=DIM)
    index_name = f"{enterprise.active_table}_emb_idx"
    store = enterprise.store(enterprise.active_id, enterprise.active_table)
    try:
        assert store.readiness_facts()["indexes_valid"] is True

        from recall.types import Chunk

        duplicates = [
            Chunk(id=f"dup-{i}", source="/corpus/dup.md", text=f"duplicate {i}", metadata={})
            for i in range(2)
        ]
        store.upsert(duplicates, [embedder.embed([c.text])[0] for c in duplicates])

        with psycopg.connect(enterprise.dsn, autocommit=True) as conn:
            original = conn.execute(
                "SELECT pg_get_indexdef(%s::regclass)", (index_name,)
            ).fetchone()[0]
            with pytest.raises(psycopg.errors.UniqueViolation):
                conn.execute(
                    f"CREATE UNIQUE INDEX CONCURRENTLY {index_name}_probe "
                    f"ON {enterprise.active_table} (source)"
                )
            left_behind = conn.execute(
                "SELECT indisvalid FROM pg_index WHERE indexrelid = %s::regclass",
                (f"{index_name}_probe",),
            ).fetchone()
            assert left_behind is not None and left_behind[0] is False, (
                "the failed concurrent build did not leave an invalid index, so this test would "
                "not be exercising the branch it names"
            )
            conn.execute(f"DROP INDEX {index_name}")
            conn.execute(f"ALTER INDEX {index_name}_probe RENAME TO {index_name}")

        assert store.readiness_facts()["indexes_valid"] is False

        with psycopg.connect(enterprise.dsn, autocommit=True) as conn:
            conn.execute(f"DROP INDEX {index_name}")
            conn.execute(original)
        assert store.readiness_facts()["indexes_valid"] is True
    finally:
        store.close()


# --------------------------------------------------------------------------------------------
# 4. The outbox: crash recovery, replay, idempotency, cutover refusal
# --------------------------------------------------------------------------------------------


def _crash_shaped_event(enterprise, sources: list[str], embedder: HashingEmbedder) -> str:
    """Append the event a crash between `append_event` and the two writes would leave behind."""
    chunks, vectors = _chunks(sources, embedder)
    operation_id = f"op-{uuid.uuid4().hex[:8]}"
    payload = {
        "active_generation": enterprise.active_id,
        "shadow_generation": enterprise.shadow_id,
        "sources": sources,
        "active_chunks": [
            {"id": c.id, "source": c.source, "text": c.text, "metadata": c.metadata,
             "embedding": v}
            for c, v in zip(chunks, vectors)
        ],
        "chunks": [
            {"id": c.id, "source": c.source, "text": c.text, "metadata": c.metadata,
             "embedding": v}
            for c, v in zip(chunks, vectors)
        ],
    }
    enterprise.control.append_event(
        enterprise.tenant, operation_id, "index", payload, active_count=len(chunks)
    )
    return operation_id


def test_replay_converges_both_generations_after_a_crash(enterprise) -> None:
    """The crash shape from the handoff: event appended, NEITHER store written, then replay."""
    embedder = HashingEmbedder(dim=DIM)
    enterprise.control.set_route(enterprise.tenant, enterprise.active_id, enterprise.shadow_id)
    operation_id = _crash_shaped_event(enterprise, ["/corpus/a.md", "/corpus/b.md"], embedder)

    active = enterprise.store(enterprise.active_id, enterprise.active_table)
    shadow = enterprise.store(enterprise.shadow_id, enterprise.shadow_table)
    try:
        assert (active.count(), shadow.count()) == (0, 0)
        completed = enterprise.control.replay_pending(
            enterprise.tenant,
            {enterprise.active_id: active, enterprise.shadow_id: shadow},
        )
        assert completed == 1
        assert active.count() == 2
        assert shadow.count() == 2
    finally:
        active.close()
        shadow.close()

    assert enterprise.control.pending_events(enterprise.tenant) == []
    with psycopg.connect(enterprise.dsn, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (enterprise.tenant,))
        status, payload, active_count, shadow_count, completed_at = conn.execute(
            "SELECT status, payload, active_count, shadow_count, completed_at "
            "FROM recall_migration_events WHERE tenant_id = %s AND operation_id = %s",
            (enterprise.tenant, operation_id),
        ).fetchone()
    assert status == "complete"
    assert payload is None, "a completed audit record still holds corpus text"
    assert (active_count, shadow_count) == (2, 2)
    assert completed_at is not None


def test_the_retained_audit_record_holds_counts_and_no_corpus(enterprise) -> None:
    """What SURVIVES completion: operation id, timestamps, status, counts. Nothing else.

    Asserted column by column against `information_schema` rather than by spot-checking the two
    columns that happen to be interesting, so a future migration that adds a text-bearing column
    to the outbox fails here instead of shipping a retention hole.
    """
    embedder = HashingEmbedder(dim=DIM)
    enterprise.control.set_route(enterprise.tenant, enterprise.active_id, enterprise.shadow_id)
    secret = "/corpus/private-memo.md"
    operation_id = _crash_shaped_event(enterprise, [secret], embedder)
    active = enterprise.store(enterprise.active_id, enterprise.active_table)
    shadow = enterprise.store(enterprise.shadow_id, enterprise.shadow_table)
    try:
        enterprise.control.replay_pending(
            enterprise.tenant,
            {enterprise.active_id: active, enterprise.shadow_id: shadow},
        )
    finally:
        active.close()
        shadow.close()

    with psycopg.connect(enterprise.dsn, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (enterprise.tenant,))
        columns = [
            name for (name,) in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'recall_migration_events' ORDER BY column_name"
            ).fetchall()
        ]
        row = conn.execute(
            f"SELECT {', '.join(columns)} FROM recall_migration_events "
            "WHERE tenant_id = %s AND operation_id = %s",
            (enterprise.tenant, operation_id),
        ).fetchone()
    retained = {name: value for name, value in zip(columns, row) if value is not None}
    assert set(retained) == {
        "sequence_id", "tenant_id", "operation_id", "operation_kind",
        "status", "active_count", "shadow_count", "created_at", "completed_at",
    }
    for name, value in retained.items():
        assert secret not in str(value), f"column {name!r} retains a source path after completion"
        assert "contents of" not in str(value), f"column {name!r} retains corpus text"


def test_replaying_the_same_event_twice_is_idempotent(enterprise) -> None:
    embedder = HashingEmbedder(dim=DIM)
    _crash_shaped_event(enterprise, ["/corpus/a.md"], embedder)
    active = enterprise.store(enterprise.active_id, enterprise.active_table)
    shadow = enterprise.store(enterprise.shadow_id, enterprise.shadow_table)
    try:
        stores = {enterprise.active_id: active, enterprise.shadow_id: shadow}
        assert enterprise.control.replay_pending(enterprise.tenant, stores) == 1
        first = (active.count(), shadow.count())
        # A second drain finds nothing pending: `complete_event` is the barrier, and the row
        # counts must not double even though `replace_sources` would happily run again.
        assert enterprise.control.replay_pending(enterprise.tenant, stores) == 0
        assert (active.count(), shadow.count()) == first == (1, 1)
    finally:
        active.close()
        shadow.close()


def test_append_event_is_idempotent_on_operation_id(enterprise) -> None:
    """A retried write must reuse its ordered sequence, not append a second one."""
    first = enterprise.control.append_event(
        enterprise.tenant, "op-retry", "index", {"sources": ["/corpus/a.md"]}
    )
    second = enterprise.control.append_event(
        enterprise.tenant, "op-retry", "index", {"sources": ["/corpus/a.md"]}
    )
    assert first == second
    assert len(enterprise.control.pending_events(enterprise.tenant)) == 1


def test_cutover_is_refused_while_an_event_is_pending(enterprise) -> None:
    """The refusal branch the handoff records as unexercised."""
    enterprise.control.set_route(enterprise.tenant, enterprise.active_id, enterprise.shadow_id)
    enterprise.control.set_generation_state(
        enterprise.shadow_id, "ready", chunk_count=1, source_count=1
    )
    enterprise.control.append_event(
        enterprise.tenant, "op-blocking", "index", {"sources": ["/corpus/a.md"]}
    )

    with pytest.raises(RuntimeError, match="pending"):
        enterprise.control.cutover(enterprise.tenant)
    route = enterprise.control.route(enterprise.tenant)
    assert route is not None and route.active.generation_id == enterprise.active_id, (
        "cutover raised but promoted anyway"
    )

    enterprise.control.complete_event(enterprise.tenant, "op-blocking", 1)
    enterprise.control.cutover(enterprise.tenant)
    promoted = enterprise.control.route(enterprise.tenant)
    assert promoted is not None and promoted.active.generation_id == enterprise.shadow_id


def test_cutover_is_refused_while_the_shadow_is_not_ready(enterprise) -> None:
    enterprise.control.set_route(enterprise.tenant, enterprise.active_id, enterprise.shadow_id)
    with pytest.raises(RuntimeError, match="ready shadow"):
        enterprise.control.cutover(enterprise.tenant)


def test_generation_rollback_restores_the_previous_active(enterprise) -> None:
    """Cutover swaps; `set-route` swaps back. The old table must still be servable."""
    enterprise.control.set_route(enterprise.tenant, enterprise.active_id, enterprise.shadow_id)
    enterprise.control.set_generation_state(
        enterprise.shadow_id, "ready", chunk_count=0, source_count=0
    )
    enterprise.control.cutover(enterprise.tenant)
    after = enterprise.control.route(enterprise.tenant)
    assert after is not None and after.active.generation_id == enterprise.shadow_id

    enterprise.control.set_route(enterprise.tenant, enterprise.active_id, enterprise.shadow_id)
    rolled_back = enterprise.control.route(enterprise.tenant)
    assert rolled_back is not None
    assert rolled_back.active.generation_id == enterprise.active_id
    assert rolled_back.active.physical_table == enterprise.active_table


def test_a_retired_generation_cannot_serve_and_cannot_be_retired_while_routed(enterprise) -> None:
    enterprise.control.set_route(enterprise.tenant, enterprise.active_id, enterprise.shadow_id)
    with pytest.raises(RuntimeError, match="active generation"):
        enterprise.control.retire_generation(enterprise.active_id, enterprise.tenant)
    with pytest.raises(RuntimeError, match="shadow generation"):
        enterprise.control.retire_generation(enterprise.shadow_id, enterprise.tenant)

    # Retire it behind the route's back, exactly as a second operator on another tenant could.
    enterprise.control.set_generation_state(enterprise.shadow_id, "retired")
    registry = StoreRegistry(
        dsn=enterprise.dsn, dim=DIM, allowed_tenants=frozenset({enterprise.tenant}),
        pool_size=2, statement_timeout_ms=5000, control_plane=enterprise.control,
    )
    try:
        with pytest.raises(RuntimeError, match="retired"):
            registry.get_shadow(enterprise.tenant)
    finally:
        registry.close()


# --------------------------------------------------------------------------------------------
# 5. Dual writes, erasure, parity
# --------------------------------------------------------------------------------------------


def test_dual_write_reaches_both_generations_and_leaves_no_pending_event(enterprise, tmp_path):
    """Both vector sets are prepared BEFORE either generation is written."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("alpha content for the shadow migration\n", encoding="utf-8")
    (corpus / "b.md").write_text("beta content for the shadow migration\n", encoding="utf-8")

    active = enterprise.store(enterprise.active_id, enterprise.active_table)
    shadow = enterprise.store(enterprise.shadow_id, enterprise.shadow_table)
    try:
        indexer = Indexer(
            active,
            HashingEmbedder(dim=DIM),
            shadow=ShadowIndexTarget(
                store=shadow, embedder=HashingEmbedder(dim=DIM),
                control_plane=enterprise.control,
            ),
        )
        stats = indexer.index_path(corpus)
        assert stats.files == 2
        assert active.count() == shadow.count() > 0
        assert enterprise.control.pending_events(enterprise.tenant) == []

        parity = validate_generation_parity(active, shadow)
        assert parity.valid, parity.failures
        assert parity.active_chunks == parity.shadow_chunks
    finally:
        active.close()
        shadow.close()


def test_a_failed_shadow_write_leaves_a_durable_pending_event(enterprise, tmp_path) -> None:
    """The active write is allowed to land only because the event survives to replay it.

    The failure is injected at the shadow store, after the outbox append and after the active
    write. That ordering is the whole design: the pending row is what makes an asymmetric state
    recoverable instead of silent.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("alpha content\n", encoding="utf-8")

    active = enterprise.store(enterprise.active_id, enterprise.active_table)
    shadow = enterprise.store(enterprise.shadow_id, enterprise.shadow_table)
    try:
        def _boom(*_args, **_kwargs):
            raise psycopg.OperationalError("shadow generation is unreachable")

        shadow.replace_sources = _boom  # type: ignore[method-assign]
        indexer = Indexer(
            active,
            HashingEmbedder(dim=DIM),
            shadow=ShadowIndexTarget(
                store=shadow, embedder=HashingEmbedder(dim=DIM),
                control_plane=enterprise.control,
            ),
        )
        with pytest.raises(psycopg.OperationalError):
            indexer.index_path(corpus)
        assert active.count() > 0, "the active write should have landed before the shadow failure"

        pending = enterprise.control.pending_events(enterprise.tenant)
        assert len(pending) == 1
        assert pending[0].operation_kind == "index"
        assert pending[0].payload is not None
        assert pending[0].active_count > 0
    finally:
        active.close()
        shadow.close()

    # And that pending event blocks cutover, which is what makes the state safe rather than merely
    # visible.
    enterprise.control.set_route(enterprise.tenant, enterprise.active_id, enterprise.shadow_id)
    enterprise.control.set_generation_state(
        enterprise.shadow_id, "ready", chunk_count=0, source_count=0
    )
    with pytest.raises(RuntimeError, match="pending"):
        enterprise.control.cutover(enterprise.tenant)


def test_forget_erases_from_both_generations_in_one_transaction(enterprise) -> None:
    embedder = HashingEmbedder(dim=DIM)
    chunks, vectors = _chunks(["/corpus/a.md", "/corpus/b.md"], embedder)
    active = enterprise.store(enterprise.active_id, enterprise.active_table)
    shadow = enterprise.store(enterprise.shadow_id, enterprise.shadow_table)
    try:
        active.upsert(chunks, vectors)
        shadow.upsert(chunks, vectors)
        removed = active.delete_sources_across(
            [active.table, shadow.table], ["/corpus/a.md"]
        )
        assert removed == 2  # one row in each generation
        assert active.count() == 1
        assert shadow.count() == 1
    finally:
        active.close()
        shadow.close()


def test_erasure_across_generations_is_atomic_when_the_second_table_fails(enterprise) -> None:
    """Force the second DELETE to fail and assert the FIRST one did not commit.

    The name of the method says atomic; a test that deletes from two healthy tables cannot tell
    an atomic implementation from a loop of autocommitted statements. The second table is made to
    fail by naming a generation table that does not exist, which raises inside the transaction
    after the first DELETE has already executed.
    """
    embedder = HashingEmbedder(dim=DIM)
    chunks, vectors = _chunks(["/corpus/a.md"], embedder)
    active = enterprise.store(enterprise.active_id, enterprise.active_table)
    try:
        active.upsert(chunks, vectors)
        assert active.count() == 1
        missing = f"c_absent_{uuid.uuid4().hex[:8]}"
        with pytest.raises(psycopg.errors.UndefinedTable):
            active.delete_sources_across([active.table, missing], ["/corpus/a.md"])
        assert active.count() == 1, (
            "the first generation's DELETE committed even though the second failed"
        )
    finally:
        active.close()


def test_erasure_reaches_a_pending_outbox_payload(enterprise) -> None:
    """Right to erasure must not leave the text sitting in the migration outbox.

    A pending event's payload carries the full chunk text and vectors. Deleting from both chunk
    tables and stopping there left the erased text in `recall_migration_events`, and a later
    replay would have written it straight back into both generations.
    """
    embedder = HashingEmbedder(dim=DIM)
    kept, erased = "/corpus/keep.md", "/corpus/erase.md"
    chunks, vectors = _chunks([kept, erased], embedder)
    active = enterprise.store(enterprise.active_id, enterprise.active_table)
    shadow = enterprise.store(enterprise.shadow_id, enterprise.shadow_table)
    try:
        active.upsert(chunks, vectors)
        shadow.upsert(chunks, vectors)
        _crash_shaped_event(enterprise, [kept, erased], embedder)

        result = forget_memory(active, [erased], shadow, enterprise.control)
        assert result.chunks_removed == 2

        pending = enterprise.control.pending_events(enterprise.tenant)
        assert len(pending) == 1, "the surviving source's replay record was discarded"
        payload = pending[0].payload or {}
        assert payload["sources"] == [kept]
        serialised = str(payload)
        assert erased not in serialised, "the erased source path survives in the outbox payload"
        assert f"contents of {erased}" not in serialised, "erased text survives in the outbox"
        assert kept in serialised, "the untouched source was scrubbed too"

        # Replaying now must not resurrect the erased source in either generation.
        enterprise.control.replay_pending(
            enterprise.tenant,
            {enterprise.active_id: active, enterprise.shadow_id: shadow},
        )
        assert active.sources_for_identifiers([erased]) == {}
        assert shadow.sources_for_identifiers([erased]) == {}
    finally:
        active.close()
        shadow.close()


def test_erasure_reaches_the_outbox_when_the_payload_is_the_ONLY_copy(enterprise) -> None:
    """The state the scrub exists for, which the first version of it could not reach.

    Three auditors found this independently. `forget_memory` gated the scrub on `to_delete`, and
    `to_delete` is resolved from the CHUNK TABLES. After a crash between `append_event` and the
    two `replace_sources` calls, the erased source has NO rows in either table, so the gate was
    closed exactly when the payload was the only copy: forget answered "no matching source(s)
    found", the text stayed in the outbox, and the next replay wrote it into both generations.

    The neighbouring test seeds both chunk tables before creating the event, so `to_delete` is
    never empty there and it passes with or without the gate. This one does NOT seed them, which
    is the whole point: it is red against the gated implementation and green against the fixed
    one.
    """
    embedder = HashingEmbedder(dim=DIM)
    only_in_outbox = "/corpus/never-written.md"
    active = enterprise.store(enterprise.active_id, enterprise.active_table)
    shadow = enterprise.store(enterprise.shadow_id, enterprise.shadow_table)
    try:
        _crash_shaped_event(enterprise, [only_in_outbox], embedder)
        assert (active.count(), shadow.count()) == (0, 0), (
            "this test is only meaningful while the chunk tables are empty"
        )
        pending = enterprise.control.pending_events(enterprise.tenant)
        assert len(pending) == 1 and only_in_outbox in str(pending[0].payload)

        result = forget_memory(active, [only_in_outbox], shadow, enterprise.control)

        # The chunk tables really had nothing, so this half is honestly reported as not-found.
        assert result.chunks_removed == 0
        # The outbox is the half that matters, and it must say so in the receipt.
        assert result.outbox_events_scrubbed == 1, (
            "the outbox was not swept, so a later replay would restore the erased text"
        )
        assert enterprise.control.pending_events(enterprise.tenant) == []

        # And the decisive check: replaying now must not resurrect it.
        enterprise.control.replay_pending(
            enterprise.tenant,
            {enterprise.active_id: active, enterprise.shadow_id: shadow},
        )
        assert (active.count(), shadow.count()) == (0, 0), "replay restored erased text"
    finally:
        active.close()
        shadow.close()


def test_retire_refuses_a_tenant_it_cannot_check(enterprise) -> None:
    """An absent route must not be the quiet path through the only check `retire` has.

    The refusal was `if row is not None and ...`, so a mistyped `--tenant` skipped it entirely and
    retired the generation unconditionally, printing success. `recall_index_generations` is not
    tenant scoped, so that is a fleet-wide state change made on no evidence at all.
    """
    with pytest.raises(KeyError, match="no route"):
        enterprise.control.retire_generation(enterprise.active_id, "tenant-that-does-not-exist")
    generation = enterprise.control.generation(enterprise.active_id)
    assert generation is not None and generation.state == "ready", (
        "the generation was retired against a tenant the control plane cannot check"
    )


def test_the_operator_cli_will_not_open_a_retired_generation(enterprise) -> None:
    """`replay` WRITES through `_open`, and the serving-path refusal did not cover it."""
    from recall.enterprise_cli import _open

    enterprise.control.set_generation_state(enterprise.shadow_id, "retired")
    retired = enterprise.control.generation(enterprise.shadow_id)
    assert retired is not None
    with pytest.raises(SystemExit, match="retired"):
        _open(enterprise.dsn, retired, enterprise.tenant)


def test_a_database_ahead_of_the_package_is_degraded_not_fatal(enterprise) -> None:
    """Migrate forward, roll the application back: that must not brick the fleet.

    `unknown` means the DATABASE has migrations this package does not ship, which is the normal
    state during a staged rollout. `missing` and a checksum mismatch stay fatal.
    """
    from recall.control_plane import ControlPlaneLedgerState

    ahead = ControlPlaneLedgerState(True, missing=(), unknown=(99,), checksum_mismatches=())
    assert ahead.current, "a database ahead of the package must still be servable"
    assert ahead.ahead
    behind = ControlPlaneLedgerState(True, missing=(1,), unknown=(), checksum_mismatches=())
    assert not behind.current
    drifted = ControlPlaneLedgerState(True, missing=(), unknown=(), checksum_mismatches=(1,))
    assert not drifted.current


def test_erasing_every_source_in_an_event_voids_it_rather_than_blocking_cutover(enterprise):
    embedder = HashingEmbedder(dim=DIM)
    source = "/corpus/only.md"
    chunks, vectors = _chunks([source], embedder)
    active = enterprise.store(enterprise.active_id, enterprise.active_table)
    shadow = enterprise.store(enterprise.shadow_id, enterprise.shadow_table)
    try:
        active.upsert(chunks, vectors)
        shadow.upsert(chunks, vectors)
        _crash_shaped_event(enterprise, [source], embedder)
        forget_memory(active, [source], shadow, enterprise.control)
    finally:
        active.close()
        shadow.close()

    assert enterprise.control.pending_events(enterprise.tenant) == []
    enterprise.control.set_route(enterprise.tenant, enterprise.active_id, enterprise.shadow_id)
    enterprise.control.set_generation_state(
        enterprise.shadow_id, "ready", chunk_count=0, source_count=0
    )
    enterprise.control.cutover(enterprise.tenant)  # no longer deadlocked


# --------------------------------------------------------------------------------------------
# 6. Routing: refusals, notification, in-flight stores
# --------------------------------------------------------------------------------------------


def _registry(enterprise, *, dim: int = DIM, profile: str | None = None) -> StoreRegistry:
    return StoreRegistry(
        dsn=enterprise.dsn, dim=dim, allowed_tenants=frozenset({enterprise.tenant}),
        pool_size=2, statement_timeout_ms=5000, control_plane=enterprise.control,
        embedding_profile=profile, route_poll_seconds=0.2,
    )


def test_a_profile_mismatch_refuses_to_open_the_generation(enterprise) -> None:
    enterprise.control.set_route(enterprise.tenant, enterprise.active_id)
    registry = _registry(enterprise, profile="profile-b")
    try:
        with pytest.raises(RuntimeError, match="profile"):
            registry.get(enterprise.tenant)
    finally:
        registry.close()


def test_a_dimension_mismatch_refuses_to_open_the_generation(enterprise) -> None:
    enterprise.control.set_route(enterprise.tenant, enterprise.active_id)
    registry = _registry(enterprise, dim=DIM * 2)
    try:
        with pytest.raises(RuntimeError, match="dimension"):
            registry.get(enterprise.tenant)
    finally:
        registry.close()


def test_no_client_parameter_can_name_a_generation_or_a_table(enterprise) -> None:
    """The registry's public surface takes a tenant and nothing else.

    `StoreRegistry.get` and `get_shadow` accept exactly one argument, and it is the tenant, which
    comes from a verified token rather than from a request body. A physical table reaches a query
    only from `recall_index_generations`, revalidated on the way out.
    """
    import inspect

    for method in (StoreRegistry.get, StoreRegistry.get_shadow):
        parameters = list(inspect.signature(method).parameters)
        assert parameters == ["self", "tenant"], (
            f"{method.__name__} grew a parameter a client could use to choose a table"
        )

    enterprise.control.set_route(enterprise.tenant, enterprise.active_id)
    registry = _registry(enterprise)
    try:
        store = registry.get(enterprise.tenant)
        assert store.table == enterprise.active_table
        assert validate_table_name(store.table) == enterprise.active_table
        with pytest.raises(PermissionError):
            registry.get("some-other-tenant")
    finally:
        registry.close()


def test_a_route_change_during_a_search_leaves_the_acquired_store_alone(enterprise) -> None:
    """In-flight requests finish on their generation; new ones pick up the new route.

    The acquired store is a Python object, so the property under test is that invalidation drops
    routing METADATA and not the store: a request already holding one keeps reading its own
    table even after the route has moved on.
    """
    embedder = HashingEmbedder(dim=DIM)
    chunks, vectors = _chunks(["/corpus/a.md"], embedder)
    seed = enterprise.store(enterprise.active_id, enterprise.active_table)
    try:
        seed.upsert(chunks, vectors)
    finally:
        seed.close()
    enterprise.control.set_generation_state(
        enterprise.shadow_id, "ready", chunk_count=0, source_count=0
    )
    enterprise.control.set_route(enterprise.tenant, enterprise.active_id)

    registry = _registry(enterprise)
    try:
        in_flight = registry.get(enterprise.tenant)
        assert in_flight.table == enterprise.active_table

        enterprise.control.set_route(enterprise.tenant, enterprise.shadow_id)
        registry.invalidate_route(enterprise.tenant)

        # The in-flight store still answers from the generation it acquired.
        assert in_flight.count() == 1
        assert in_flight.table == enterprise.active_table
        # A new acquisition uses the new route, and is a DIFFERENT store object: the cache is
        # keyed on (tenant, generation), so one tenant holds two generations at once.
        fresh = registry.get(enterprise.tenant)
        assert fresh.table == enterprise.shadow_table
        assert fresh is not in_flight
        assert in_flight.table == enterprise.active_table
    finally:
        registry.close()


def test_the_route_cache_is_keyed_on_tenant_and_generation(enterprise) -> None:
    enterprise.control.set_generation_state(
        enterprise.shadow_id, "ready", chunk_count=0, source_count=0
    )
    enterprise.control.set_route(enterprise.tenant, enterprise.active_id)
    registry = _registry(enterprise)
    try:
        first = registry.get(enterprise.tenant)
        enterprise.control.set_route(enterprise.tenant, enterprise.shadow_id)
        registry.invalidate_route(enterprise.tenant)
        second = registry.get(enterprise.tenant)
        keys = set(registry._stores)
        assert keys == {
            (enterprise.tenant, enterprise.active_id),
            (enterprise.tenant, enterprise.shadow_id),
        }
        assert first.generation_id == enterprise.active_id
        assert second.generation_id == enterprise.shadow_id
    finally:
        registry.close()


def test_listen_delivers_a_route_change_and_the_ttl_bounds_a_missed_one(enterprise) -> None:
    """Both halves of the invalidation contract, against a real LISTEN/NOTIFY.

    `set_route` and `cutover` each send `pg_notify('recall_route_changed', tenant)`. The listener
    thread turns that into `invalidate_route`. The cache TTL is the fallback for a notification
    that never arrives, so it is tested with the listener deliberately not running: a registry
    whose LISTEN is dead must still converge within `route_poll_seconds`.
    """
    enterprise.control.set_route(enterprise.tenant, enterprise.active_id)

    received: list[str] = []
    stop = threading.Event()
    listener = threading.Thread(
        target=enterprise.control.watch_routes, args=(received.append, stop), daemon=True
    )
    listener.start()
    try:
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and not received:
            enterprise.control.set_route(enterprise.tenant, enterprise.active_id)
            time.sleep(0.3)
        assert enterprise.tenant in received, "LISTEN never delivered the route change"
    finally:
        stop.set()
        listener.join(timeout=10.0)

    # The polling fallback: no listener, no invalidation call, route still converges on TTL.
    registry = StoreRegistry(
        dsn=enterprise.dsn, dim=DIM, allowed_tenants=frozenset({enterprise.tenant}),
        pool_size=2, statement_timeout_ms=5000, control_plane=enterprise.control,
        route_poll_seconds=0.2,
    )
    try:
        assert registry.get(enterprise.tenant).table == enterprise.active_table
        enterprise.control.set_generation_state(
            enterprise.shadow_id, "ready", chunk_count=0, source_count=0
        )
        enterprise.control.set_route(enterprise.tenant, enterprise.shadow_id)
        time.sleep(0.4)  # longer than route_poll_seconds, no invalidate_route call made
        assert registry.get(enterprise.tenant).table == enterprise.shadow_table
    finally:
        registry.close()


def test_concurrent_route_changes_never_serve_a_torn_route(enterprise) -> None:
    """Hammer `set_route` from one thread while another opens stores.

    Every observation must be one of the two configured generations and must match its own
    table: a reader that saw generation A's id with generation B's table would be reading a
    half-applied route.
    """
    enterprise.control.set_generation_state(
        enterprise.shadow_id, "ready", chunk_count=0, source_count=0
    )
    enterprise.control.set_route(enterprise.tenant, enterprise.active_id)
    expected = {
        enterprise.active_id: enterprise.active_table,
        enterprise.shadow_id: enterprise.shadow_table,
    }
    registry = _registry(enterprise)
    observed: list[tuple[str, str]] = []
    errors: list[BaseException] = []
    stop = threading.Event()

    def _flip() -> None:
        generations = [enterprise.active_id, enterprise.shadow_id]
        index = 0
        while not stop.is_set():
            try:
                enterprise.control.set_route(enterprise.tenant, generations[index % 2])
                index += 1
            except BaseException as exc:  # pragma: no cover - surfaced through `errors`
                errors.append(exc)
                return

    def _read() -> None:
        while not stop.is_set():
            try:
                registry.invalidate_route(enterprise.tenant)
                store = registry.get(enterprise.tenant)
                observed.append((store.generation_id, store.table))
            except BaseException as exc:  # pragma: no cover - surfaced through `errors`
                errors.append(exc)
                return

    threads = [threading.Thread(target=_flip), threading.Thread(target=_read)]
    try:
        for thread in threads:
            thread.start()
        time.sleep(3.0)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=15.0)
        registry.close()

    assert not errors, errors
    assert observed, "the reader thread never acquired a store"
    for generation_id, table in observed:
        assert expected.get(generation_id) == table, (
            f"generation {generation_id!r} was served table {table!r}"
        )


# --------------------------------------------------------------------------------------------
# 7. Migrator concurrency and both ledgers
# --------------------------------------------------------------------------------------------


def test_a_second_control_plane_migrator_is_refused_rather_than_interleaved(enterprise) -> None:
    """The advisory lock the two-ledger decision required.

    Held from another session so the refusal is real contention and not a same-connection
    re-entry, which PostgreSQL advisory locks would grant.
    """
    from recall.control_plane import CONTROL_PLANE_LOCK_NAME, ConcurrentControlPlaneMigrator

    with psycopg.connect(enterprise.dsn, autocommit=True) as holder:
        got = holder.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))", (CONTROL_PLANE_LOCK_NAME,)
        ).fetchone()
        assert got and got[0], "could not take the lock to simulate a concurrent migrator"
        try:
            with pytest.raises(ConcurrentControlPlaneMigrator):
                enterprise.control.apply_migrations()
        finally:
            holder.execute(
                "SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (CONTROL_PLANE_LOCK_NAME,)
            )
    # Lock released, so the migrator runs again. A lock that is never released would make the
    # refusal above permanent and this line is what distinguishes the two.
    enterprise.control.apply_migrations()


def test_the_migrator_recovers_from_a_partially_recorded_ledger(enterprise) -> None:
    """Crash recovery for the control-plane migrator: re-running converges, it does not blow up.

    The crash shape is a migrator that died between applying one migration's SQL and recording
    the next. Deleting a ledger row reproduces the worse half of it, where the objects exist and
    the ledger says they do not, and the recovery is that every statement in the shipped SQL is
    idempotent (`CREATE TABLE IF NOT EXISTS`, `DROP POLICY IF EXISTS` before `CREATE POLICY`,
    `CREATE INDEX IF NOT EXISTS`). If any statement were not, this test fails with the real error
    rather than a hypothetical one.

    A route is created first and asserted afterwards, so a "recovery" that silently recreated the
    tables and dropped their contents cannot pass.
    """
    enterprise.control.set_route(enterprise.tenant, enterprise.active_id)
    assert enterprise.control.ledger_state().current

    with psycopg.connect(enterprise.dsn, autocommit=True) as conn:
        conn.execute("DELETE FROM recall_schema_versions WHERE version = 1")
    behind = enterprise.control.ledger_state()
    assert not behind.current and 1 in behind.missing

    enterprise.control.apply_migrations()

    assert enterprise.control.ledger_state().current
    route = enterprise.control.route(enterprise.tenant)
    assert route is not None and route.active.generation_id == enterprise.active_id, (
        "recovery recreated the control plane and lost the existing route"
    )


def test_the_control_plane_ledger_reports_drift(enterprise) -> None:
    state = enterprise.control.ledger_state()
    assert state.current, state.describe()

    with psycopg.connect(enterprise.dsn, autocommit=True) as conn:
        conn.execute("UPDATE recall_schema_versions SET checksum = 'tampered' WHERE version = 1")
        try:
            drifted = enterprise.control.ledger_state()
            assert not drifted.current
            assert 1 in drifted.checksum_mismatches
            assert "checksum" in drifted.describe()
        finally:
            conn.execute(
                "UPDATE recall_schema_versions SET checksum = %s WHERE version = 1",
                (dict(
                    (version, checksum)
                    for version, _sql, checksum in ControlPlane.bundled_migrations()
                )[1],),
            )
    assert enterprise.control.ledger_state().current


# --------------------------------------------------------------------------------------------
# 8. Filesystem confinement (the ingest boundary the control plane sits behind)
# --------------------------------------------------------------------------------------------


def test_a_symlink_out_of_the_corpus_is_not_indexed(tmp_path) -> None:
    """Resolved-path confinement, not the glob and not the argument check.

    `**` follows directory symlinks on the Python versions this package supports, so a link
    planted inside an allowed root yields files from anywhere on the filesystem while every
    argument-level check still passes.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("exfiltrated content\n", encoding="utf-8")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "real.md").write_text("legitimate content\n", encoding="utf-8")
    try:
        (corpus / "link").symlink_to(outside, target_is_directory=True)
        (corpus / "direct.md").symlink_to(outside / "secret.md")
    except (OSError, NotImplementedError):
        pytest.skip("this platform does not allow creating symlinks unprivileged")

    # The control first. `candidate_files` returning only `real.md` is evidence of confinement
    # ONLY if the unconfined walk actually reached something outside the root; if the glob never
    # got there, an implementation with no confinement at all would produce the same answer.
    # The escape appears under the LINK's name, not its target's, which is what makes this easy
    # to assert wrongly.
    raw = sorted(corpus.glob("**/*.md"))
    escapes = [p for p in raw if not p.resolve().is_relative_to(corpus.resolve())]
    assert escapes, (
        "the unconfined glob reached no path outside the root on this platform, so this test "
        "cannot distinguish confinement from an empty walk"
    )

    found = {p.name for p in candidate_files(corpus)}
    assert found == {"real.md"}, f"symlink escape reached {sorted(found)}"
    assert not {p.name for p in escapes} & found, "confinement kept an escaping link"
    assert "exfiltrated" not in "".join(p.read_text(encoding="utf-8") for p in candidate_files(corpus))


def test_index_root_confinement_refuses_a_path_outside_the_root(tmp_path, monkeypatch) -> None:
    from recall_mcp.service import index_memory

    root = tmp_path / "root"
    root.mkdir()
    (root / "ok.md").write_text("fine\n", encoding="utf-8")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "secret.md").write_text("VOYAGE_API_KEY=sk-live-EXAMPLE\n", encoding="utf-8")

    monkeypatch.setenv("RECALL_INDEX_ROOT", str(root))
    monkeypatch.delenv("RECALL_ENV", raising=False)

    class _Unreachable:
        def __getattr__(self, name):  # pragma: no cover - reaching this IS the failure
            raise AssertionError(f"confinement let the request reach the store ({name})")

    with pytest.raises(ValueError) as exc:
        index_memory(_Unreachable(), HashingEmbedder(dim=DIM), str(outside / "secret.md"))
    message = str(exc.value)
    assert "sk-live-EXAMPLE" not in message, "the refusal echoed the file's contents"
    assert str(root.resolve()) not in message, "the refusal handed back the resolved root"


def test_the_physical_table_allowlist_refuses_what_isidentifier_allowed() -> None:
    """`str.isidentifier()` was the old gate. These are the three classes it let through."""
    for accepted in ("chunks_g2026_08", "_staging", "c" * 46):
        assert validate_table_name(accepted) == accepted
    for refused, why in (
        ("Chunks_G1", "PostgreSQL folds unquoted uppercase, so the row and the table diverge"),
        ("café_chunks", "non-ASCII is a valid Python identifier"),
        ("c" * 47, "47 + the 17-byte _tenant_isolation suffix exceeds PostgreSQL's 63"),
        ("c" * 64, "PostgreSQL truncates at 63 bytes, so two rows could map to one table"),
        ("chunks; DROP TABLE users", "not an identifier at all"),
        ("", "empty"),
        ("2026_chunks", "leading digit"),
    ):
        assert not refused.isidentifier() or True  # documents that some of these DID pass before
        with pytest.raises(ValueError, match="physical table"):
            validate_table_name(refused)
        assert why  # keeps the reason attached to the case in the source


def test_delete_sources_across_uses_the_same_allowlist(enterprise) -> None:
    """A second, weaker identifier check in the erasure path is a second attack surface."""
    store = enterprise.store(enterprise.active_id, enterprise.active_table)
    try:
        with pytest.raises(ValueError, match="physical table"):
            store.delete_sources_across([store.table, "Bad_Table"], ["/corpus/a.md"])
    finally:
        store.close()


def test_the_suite_would_notice_if_the_dsn_became_privileged() -> None:
    """`role_is_unprivileged` must be able to answer both ways, or it proves nothing.

    `TEST_DSN` is whatever the developer configured, so this asserts the PREDICATE rather than
    the answer: a helper that returned True unconditionally would pass every isolation test in
    this file, and it is checked here against a role whose privilege is known from the catalog.
    """
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        row = conn.execute(
            "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user"
        ).fetchone()
    assert role_is_unprivileged(TEST_DSN) is (not bool(row[0]))


def test_recall_test_dsn_is_the_variable_in_use() -> None:
    """Never `RECALL_DSN`. The suite drops tables; that variable names a real database."""
    configured = os.environ.get("RECALL_TEST_DSN")
    if configured is not None:
        assert configured != os.environ.get("RECALL_DSN")
        assert TEST_DSN == configured
