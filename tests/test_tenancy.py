"""Tenant isolation, at both layers.

Two independent mechanisms, tested separately because they fail independently:

1. **Every query filters on `tenant_id`.** This is the correctness mechanism and it holds for
   any database role, including the superuser the dev container ships.
2. **A row-level-security policy.** This is the CONTROL: it makes a forgotten predicate — in
   future code, a migration script, someone's psql session — return nothing rather than another
   tenant's memories.

Layer 2 is the one that is easy to believe without it being true, because **a superuser bypasses
RLS entirely** and the default `docker-compose.yml` role is a superuser. Testing the policy as
that role would pass vacuously, so the RLS tests below create a dedicated unprivileged role and
connect as it. Without that, this file would be theatre.
"""
from __future__ import annotations

import uuid
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest

from recall.store import PgVectorStore
from recall.types import Chunk

from tests.conftest import TEST_DSN, requires_db


def _vec(i: int, dim: int = 4) -> list[float]:
    v = [0.0] * dim
    v[i % dim] = 1.0
    return v


@pytest.fixture
def tenant_table():
    """A shared table used by two differently-tenanted stores."""
    name = "tn_" + uuid.uuid4().hex[:8]
    yield name
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {name}")


# --------------------------------------------------------------------------------------------
# Layer 1: the application-level predicate
# --------------------------------------------------------------------------------------------


@requires_db
def test_a_tenant_cannot_see_another_tenants_rows(tenant_table):
    a = PgVectorStore(TEST_DSN, dim=4, table=tenant_table, tenant="acme")
    b = PgVectorStore(TEST_DSN, dim=4, table=tenant_table, tenant="globex")
    try:
        a.ensure_schema()
        b.ensure_schema()
        a.upsert([Chunk("x", "a.md", "acme private note")], [_vec(0)])
        b.upsert([Chunk("y", "b.md", "globex private note")], [_vec(1)])

        assert a.count() == 1
        assert b.count() == 1
        assert [h.chunk.text for h in a.query_dense(_vec(0), k=10)] == ["acme private note"]
        assert [h.chunk.text for h in b.query_dense(_vec(1), k=10)] == ["globex private note"]
    finally:
        a.close()
        b.close()


@requires_db
def test_the_same_chunk_id_can_exist_for_two_tenants(tenant_table):
    """Chunk ids derive from the file path, so two tenants indexing the same layout collide.

    With the pre-tenancy single-column primary key, one tenant's re-index OVERWROTE the other's
    row. The key is `(tenant_id, id)` for exactly this reason.
    """
    a = PgVectorStore(TEST_DSN, dim=4, table=tenant_table, tenant="acme")
    b = PgVectorStore(TEST_DSN, dim=4, table=tenant_table, tenant="globex")
    try:
        a.ensure_schema()
        a.upsert([Chunk("same-id", "notes.md", "acme version")], [_vec(0)])
        b.upsert([Chunk("same-id", "notes.md", "globex version")], [_vec(1)])

        assert a.query_dense(_vec(0), k=5)[0].chunk.text == "acme version"
        assert b.query_dense(_vec(1), k=5)[0].chunk.text == "globex version"
    finally:
        a.close()
        b.close()


@requires_db
def test_sparse_search_is_also_tenant_scoped(tenant_table):
    """The full-text leg is a separate SQL statement and a separate chance to forget."""
    a = PgVectorStore(TEST_DSN, dim=4, table=tenant_table, tenant="acme")
    b = PgVectorStore(TEST_DSN, dim=4, table=tenant_table, tenant="globex")
    try:
        a.ensure_schema()
        a.upsert([Chunk("x", "a.md", "quarterly revenue figures")], [_vec(0)])
        b.upsert([Chunk("y", "b.md", "quarterly revenue figures")], [_vec(1)])

        assert len(a.query_sparse("quarterly revenue", k=10)) == 1
        assert len(b.query_sparse("quarterly revenue", k=10)) == 1
    finally:
        a.close()
        b.close()


@requires_db
def test_delete_and_touch_do_not_reach_across_tenants(tenant_table):
    """Write paths matter more than reads: a cross-tenant DELETE is unrecoverable."""
    a = PgVectorStore(TEST_DSN, dim=4, table=tenant_table, tenant="acme")
    b = PgVectorStore(TEST_DSN, dim=4, table=tenant_table, tenant="globex")
    try:
        a.ensure_schema()
        a.upsert([Chunk("x", "shared.md", "acme note")], [_vec(0)])
        b.upsert([Chunk("y", "shared.md", "globex note")], [_vec(1)])

        assert a.delete_sources(["shared.md"]) == 1  # not 2
        assert b.count() == 1
    finally:
        a.close()
        b.close()


@requires_db
def test_freshness_and_supersession_are_tenant_scoped(tenant_table):
    """`newest_indexed_at` drives the staleness report and the supersession map drives verdicts;
    reading either across tenants leaks one tenant's activity into another's answers."""
    a = PgVectorStore(TEST_DSN, dim=4, table=tenant_table, tenant="acme")
    b = PgVectorStore(TEST_DSN, dim=4, table=tenant_table, tenant="globex")
    try:
        a.ensure_schema()
        b.upsert(
            [Chunk("y", "b.md", "globex", {"file": "v2.md", "ord": 0, "supersedes": "v1.md"})],
            [_vec(1)],
        )
        assert a.newest_indexed_at() is None  # b's write must not count as a's freshness
        assert a.supersession_map() == {}
        assert b.supersession_map() == {"v1.md": "v2.md"}
    finally:
        a.close()
        b.close()


@requires_db
def test_default_tenant_keeps_an_existing_single_tenant_install_working(tenant_table):
    """An upgrade must be invisible: rows written before tenancy land in `default`, which is
    also the default tenant, so an existing deployment keeps reading its own data."""
    from recall.store import DEFAULT_TENANT

    s = PgVectorStore(TEST_DSN, dim=4, table=tenant_table)
    other = PgVectorStore(TEST_DSN, dim=4, table=tenant_table, tenant="somebody-else")
    try:
        s.ensure_schema()
        s.upsert([Chunk("x", "a.md", "pre-existing note")], [_vec(0)])
        # Asserted through two differently-tenanted stores rather than a raw connection. A raw
        # connection does not set the tenant GUC, so under REAL row-level security (an
        # unprivileged role) the policy hides every row and the read returns None — which the
        # previous version mistook for "the row is missing". It only worked because the local
        # dev role is a superuser that bypasses RLS.
        assert s._tenant == DEFAULT_TENANT
        assert s.count() == 1
        assert other.count() == 0
    finally:
        s.close()
        other.close()


@requires_db
def test_migrating_a_pre_tenancy_table_preserves_its_rows(tenant_table):
    """The upgrade path: a table created by an older version, then opened by this one."""
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            f"""CREATE TABLE {tenant_table} (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding vector(4),
                indexed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
            )"""
        )
        conn.execute(
            f"INSERT INTO {tenant_table} (id, source, text, embedding) "
            f"VALUES ('old', 'legacy.md', 'a memory from before tenancy', '[1,0,0,0]')"
        )

    s = PgVectorStore(TEST_DSN, dim=4, table=tenant_table)
    try:
        s.ensure_schema()  # must migrate, not fail
        assert s.count() == 1
        assert s.query_dense(_vec(0), k=5)[0].chunk.text == "a memory from before tenancy"
    finally:
        s.close()


@requires_db
def test_a_half_migrated_table_is_repaired_not_reported_done(tenant_table):
    """An interrupted migration must be finished on the next open, not treated as complete.

    The migration is three ALTERs. Interrupted after the DROP CONSTRAINT — crash, restart,
    cancelled statement — the table has `tenant_id` but NO primary key. A completion check that
    only looks for the column reports "already migrated" forever, and every subsequent upsert
    dies on `ON CONFLICT (tenant_id, id)` with no matching constraint. Reproduced here by
    building that exact half-migrated state directly.
    """
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            f"""CREATE TABLE {tenant_table} (
                id TEXT NOT NULL,
                source TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding vector(4),
                indexed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
                tenant_id TEXT NOT NULL DEFAULT 'default'
            )"""
        )  # column added, primary key never re-added: the interrupted state

    s = PgVectorStore(TEST_DSN, dim=4, table=tenant_table)
    try:
        s.ensure_schema()

        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            pkey = conn.execute(
                "SELECT array_length(conkey, 1) FROM pg_constraint "
                "WHERE conrelid = %s::regclass AND contype = 'p'",
                (tenant_table,),
            ).fetchone()
        assert pkey is not None, "half-migrated table was left without a primary key"
        assert pkey[0] == 2, "primary key was not repaired to the composite (tenant_id, id)"

        # The consequence the missing key actually causes: an upsert must work.
        s.upsert([Chunk(id="c1", source="a.md", text="hello", metadata={})], [_vec(0)])
        assert s.count() == 1
    finally:
        s.close()


# --------------------------------------------------------------------------------------------
# Layer 2: row-level security, verified as a role that cannot bypass it
# --------------------------------------------------------------------------------------------


def _role_can_bypass_rls(dsn: str) -> bool:
    with psycopg.connect(dsn, autocommit=True) as conn:
        row = conn.execute(
            "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user"
        ).fetchone()
    return bool(row and row[0])


@pytest.fixture
def unprivileged_dsn(tenant_table):
    """A DSN whose role cannot bypass RLS, so the policy actually applies.

    Three cases, in order, because the previous version handled only the middle one and therefore
    could ONLY run as a superuser — the single configuration in which RLS does not apply. The
    security property was verifiable exactly where it did not matter.

    1. The configured role already cannot bypass RLS (a correctly-configured deployment). Use it
       directly: no new role, no CREATEROLE needed.
    2. It can bypass, but we may create roles — make a throwaway one, as before.
    3. It can bypass and we cannot create roles — SKIP, loudly. Silently passing here would
       report the policy as verified when nothing was tested.
    """
    if not _role_can_bypass_rls(TEST_DSN):
        yield "current_user", TEST_DSN
        return

    role = "rls_" + uuid.uuid4().hex[:8]
    try:
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute(f"CREATE ROLE {role} LOGIN PASSWORD 'pw' NOSUPERUSER NOBYPASSRLS")
            conn.execute(f"GRANT ALL ON SCHEMA public TO {role}")
    except psycopg.errors.InsufficientPrivilege:
        pytest.skip(
            "cannot verify RLS: the configured role bypasses it and lacks CREATEROLE to make "
            "one that does not"
        )
    parts = urlsplit(TEST_DSN)
    dsn = urlunsplit(parts._replace(netloc=f"{role}:pw@{parts.hostname}:{parts.port or 5432}"))
    yield role, dsn
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {tenant_table}")
        conn.execute(f"DROP OWNED BY {role}")
        conn.execute(f"DROP ROLE IF EXISTS {role}")


@requires_db
def test_check_rls_effective_agrees_with_the_roles_actual_privileges():
    """It must report what the CONNECTED ROLE really is, in any environment.

    The previous version asserted `is False` — i.e. it asserted that the local dev role is a
    superuser. That is a property of docker-compose.yml, not of this code, and it failed on a
    correctly-configured deployment where the role is unprivileged. A test that encodes the
    developer's environment fails exactly where the software is meant to run.
    """
    expected = not _role_can_bypass_rls(TEST_DSN)
    s = PgVectorStore(TEST_DSN, dim=4, table="chunks")
    try:
        assert s.check_rls_effective() is expected
    finally:
        s.close()


@requires_db
def test_rls_blocks_a_raw_cross_tenant_query(unprivileged_dsn, tenant_table):
    """The control: a hand-written SELECT with NO tenant predicate returns only this tenant.

    This is the query the application would issue if someone forgot the filter. Under the
    application-level predicate alone it would return every tenant's rows.
    """
    role, dsn = unprivileged_dsn
    a = PgVectorStore(dsn, dim=4, table=tenant_table, tenant="acme")
    b = PgVectorStore(dsn, dim=4, table=tenant_table, tenant="globex")
    try:
        a.ensure_schema()
        assert a.check_rls_effective() is True, "this role must not bypass RLS"
        a.upsert([Chunk("x", "a.md", "acme private note")], [_vec(0)])
        b.upsert([Chunk("y", "b.md", "globex private note")], [_vec(1)])

        # Guard against a VACUOUS pass: if globex's write had silently failed, the unfiltered
        # query below would return one row for the wrong reason and the test would still be
        # green. Prove there really is another tenant's row in this table first.
        assert b.count() == 1
        as_owner = a._with_retry(
            lambda conn: conn.execute(
                f"SELECT count(*) FROM {tenant_table} WHERE tenant_id = 'globex'"
            ).fetchone()
        )
        assert as_owner[0] == 0, "acme can see globex's row even WITH a predicate — RLS is off"

        # deliberately unfiltered — the mistake the policy exists to survive
        rows = a._with_retry(
            lambda conn: conn.execute(f"SELECT text FROM {tenant_table}").fetchall()
        )
        assert [r[0] for r in rows] == ["acme private note"]
    finally:
        a.close()
        b.close()


@requires_db
def test_rls_blocks_writing_a_row_for_another_tenant(unprivileged_dsn, tenant_table):
    """WITH CHECK: a tenant must not be able to plant a row that another tenant will read."""
    role, dsn = unprivileged_dsn
    a = PgVectorStore(dsn, dim=4, table=tenant_table, tenant="acme")
    try:
        a.ensure_schema()
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            a._with_retry(
                lambda conn: conn.execute(
                    f"INSERT INTO {tenant_table} (tenant_id, id, source, text, embedding) "
                    f"VALUES ('globex', 'evil', 's.md', 'planted', '[1,0,0,0]')"
                )
            )
    finally:
        a.close()


@requires_db
def test_a_tampered_policy_predicate_is_repaired_on_the_next_open(tenant_table):
    """The isolation policy must converge on its DEFINITION, not merely on its name.

    `ensure_schema` runs on every store open, which is what makes it a repair mechanism. If it
    only asked "does a policy with this name exist?", a policy whose predicate had been altered —
    by hand, by a migration, by an older version of this code — would be left in place because
    the name matched. A changed predicate here is a changed isolation boundary, so `USING (true)`
    is the whole control silently switched off while the table still reports RLS as enabled.
    """
    s = PgVectorStore(TEST_DSN, dim=4, table=tenant_table)
    try:
        s.ensure_schema()
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute(
                f"ALTER POLICY {tenant_table}_tenant_isolation ON {tenant_table} "
                f"USING (true) WITH CHECK (true)"
            )
            tampered = conn.execute(
                "SELECT pg_get_expr(polqual, polrelid) FROM pg_policy WHERE polname = %s",
                (f"{tenant_table}_tenant_isolation",),
            ).fetchone()[0]
        assert tampered == "true", "fixture failed to tamper with the policy"

        s.ensure_schema()  # the repair

        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            repaired = conn.execute(
                "SELECT pg_get_expr(polqual, polrelid) FROM pg_policy WHERE polname = %s",
                (f"{tenant_table}_tenant_isolation",),
            ).fetchone()[0]
        assert "tenant_id" in repaired and "current_setting" in repaired, (
            f"tampered policy was not repaired: {repaired!r}"
        )
    finally:
        s.close()


@requires_db
def test_reopening_a_correct_table_does_not_recreate_the_policy(tenant_table):
    """`ensure_schema` must recognise its OWN policy as already correct.

    It compares the stored predicate against a literal built in `_enable_rls`, so that literal
    has to match how Postgres deparses the policy it just created — down to the `::text` cast it
    inserts. If it ever stops matching, nothing fails: the comparison simply never succeeds and
    every store open silently DROPs and re-CREATEs the policy, taking an ACCESS EXCLUSIVE lock on
    a live table each time, once per tenant. The unit tests cannot see this — they compare the
    literal against another copy of itself — so it is pinned here against a real server, on the
    policy's OID, which changes if and only if the policy was actually recreated.

    This is also the regression test for a future PostgreSQL upgrade that renders the expression
    differently.
    """
    def policy_oid() -> int | None:
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            row = conn.execute(
                "SELECT oid FROM pg_policy WHERE polname = %s",
                (f"{tenant_table}_tenant_isolation",),
            ).fetchone()
        return row[0] if row else None

    s = PgVectorStore(TEST_DSN, dim=4, table=tenant_table)
    try:
        s.ensure_schema()
        first = policy_oid()
        assert first is not None, "no policy was created"

        s.ensure_schema()
        s.ensure_schema()
        assert policy_oid() == first, (
            "the policy was recreated on a steady-state open — the expected predicate no longer "
            "matches what Postgres stores, so every open now churns it under an exclusive lock"
        )
    finally:
        s.close()


@requires_db
def test_a_policy_narrowed_to_a_role_is_repaired(tenant_table):
    """`ALTER POLICY ... TO <role>` must not survive a re-open.

    The policy is created for PUBLIC and ALL commands. Narrowing its role set leaves the name,
    the predicate and `relrowsecurity` all reporting healthy while the policy no longer applies
    to the role the application actually connects as — an isolation boundary switched off in a
    way that a name-or-predicate check cannot see.
    """
    s = PgVectorStore(TEST_DSN, dim=4, table=tenant_table)
    try:
        s.ensure_schema()
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute(
                f"ALTER POLICY {tenant_table}_tenant_isolation ON {tenant_table} TO recall"
            )
            narrowed = conn.execute(
                "SELECT polroles = '{0}'::oid[] FROM pg_policy WHERE polname = %s",
                (f"{tenant_table}_tenant_isolation",),
            ).fetchone()[0]
        assert narrowed is False, "fixture failed to narrow the policy's role set"

        s.ensure_schema()  # the repair

        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            restored = conn.execute(
                "SELECT polroles = '{0}'::oid[] FROM pg_policy WHERE polname = %s",
                (f"{tenant_table}_tenant_isolation",),
            ).fetchone()[0]
        assert restored is True, "a policy narrowed to one role was left in place"
    finally:
        s.close()
