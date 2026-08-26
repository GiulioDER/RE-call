"""`recall doctor` against a REAL PostgreSQL, because a fake connection is what hid the worst bug.

⛔ **This file exists because its absence was itself a finding.** `tests/test_doctor.py`'s
`_FakeConn` docstring said "The queries themselves are covered by the DB-backed suite", and no such
suite existed. That sentence was the stated justification for testing the module against a fake,
and three defects walked through the gap it left:

* every count returned zero under row-level security, so a full corpus reported as empty;
* a `::regclass` cast aborted the whole report on a table whose name needed quoting;
* a table without a `tenant_id` column raised `UndefinedColumn` out of `run_checks`.

None of them is expressible against a fake, because each is a fact about what PostgreSQL does.

⚠️ **The RLS test needs a NON-SUPERUSER role, and that is the whole point.** `docker-compose.yml`
ships `POSTGRES_USER=recall`, which is the cluster superuser, and a superuser bypasses
`FORCE ROW LEVEL SECURITY`. Running these checks as the default role is exactly how the defect
stayed green in development for the life of the feature. `tests/conftest.py` records the same trap
for the RLS tests it already guards.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
from psycopg import sql

from recall.doctor import run_checks
from recall.schema import LEDGER_TABLE, apply_migrations
from recall.store import DEFAULT_TENANT, PgVectorStore
from recall.types import Chunk
from tests.conftest import TEST_DSN, requires_db

# `requires_db` rather than a bare `pytest.mark.requires_db`: conftest owns the marker AND the
# per-worker DSN rewriting that keeps xdist workers off each other's tables.
pytestmark = requires_db

#: Fixture tables. Named as constants so every statement below composes an identifier from a
#: literal this module owns, never from anything a caller supplies.
CORPUS_TABLE = "doctor_rls_chunks"
EMPTY_TABLE = "doctor_empty_chunks"
NOT_A_CORPUS = "doctor_not_a_corpus"
ODD_NAME = "DoctorMixedCase"
PROBE_ROLE = "doctorprobe"


def _ddl(conn: psycopg.Connection, statement: str, *identifiers: str) -> None:
    """Run a fixture statement, composing identifiers rather than interpolating them."""
    conn.execute(sql.SQL(statement).format(*(sql.Identifier(i) for i in identifiers)))


def _rebuild(table: str, dim: int) -> None:
    """Drop `table` and forget it in the ledger, then migrate it fresh.

    ⚠️ **Dropping a table is not enough to be able to recreate it.** `apply_migrations` records
    what it applied in `recall_schema_migrations`, keyed by target table, so a fixture that dropped
    its table left the ledger claiming those migrations were still applied and the next run found
    `relation ... does not exist`. Clearing both makes the fixture deterministic from ANY prior
    state, which matters here because the session container outlives a single pytest run.
    """
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        _ddl(conn, "DROP TABLE IF EXISTS {} CASCADE", table)
        conn.execute(
            sql.SQL("DELETE FROM {} WHERE target_table = %s").format(sql.Identifier(LEDGER_TABLE)),
            (table,),
        )
    apply_migrations(TEST_DSN, table=table, dim=dim)


def _dsn_as(role: str, password: str) -> str:
    """The same database, reached as another role."""
    parts = urlsplit(TEST_DSN)
    netloc = f"{role}:{password}@{parts.hostname}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _corpus_detail(dsn: str, table: str) -> str:
    report = run_checks(
        dsn=dsn,
        embedder="hashing",
        table=table,
        tenant=DEFAULT_TENANT,
        trust_mode="development",
    )
    return next(c for c in report.checks if c.name == "corpus").detail


@pytest.fixture(scope="module")
def corpus_with_one_chunk():
    """A migrated corpus holding exactly one chunk, plus an unprivileged role that may read it.

    ⚠️ **Module-scoped, and the reason is the migration ledger.** A function-scoped version that
    dropped the table in teardown left the ledger still recording those migrations as applied, so
    the next `apply_migrations` was a no-op and the second test found `relation ... does not
    exist`. Building once per module is simpler than teaching the fixture to unwind a ledger, and
    nothing here mutates the corpus.
    """
    from recall.embeddings import HashingEmbedder

    _rebuild(CORPUS_TABLE, 64)
    embedder = HashingEmbedder(dim=64)
    with PgVectorStore(TEST_DSN, dim=64, table=CORPUS_TABLE, tenant=DEFAULT_TENANT) as store:
        store.check_schema()
        chunk = Chunk("doctorprobe0001", "doctor_probe.md", "the cache ttl is sixty seconds")
        store.upsert([chunk], embedder.embed([chunk.text]))
        assert store.count() == 1, "the fixture must actually hold a row or every test is vacuous"

    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        _ddl(conn, "DROP ROLE IF EXISTS {}", PROBE_ROLE)
        conn.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD 'probe' NOSUPERUSER NOBYPASSRLS").format(
                sql.Identifier(PROBE_ROLE)
            )
        )
        _ddl(conn, "GRANT SELECT ON {} TO {}", CORPUS_TABLE, PROBE_ROLE)
        _ddl(conn, "GRANT USAGE ON SCHEMA public TO {}", PROBE_ROLE)
    try:
        yield CORPUS_TABLE
    finally:
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            _ddl(conn, "REVOKE ALL ON {} FROM {}", CORPUS_TABLE, PROBE_ROLE)
            _ddl(conn, "REVOKE USAGE ON SCHEMA public FROM {}", PROBE_ROLE)
            _ddl(conn, "DROP ROLE IF EXISTS {}", PROBE_ROLE)


def test_an_unprivileged_role_sees_the_same_corpus_a_superuser_does(corpus_with_one_chunk) -> None:
    """⛔ **The red→green test for the worst defect in this command.**

    Measured before the fix, one database holding one chunk, same table and tenant, two roles:

        role recall      (superuser)                OK    corpus  1 chunk(s)       exit 0
        role doctorprobe (NOSUPERUSER NOBYPASSRLS)  FAIL  corpus  holds NO chunks  exit 1

    The chunk table carries `FORCE ROW LEVEL SECURITY` with
    `USING (tenant_id = current_setting('recall.tenant_id', true))`, and two-argument
    `current_setting` returns NULL when the GUC is unset, so the policy hid every row from the
    diagnostic. `PgVectorStore._prepare` sets that GUC; `recall.doctor._connect` did not.

    🔑 The reason this is a P1 rather than a wrong number: the repair it printed was
    `recall index <folder>`, and `recall index` PRUNES sources absent from disk. A user on a
    correctly hardened install was told their full corpus was empty and handed a command that can
    delete it.

    The assertion is deliberately an EQUALITY between the two roles rather than "the unprivileged
    role reports ok". A fix that made both roles wrong in the same direction would satisfy the
    weaker claim.
    """
    as_owner = _corpus_detail(TEST_DSN, corpus_with_one_chunk)
    as_unprivileged = _corpus_detail(_dsn_as(PROBE_ROLE, "probe"), corpus_with_one_chunk)

    assert "1 chunk(s)" in as_owner, as_owner
    assert "1 chunk(s)" in as_unprivileged, (
        "row-level security hid the corpus from the unprivileged role, so the doctor reported a "
        f"populated corpus as empty and would have advised a pruning re-index. Got: "
        f"{as_unprivileged}"
    )


def test_a_table_whose_name_needs_quoting_does_not_abort_the_report(corpus_with_one_chunk) -> None:
    """A mixed-case sibling table used to kill the whole run with a psycopg traceback.

    `'MyTable'::regclass` reparses the string as an identifier and downcases it, so the lookup
    raised `UndefinedTable` — outside the `try` that guarded the count. One oddly-named table
    anywhere in the search path cost the reader every other check.
    """
    # ⚠️ Deliberately an EMPTY corpus. The `::regclass` cast lives in `_populated_corpora`, which
    # only runs when the configured corpus holds nothing — the first version of this test used the
    # populated fixture, never reached the cast, and passed against the pre-fix code. The red-state
    # gate caught it; see .claude/audits/FIX_JOURNAL.md.
    _rebuild(EMPTY_TABLE, 64)
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        _ddl(conn, "CREATE TABLE IF NOT EXISTS {} (tenant_id text)", ODD_NAME)
    try:
        report = run_checks(
            dsn=TEST_DSN,
            embedder="hashing",
            table=EMPTY_TABLE,
            tenant=DEFAULT_TENANT,
            trust_mode="development",
        )
        assert any(c.name == "corpus" for c in report.checks), (
            "one oddly-named sibling table aborted the entire report with a traceback"
        )
        # The populated fixture corpus must still be found and named, which proves the scan ran to
        # completion rather than merely not crashing.
        corpus = next(c for c in report.checks if c.name == "corpus")
        assert corpus_with_one_chunk in corpus.detail, corpus.detail
    finally:
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            _ddl(conn, "DROP TABLE IF EXISTS {}", ODD_NAME)


def test_a_table_that_is_not_a_corpus_is_reported_rather_than_raised() -> None:
    """Naming a non-recall table is the input this command advertises catching, not a crash.

    `_count_for_tenant` hardcodes `WHERE tenant_id = %s`, so a table without that column raised
    `UndefinedColumn` out of `run_checks`. The report got as far as printing `schema ok` and then
    aborted.
    """
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        _ddl(conn, "CREATE TABLE IF NOT EXISTS {} (id int)", NOT_A_CORPUS)
    try:
        report = run_checks(
            dsn=TEST_DSN,
            embedder="hashing",
            table=NOT_A_CORPUS,
            tenant=DEFAULT_TENANT,
            trust_mode="development",
        )
        schema = next(c for c in report.checks if c.name == "schema")
        assert schema.status == "fail"
        assert "not a recall corpus" in schema.detail, schema.detail
    finally:
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            _ddl(conn, "DROP TABLE IF EXISTS {}", NOT_A_CORPUS)


def test_bookkeeping_tables_are_not_reported_as_populated_corpora(corpus_with_one_chunk) -> None:
    """An EMPTY database used to say `These do hold rows: recall_tenant_state/default (1)`.

    A migrated database carries fourteen tenant-scoped tables and only three are corpora. Counting
    on a `tenant_id` column alone meant one row of bookkeeping — which every real install has —
    suppressed the only correct instruction (`recall index <folder>`) and advised pointing
    `--table` at a table that is not a corpus.
    """
    _rebuild(EMPTY_TABLE, 64)
    # ⚠️ The bookkeeping table must actually hold a row. The old shape-blind scan reported only
    # tables with a non-zero count, so on a pristine database it named nothing and the first
    # version of this test passed against the pre-fix code. Every real install has tenant state;
    # this makes the fixture match one.
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute(
            "INSERT INTO recall_tenant_state (tenant_id) VALUES (%s) "
            "ON CONFLICT DO NOTHING",
            (DEFAULT_TENANT,),
        )
    try:
        report = run_checks(
            dsn=TEST_DSN,
            embedder="hashing",
            table=EMPTY_TABLE,
            tenant=DEFAULT_TENANT,
            trust_mode="development",
        )
        corpus = next(c for c in report.checks if c.name == "corpus")
        assert corpus.status == "fail"
        assert "recall_tenant_state" not in corpus.detail, corpus.detail
        assert "recall_generations" not in corpus.detail, corpus.detail
        # The real corpus from the fixture IS a corpus and must still be named, or the filter has
        # been tightened into uselessness rather than into correctness.
        assert corpus_with_one_chunk in corpus.detail, corpus.detail
    finally:
        pass  # the table is left in place; see the ledger note on `corpus_with_one_chunk`


def test_the_doctor_writes_nothing() -> None:
    """The module's central promise, asserted against the catalog rather than by reading the code.

    Compares every table's row count before and after a full run. A diagnostic that writes cannot
    be run to find out what is wrong, because running it changes the answer.
    """

    def snapshot() -> dict[str, int]:
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            names = [
                r[0]
                for r in conn.execute(
                    "SELECT tablename FROM pg_tables WHERE schemaname = current_schema()"
                ).fetchall()
            ]
            counts: dict[str, int] = {}
            for name in names:
                row = conn.execute(
                    sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(name))
                ).fetchone()
                counts[name] = int(row[0]) if row else 0
            return counts

    before = snapshot()
    run_checks(dsn=TEST_DSN, embedder="hashing", trust_mode="development")
    assert snapshot() == before


def test_a_role_that_can_read_almost_nothing_still_gets_the_index_command(
    corpus_with_one_chunk,
) -> None:
    """⛔ **RR1, attempt 3. The first two repairs each fixed one producer and the defect returned.**

    The scan reports two different things about other corpora: what it found, and what it could not
    read. Both were once concatenated into one string, and every caller tested that string for
    truth. So a least-privilege serving role — the role `docs/MIGRATIONS.md` tells operators to use,
    and the role the RLS fix exists for — was told:

        'doctor_empty_chunks'/'default' holds NO chunks ...
        These do hold rows: [not counted: chunks, doctor_rls_chunks]

    "These do hold rows" asserted about tables that were never read, and `recall index <folder>`
    suppressed. Attempt 1 appended the RLS caveat and caused it; attempt 2 moved the RLS caveat out
    and left the "not counted" appender, so it came straight back through the other producer.

    🔑 Attempt 3 changed the TYPE rather than the producers: `_populated_corpora` returns a pair, so
    there is no longer any string that means both "found" and "could not check", and a future caller
    cannot recombine them by accident.

    This role holds SELECT on one EMPTY corpus and nothing else, so every sibling lands in the
    unreadable list — which is exactly the shape that defeated the previous two fixes.
    """
    blind = "doctorblind"
    _rebuild(EMPTY_TABLE, 64)
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        _ddl(conn, "DROP ROLE IF EXISTS {}", blind)
        conn.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD 'blind' NOSUPERUSER NOBYPASSRLS").format(
                sql.Identifier(blind)
            )
        )
        _ddl(conn, "GRANT USAGE ON SCHEMA public TO {}", blind)
        _ddl(conn, "GRANT SELECT ON {} TO {}", EMPTY_TABLE, blind)
    try:
        report = run_checks(
            dsn=_dsn_as(blind, "blind"),
            embedder="hashing",
            table=EMPTY_TABLE,
            tenant=DEFAULT_TENANT,
            trust_mode="development",
        )
        corpus = next(c for c in report.checks if c.name == "corpus")
        assert corpus.status == "fail"
        assert corpus.fix is not None
        assert "index <folder>" in corpus.fix, (
            "a role that could read nothing lost the only instruction that applies to it: "
            + repr(corpus.fix)
        )
        assert "These do hold rows" not in corpus.detail, corpus.detail

        # The facts are not dropped, only moved somewhere they cannot be read as findings.
        names = {c.name for c in report.checks}
        assert "corpus scan" in names or "rls visibility" in names, sorted(names)
    finally:
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            _ddl(conn, "REVOKE ALL ON {} FROM {}", EMPTY_TABLE, blind)
            _ddl(conn, "REVOKE USAGE ON SCHEMA public FROM {}", blind)
            _ddl(conn, "DROP ROLE IF EXISTS {}", blind)
