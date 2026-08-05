from __future__ import annotations

import logging
import os
import uuid

import psycopg
import pytest

from recall.store import PgVectorStore
from recall.schema import LEDGER_TABLE, apply_migrations

#: The local dev database from docker-compose.yml — the same one the README quickstart uses.
_LOCAL_DEV_DSN = "postgresql://recall:recall@localhost:5432/recall"

#: The test database. Deliberately NOT read from `RECALL_DSN`.
#:
#: These tests DROP TABLES. `RECALL_DSN` is the variable the README tells users to point at their
#: real database, so resolving the test DSN from it meant that exporting it and running `pytest`
#: destroyed production data — no flag, no prompt, no way back. The suite now reads a dedicated
#: `RECALL_TEST_DSN` and otherwise falls back to the local dev container, so a `RECALL_DSN`
#: pointing at anything real is simply never consulted.
TEST_DSN = os.environ.get("RECALL_TEST_DSN", _LOCAL_DEV_DSN)


def _reject_unsafe_test_dsn() -> None:
    """Refuse to run destructive tests against a database that might not be disposable.

    Two ways to get here: pointing `RECALL_TEST_DSN` at the same database as `RECALL_DSN`, or
    pointing it at a remote host. Both are refused at import time rather than discovered
    afterwards, because the damage is not recoverable from a test report.
    """
    from urllib.parse import urlsplit

    from recall.store import _is_local_host

    configured = os.environ.get("RECALL_TEST_DSN")
    if configured is None:
        return
    if configured == os.environ.get("RECALL_DSN"):
        raise RuntimeError(
            "RECALL_TEST_DSN is the same database as RECALL_DSN. These tests DROP TABLES — "
            "point RECALL_TEST_DSN at a throwaway database."
        )
    host = (urlsplit(configured).hostname or "").lower()
    if not _is_local_host(host) and not os.environ.get("RECALL_ALLOW_REMOTE_TEST_DB"):
        raise RuntimeError(
            f"RECALL_TEST_DSN points at the non-local host {host!r}. These tests DROP TABLES — "
            "set RECALL_ALLOW_REMOTE_TEST_DB=1 only if that database is genuinely disposable."
        )


_reject_unsafe_test_dsn()


def _db_available() -> bool:
    try:
        psycopg.connect(TEST_DSN, connect_timeout=2).close()
        return True
    except Exception:
        return False


requires_db = pytest.mark.skipif(
    not _db_available(),
    reason="pgvector DB not reachable (run `docker compose up -d`)",
)

#: Role provisioned by `unprivileged_dsn` when `TEST_DSN` turns out to be privileged.
UNPRIVILEGED_ROLE = "recall_rls_probe"
_UNPRIVILEGED_PASSWORD = "recall_rls_probe"  # noqa: S105 - throwaway local test role


def role_is_unprivileged(dsn: str) -> bool:
    """True when this DSN's role is provably neither superuser nor BYPASSRLS.

    Fails CLOSED. The earlier form was `not (row and row[0])`, which reported a MISSING row as
    unprivileged: absence of evidence read as evidence of safety, in the one helper whose entire
    job is to prove a negative capability. Every RLS assertion in the suite rests on this answer,
    so an unknown privilege must never be the safe one.
    """
    with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as conn:
        row = conn.execute(
            "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user"
        ).fetchone()
    if row is None:  # pragma: no cover - current_user is always in pg_roles
        raise RuntimeError("could not determine whether current_user is privileged")
    return not row[0]


@pytest.fixture(scope="session")
def unprivileged_dsn() -> str:
    """A DSN whose role is provably neither superuser nor `BYPASSRLS`.

    Row level security is INERT for a superuser and for a `BYPASSRLS` role. An isolation test run
    on such a role passes whether or not a single policy exists, which makes it a check that
    cannot fail, the most expensive kind, because it reads as protection. `docker-compose.yml`
    ships `POSTGRES_USER=recall`, which IS the cluster superuser, so the default developer
    configuration is exactly the one where these assertions mean nothing.

    Two paths, and the caller cannot tell them apart:

    - `RECALL_TEST_DSN` already points at an unprivileged role. Use it as is.
    - It points at a privileged one. Provision `recall_rls_probe` (`NOSUPERUSER NOBYPASSRLS`),
      grant it the connected role's privileges so it can reach objects that role already owns,
      and hand back a DSN for it.

    Provisioning needs `CREATEROLE` or superuser. A role that has neither, and is itself
    privileged, cannot produce an unprivileged DSN at all; that skips, rather than quietly
    falling back to the privileged DSN, which is how this check would become vacuous again.

    Every test that uses this fixture still asserts the property itself. A fixture named
    "unprivileged" is a claim; `SELECT rolsuper OR rolbypassrls` is evidence.
    """
    if not _db_available():
        pytest.skip("pgvector DB not reachable")
    if role_is_unprivileged(TEST_DSN):
        return TEST_DSN

    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        owner = conn.execute("SELECT current_user").fetchone()[0]
        may_provision = conn.execute(
            "SELECT rolsuper OR rolcreaterole FROM pg_roles WHERE rolname = current_user"
        ).fetchone()
        if not (may_provision and may_provision[0]):
            pytest.skip(
                f"RECALL_TEST_DSN role {owner!r} bypasses RLS and cannot create a role to "
                "stand in for one; point RECALL_TEST_DSN at an unprivileged role"
            )
        exists = conn.execute(
            "SELECT 1 FROM pg_roles WHERE rolname = %s", (UNPRIVILEGED_ROLE,)
        ).fetchone()
        if not exists:
            conn.execute(
                f"CREATE ROLE {UNPRIVILEGED_ROLE} LOGIN PASSWORD '{_UNPRIVILEGED_PASSWORD}' "
                f"NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE"
            )
        # Membership in the connected role, so the probe reaches the objects that role owns,
        # but WITH SET FALSE so it cannot `SET ROLE` back into it. That distinction is the whole
        # security of this fixture: on the shipped docker-compose the connected role IS the
        # cluster superuser, and a member that can SET ROLE to a superuser is a superuser, for
        # whom every RLS policy below is inert. `rolsuper` on the probe stays false either way,
        # so the fixture's own self-check cannot see the difference; the grant option is what
        # makes the check honest.
        #
        # FORCE ROW LEVEL SECURITY binds an owner too, so inheriting ownership privileges does
        # not reopen the bypass on the tables under test.
        try:
            conn.execute(f'GRANT "{owner}" TO {UNPRIVILEGED_ROLE} WITH SET FALSE')
        except psycopg.errors.SyntaxError:
            # WITH SET requires PostgreSQL 16. On 15 and earlier, membership always implies
            # SET ROLE, so refuse rather than silently provisioning an escalatable probe.
            pytest.skip(
                "this PostgreSQL is too old for GRANT ... WITH SET FALSE; point "
                "RECALL_TEST_DSN at an unprivileged role instead"
            )
        conn.execute(f"GRANT USAGE, CREATE ON SCHEMA public TO {UNPRIVILEGED_ROLE}")

    parts = conninfo_to_dict(TEST_DSN)
    parts["user"] = UNPRIVILEGED_ROLE
    parts["password"] = _UNPRIVILEGED_PASSWORD
    dsn = make_conninfo(**parts)
    if not role_is_unprivileged(dsn):  # pragma: no cover - provisioning contradicted itself
        pytest.fail(f"provisioned role {UNPRIVILEGED_ROLE} is still privileged")
    return dsn


@pytest.fixture(scope="session", autouse=True)
def _bootstrap_default_test_schema():
    """Provision the default MCP table explicitly for subprocess/server integration tests."""
    if not _db_available():
        yield
        return
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks CASCADE")
        if conn.execute("SELECT to_regclass(%s)", (LEDGER_TABLE,)).fetchone()[0]:
            conn.execute(f"DELETE FROM {LEDGER_TABLE} WHERE target_table = 'chunks'")
    apply_migrations(TEST_DSN, table="chunks", dim=64)
    yield


def _fastembed_available() -> bool:
    try:
        import fastembed  # noqa: F401
        return True
    except ImportError:
        return False


#: The `floor` and `test` CI jobs install WITHOUT the optional extras, deliberately — their absence
#: is what proves the import guards work (see pyproject's mypy overrides for the same reasoning).
#: A test that needs a real local embedder must therefore skip rather than fail, or it turns an
#: intentional CI condition into a red build.
requires_fastembed = pytest.mark.skipif(
    not _fastembed_available(),
    reason="needs the fastembed extra (pip install recall[fastembed])",
)


@pytest.fixture(autouse=True)
def _isolate_recall_logger():
    """Restore the `recall` logger around every test.

    `configure_logging()` is an entry-point function: it attaches a handler and sets
    `propagate = False`, deliberately, so that a record cannot reach a root handler and be
    re-emitted onto stdout — which on the MCP stdio transport would corrupt JSON-RPC. That is
    right in a process and wrong in a test session, because it is global and never undone: the
    moment one test calls it, `caplog` stops seeing records from `recall.*` for every test that
    follows, since caplog captures by propagation to the root.

    The symptom is an order-dependent failure — each affected test passes alone and fails in the
    suite — which is why it went unnoticed: the pytest version in use happened to order or handle
    capture in a way that hid it, while the DECLARED floor (`pytest>=8`) did not. Snapshotting
    here fixes the isolation itself rather than pinning a version that happens to mask it.
    """
    logger = logging.getLogger("recall")
    saved = (list(logger.handlers), logger.level, logger.propagate, logger.disabled)
    try:
        yield
    finally:
        logger.handlers[:] = saved[0]
        logger.setLevel(saved[1])
        logger.propagate = saved[2]
        logger.disabled = saved[3]


@pytest.fixture
def make_store():
    created: list[PgVectorStore] = []

    def _factory(dim: int) -> PgVectorStore:
        table = "t_" + uuid.uuid4().hex[:8]
        apply_migrations(TEST_DSN, table=table, dim=dim)
        store = PgVectorStore(TEST_DSN, dim=dim, table=table)
        store.check_schema()
        created.append(store)
        return store

    yield _factory

    for store in created:
        if store._closed:
            # A test may close its store deliberately (close() is sticky). Still drop the
            # table — skipping teardown entirely would leak a uuid-named table per run.
            with psycopg.connect(TEST_DSN, autocommit=True) as conn:
                conn.execute(f"DROP TABLE IF EXISTS {store.table}")
                conn.execute(
                    f"DELETE FROM {LEDGER_TABLE} WHERE target_table = %s", (store.table,)
                )
            continue
        store.drop_table()
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute(f"DELETE FROM {LEDGER_TABLE} WHERE target_table = %s", (store.table,))
        store.close()


@pytest.fixture
def cli_table():
    """A uuid-named table for CLI end-to-end tests, dropped afterwards.

    The CLI tests used to run against the default `chunks` table and `DROP TABLE IF EXISTS
    chunks` to isolate themselves — which is what made `pytest` destructive against whatever
    database was configured. A throwaway table per test isolates without dropping anything a
    user owns.
    """
    name = "cli_" + uuid.uuid4().hex[:8]
    apply_migrations(TEST_DSN, table=name, dim=64)
    yield name
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {name}")
        conn.execute(f"DELETE FROM {LEDGER_TABLE} WHERE target_table = %s", (name,))


def dev_search(*args, **kwargs):
    """`trusted_search` in development mode, for tests that exercise UNCALIBRATED retrieval.

    Strict is the library's production default as of the strict-trust work, so a plain
    `PgVectorStore` with no generation and no published calibration now refuses rather than
    scoring against the 0.50 floor. That is the point of the change, not a side effect of it.

    Tests whose subject is the retrieval, supersession or provenance machinery still have to
    reach that machinery, so they opt into development mode HERE, visibly, one call at a time.
    Reading `dev_search` in a test is meant to prompt the question "why is this one not strict?",
    and the answer is always the same: the test is about retrieval mechanics, not trust policy.
    Tests whose subject IS trust policy call `trusted_search` directly and assert the refusal.

    Two variants, because "development mode" and "no threshold at all" are different states and
    conflating them is what makes these tests confusing to read:

    - `dev_search` also supplies an explicit threshold, so the verdict machinery (superseded,
      expired, ambiguous_supersession, low_confidence) actually runs. This is what tests of
      retrieval mechanics need, and it reproduces exactly the behaviour they were written
      against, since the threshold is the same 0.50 the library used to fall back to silently.
    - `dev_search_uncalibrated` supplies none, so every hit degrades to `unverified`. Use it only
      when the absence of a calibration is itself the subject of the test.
    """
    from recall.calibration import Calibration
    from recall.guards import DEFAULT_GAP_THRESHOLD
    from recall.trust import trusted_search
    from recall.trust_policy import TrustPolicy

    kwargs.setdefault("policy", TrustPolicy.development())
    kwargs.setdefault(
        "calibration",
        Calibration(embedder="test-development", threshold=DEFAULT_GAP_THRESHOLD),
    )
    return trusted_search(*args, **kwargs)


def dev_search_memory(*args, **kwargs):
    """`recall_mcp.service.search_memory` in development mode with an explicit threshold.

    The MCP service defaults to strict for the same reason the library does, so these tests have
    to opt out the same way. See `dev_search`; this is the service-layer twin of it.
    """
    from recall.calibration import Calibration
    from recall.guards import DEFAULT_GAP_THRESHOLD
    from recall.trust_policy import TrustPolicy
    from recall_mcp.service import search_memory

    kwargs.setdefault("policy", TrustPolicy.development())
    kwargs.setdefault(
        "calibration",
        Calibration(embedder="test-development", threshold=DEFAULT_GAP_THRESHOLD),
    )
    return search_memory(*args, **kwargs)


def dev_search_uncalibrated(*args, **kwargs):
    """`trusted_search` in development mode with NO threshold: every hit comes back unverified.

    For tests whose subject is the absence of a calibration, rather than tests that merely need
    to get past it. See `dev_search`.
    """
    from recall.trust import trusted_search
    from recall.trust_policy import TrustPolicy

    kwargs.setdefault("policy", TrustPolicy.development())
    return trusted_search(*args, **kwargs)
