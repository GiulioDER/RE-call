from __future__ import annotations

import logging
import os
import sys
import uuid
from typing import TYPE_CHECKING, Any

# ─── Import-time environment neutralisation ──────────────────────────────────────────────────────
# This block runs BEFORE the `recall.*` imports below, and that ordering is load-bearing — see
# `_neutralise_deployment_env`. Nothing above it may import anything that can reach
# `recall_mcp.server`, which is why the imports below it are exempted from the import-order rule.

#: Prefixes owning every variable that configures authentication. Matched as a PREFIX rather than
#: enumerated, so a variable added to either family is neutralised without anyone remembering to
#: come back here — the failure mode `recall_mcp.oidc.oidc_non_issuer_env_keys` documents about the
#: hand-written tuple it replaced. `tests/test_mcp_tool_authorization.py` asserts that this rule
#: still covers every key those modules actually read.
_AUTH_ENV_PREFIXES = ("RECALL_AUTH_", "RECALL_OIDC_")

#: Variables that break the IMPORT without carrying an auth-shaped name. The last three are not auth
#: at all: `recall_mcp/server.py` parses them into module constants via `_read_int_env` at import, so
#: `RECALL_PORT=99999`, `RECALL_POOL_SIZE=0` or `RECALL_STATEMENT_TIMEOUT_MS=abc` each raise
#: `ValueError` during collection and take the same five modules to zero tests run. The transport was
#: simply the first one anyone tripped over; the failure CLASS is "read at import", not "auth".
_IMPORT_TIME_ENV_EXACT = (
    "RECALL_TRANSPORT",
    "RECALL_ENV",
    "RECALL_PORT",
    "RECALL_POOL_SIZE",
    "RECALL_STATEMENT_TIMEOUT_MS",
)


def neutralised_env_keys(name: str) -> bool:
    """Whether `name` is cleared by `_neutralise_deployment_env` below."""
    return name in _IMPORT_TIME_ENV_EXACT or name.startswith(_AUTH_ENV_PREFIXES)


def _neutralise_deployment_env() -> None:
    """Clear the deployment env BEFORE any test module imports `recall_mcp.server`.

    `recall_mcp/server.py` builds its server at module scope (`mcp = build_server()`) and parses
    several env vars into module constants, all at import. A developer or CI shell carrying
    `RECALL_TRANSPORT=streamable-http`, a partial OIDC block, or a bad `RECALL_PORT` therefore raises
    during IMPORT, which pytest reports as a collection error: five modules — including the entire
    MCP authorisation suite — run ZERO tests. A security file that silently does not execute is
    worse than one that fails.

    This cannot be a fixture, which is what the previous attempt was. Fixtures run after collection
    and the import that raises happens during it, so the fixture whose docstring named collection
    failure as the thing it prevented had never once been in a position to prevent it. It cannot be
    module-scope code in the test file either: by then another module may already have imported
    `recall_mcp.server` and failed. `conftest.py` is imported before any test module, so this is the
    last point that is still early enough.

    ⚠️ "Early enough" is a property of WHERE THIS CALL SITS, and the first version of it sat below
    `from recall.store import PgVectorStore`. That was luck, not design: the day any `recall.*` module
    acquires a transitive import of `recall_mcp.server`, the clearing happens after the crash it
    exists to prevent, and nothing in the suite would look different. The call now runs above every
    non-stdlib import, and the assertion below pins the precondition instead of trusting it.

    `RECALL_TRANSPORT` makes the point twice over: `server.py` freezes it into the module-level
    `TRANSPORT` constant at import and `build_server()` passes that to `build_auth()`, so a
    `monkeypatch.delenv` would not have undone it even if it had run in time.

    Clearing rather than preserving-and-restoring is deliberate. A test that needs one of these set
    uses `monkeypatch.setenv`, and a suite whose behaviour depends on what the operator happened to
    export is not a suite whose green means anything.
    """
    assert "recall_mcp.server" not in sys.modules, (
        "recall_mcp.server was imported before conftest could neutralise the environment, so an "
        "operator's RECALL_* settings have already been baked into its module constants. Move this "
        "call earlier, or remove whatever now imports it during conftest import."
    )
    for key in [key for key in os.environ if neutralised_env_keys(key)]:
        del os.environ[key]


_neutralise_deployment_env()
# ─── end import-time environment neutralisation ──────────────────────────────────────────────────

import psycopg  # noqa: E402
import pytest  # noqa: E402

from recall.store import PgVectorStore  # noqa: E402
from recall.schema import LEDGER_TABLE, apply_migrations  # noqa: E402

if TYPE_CHECKING:
    # Annotation-only, and deliberately not imported at runtime: the `dev_search*` helpers below
    # defer their own imports, and this block must not undo that by pulling the same modules in at
    # conftest import time. `from __future__ import annotations` above makes the names strings.
    from collections.abc import Callable, Iterator

    from recall.trust import TrustedResult
    from recall_mcp.service import SearchResult

#: Where an unconfigured run points instead of at a real database.
#:
#: Port 1 is reserved and nothing listens on it, on any platform, so every connection attempt is
#: refused immediately. That property is the whole point. `TEST_DSN` is imported directly by test
#: modules and used in `psycopg.connect` and `apply_migrations` calls all over the suite, so a
#: default that merely *usually* gets skipped is not good enough: any path that slips past a
#: `requires_db` mark has to fail loudly rather than quietly connect to somebody else's database.
#: The worst outcome here is a confusing connection error. It is never a dropped table.
_UNCONFIGURED_DSN = "postgresql://recall:recall@127.0.0.1:1/recall"

#: The test database. Deliberately NOT read from `RECALL_DSN`, and deliberately NOT defaulted to
#: the shared dev container on port 5432.
#:
#: These tests DROP TABLES. Two separate ways that has destroyed data that was not ours:
#:
#: 1. `RECALL_DSN` is the variable the README tells users to point at their real database, so
#:    resolving the test DSN from it meant exporting it and running `pytest` destroyed production
#:    data — no flag, no prompt, no way back. The suite reads a dedicated `RECALL_TEST_DSN`, so a
#:    `RECALL_DSN` pointing at anything real is simply never consulted.
#:
#: 2. Defaulting to `localhost:5432` pointed every concurrent checkout at the *same* container.
#:    Two sessions running the suite dropped each other's tables mid-run, and the resulting
#:    failures described the other session's timing rather than anything about the code. There is
#:    no default any more: start a container scoped to this checkout with `scripts/session-db.sh
#:    up`, which prints the `RECALL_TEST_DSN` to export.
TEST_DSN = os.environ.get("RECALL_TEST_DSN") or _UNCONFIGURED_DSN


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


#: One wording, used by the collection-time mark and by every fixture that refuses at setup, so a
#: DB-less run reports the same reason however the test was skipped.
#:
#: The two cases are worded differently on purpose. "Not reachable" and "never configured" look
#: identical in a skip summary but call for opposite responses, and reading several hundred skips
#: that say the container is down, when in fact no DSN was ever exported, wastes the time it takes
#: to go and inspect a healthy container.
DB_UNREACHABLE = (
    "RECALL_TEST_DSN is not set. Start a database scoped to this checkout with "
    '`eval "$(scripts/session-db.sh up)"`. The suite DROPs tables and no longer defaults to the '
    "shared container on port 5432, because concurrent checkouts dropped each other's tables."
    if not os.environ.get("RECALL_TEST_DSN")
    else "pgvector DB at RECALL_TEST_DSN is not reachable (`scripts/session-db.sh status`)"
)

requires_db = pytest.mark.skipif(not _db_available(), reason=DB_UNREACHABLE)


def require_db() -> None:
    """Skip the calling test unless the database is reachable.

    The single refusal site. `@requires_db` is a collection-time optimisation and only protects
    tests whose author remembered it: `test_store_cosines_for.py` and `test_store_query_latency.py`
    both reached this database without it and spent 213 s collecting `ConnectionTimeout` failures
    instead of skipping. Anything here that can touch the database calls this FIRST, so the
    refusal does not depend on anyone remembering.

    Exported deliberately. A module-local fixture that opens its own connection, and there are 22
    of them, is outside this file's reach; calling `require_db()` at the top of one buys it the
    same protection.
    """
    if not _db_available():
        pytest.skip(DB_UNREACHABLE)

#: Fixtures below that hand a test access to the database. Each one REFUSES to run without a
#: reachable DB, which is what makes `@requires_db` an optimisation (skip at collection, before the
#: fixture is ever set up) rather than the thing standing between a missing mark and a 213 s red.
#:
#: Hand-maintained, so `test_requires_db_coverage.py` derives the same set from this file's source
#: and requires both the membership and the refusal to match. A new DB fixture added here without
#: the refusal is exactly the hole this pair exists to close.
DB_BACKED_FIXTURES = ("cli_table", "make_store", "unprivileged_dsn")

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
    # Tests import and call this directly, not only through `unprivileged_dsn`, so the refusal
    # belongs here too rather than only at the fixture that happens to be the usual caller.
    require_db()
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
    require_db()
    if role_is_unprivileged(TEST_DSN):
        return TEST_DSN

    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        owner_row = conn.execute("SELECT current_user").fetchone()
        if owner_row is None:  # pragma: no cover - SELECT current_user always returns a row
            raise RuntimeError("could not determine the connected role")
        owner = owner_row[0]
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

    parts: dict[str, Any] = dict(conninfo_to_dict(TEST_DSN))
    parts["user"] = UNPRIVILEGED_ROLE
    parts["password"] = _UNPRIVILEGED_PASSWORD
    dsn = make_conninfo(**parts)
    if not role_is_unprivileged(dsn):  # pragma: no cover - provisioning contradicted itself
        pytest.fail(f"provisioned role {UNPRIVILEGED_ROLE} is still privileged")
    return dsn


@pytest.fixture(scope="session", autouse=True)
def _bootstrap_default_test_schema() -> Iterator[None]:
    """Provision the default MCP table explicitly for subprocess/server integration tests."""
    if not _db_available():
        yield
        return
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks CASCADE")
        ledger = conn.execute("SELECT to_regclass(%s)", (LEDGER_TABLE,)).fetchone()
        if ledger is not None and ledger[0]:
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
    reason='needs the fastembed extra (pip install "recall-rag[fastembed]")',
)


def _openai_available() -> bool:
    try:
        import openai  # noqa: F401
        return True
    except ImportError:
        return False


#: The same rule as `requires_fastembed`, for the `extract` and `bench` extras that carry the
#: `openai` SDK. It is stated twice because it was learned twice: six tests in
#: `test_truth_extraction_engine_openai.py` reached the real package with no guard, so the
#: deliberate absence of the extras turned the `test` and `floor` jobs red on master itself
#: rather than skipping. Note what must NOT be guarded this way — that file's
#: `test_importing_the_package_does_not_import_openai` is only meaningful WITHOUT the extra, so a
#: module-level skip would silently retire the test that proves the import stays lazy.
requires_openai = pytest.mark.skipif(
    not _openai_available(),
    reason='needs the extract extra (pip install "recall-rag[extract]")',
)


@pytest.fixture(autouse=True)
def _isolate_recall_logger() -> Iterator[None]:
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
def make_store() -> Iterator[Callable[[int], PgVectorStore]]:
    """Hands out throwaway tables, and REFUSES to run at all when there is no database.

    The refusal is the point. `@requires_db` is the convention, but a convention only protects the
    tests whose author remembered it: `test_store_cosines_for.py` and `test_store_query_latency.py`
    both requested this fixture without the mark, and instead of skipping they spent 213 s
    collecting `psycopg.errors.ConnectionTimeout` failures. Skipping HERE makes that impossible
    rather than merely discouraged, because every test that reaches this database through conftest
    must come through this line, whether or not anyone marked it.
    """
    require_db()
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
def cli_table() -> Iterator[str]:
    """A uuid-named table for CLI end-to-end tests, dropped afterwards.

    The CLI tests used to run against the default `chunks` table and `DROP TABLE IF EXISTS
    chunks` to isolate themselves — which is what made `pytest` destructive against whatever
    database was configured. A throwaway table per test isolates without dropping anything a
    user owns.
    """
    require_db()  # see `make_store`: the refusal is what makes the mark optional
    name = "cli_" + uuid.uuid4().hex[:8]
    apply_migrations(TEST_DSN, table=name, dim=64)
    yield name
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {name}")
        conn.execute(f"DELETE FROM {LEDGER_TABLE} WHERE target_table = %s", (name,))


def dev_search(*args: Any, **kwargs: Any) -> TrustedResult:
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


def dev_search_memory(*args: Any, **kwargs: Any) -> SearchResult:
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


def dev_search_uncalibrated(*args: Any, **kwargs: Any) -> TrustedResult:
    """`trusted_search` in development mode with NO threshold: every hit comes back unverified.

    For tests whose subject is the absence of a calibration, rather than tests that merely need
    to get past it. See `dev_search`.
    """
    from recall.trust import trusted_search
    from recall.trust_policy import TrustPolicy

    kwargs.setdefault("policy", TrustPolicy.development())
    return trusted_search(*args, **kwargs)
