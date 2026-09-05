from __future__ import annotations

import functools
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
#: `RECALL_TRUST_MODE` is the odd one out and the most important to clear. Every other key here
#: announces itself by raising during collection. This one does not: exported as `development` it
#: makes `recall_mcp.server.TRUST_POLICY` relaxed at import, and the suite then runs green against a
#: server whose trust gate is open. README.md, CLAUDE.md and docs/USING_WITH_CLAUDE.md all tell
#: developers to export exactly this variable for local work, so the shell that is most likely to
#: run the suite is the one most likely to have it set. A suite whose behaviour depends on what the
#: operator happened to export is not a suite whose green means anything.
_IMPORT_TIME_ENV_EXACT = (
    "RECALL_TRANSPORT",
    "RECALL_ENV",
    "RECALL_PORT",
    "RECALL_POOL_SIZE",
    "RECALL_STATEMENT_TIMEOUT_MS",
    "RECALL_TRUST_MODE",
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
from recall.schema import (  # noqa: E402
    GLOBAL_MIGRATION_TARGET,
    LEDGER_TABLE,
    apply_migrations,
    load_migrations,
)

if TYPE_CHECKING:
    # Annotation-only, and deliberately not imported at runtime: the `dev_search*` helpers below
    # defer their own imports, and this block must not undo that by pulling the same modules in at
    # conftest import time. `from __future__ import annotations` above makes the names strings.
    from collections.abc import Callable, Iterator

    from recall.trust import TrustedResult
    from recall_mcp.service import SearchResult

#: Where an unconfigured run points instead of at a real database.
#:
#: Port 1 is reserved and nothing listens on it, so a connection there is refused without reaching
#: any database. That property is the whole point. `TEST_DSN` is imported directly by test modules
#: and used in `psycopg.connect` and `apply_migrations` calls all over the suite, so a default that
#: merely *usually* gets skipped is not good enough: any path that slips past a `requires_db` mark
#: has to fail rather than quietly connect to somebody else's database. The worst outcome here is a
#: connection error. It is never a dropped table.
#:
#: `connect_timeout` is part of the constant rather than left to each call site, and that is not
#: decoration. "Refused" is not the same as "refused promptly": measured on the Windows host this
#: is developed on, a refusal on 127.0.0.1 takes ~2.0 s, and libpq's default is to wait
#: indefinitely, which turned a `psycopg.connect` with no timeout into a **130 second** stall. The
#: suite has ~110 `psycopg.connect(TEST_DSN...)` sites that pass no timeout of their own; carrying
#: the bound in the DSN is what stops one forgotten `requires_db` mark from becoming a hang that
#: reads like a deadlock in the code under test.
_UNCONFIGURED_DSN = "postgresql://recall:recall@127.0.0.1:1/recall?connect_timeout=2"

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


def _isolate_xdist_worker(dsn: str) -> str:
    """Give each `pytest -n` worker its own DATABASE inside the checkout's own container.

    Parallel workers are separate processes with separate pytest sessions, so every guarantee this
    file makes about isolation holds WITHIN a worker and none of it holds ACROSS workers. Three
    things collide otherwise, and all three are silent:

    1. `chunks`. The session bootstrap below creates it once per session, which under `-n` means
       once per worker against the same database, and `tests/test_wizard_database.py` DROPs and
       rebuilds it mid-run because `wizard.database.probe_database` inspects it by name. A worker
       reading `chunks` while another rebuilds it fails with `relation "chunks" does not exist`,
       in a test that has nothing to do with the wizard.
    2. The migration ledger. `apply_migrations` is idempotent by consulting it, so two workers
       racing on the same row can leave a table unbuilt and no error behind.
    3. `recall_rls_probe`, provisioned by `unprivileged_dsn`. A role is CLUSTER-wide, so a
       check-then-create in two workers at once raises `DuplicateObject` in one of them.

    A database per worker removes all three at once, rather than fixing them one at a time and
    waiting to discover the fourth. It is cheap: `CREATE DATABASE` off the empty template, with
    `CREATE EXTENSION IF NOT EXISTS vector` arriving in migration 0001 like everywhere else.

    ⚠️ **`RECALL_TEST_DSN` is rewritten in the environment too, not just here.** ~30 tests spawn a
    subprocess (`python -m recall.cli`, the MCP server) that reads the variable itself, and a
    subprocess left on the shared database would reintroduce exactly the collisions above, from
    the one place this module cannot see.

    The per-worker databases are left behind on purpose. They live in a container this checkout
    started and `scripts/session-db.sh down` removes, so dropping them per run would buy nothing
    and would need a connection-terminating `DROP DATABASE` that could hit a worker still finishing.
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if not worker or not os.environ.get("RECALL_TEST_DSN"):
        return dsn

    from urllib.parse import urlsplit, urlunsplit

    # `gw0`, `gw1`, ... from xdist. Sanitised rather than trusted: it lands in DDL that cannot be
    # parameterised, and a database name is not a place to find out what xdist calls its workers.
    suffix = "".join(c for c in worker if c.isalnum())[:16] or "w"

    # ⚠️ **Rewritten as a URL, NOT through `psycopg.conninfo.make_conninfo`.** Both spellings are
    # valid libpq and psycopg accepts either, but the suite does not: five tests take
    # `TEST_DSN` apart with `urlsplit`, `rsplit("/", 1)` or an f-string to build a DSN for a role
    # or a database of their own, and a keyword-form DSN turns those into
    # `invalid connection option "//recall_serve_x:pw@None:5432/user"` — a failure that names the
    # test's own string handling and says nothing about the worker isolation that caused it.
    # Measured: exactly that, in `test_schema_migrations`, `test_tenancy`, `test_store` and
    # `test_beam_transfer_index_guards`, on the first parallel run.
    split = urlsplit(dsn)
    if not split.scheme.startswith("postgres"):
        return dsn
    base = (split.path or "/recall").lstrip("/") or "recall"
    isolated_db = f"{base}_{suffix}"

    try:
        with psycopg.connect(dsn, autocommit=True) as conn:
            exists = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (isolated_db,)
            ).fetchone()
            if not exists:
                conn.execute(f'CREATE DATABASE "{isolated_db}"')
    except psycopg.errors.DuplicateDatabase:  # pragma: no cover - two workers, same instant
        pass
    except psycopg.errors.InsufficientPrivilege as exc:
        # Deliberately fatal, and deliberately NOT a fallback to the shared database. Falling back
        # would hand every worker the same database and reintroduce the collisions this exists to
        # prevent, as a green run with occasional inexplicable failures. Serial needs no such
        # privilege, so the fix is to drop `-n` or to use a role that has CREATEDB.
        raise RuntimeError(
            f"the RECALL_TEST_DSN role cannot CREATE DATABASE, which `pytest -n` needs so that "
            f"worker {worker} does not share one with the others. Run the suite serially, or "
            f"point RECALL_TEST_DSN at a role with CREATEDB."
        ) from exc
    except psycopg.OperationalError:
        # No database reachable at all. Hand back the DSN unchanged and let `require_db` skip with
        # its own message, which names the reason; failing here would replace it with a worse one.
        return dsn

    isolated = urlunsplit(split._replace(path=f"/{isolated_db}"))
    os.environ["RECALL_TEST_DSN"] = isolated
    return isolated


TEST_DSN = _isolate_xdist_worker(TEST_DSN)


#: How long one probe waits, and how many probes are tried before the answer is believed.
#:
#: ⚠️ **What this replaces is a 2 second timeout tried once PER TEST, and its failure mode was a
#: false GREEN.** `require_db()` re-probed on every test reaching a DB fixture, so on a loaded
#: machine a probe that lost the race turned a database test into a SKIP. Nothing failed and
#: nothing said so, which is the signature `CLAUDE.md` warns about under "read the skip count
#: before calling a run green".
#:
#: Measured 2026-08-21, same commit, same machine, same container: a run with nothing else
#: competing reported `6209 passed, 34 skipped`; a run sharing the host with a type check and a
#: doc gate reported `6176 passed, 88 skipped`. 21 of that difference is tests added in between,
#: and **54 is tests that had passed and now skipped**.
#:
#: Both numbers only apply when a DSN is CONFIGURED. With `RECALL_TEST_DSN` unset there is nothing
#: to wait for: the probe goes to a reserved port, is refused immediately, and retrying a refusal
#: three times would just triple a cost every DB-less run already pays.
_PROBE_TIMEOUT_SECONDS = 10
_PROBE_ATTEMPTS = 3


def _is_timeout(exc: BaseException) -> bool:
    """Did this connection attempt run out of time, as opposed to being refused outright?

    Matched on the message as well as the class. `psycopg.errors.ConnectionTimeout` is the precise
    type, and it is not the only way libpq reports the condition: a wrapped `OperationalError`
    carrying "timeout expired" arrives from the same cause through a different path. Missing one
    would silently turn the retry off for the exact case it exists to handle.
    """
    return isinstance(exc, psycopg.errors.ConnectionTimeout) or "timeout" in str(exc).lower()


@functools.cache
def _probe_database() -> str | None:
    """`None` when the database answered, otherwise a one-line diagnosis of why it did not.

    **Cached, and the cache is the fix rather than an optimisation.** The answer is decided once,
    at conftest import, before the suite has had a chance to load the machine, so every test in a
    run agrees about whether a database exists. Deciding it per test made the skip count a function
    of what else the host happened to be doing at that second.

    Retried, because one decision for a whole run must not itself be a coin flip.

    The failure is RETURNED rather than swallowed. `_db_available()` reduces it to a boolean for
    the callers that only need one, but the text reaches the skip reason, and that is the
    difference between "nothing is listening" and "something is there and too slow to answer".
    Those want opposite responses from whoever reads the report, and the old probe collapsed them.
    """
    # A shorter wait when nobody asked for a database. `_UNCONFIGURED_DSN` points at a reserved
    # port so the answer is normally instant, but a host that DROPs rather than refuses turns that
    # into a full wait, and every database-less run on every contributor's machine would pay it.
    configured = os.environ.get("RECALL_TEST_DSN") is not None
    timeout = _PROBE_TIMEOUT_SECONDS if configured else 2
    failure = "no attempt was made"
    for _attempt in range(_PROBE_ATTEMPTS):
        try:
            psycopg.connect(TEST_DSN, connect_timeout=timeout).close()
            return None
        except Exception as exc:  # noqa: BLE001 - any failure to connect is the same answer here
            # The class name as well as the message: `str(exc)` is the empty string for several
            # psycopg errors, and a reason ending in "Probe saw: " reads as a mechanism that ran
            # and found nothing worth saying. Collapsed to one line because this text becomes a
            # pytest skip reason, and psycopg's connection errors span lines, which would break the
            # `-rs` report into fragments that no longer read as one cause.
            failure = " ".join(f"{type(exc).__name__}: {exc}".split())
            if not _is_timeout(exc):
                # ⛔ Retry AMBIGUITY, never certainty. A refused connection is a complete answer:
                # nothing is listening on that port, and asking twice more cannot change it. Only a
                # timeout is the ambiguous case this retry exists for, and it is the only one a
                # loaded host manufactures.
                #
                # Not merely tidy. `test_requires_db_coverage.py` deliberately runs a subprocess
                # against a dead port and requires it to skip cleanly; retrying that refusal would
                # triple the fixed cost of the guard that protects the whole fixture set.
                break
    return failure


def _db_available() -> bool:
    return _probe_database() is None


#: One wording, used by the collection-time mark and by every fixture that refuses at setup, so a
#: DB-less run reports the same reason however the test was skipped.
#:
#: Deliberately a single constant rather than one message per cause. Branching it on
#: `RECALL_TEST_DSN` reads better in isolation and breaks the guard in
#: `test_requires_db_coverage.py` that compares this process's constant against a subprocess run
#: with an explicit DSN: the two processes would compute different text and the comparison would
#: fail for a reason that has nothing to do with what it is checking. One wording, both causes
#: named, no environment read at import time.
DB_UNREACHABLE = (
    "pgvector DB not reachable: RECALL_TEST_DSN is unset or nothing is listening on it. "
    'Start one scoped to this checkout with `eval "$(scripts/session-db.sh up)"`. '
    "The suite DROPs tables, so it has no default DSN. It used to fall back to the shared "
    "`docker compose up -d` container on port 5432, and concurrent checkouts dropped each "
    "other's tables."
)

def db_unreachable_reason() -> str:
    """`DB_UNREACHABLE`, plus what the probe actually saw.

    The constant stays the PREFIX and is never rebuilt, because `test_requires_db_coverage.py`
    compares it against a subprocess's output and the comment above explains why it must not branch
    on the environment. What is appended is a diagnosis, not a second wording: a refused connection
    and a connection that timed out are different states, and a report that spells both "not
    reachable" cannot tell a machine with no database from a machine too busy to answer one.
    """
    probe = _probe_database()
    return DB_UNREACHABLE if probe is None else f"{DB_UNREACHABLE} Probe saw: {probe}"


requires_db = pytest.mark.skipif(not _db_available(), reason=db_unreachable_reason())


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
        pytest.skip(db_unreachable_reason())

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


def restore_default_chunks_table() -> None:
    """Put the shared `chunks` table back exactly as the session bootstrap leaves it.

    ⚠️ **Any test that replaces `chunks` MUST call this afterwards.** This file already documents
    the same lesson thirty lines below, for the CLI tests, and it was reintroduced anyway. `chunks`
    is SHARED: the session fixture creates it once at dim 64 and a dozen integration tests assume
    it. A test that drops it to build its own leaves everything that runs later failing with
    `relation "chunks" does not exist` — and only when the random order puts it first, so it passes
    locally, passes on a re-run, and fails in CI.

    Measured: `tests/test_wizard_database.py` did exactly that and took three
    `tests/test_wizard_pipeline.py` tests down with it.

    **Prefer a uuid-named table wherever the code under test allows it** — see `cli_table`. This
    exists for the case that cannot: `wizard.database.probe_database` inspects `chunks` by name, so
    a test of the dimension check has to use that name and put it back.

    Clearing the ledger row matters as much as the drop: `apply_migrations` is idempotent by
    consulting it, so re-applying over a stale row would skip the work and leave no table at all.
    """
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks CASCADE")
        ledger = conn.execute("SELECT to_regclass(%s)", (LEDGER_TABLE,)).fetchone()
        if ledger is not None and ledger[0]:
            conn.execute(f"DELETE FROM {LEDGER_TABLE} WHERE target_table = 'chunks'")
    apply_migrations(TEST_DSN, table="chunks", dim=64)


@pytest.fixture(scope="session", autouse=True)
def _bootstrap_default_test_schema() -> Iterator[None]:
    """Provision the default MCP table explicitly for subprocess/server integration tests."""
    if not _db_available():
        yield
        return
    restore_default_chunks_table()
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


@pytest.fixture(scope="session")
def _suite_index_root(tmp_path_factory):
    """One disposable directory that `RECALL_INDEX_ROOT` points at for the whole session.

    Deliberately NOT under each test's own `tmp_path`, which was the first version of this and cost
    two failures to learn: `tests/test_fix.py::test_apply_proposal_preserves_the_memo_when_the_write_fails`
    and `tests/test_bench_systems.py::test_conversation_to_messages_mirrors_recall_turn_walk` both
    ENUMERATE `tmp_path` and assert on everything in it, so a fixture that creates one directory
    there fails them without either test having anything to do with uploads. `tmp_path` belongs to
    the test; a fixture that writes into it is changing the subject.

    Session scope is safe because nothing collides inside it: `stage_uploads` keys every staging
    directory by a fresh uuid, and any test that cares about the value sets its own.

    ⚠️ The directory is explicitly created here so the upload code never needs to guess whether it
    should or must create nested subdirectories. It is created once, at session start, and exists
    for the life of every test.
    """
    root = tmp_path_factory.mktemp("recall-index-root")
    # Explicitly ensure the directory exists, even though mktemp() should have created it.
    # This is defensive: the upload code requires it before writing.
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(autouse=True)
def _confine_index_root(_suite_index_root, monkeypatch) -> Iterator[None]:
    """Point `RECALL_INDEX_ROOT` somewhere disposable for EVERY test.

    `RECALL_INDEX_ROOT` defaults to `.`, the server's working directory, and that default is
    documented, deliberate and correct for the desktop app (docs/SECURITY_MODEL.md,
    docs/USING_WITH_CLAUDE.md), so it is not the thing to change. It is wrong for a test session
    only because the working directory of a test session is the checkout: `recall.desktop.uploads`
    resolves its staging root from the same variable, so any test that reaches `stage_uploads`
    without setting it decodes its upload into `uploads/<tenant>/<job_id>/` at the repository root.

    Three such directories were left behind by `tests/test_mcp_tool_authorization.py` alone, which
    calls `recall_ingest` for its authorised cases and stops at the service boundary, which is AFTER
    the staging write. They are untracked, they survive the run, and `git add` by pathspec is the
    only thing standing between them and a commit.

    Autouse rather than a per-test `monkeypatch.setenv` because the defect is a default reached by
    OMISSION: a new test that touches an upload or index path inherits the confinement without
    knowing this variable exists, which is the only version of this that cannot regress. Tests that
    care about the value still set their own, and win: `monkeypatch.setenv` in the test body runs
    after this fixture.
    """
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(_suite_index_root))
    yield


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
        # A few schema-boundary tests deliberately leave the shared migration ledger in a
        # partial state while exercising a refusal. Restore the session's default target before
        # creating a disposable table, otherwise an unrelated later test gets SchemaTooOld from
        # its normal fixture setup. Keep the product guard intact: this is test-database hygiene,
        # not a change to apply_migrations' requirement that custom tables follow the default.
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            expected_global = {
                migration.version
                for migration in load_migrations()
                if migration.version >= "0008"
            }
            actual_global = {
                str(row[0])
                for row in conn.execute(
                    f"SELECT version FROM {LEDGER_TABLE} WHERE target_table = %s "
                    "AND state = 'applied'",
                    (GLOBAL_MIGRATION_TARGET,),
                ).fetchall()
            }
        if actual_global != expected_global:
            restore_default_chunks_table()
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


@pytest.fixture(scope="session")
def voyageai_sdk() -> Any:
    """The `voyageai` package, imported on demand and NEVER during collection.

    ⚠️ **Importing this costs ~44s warm and ~75s cold, and the module that pays it is not the one
    you expect.** `voyageai/__init__.py` imports `voyageai.chunking`, which imports
    `langchain_text_splitters`, which imports `transformers`, which imports `torch`. Measured
    2026-08-23 with `python -X importtime`: 74.8s cumulative for `voyageai` on a cold cache, of
    which 31.4s is `transformers`.

    Three test modules used to pay that at MODULE scope, and each carried a comment explaining
    why: an `import` inside a test is billed to that test's 120s timeout, and one of them had
    already timed out that way. That reasoning is still right, which is why the four tests using
    this fixture carry `@pytest.mark.timeout(300)`.

    What it cost, measured back to back against this branch's base and warm, so that the two
    halves are comparable:

    | | with the module-scope import | with this fixture |
    |---|---|---|
    | `pytest tests/test_embeddings_retry_after.py --collect-only` | **45.33s** | **1.04s** |
    | whole-suite collection | 55.6 / 59.6 / 59.1s | 50.9 / 54.8 / 50.0 / 51.3s |

    So the win is on the SINGLE-FILE run, which is what you do while working on this file. It
    nearly vanishes from the whole-suite figure because a module collected earlier has already
    pulled `transformers` in, and voyageai then only adds its margin.

    🔁 This docstring first claimed 154.1s of collection falling to 75.1s, and that "every
    `pytest` invocation paid it, including `pytest tests/test_cli.py`". Both were wrong. The pair
    compared a cold cache against a warm one, and pytest imports only the modules it COLLECTS, so
    an unrelated single-file run never touched `voyageai` at all.
    `docs/preregistrations/2026-08-23-test-suite-wall-clock.md` carries the full correction.

    A session-scoped fixture keeps both halves: the import happens once per session, and only if
    one of the four tests that actually need the SDK is selected. Those four carry
    `@pytest.mark.timeout(300)` because whichever of them runs first is billed for the import.
    """
    try:
        import voyageai
        import voyageai.error  # noqa: F401  # the submodule the error-shape tests read
    except ImportError:
        pytest.skip('needs the voyage extra (pip install "recall-rag[voyage]")')
    return voyageai


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


@pytest.fixture(autouse=True)
def _confine_claude_client_config(tmp_path_factory, monkeypatch) -> None:
    """Keep every test away from the user's REAL `~/.claude.json`.

    ⚠️ **Written after a test run put five junk entries into the developer's own client config.**
    `wiring.register_local_scope` writes the wizard's servers into Claude Code's own config so
    they load in that project without an approval prompt, and its default target is
    `Path.home() / ".claude.json"` — a user-global file holding every project the user has. Five
    `run_headless` tests reached it with a `project_root` under `pytest-of-.../`, and each one
    appended an entry pointing at a temp directory that no longer exists.

    🔁 The writer named above has changed twice (it recorded an APPROVAL for a project-scoped
    `.mcp.json` when this was written, then registered the servers at user scope, and now at
    local scope).
    The hazard did not change with it: the default target is still another application's
    user-global file, which is the only reason this fixture exists.

    Nothing was corrupted (the writer is atomic and backs up first, and no existing project was
    modified), which is precisely why it went unnoticed: the suite was green and the damage was
    additive. `run_headless` now takes an explicit `claude_config_path`, but a parameter only
    protects the callers that remember it, and remembering is the thing that failed. This makes
    forgetting harmless.

    `Path.home` rather than the HOME variable, because that is what the writer calls and because
    `Path.home()` on Windows reads USERPROFILE, so patching one environment name would miss it.

    ⚠️ **The environment is redirected AS WELL, and that is not redundancy.** A patched `Path.home`
    lives in this process and does not survive into a subprocess: anything that shells out resolves
    the real home itself and writes to the developer's own config. That is not hypothetical — a peer
    session hit exactly it, and the way it happened is the part worth keeping. Their test was safe
    when written, because it exercised a faked CLI arm that wrote no file; a later change flipped
    the primary path, the test stopped taking that branch, fell through to a direct writer, and
    started leaking. **The test did not change. The code beneath it changed branches**, which is
    invisible in a diff and which no per-test opt-in can cover, because the opt-in was a decision
    made against the code as it was.

    Nothing under `recall/wizard/` spawns a process today, so the `Path.home` patch alone is
    currently sufficient. That is a property of the implementation, not of this guard, and it is
    exactly the kind of property that stops being true without anyone noticing. Both names are set
    because `Path.home()` reads USERPROFILE on Windows and HOME elsewhere, and a subprocess may
    consult either.
    """
    from pathlib import Path as _Path

    home = tmp_path_factory.mktemp("fake-home")
    monkeypatch.setattr(_Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
