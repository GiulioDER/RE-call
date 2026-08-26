"""`recall doctor`: one command that says which of the six things is wrong.

**Why this exists.** Six independent failures in this product present to the user as one of two
symptoms, and neither symptom names its cause:

| What is actually wrong | What the user sees |
|---|---|
| `recall-mcp` is not on the PATH the agent client sees | the tools are absent |
| the MCP server is registered but not approved by the client | the tools are absent (NOT checked here: `claude mcp list` is the only thing that reports approval state) |
| the database is unreachable | the tools are absent (startup raised into a log nobody reads) |
| migrations are pending | the tools are absent |
| the table or tenant names a corpus nobody indexed | `0 relevant memory hit(s)` |
| the corpus is uncalibrated and the server is strict | `DEGRADED:INDEX_NOT_READY` |

Only the last of those names itself. The fifth is the worst, because a well-formed empty answer is
indistinguishable from an empty corpus, and it was live on the documented quickstart-to-plugin path
until 2026-08-25: the plugin collected a DSN, a tenant and a trust mode, `recall quickstart` had
written to `quickstart_chunks`, and nothing anywhere read a table. One session measured
`recall_search` returning zero hits against a database that held the corpus, with no error printed
by anything.

A person hitting any of these has no ordering to work through and no command to run. That is what
this is: **it inspects, it explains, and it writes nothing at all.**

Four decisions, each of which is a place this would otherwise go wrong.

**It never constructs an embedder.** `resolve_embedder("fastembed")` downloads a model on a cold
cache, and a diagnostic that takes minutes and 130 MB of network before its first line is one
people stop running. The embedder is reported by NAME and its backend by importability; the vector
width comes from the database, which is where the width that actually matters lives.

**It reads the database rather than the configuration wherever both could answer.** A config file
records what somebody intended; `pg_attribute` and `count(*)` record what is true. The whole value
of the corpus check is that it can contradict the configuration.

**It never writes, not even a fix it is certain of.** Every check carries the command that repairs
it, and the operator runs it. A doctor that repairs is a doctor nobody can run to find out what is
wrong, because running it changes the answer.

**Exit status separates blocked from imperfect.** A missing calibration and an unapproved MCP
server are warnings: the product works, less well. An unreachable database is a failure. Exiting
non-zero on every imperfection would make the exit code useless in a script, which is the only
place an exit code is read.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

from psycopg import sql

from recall.store import (
    DEFAULT_TABLE,
    DEFAULT_TENANT,
    TENANT_GUC,
    redacted_dsn,
    scrub_dsn_secrets,
)

__all__ = [
    "Check",
    "Report",
    "Status",
    "run_checks",
]

Status = Literal["ok", "warn", "fail", "skip"]

#: Printed beside each status. ASCII, because this runs in whatever terminal the person has, and a
#: box-drawing character that renders as a replacement glyph in a Windows console makes a
#: diagnostic look broken at the exact moment its credibility matters.
MARKERS: dict[Status, str] = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL", "skip": "----"}

#: Console scripts a working install places on PATH. `recall-mcp` and `recall-hooks` are the two
#: the Claude Code plugin invokes BY BARE NAME: a plugin's manifest is written once and shipped to
#: every machine, so it cannot name an interpreter, and a marketplace install with no
#: `pip install recall-rag` behind it fails to spawn its own server with nothing said about why.
CONSOLE_SCRIPTS = ("recall", "recall-mcp", "recall-hooks")


@dataclass(frozen=True)
class Check:
    """One question, its answer, and the command that repairs it.

    `fix` is not optional decoration. A diagnostic that reports a problem it cannot tell you how to
    resolve has moved the work rather than done it, and every `fail` here has been through the
    question "what would I type next?".
    """

    name: str
    status: Status
    detail: str
    fix: str | None = None

    def render(self) -> str:
        line = f"{MARKERS[self.status]}  {self.name:22} {self.detail}"
        if self.fix and self.status in ("fail", "warn"):
            line += f"\n      -> {self.fix}"
        return line

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "detail": self.detail, "fix": self.fix}


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, check: Check) -> Check:
        self.checks.append(check)
        return check

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == "fail"]

    @property
    def warned(self) -> list[Check]:
        return [c for c in self.checks if c.status == "warn"]

    def exit_code(self) -> int:
        """Non-zero only when something is BLOCKED, never merely imperfect.

        An uncalibrated corpus and an unregistered client are `warn`, because the product works
        without them. If this returned 1 for those, every CI job that ran it would have to ignore
        the exit code, which is the same as not having one.
        """
        return 1 if self.failed else 0

    def render(self) -> str:
        lines = [c.render() for c in self.checks]
        lines.append("")
        if self.failed:
            lines.append(
                f"{len(self.failed)} blocking problem(s). Fix the first one and run this again: "
                "later checks are skipped when an earlier one makes them unanswerable."
            )
        elif self.warned:
            lines.append(f"Nothing is blocked. {len(self.warned)} thing(s) could be better.")
        else:
            lines.append("Everything this can check is healthy.")
        return "\n".join(lines)


def _python_check() -> Check:
    version = ".".join(str(p) for p in sys.version_info[:3])
    if sys.version_info < (3, 11):
        return Check(
            "python",
            "fail",
            f"{version} at {sys.executable}; recall needs 3.11 or newer",
            "install recall into a 3.11+ interpreter",
        )
    return Check("python", "ok", f"{version} at {sys.executable}")


def _package_check() -> Check:
    from recall.version import __version__

    return Check("recall package", "ok", f"{__version__} from {Path(__file__).parent}")


def _console_scripts_check() -> Check:
    """⛔ The check that explains a plugin install with no tools, and nothing else does.

    `shutil.which` resolves against this process's PATH, which is not necessarily the client's.
    That weakens the check in one direction only, and the useful direction: an absent script here
    is conclusive, while a present one is merely likely. It is stated that way rather than
    overclaimed.
    """
    missing = [name for name in CONSOLE_SCRIPTS if shutil.which(name) is None]
    if not missing:
        return Check(
            "console scripts",
            "ok",
            f"{', '.join(CONSOLE_SCRIPTS)} all resolve on PATH",
        )
    return Check(
        "console scripts",
        "fail",
        f"not on PATH: {', '.join(missing)}. The Claude Code plugin invokes recall-mcp and "
        "recall-hooks by bare name, so it cannot start them",
        'pip install "recall-rag[fastembed]" into the interpreter whose scripts directory is on '
        "PATH, or add that scripts directory to PATH",
    )


def _embedder_check(embedder_name: str) -> Check:
    """Report the embedder without building one, because building one downloads a model.

    A cold `fastembed` costs about 130 MB and a first-run wait. A diagnostic people will not wait
    for is a diagnostic people do not run, so this answers the question that can be answered
    cheaply (is the backend even installed?) and leaves the width to the database check, which
    reads the width that is actually in force.
    """
    backend = embedder_name.split(":", 1)[0]
    modules = {
        "fastembed": "fastembed",
        "st": "sentence_transformers",
        "sfr-code": "sentence_transformers",
        "voyage": "voyageai",
        "openai": "openai",
        "openrouter": "openai",
    }
    module = modules.get(backend)
    if module is None:
        # `hashing` and anything unrecognised. Hashing needs nothing; an unknown spelling is the
        # CLI's error to raise when it resolves, not this command's to guess at.
        return Check("embedder", "ok", f"{embedder_name} (no optional backend needed)")
    if importlib.util.find_spec(module) is None:
        extra = {"fastembed": "fastembed", "sentence_transformers": "rerank"}.get(module, module)
        return Check(
            "embedder",
            "fail",
            f"{embedder_name} needs the {module!r} package and it is not importable",
            f'pip install "recall-rag[{extra}]"',
        )
    return Check("embedder", "ok", f"{embedder_name} ({module} importable; not loaded)")


def _docker_check() -> Check:
    from recall.quickstart import docker_unavailable_reason

    reason = docker_unavailable_reason()
    if reason:
        return Check(
            "docker",
            "warn",
            reason.splitlines()[0],
            "only needed for `recall quickstart` and the managed stack; an existing PostgreSQL "
            "with pgvector works without it",
        )
    return Check("docker", "ok", "available")


#: Bound on any single statement this command issues, milliseconds.
#:
#: ⚠️ Applied to the SESSION rather than per statement, and that ordering was the defect. The
#: sibling-table scan was bounded and `_count_for_tenant` — the query that runs on every HEALTHY
#: install — was not, so the bounded query was the fallback and the unbounded one was the default.
#: A `count(*)` over a multi-million-row corpus is a scan, and the audience for this command is
#: someone already stuck.
STATEMENT_TIMEOUT_MS = 5000


def _connect(dsn: str, *, tenant: str | None = None) -> Any:
    """Open the diagnostic connection in the same tenant context every real reader uses.

    ⛔ **Setting the GUC is not optional, and omitting it made this command LIE.** The chunk table
    carries `ENABLE` and `FORCE ROW LEVEL SECURITY` with
    `USING (tenant_id = current_setting('recall.tenant_id', true))`. Two-argument `current_setting`
    returns NULL when the GUC is unset, so the predicate is NULL for every row and every row is
    invisible: no error, just zero. `PgVectorStore._prepare` sets it on every connection it opens
    and `recall.pool` sets it on every checkout. This function did not.

    Measured 2026-08-26, one database holding one chunk, same table and tenant, two roles:

        role recall      (superuser)                OK    corpus  1 chunk(s)       exit 0
        role doctorprobe (NOSUPERUSER NOBYPASSRLS)  FAIL  corpus  holds NO chunks  exit 1

    FORCE means even the table owner is subject, so the false answer appeared on exactly the roles
    `docs/MIGRATIONS.md` tells operators to serve with. Every doctor test used a fake connection,
    and `docker-compose.yml` ships `POSTGRES_USER=recall`, the cluster superuser, so nothing saw it.

    🔑 **Why this was a P1 rather than a wrong number: the repair it printed was
    `recall index <folder>`, and `recall index` PRUNES sources that have vanished from disk.** A
    user on a correctly hardened install was told their full corpus was empty and handed a command
    that can delete it.

    `autocommit` because every statement here is a read and nothing needs a shared snapshot; it
    also keeps a failed statement from poisoning the rest of the report, and makes the per-scan
    `SET LOCAL` scope the way its comment claims rather than surviving to the end of an enclosing
    transaction.
    """
    import psycopg

    conn = psycopg.connect(
        dsn,
        connect_timeout=5,
        autocommit=True,
        options=f"-c statement_timeout={STATEMENT_TIMEOUT_MS}",
    )
    if tenant is not None:
        conn.execute(f"SELECT set_config('{TENANT_GUC}', %s, false)", (tenant,))
    return conn


def _rls_can_see_every_tenant(conn: Any) -> bool:
    """Can this role read past row-level security, i.e. is a cross-tenant scan meaningful at all?

    A role without `BYPASSRLS` sees only the tenant its GUC names, so "which OTHER corpora hold
    rows" is a question it cannot answer. Reporting an empty list there would repeat the original
    defect one level up: a confident negative standing in for an unanswerable question.
    """
    row = conn.execute(
        "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user"
    ).fetchone()
    return bool(row and row[0])


def _database_checks(dsn: str, conn: Any) -> Iterator[Check]:
    """Facts about the server, given a connection the caller already opened.

    ⚠️ Takes the connection rather than opening one, and that is not tidiness. This and
    `_corpus_checks` used to open one each, so a database that was down cost the reader TWO connect
    timeouts back to back before the first line appeared. The one person guaranteed to be on that
    path is the one whose database is broken, which is the entire audience for this command.

    The DSN is still passed, for the redacted display name only. `redacted_dsn` is what keeps a
    real password out of the reader's scrollback, and a connection object cannot be asked for the
    string that made it without recovering the password with it.
    """
    version = conn.execute("SELECT version()").fetchone()
    assert version is not None
    yield Check("database", "ok", f"{version[0].split(',')[0]} at {redacted_dsn(dsn)}")

    installed = conn.execute(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
    ).fetchone()
    if installed is None:
        available = conn.execute(
            "SELECT 1 FROM pg_available_extensions WHERE name = 'vector'"
        ).fetchone()
        yield Check(
            "pgvector",
            "fail",
            "the vector extension is not installed in this database"
            + ("" if available else ", and the server does not ship it"),
            "CREATE EXTENSION vector;  (or use the pgvector/pgvector image)"
            if available
            else "use a PostgreSQL image that carries pgvector, such as pgvector/pgvector:pg18",
        )
    else:
        yield Check("pgvector", "ok", f"extension version {installed[0]}")


def _chunk_tables(conn: Any) -> dict[str, set[str]]:
    """Every table in the search path that is SHAPED like a chunk corpus, with its columns.

    Three defects collapse into this one query, and each was measured rather than reasoned:

    **Shape, not just `tenant_id` (F03).** Counting on a `tenant_id` column alone was wrong by a
    factor of four: a migrated database carries FOURTEEN tenant-scoped tables and only three are
    corpora. With one row in `recall_tenant_state`, which every real install has, an EMPTY database
    reported ``These do hold rows: recall_tenant_state/default (1)`` and advised pointing `--table`
    at a bookkeeping table, suppressing the one correct instruction (`recall index <folder>`).

    **No `::regclass` cast (F17).** The old probe cast a raw `pg_tables.tablename` to `regclass`,
    outside its `try`. That cast reparses the string as an identifier and downcases it, so a table
    stored as ``MyTable`` raises `UndefinedTable` and a hyphenated one is a parse error — aborting
    the whole report with a traceback, in the command whose premise is that it works when nothing
    else does. Joining the catalog by oid needs no cast.

    **The whole search path, not `current_schema()` (F19).** `PgVectorStore` interpolates a bare
    table name, which PostgreSQL resolves across every entry of `search_path`. Looking only at the
    first schema meant the doctor could report ``table 'chunks' does not exist`` about a table the
    server opens successfully, and then advise `schema apply`, which would create a SECOND empty
    `chunks` shadowing the real corpus.

    One round trip for the whole schema, rather than one per table (F30).
    """
    rows = conn.execute(
        "SELECT c.relname, a.attname FROM pg_class c "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "JOIN pg_attribute a ON a.attrelid = c.oid "
        "WHERE c.relkind = 'r' AND n.nspname = ANY(current_schemas(false)) "
        "AND NOT a.attisdropped AND a.attnum > 0"
    ).fetchall()
    columns: dict[str, set[str]] = {}
    for name, attname in rows:
        columns.setdefault(name, set()).add(attname)
    return columns


#: The columns a table must have before this command will call it a corpus and count it.
#: `tenant_id` alone admitted eleven bookkeeping tables; see `_chunk_tables`.
CHUNK_SHAPE = frozenset({"tenant_id", "embedding", "source"})


def _count_for_tenant(conn: Any, table: str, tenant: str) -> int | None:
    """`SELECT count(*)` for one corpus, or `None` when the database would not answer.

    `None` rather than `0`, because "could not count" and "counted zero" are the two things this
    entire command exists to keep apart. Returning 0 for a statement timeout would reproduce the
    original defect exactly: a confident empty answer standing in for an unanswerable question.

    The table name is composed with `psycopg.sql.Identifier`, which quotes and escapes it at
    composition time. `run_checks` has already refused anything that is not an identifier, so this
    is the second of two gates rather than the only one.
    """
    query = sql.SQL("SELECT count(*) FROM {} WHERE tenant_id = %s").format(sql.Identifier(table))
    try:
        row = conn.execute(query, (tenant,)).fetchone()
    except Exception:
        return None
    return int(row[0]) if row else 0


def _scan_caveats(conn: Any, unreadable: list[str]) -> Iterator[Check]:
    """Everything the cross-corpus scan could NOT establish, reported where it cannot be misread.

    ⚠️ Emitted at EVERY branch that reports on other corpora, including the two early returns. Those
    branches printed an unqualified "Nothing holds rows." to a role that had not been able to read
    anything, which is the same confident-negative failure one level along.

    Two distinct facts, kept distinct:

    * a role without `BYPASSRLS` sees only its own tenant, so "no other corpus holds rows" is a
      question it cannot answer at all;
    * a table the scan could not read (no SELECT for this role, or too large to count inside the
      session bound) is not a table that is empty.
    """
    if not _rls_can_see_every_tenant(conn):
        yield Check(
            "rls visibility",
            "warn",
            "this role cannot read past row-level security, so a cross-corpus scan can only see "
            "the tenant it is configured for. An empty result there means 'not visible to this "
            "role', not 'not there'",
            "re-run with the migration/owner DSN to see every tenant",
        )
    if unreadable:
        yield Check(
            "corpus scan",
            "warn",
            f"could not read {len(unreadable)} candidate corpus/corpora: "
            f"{', '.join(sorted(unreadable))}. Either this role has no SELECT on them, or they are "
            "too large to count inside the statement bound. NOT counted is not the same as empty",
            "re-run with the migration/owner DSN, or count those tables directly with psql",
        )


def _embedding_width(conn: Any, table: str) -> int | None:
    """The declared `vector(N)` width of a table's embedding column, or None.

    ⚠️ **No `::regclass` cast, and that is F17 finished rather than two-thirds done.** The cast
    reparses its argument as an identifier and downcases it, so a table stored as `MyChunks` raises
    `UndefinedTable`. Two of these survived the first pass, and one sat outside a `try`, which meant
    a quoted mixed-case configured table turned a healthy database into `FAIL database`. Joining
    `pg_class` by name inside the current search path needs no cast and cannot raise on a name.

    `atttypmod` carries the dimension directly for pgvector — verified against pgvector/pgvector:pg18
    (`vector(1024)` reports 1024). The `+ VARHDRSZ` convention that would make this `N + 4` is
    varchar and bpchar only, and an un-dimensioned `vector` reports -1, which the guard rejects.
    """
    row = conn.execute(
        "SELECT a.atttypmod FROM pg_class c "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "JOIN pg_attribute a ON a.attrelid = c.oid "
        "WHERE c.relname = %s AND n.nspname = ANY(current_schemas(false)) "
        "AND a.attname = 'embedding' AND NOT a.attisdropped",
        (table,),
    ).fetchone()
    return row[0] if row and row[0] and row[0] > 0 else None


def _corpus_checks(conn: Any, *, table: str, tenant: str) -> Iterator[Check]:
    """⚠️ The headline check, and the one the other tools cannot substitute for.

    `recall search` answers "what did you find". This answers "where were you looking", which is
    the question nobody thinks to ask, because a search against the wrong table succeeds. When the
    configured corpus is empty and some OTHER corpus in the same database is not, the detail names
    the populated one with its exact table and tenant, so the repair is a copy rather than a hunt.
    """
    columns = _chunk_tables(conn)

    if table not in columns:
        yield Check(
            "schema",
            "fail",
            f"table {table!r} does not exist anywhere on this connection's search_path",
            f"recall --table {table} --migration-dsn <owner dsn> schema apply   "
            "(or point --table at a table that does exist)",
        )
        populated, unreadable = _populated_corpora(conn, columns)
        yield from _scan_caveats(conn, unreadable)
        yield Check(
            "corpus",
            "skip",
            "skipped: there is no such table. "
            + (
                f"What does hold rows: {populated}"
                if populated
                else "No corpus this role can read holds rows."
            ),
        )
        return

    present_columns = columns[table]
    missing = CHUNK_SHAPE - present_columns
    if missing:
        # F20: the old code assumed `tenant_id` existed and let psycopg's UndefinedColumn escape
        # as a traceback. Naming a table that is not a recall table is EXACTLY the input this
        # command advertises catching, so it has to be a Check rather than a crash.
        yield Check(
            "schema",
            "fail",
            f"table {table!r} exists but is not a recall corpus: missing {sorted(missing)}",
            f"recall --table {table} --migration-dsn <owner dsn> schema apply   "
            "(or point --table at a real corpus)",
        )
        populated, unreadable = _populated_corpora(conn, columns)
        yield from _scan_caveats(conn, unreadable)
        yield Check(
            "corpus",
            "skip",
            "skipped: that table is not a corpus. "
            + (
                f"What does hold rows: {populated}"
                if populated
                else "No corpus this role can read holds rows."
            ),
        )
        return

    dim = _embedding_width(conn, table)
    yield Check(
        "schema",
        "ok",
        f"table {table!r} exists"
        + (f" at vector({dim})" if dim else " (no embedding column width recorded)"),
    )

    held = _count_for_tenant(conn, table, tenant)
    if held is None:
        yield Check(
            "corpus",
            "warn",
            f"{table!r}/{tenant!r} could not be counted within "
            f"{STATEMENT_TIMEOUT_MS / 1000:.0f}s. That is not the same as empty, and this command "
            "will not guess which",
            "re-run against a quieter database, or count it directly with psql",
        )
        return
    if held:
        yield Check("corpus", "ok", f"{held} chunk(s) in {table!r}/{tenant!r}")
        return

    populated, unreadable = _populated_corpora(conn, columns)
    yield from _scan_caveats(conn, unreadable)
    yield Check(
        "corpus",
        "fail",
        f"{table!r}/{tenant!r} holds NO chunks, so every search returns nothing and says "
        "nothing about why. "
        + (
            f"These do hold rows: {populated}"
            if populated
            else "No corpus in this database holds rows yet."
        ),
        f"recall --table {table} --tenant {tenant} index <folder>"
        if not populated
        # ⚠️ The two audiences take DIFFERENT knobs, and conflating them made this line inert (F02).
        # `RECALL_TABLE` is read by the MCP server and by nothing in this CLI, so a reader who
        # exported it and re-ran got byte-identical output. Naming a variable the command ignores
        # is worse than naming none.
        else "re-run this command with --table/--tenant, or set RECALL_TABLE/RECALL_TENANT in the "
        "MCP server's env block and restart the client",
    )


def _populated_corpora(conn: Any, columns: dict[str, set[str]]) -> tuple[str, list[str]]:
    """Every corpus in this database that actually holds chunks, as one printable string.

    Deliberately scans every candidate rather than only the configured one: the entire point is to
    be able to say "not there, but here", and a check that could only inspect where it was already
    looking could never say that. Candidates come from the catalog and are composed as identifiers,
    so the scan cannot be steered by anything a caller passes.
    """
    found: list[str] = []
    slow: list[str] = []
    for candidate in sorted(name for name, cols in columns.items() if CHUNK_SHAPE <= cols):
        query = sql.SQL("SELECT tenant_id, count(*) FROM {} GROUP BY tenant_id ORDER BY 2 DESC")
        try:
            rows = conn.execute(query.format(sql.Identifier(candidate))).fetchall()
        except Exception:
            # A corpus too large to count inside the session bound, or one this role may not read.
            # Named rather than dropped: silently omitting a table from "these do hold rows" is how
            # this check would come to give the confidently wrong answer it exists to prevent.
            slow.append(candidate)
            continue
        found.extend(f"{candidate}/{name} ({count})" for name, count in rows if count)
    # ⛔ **Returned SEPARATELY, and the type is the point.** Twice now a "could not read this" note
    # was appended to `found`, which every caller tests for truth — so a role that could read
    # nothing was told `These do hold rows: [not counted: chunks, ...]` about tables nobody had
    # read, and lost the one instruction it needed (`recall index <folder>`). The first repair moved
    # the RLS caveat out and left this appender; the defect came straight back through it.
    #
    # Patching producers one at a time was the wrong move. A pair cannot be mixed up: there is no
    # longer any string that means both "here is what I found" and "here is what I could not
    # check", so a future caller cannot reintroduce the confusion by accident.
    return ", ".join(found), slow


def _calibration_check(embedder_name: str, *, strict: bool, dsn: str) -> Check:
    """A missing calibration is a WARNING in every mode, and the docs are why.

    ⚠️ **This returned `fail` under the default mode, which made five published sentences false.**
    `README.md`, `docs/API.md`, `recall/cli_commands/doctor_cmd.py`, `site/troubleshooting.html`
    and `CHANGELOG.md` all promise the exit code means BLOCKED and that "a missing calibration will
    not fail a script". `RECALL_TRUST_MODE` defaults to strict, so an ordinary uncalibrated install
    exited 1 and every one of those sentences was wrong.

    Of the two ways to reconcile that, the docs describe the better behaviour and the code was
    changed to match. An uncalibrated corpus is the honest starting state of every fresh install:
    `recall quickstart` builds one deliberately. Exiting non-zero on it would make the exit code
    fire on the common case, and a code that fires on the common case is one every script learns
    to ignore, which is the same as not having one. The strictness is still REPORTED, because it
    is what decides whether queries are refused.

    `strict` is passed in rather than re-derived: the rule lives in `TrustPolicy.from_env`, and
    this module had hand-rolled it twice (F22).
    """
    from recall.calibration import load_for

    try:
        calibration = load_for(embedder_name)
    except Exception as exc:
        return Check(
            "calibration",
            "warn",
            f"could not be read: {scrub_dsn_secrets(f'{type(exc).__name__}: {exc}', dsn)}",
        )
    if calibration is None:
        return Check(
            "calibration",
            "warn",
            f"none fitted for {embedder_name!r}"
            + (
                ", and this server is STRICT, so it will refuse every query until one is published"
                if strict
                else ", and this server is in development mode, so queries answer as DEGRADED"
            ),
            "recall setup   (fits one against your corpus)",
        )
    return Check("calibration", "ok", f"threshold {calibration.threshold:.3f} for {embedder_name}")


def _migration_check(dsn: str, *, table: str, dim: int | None) -> Check:
    """Ask `schema_status` whether migrations are pending, which is what the server refuses on.

    ⛔ **The check named `schema` never asked this, and "migrations are pending" is one of the six
    symptoms this module's own docstring says the command separates.** `recall_mcp/server.py`
    calls `schema_status` at startup and raises `SchemaTooOld`, which an MCP client renders as a
    server with no tools. So a user in exactly that state ran the diagnostic written for it and was
    told the schema was fine.

    `schema_status` is SELECT-only, so this costs one connection and can never change anything.
    """
    from recall.schema import SchemaError, schema_status

    if dim is None:
        return Check(
            "migrations",
            "skip",
            "skipped: this table records no embedding width, so the migration check has no "
            "dimension to ask about",
        )
    try:
        status = schema_status(dsn, table=table, dim=dim)
    except SchemaError as exc:
        # SchemaTooNew and MigrationChecksumMismatch both land here, and both mean the install and
        # the database disagree about who is ahead. Reporting the exception verbatim is more useful
        # than any summary this function could write.
        return Check(
            "migrations",
            "fail",
            f"{type(exc).__name__}: {scrub_dsn_secrets(str(exc), dsn)}",
            "upgrade or downgrade recall so it matches this database, then re-run",
        )
    except Exception as exc:
        return Check(
            "migrations",
            "warn",
            f"could not be read: {scrub_dsn_secrets(f'{type(exc).__name__}: {exc}', dsn)}",
        )
    if status.compatible:
        return Check("migrations", "ok", f"{table!r} is at the schema this install expects")
    pending = [m.version for m in status.pending]
    return Check(
        "migrations",
        "fail",
        f"{len(pending)} migration(s) pending on {table!r}: {pending}. The MCP server refuses to "
        "start in this state, which a client shows as a server with no tools",
        f"recall --table {table} --migration-dsn <owner dsn> schema apply",
    )


def _claude_code_checks(project_root: Path) -> Iterator[Check]:
    import json

    from recall.claude_code import claude_code_detected, client_config_path

    if not claude_code_detected():
        yield Check("claude code", "skip", "no Claude Code install found on this machine")
        return
    config = client_config_path()
    if not config.is_file():
        yield Check(
            "claude code",
            "warn",
            f"detected, but {config} does not exist yet",
            "recall setup   (registers the MCP server and installs the session hooks)",
        )
        return
    try:
        document = json.loads(config.read_text(encoding="utf-8"))
    except Exception as exc:
        yield Check("claude code", "warn", f"{config} could not be parsed: {type(exc).__name__}")
        return
    project = (document.get("projects") or {}).get(str(project_root)) or {}
    servers = project.get("mcpServers") or {}
    recall_servers = sorted(k for k, v in servers.items() if _is_recall_server(v))
    if recall_servers:
        yield Check(
            "claude code",
            "ok",
            f"MCP server(s) registered for this project: {', '.join(recall_servers)}",
        )
    else:
        yield Check(
            "claude code",
            "warn",
            f"no recall MCP server in the LOCAL scope of {project_root}. A plugin install or a "
            "user-scope server lives elsewhere in the client config and is not checked here, so "
            "this is not proof that nothing is registered",
            "claude mcp list   (names the scope and the approval state, which this cannot), "
            "then recall setup or /plugin marketplace add GiulioDER/RE-call",
        )


def _is_recall_server(block: object) -> bool:
    """Recognise a recall server block however it was written.

    Three spellings are in the wild and all three are ours: `recall-mcp` as a console script, and
    `python -m recall_mcp.server` written either by the wizard or by hand. Matching only the first
    would report a hand-registered server as absent, which is the false negative that sends
    somebody re-running an installer that already worked.
    """
    if not isinstance(block, dict):
        return False
    command = str(block.get("command", ""))
    args = " ".join(str(a) for a in block.get("args", []) or [])
    return "recall-mcp" in command or "recall_mcp" in args


def run_checks(
    *,
    dsn: str,
    embedder: str,
    table: str = DEFAULT_TABLE,
    tenant: str = DEFAULT_TENANT,
    trust_mode: str | None = None,
    project_root: Path | None = None,
) -> Report:
    """Every check, in dependency order, writing nothing.

    Ordered so that a failure explains the skips beneath it rather than producing a second page of
    unrelated red: there is no useful corpus answer while the database is unreachable, and saying
    "skipped: the database is unreachable" is more honest than saying "0 chunks".
    """
    if not table.isidentifier():
        raise ValueError(f"table {table!r} is not a valid SQL identifier")

    from recall.trust_policy import TrustPolicy

    # F22: the RECALL_TRUST_MODE rule lives in ONE place. This module had re-derived it twice, and
    # `recall/_env.py` records that the same literal was re-implemented eleven times before the
    # project made delegation the rule.
    policy = TrustPolicy.from_env(
        {"RECALL_TRUST_MODE": trust_mode} if trust_mode is not None else None
    )

    report = Report()
    report.add(_python_check())
    report.add(_package_check())
    report.add(_console_scripts_check())
    report.add(_embedder_check(embedder))
    report.add(_docker_check())

    # ⚠️ **Read the server's variables, but do NOT let them steer this CLI (F02).** Defaulting
    # `--table`/`--tenant` from the environment in `recall/cli.py` was the obvious fix and is
    # dangerous: the same defaults feed `recall index`, which PRUNES, and `recall forget`, which is
    # irreversible, so anyone with `RECALL_TABLE` exported would have those silently redirected.
    # The diagnostic reports the divergence instead of inheriting it, which is also strictly more
    # informative: "you are looking at a different corpus from the one being served" is the answer,
    # not a step toward it.
    server_table = os.environ.get("RECALL_TABLE", "").strip()
    server_tenant = os.environ.get("RECALL_TENANT", "").strip()
    if (server_table and server_table != table) or (server_tenant and server_tenant != tenant):
        report.add(
            Check(
                "server config",
                "warn",
                f"this report is for {table!r}/{tenant!r}, but RECALL_TABLE/RECALL_TENANT in this "
                f"environment point the MCP server at "
                f"{(server_table or table)!r}/{(server_tenant or tenant)!r}",
                f"recall --table {server_table or table} --tenant {server_tenant or tenant} "
                "doctor   (to diagnose the corpus the server actually opens)",
            )
        )

    # ONE connection for the server facts and the corpus counts both, opened here so an unreachable
    # database is reported once and waited for once. See `_database_checks`.
    try:
        conn = _connect(dsn, tenant=tenant)
    except Exception as exc:
        # ⛔ `scrub_dsn_secrets`, not only `redacted_dsn`, and the difference is a real leak.
        # `redacted_dsn` cleans the string WE format; the password can be inside the EXCEPTION,
        # which psycopg builds from the connection string it was handed. `scrub_dsn_secrets`
        # documents two measured cases: a password containing `%` comes back verbatim inside
        # `invalid percent-encoded token`, and a malformed DSN is echoed whole. This command's
        # output is the single most likely thing in this package to be pasted into a bug report.
        detail = scrub_dsn_secrets(f"{type(exc).__name__}: {exc}", dsn)
        report.add(
            Check(
                "database",
                "fail",
                f"cannot connect to {redacted_dsn(dsn)}: {detail}",
                "start one with `recall quickstart`, or pass --serving-dsn / set "
                "RECALL_SERVING_DSN",
            )
        )
        report.add(Check("corpus", "skip", "skipped: the database is unreachable"))
    else:
        # ⛔ F31: this block had NO handler, so any post-connect failure escaped `run_checks` as a
        # traceback — replacing a curated, secret-scrubbed report with raw exception text, in the
        # one command whose contract is to explain problems rather than throw them. Two reachable
        # instances were found by audit (a `::regclass` cast on an odd table name, and a table
        # without a `tenant_id` column); both are fixed at source, and this catches the next one.
        try:
            with conn:
                for check in _database_checks(dsn, conn):
                    report.add(check)
                for check in _corpus_checks(conn, table=table, tenant=tenant):
                    report.add(check)
                report.add(_migration_check_for(conn, dsn, table=table))
        except Exception as exc:
            report.add(
                Check(
                    "database",
                    "fail",
                    "a check failed against the database: "
                    + scrub_dsn_secrets(f"{type(exc).__name__}: {exc}", dsn),
                    "re-run with --json for the checks that did complete, and report this message",
                )
            )

    report.add(
        Check(
            "trust mode",
            "ok",
            "strict (refuses an uncalibrated corpus)"
            if policy.strict
            else "development (answers an uncalibrated corpus as DEGRADED)",
        )
    )
    report.add(_calibration_check(embedder, strict=policy.strict, dsn=dsn))
    for check in _claude_code_checks(Path(project_root or Path.cwd()).resolve()):
        report.add(check)
    return report


def _migration_check_for(conn: Any, dsn: str, *, table: str) -> Check:
    """Read the table's declared vector width, then ask `schema_status` about it.

    Split out so the width read happens on the connection that is already open, while
    `schema_status` opens its own (it takes a DSN, and it is SELECT-only).
    """
    return _migration_check(dsn, table=table, dim=_embedding_width(conn, table))
