"""`recall doctor`: one command that says which of the six things is wrong.

**Why this exists.** Six independent failures in this product present to the user as one of two
symptoms, and neither symptom names its cause:

| What is actually wrong | What the user sees |
|---|---|
| `recall-mcp` is not on the PATH the agent client sees | the tools are absent |
| the MCP server is registered but not approved by the client | the tools are absent |
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

from recall.store import DEFAULT_TABLE, DEFAULT_TENANT, redacted_dsn, scrub_dsn_secrets

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


def _connect(dsn: str):
    import psycopg

    return psycopg.connect(dsn, connect_timeout=5)


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


def _count_for_tenant(conn: Any, table: str, tenant: str) -> int:
    """`SELECT count(*)` for one corpus, with the table name composed rather than interpolated.

    `psycopg.sql.Identifier` quotes and escapes the name at composition time. `run_checks` has
    already refused anything that is not an identifier, so this is the second of two gates rather
    than the only one: the validation states the intent, and the composition is what enforces it
    even if a future caller reaches this function directly.
    """
    query = sql.SQL("SELECT count(*) FROM {} WHERE tenant_id = %s").format(sql.Identifier(table))
    row = conn.execute(query, (tenant,)).fetchone()
    return int(row[0]) if row else 0


def _corpus_checks(conn: Any, *, table: str, tenant: str) -> Iterator[Check]:
    """⚠️ The headline check, and the one the other tools cannot substitute for.

    `recall search` answers "what did you find". This answers "where were you looking", which is
    the question nobody thinks to ask, because a search against the wrong table succeeds. When the
    configured corpus is empty and some OTHER corpus in the same database is not, the detail names
    the populated one with its exact table and tenant, so the repair is a copy rather than a hunt.

    Takes an open connection rather than a DSN: this and `_database_checks` used to open one each,
    so a database that was down cost the reader two connect timeouts back to back before anything
    printed, and the person on that path is the one whose database is broken.
    """
    present = {
        row[0]
        for row in conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = current_schema()"
        ).fetchall()
    }
    if table not in present:
        yield Check(
            "schema",
            "fail",
            f"table {table!r} does not exist in this database",
            f"recall --table {table} schema apply   (or point --table at one that does)",
        )
        populated = _populated_corpora(conn, present)
        yield Check(
            "corpus",
            "skip",
            "skipped: there is no such table. "
            + (f"What does hold rows: {populated}" if populated else "Nothing holds rows."),
        )
        return

    width = conn.execute(
        "SELECT atttypmod FROM pg_attribute "
        "WHERE attrelid = %s::regclass AND attname = 'embedding'",
        (table,),
    ).fetchone()
    dim = width[0] if width and width[0] and width[0] > 0 else None
    yield Check(
        "schema",
        "ok",
        f"table {table!r} exists"
        + (f" at vector({dim})" if dim else " (no embedding column width recorded)"),
    )

    held = _count_for_tenant(conn, table, tenant)
    if held:
        yield Check("corpus", "ok", f"{held} chunk(s) in {table!r}/{tenant!r}")
        return

    populated = _populated_corpora(conn, present)
    yield Check(
        "corpus",
        "fail",
        f"{table!r}/{tenant!r} holds NO chunks, so every search returns nothing and says "
        "nothing about why. "
        + (
            f"These do hold rows: {populated}"
            if populated
            else "Nothing in this database holds rows yet."
        ),
        f"recall --table {table} --tenant {tenant} index <folder>"
        if not populated
        else "point --table/--tenant (or RECALL_TABLE/RECALL_TENANT) at one of the above",
    )


def _populated_corpora(conn: Any, present: set[str]) -> str:
    """Every (table, tenant) in this database that actually holds chunks, as one printable string.

    Deliberately scans every candidate table rather than only the configured one: the entire point
    is to be able to say "not there, but here", and a check that could only inspect where it was
    already looking could never say that. Names come from `pg_tables` and are composed as
    identifiers, so the scan cannot be steered by anything a caller passes.
    """
    found: list[str] = []
    slow: list[str] = []
    for candidate in sorted(present):
        has_tenant = conn.execute(
            "SELECT 1 FROM pg_attribute WHERE attrelid = %s::regclass AND attname = 'tenant_id'",
            (candidate,),
        ).fetchone()
        if not has_tenant:
            continue
        query = sql.SQL("SELECT tenant_id, count(*) FROM {} GROUP BY tenant_id ORDER BY 2 DESC")
        try:
            # ⚠️ BOUNDED, because this is a grouped `count(*)` over every sibling table and there
            # is no index that answers it. On a large corpus that is a full scan each, and the
            # person running this command is by definition already stuck: a diagnostic that hangs
            # is worse than one that says it could not finish. `SET LOCAL` so the bound dies with
            # the transaction rather than leaking into anything else on this connection.
            with conn.transaction():
                conn.execute("SET LOCAL statement_timeout = 5000")
                rows = conn.execute(query.format(sql.Identifier(candidate))).fetchall()
        except Exception:
            # Two different events land here and the distinction does not change what to do: a
            # table with a `tenant_id` column that will not group, and one too large to count in
            # five seconds. Either way, reporting the tables that DID answer beats failing the
            # whole diagnostic over one that did not — but the name is kept, because silently
            # omitting a table from "these do hold rows" is how this check would come to give the
            # confidently wrong answer it exists to prevent.
            slow.append(candidate)
            continue
        found.extend(f"{candidate}/{name} ({count})" for name, count in rows if count)
    if slow:
        found.append(f"[not counted in time: {', '.join(slow)}]")
    return ", ".join(found)


def _calibration_check(embedder_name: str, trust_mode: str) -> Check:
    from recall.calibration import load_for

    try:
        calibration = load_for(embedder_name)
    except Exception as exc:
        return Check("calibration", "warn", f"could not be read: {type(exc).__name__}: {exc}")
    if calibration is None:
        strict = trust_mode.strip().lower() != "development"
        return Check(
            "calibration",
            "fail" if strict else "warn",
            f"none fitted for {embedder_name!r}, and this trust mode is "
            + ("strict, which correctly REFUSES every query" if strict else "development"),
            "recall setup   (fits one against your corpus), or set RECALL_TRUST_MODE=development "
            "for local evaluation only",
        )
    return Check("calibration", "ok", f"threshold {calibration.threshold:.3f} for {embedder_name}")


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
            f"no recall MCP server registered for {project_root}",
            "recall setup, or install the plugin: /plugin marketplace add GiulioDER/RE-call",
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

    report = Report()
    report.add(_python_check())
    report.add(_package_check())
    report.add(_console_scripts_check())
    report.add(_embedder_check(embedder))
    report.add(_docker_check())
    # ONE connection for the server facts and the corpus counts both, opened here so an unreachable
    # database is reported once and waited for once. See `_database_checks`.
    try:
        conn = _connect(dsn)
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
        with conn:
            for check in _database_checks(dsn, conn):
                report.add(check)
            for check in _corpus_checks(conn, table=table, tenant=tenant):
                report.add(check)
    mode = trust_mode if trust_mode is not None else os.environ.get("RECALL_TRUST_MODE", "strict")
    report.add(Check("trust mode", "ok", mode.strip().lower() or "strict"))
    report.add(_calibration_check(embedder, mode))
    for check in _claude_code_checks(Path(project_root or Path.cwd()).resolve()):
        report.add(check)
    return report
