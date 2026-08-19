"""Decide whether a database somebody else provisioned can host this install, before building.

The wizard has always accepted an existing database: `HeadlessConfig` takes `dsn` OR `data_root`,
and with `dsn` set no container is ever started. What it did not do is CHECK that database, because
the only caller was CI, pointing at a container it had just created to its own specification. The
moment a person supplies their own Postgres that assumption stops holding, and every way it can be
wrong fails late, expensively, and with a message about something else:

* **No pgvector** surfaces as a syntax error on `vector(384)` partway through applying the schema.
* **A role that cannot create objects** surfaces the same way, one statement later.
* **A dimension that does not match the chosen embedder** does not surface during setup at ALL. The
  schema applies cleanly, the build runs, and the first insert fails minutes in — the failure mode
  `headless.py` already documents for a hardcoded dimension, reached here by a different road.
* **An unreachable host** is indistinguishable from a wrong password until you read the driver's
  exception, which a first-run user should never be asked to do.

So this module answers those questions FIRST, in one connection, and reports them as findings a
person can act on. Every check returns rather than raises: a preflight that dies on its own
diagnostic has told the user nothing, and "I could not determine this" is a real answer that must
be distinguishable from "this is fine".

**Nothing here writes.** It is safe to run against a production database, and safe to run twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

__all__ = [
    "DatabaseReport",
    "Finding",
    "LOCAL_HOSTS",
    "is_local_host",
    "probe_database",
]

#: Hosts that mean "this machine". Kept beside `is_local_host` rather than inlined because the
#: credentials guard in `recall.store` makes the same distinction, and two lists would drift.
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", ""})

#: How long to wait for a connection before calling the host unreachable. Short: a person is
#: watching, and a wrong hostname should not cost them a minute of silence.
CONNECT_TIMEOUT_SECONDS = 8


@dataclass(frozen=True)
class Finding:
    """One thing that was checked, and what it means for the install.

    `blocking` and `ok` are separate on purpose. A finding can be not-ok and not-blocking (the
    schema is absent, which is normal on a first install and merely worth saying), and it can be
    undetermined, which is neither. Collapsing the three into a boolean is how a preflight starts
    reporting "fine" for states it never established.
    """

    name: str
    #: True when checked and good, False when checked and bad, None when it could not be determined.
    ok: bool | None
    detail: str
    #: Whether the install must not proceed. Never true for an undetermined finding: refusing on
    #: something that was not established would block installs for a broken probe.
    blocking: bool = False
    #: What the user should do. Empty when there is nothing to do.
    advice: str = ""


@dataclass(frozen=True)
class DatabaseReport:
    """Everything the preflight learned, and the one question the caller actually asks."""

    dsn: str
    findings: tuple[Finding, ...] = ()
    #: Reported separately because the credentials guard keys on it and because a remote database
    #: changes what advice makes sense.
    remote: bool = False
    server_version: str = ""
    existing_dimension: int | None = None

    @property
    def usable(self) -> bool:
        """True when nothing found is blocking. Undetermined findings do not block."""
        return not any(f.blocking for f in self.findings)

    @property
    def blockers(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.blocking)

    def render(self) -> str:
        lines: list[str] = []
        for finding in self.findings:
            mark = {True: "ok  ", False: "FAIL", None: "??  "}[finding.ok]
            lines.append(f"  {mark} {finding.name}: {finding.detail}")
            if finding.advice:
                lines.append(f"       -> {finding.advice}")
        head = "database usable" if self.usable else "database NOT usable"
        return "\n".join([head, *lines])


def is_local_host(dsn: str) -> bool:
    """Whether `dsn` points at this machine.

    Parsed rather than substring-matched: `postgresql://user@db.example.invalid/localhost` contains
    the word and is not local, and a database named after a local host is not a rare joke, it is
    what happens when somebody names a database after the machine it replaced.
    """
    try:
        host = urlsplit(dsn).hostname
    except ValueError:
        # An unparseable DSN is not local, because we cannot show that it is, and the consequence
        # of guessing "local" is skipping the credentials guard.
        return False
    return (host or "") in LOCAL_HOSTS


def probe_database(dsn: str, *, expected_dimension: int | None = None) -> DatabaseReport:
    """Check an existing database, read-only, and report what would go wrong.

    `expected_dimension` is the embedder's vector width. Passing it is what turns the most
    expensive failure in this list — a schema whose vector column does not match the chosen
    embedder — from an insert error minutes into a build into a sentence before anything is built.
    """
    remote = not is_local_host(dsn)
    findings: list[Finding] = []

    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - psycopg is a hard dependency of the package
        return DatabaseReport(
            dsn=dsn,
            remote=remote,
            findings=(
                Finding(
                    name="driver",
                    ok=None,
                    detail=f"psycopg is not importable ({exc})",
                    advice="reinstall recall with its database extra",
                ),
            ),
        )

    try:
        connection = psycopg.connect(dsn, connect_timeout=CONNECT_TIMEOUT_SECONDS, autocommit=True)
    except Exception as exc:  # noqa: BLE001 - the driver raises several unrelated types here, and
        # every one of them means the same thing to the person waiting: this address did not work.
        # Re-raising would make the preflight the thing that fails rather than the thing that
        # reports, which is the opposite of its job.
        return DatabaseReport(
            dsn=dsn,
            remote=remote,
            findings=(
                Finding(
                    name="reachable",
                    ok=False,
                    detail=f"{type(exc).__name__}: {exc}".strip(),
                    blocking=True,
                    advice=(
                        "check the host, port, database name and password. For a database behind "
                        "SSH, open a tunnel first (`ssh -L 5433:localhost:5432 user@host`) and "
                        "point the DSN at 127.0.0.1:5433."
                    ),
                ),
            ),
        )

    version = ""
    dimension: int | None = None
    with connection:
        findings.append(Finding(name="reachable", ok=True, detail="connected"))

        reported = _scalar(connection, "SELECT version()")
        # Narrowed rather than coerced. `str(row[0])` would turn a None or an unexpected type into
        # the strings "None" or a repr, and a version string nobody can parse reads as a real
        # answer. Anything that is not a string is "not reported", which is a state this already
        # distinguishes.
        version = reported if isinstance(reported, str) else ""
        if version:
            findings.append(Finding(name="server", ok=True, detail=version.split(" on ")[0]))
        else:
            findings.append(Finding(name="server", ok=None, detail="version not reported"))

        findings.append(_check_vector(connection))
        findings.append(_check_create_privilege(connection))
        dimension, dimension_finding = _check_dimension(connection, expected_dimension)
        findings.append(dimension_finding)

    return DatabaseReport(
        dsn=dsn,
        findings=tuple(findings),
        remote=remote,
        server_version=version,
        existing_dimension=dimension,
    )


def _scalar(connection: object, sql: str) -> object:
    """One value, or None if the statement fails for any reason.

    Swallowing here is deliberate and bounded: every caller treats None as "could not determine"
    and reports it as such, so a permission error on one catalogue query degrades one finding
    rather than ending the preflight.
    """
    try:
        row = connection.execute(sql).fetchone()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - see the docstring; a failed probe is a finding, not a crash.
        return None
    if not row:
        return None
    return row[0]


def _check_vector(connection: object) -> Finding:
    """Is pgvector usable, and if not, can this database install it?

    Three states, not two, because the advice differs completely. Installed is fine. Available but
    not created is a one-line fix the user can perform. Not available at all means the extension is
    absent from the server and needs a package installed by whoever administers it — which, on a
    remote database, may not be the person running this.
    """
    if _scalar(connection, "SELECT to_regtype('vector')") is not None:
        return Finding(name="pgvector", ok=True, detail="extension installed")

    available = _scalar(
        connection, "SELECT 1 FROM pg_available_extensions WHERE name = 'vector'"
    )
    if available is not None:
        return Finding(
            name="pgvector",
            ok=False,
            detail="available on the server but not created in this database",
            blocking=True,
            advice="run `CREATE EXTENSION vector;` in this database, then re-run",
        )
    return Finding(
        name="pgvector",
        ok=False,
        detail="not available on this server",
        blocking=True,
        advice=(
            "install the pgvector package for this PostgreSQL (for example `apt install "
            "postgresql-16-pgvector`), then `CREATE EXTENSION vector;`. On a managed or remote "
            "server this is a job for whoever administers it."
        ),
    )


def _check_create_privilege(connection: object) -> Finding:
    """Can this role create the objects the schema needs?

    Asked of the CURRENT database rather than assumed from the role name. A read-only user is a
    perfectly ordinary thing to have in a connection string, and discovering it during migration
    means discovering it after the user has been told the install started.
    """
    allowed = _scalar(
        connection,
        "SELECT has_database_privilege(current_user, current_database(), 'CREATE')",
    )
    role = _scalar(connection, "SELECT current_user") or "this role"
    if allowed is True:
        return Finding(name="privileges", ok=True, detail=f"{role} may create objects")
    if allowed is False:
        return Finding(
            name="privileges",
            ok=False,
            detail=f"{role} may not create objects in this database",
            blocking=True,
            advice=(
                f"grant it: `GRANT CREATE ON DATABASE <db> TO {role};`, or supply a separate "
                "migration DSN whose role owns the schema"
            ),
        )
    return Finding(
        name="privileges",
        ok=None,
        detail="could not be determined",
        advice="the install will find out when it applies the schema",
    )


def _check_dimension(connection: object, expected: int | None) -> tuple[int | None, Finding]:
    """Does an existing `chunks.embedding` match the embedder we are about to use?

    ⚠️ **This is the check that pays for the module.** A dimension mismatch does not fail during
    setup. The schema is already there, so nothing is applied, the build starts, and the FIRST
    INSERT fails — minutes in, with a driver error about vector width that names neither the
    embedder nor the schema that disagree. `headless.py` documents that failure for a hardcoded
    dimension; pointing at a database someone else built reaches it by another road.
    """
    found = _scalar(
        connection,
        """
        SELECT a.atttypmod
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = 'chunks'
          AND a.attname = 'embedding'
          AND n.nspname = current_schema()
        """,
    )
    if not isinstance(found, int):
        # None means no such column, which is the first-install case. A non-integer means the
        # catalogue answered something this code does not understand, and guessing a dimension from
        # it is how the check it exists to perform gets skipped silently.
        return None, Finding(
            name="schema",
            ok=True,
            detail="no recall schema yet, which is normal for a first install",
        )

    if found < 0:
        # ⚠️ `atttypmod` is -1 for a `vector` column declared with NO dimension. Verified against a
        # real pgvector: `vector(384)` gives exactly 384 (NOT 384 + 4, which is what varchar's
        # header-size convention would suggest and what this code would otherwise have assumed),
        # and a bare `vector` gives -1. Treating -1 as a dimension would report every such column
        # as a mismatch against a number nobody chose.
        return None, Finding(
            name="schema",
            ok=None,
            detail="a `chunks.embedding` column exists but declares no dimension",
            advice="the install will use it as-is; recall's own schema always declares one",
        )

    dimension = found
    if expected is None:
        return dimension, Finding(
            name="schema",
            ok=None,
            detail=f"already present, vector({dimension}); no embedder given to compare against",
        )
    if dimension == expected:
        return dimension, Finding(
            name="schema",
            ok=True,
            detail=f"already present and matches the embedder, vector({dimension})",
        )
    return dimension, Finding(
        name="schema",
        ok=False,
        detail=f"already present as vector({dimension}), but the embedder produces {expected}",
        blocking=True,
        advice=(
            "these cannot coexist in one schema. Choose an embedder with the matching dimension, "
            "or use a different database or PostgreSQL schema for this install. Do NOT drop the "
            "existing one without knowing what indexed it."
        ),
    )

