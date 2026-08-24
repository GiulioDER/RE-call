"""One indexing process per corpus, enforced in the database rather than asked for in a doc.

⛔ **Why this is a lock and not a convention.** Embedding is the expensive half of indexing and it
is the half that allocates: bge-large at sequence 512 costs `batch x 16 heads x 512 x 512 x 4
bytes` for the attention scores alone, so one long chunk in a 256-item batch asks onnxruntime for
4.3 GB. Two indexers against one corpus therefore do not merely race on rows, they multiply the
peak allocation on whichever host is embedding, and the machine that hosts this project's corpus
also runs live services. The second cost is correctness: `Indexer.index_path` reads
`source_content_hashes()` once, decides what to skip, then prunes what is no longer on disk. Two
runs interleaved on that sequence both read the same "what is already here", and the one that
finishes second replaces rows the first had just written, having decided to skip them.

**Session-scoped, not `pg_advisory_xact_lock`.** An index run spans many transactions, so a
transaction lock is released at the first commit, before the interesting part starts. This holds
`pg_try_advisory_lock` on its OWN connection for the whole run, and closing that connection
releases it — so a killed indexer never leaves a lock behind, which is what makes it safe to have
no `--force` for stale locks. There is nothing to make stale.

The idiom is `GenerationManager.tenant_ingest_lock` (`recall/generations.py`), which serialises the
MCP upload path for the same reason. This is its sibling for the `index` path, keyed on
`(table, tenant)` rather than on the tenant alone, because two tables in one database are two
corpora.

⚠️ **What it does NOT cover, stated because a guard that overclaims is worse than none.** It
serialises indexing *into one corpus in one database*. It does not know about an embedding process
that writes nowhere — a benchmark sweep, a calibration run — and it cannot: those share the host's
memory, not the corpus, so they are the host's problem and belong to a host-level lock (on the
embedding host, `flock`). Two indexers against *different* tenants still run at once by design:
they are different corpora, and refusing that would serialise unrelated work.

**It refuses rather than waits, after a bounded poll.** A short wait absorbs the ordinary case of
two sessions closing seconds apart. Past that, an unbounded wait on an interactive command is
indistinguishable from a hang, and the refusal names the holder so the next question ("whose run is
it?") has an answer.
"""
from __future__ import annotations

import os
import socket
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any

import psycopg

from recall.observability import get_logger
from recall.errors import RecallError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from recall.store import PgVectorStore

_log = get_logger(__name__)

#: Bare word that lets a run proceed while another holds the lock. Loud rather than silent: it
#: logs a warning naming itself, because an escape nobody can see is an escape nobody audits.
ENV_ALLOW_CONCURRENT = "RECALL_INDEX_ALLOW_CONCURRENT"

#: How long to poll before refusing. Chosen for the case this actually protects: two sessions
#: closing within seconds of each other, where the second waits and then proceeds. It is NOT
#: sized to outwait a real index run (those are minutes to hours), and it must not be — waiting
#: out a long run is what makes a command look hung.
DEFAULT_WAIT_SECONDS = 20.0

_POLL_SECONDS = 0.25


class ConcurrentIndex(RuntimeError, RecallError):
    """Another process is indexing this corpus. Carries the holder's identity in its message."""


def lock_name(table: str, tenant: str) -> str:
    """The advisory-lock name for one corpus.

    `\\x1f` (unit separator) joins the parts rather than `:`, so a tenant that contains the
    separator cannot be made to collide with another table's key.
    """
    return f"index\x1f{table}\x1f{tenant}"


def _application_name(table: str, tenant: str) -> str:
    """What a blocked run will read in `pg_stat_activity` when it asks who holds the lock.

    Postgres truncates `application_name` at 63 bytes, so the identifying part (host and pid)
    comes FIRST: a truncated name that still says which machine and which process is useful,
    while one truncated to "recall index chunks/very-long-tena…" is not.
    """
    host = socket.gethostname()[:20]
    return f"{host} pid={os.getpid()} recall index {table}/{tenant}"[:63]


def _holder(conn: "psycopg.Connection", key: str) -> str:
    """Describe whoever holds `key`, or say plainly that it could not be identified.

    The join is on the lock's own classid/objid rather than on a `<<`-recombined bigint:
    `hashtextextended` returns a signed bigint, and the two halves land in unsigned `oid` columns,
    so recombining them gets the sign wrong for half of all keys. Masking each half instead is
    correct for both signs.
    """
    sql = """
        WITH k AS (SELECT hashtextextended(%s, 0) AS key)
        SELECT a.pid, a.application_name, a.client_addr::text, a.backend_start::text, a.state
        FROM pg_locks l
        JOIN pg_stat_activity a USING (pid), k
        WHERE l.locktype = 'advisory'
          AND l.granted
          AND l.objsubid = 1
          AND l.classid = ((k.key >> 32) & 4294967295)::oid
          AND l.objid = (k.key & 4294967295)::oid
        LIMIT 1
    """
    try:
        row = conn.execute(sql, (key,)).fetchone()
    except Exception:  # noqa: BLE001 - diagnostics must never REPLACE the refusal
        # Deliberately wider than `psycopg.Error`. This runs inside the branch that is about to
        # raise `ConcurrentIndex`, so anything raised here would take the place of the error the
        # caller needs, and the caller would be told the wrong thing about their own run.
        return "another connection (the holder could not be looked up)"
    if row is None:
        # Reachable and not an error: the holder released between the failed acquire and this
        # query, or `pg_stat_activity` hides it because it belongs to a different role.
        return "another connection (it did not appear in pg_stat_activity)"
    pid, app, addr, started, state = row
    where = f" from {addr}" if addr else ""
    return (
        f"pid {pid}{where}, connected {started}, state {state}, "
        f"application_name {app or '(unset)'!s}"
    )


@contextmanager
def single_writer(
    store: "PgVectorStore | Any",
    *,
    wait_seconds: float = DEFAULT_WAIT_SECONDS,
) -> Iterator[bool]:
    """Hold this corpus's index lock for the duration of the block. Yields whether it was taken.

    Yields `False`, having locked nothing, in exactly two cases, and both are reported rather than
    assumed:

    * **the store has no DSN** — an in-memory or stubbed store used by a test. There is no shared
      database, so there is nothing to serialise against;
    * **the escape hatch is set** — `RECALL_INDEX_ALLOW_CONCURRENT`, which logs a warning.

    A failure to CONNECT is not one of those cases and is not swallowed: the caller is about to
    write to that same database, so a DSN this cannot reach is a problem the caller wants raised
    now rather than discovered halfway through an embedding run.
    """
    dsn = getattr(store, "_dsn", None)
    # `str(... or default)`: a store is duck-typed here on purpose (tests pass stubs, and the
    # generation store is a subclass), so neither attribute is guaranteed to be a string. A `None`
    # tenant reaching the key would build a lock name that reads plausibly and matches nothing.
    table = str(getattr(store, "_table", None) or "chunks")
    tenant = str(getattr(store, "_tenant", None) or getattr(store, "tenant", None) or "default")
    if not dsn:
        yield False
        return
    if os.environ.get(ENV_ALLOW_CONCURRENT, "").strip():
        _log.warning(
            "%s is set: indexing %s/%s WITHOUT the single-writer lock. Two runs against one "
            "corpus multiply the embedder's peak memory and can overwrite each other's rows.",
            ENV_ALLOW_CONCURRENT, table, tenant,
        )
        yield False
        return

    key = lock_name(table, tenant)
    conn = psycopg.connect(
        dsn, autocommit=True, connect_timeout=10, application_name=_application_name(table, tenant)
    )
    try:
        deadline = time.monotonic() + wait_seconds
        while True:
            row = conn.execute(
                "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))", (key,)
            ).fetchone()
            if row is not None and row[0]:
                break
            if time.monotonic() >= deadline:
                raise ConcurrentIndex(
                    f"another process is already indexing {table}/{tenant}: {_holder(conn, key)}.\n"
                    "Indexing one corpus is serialised on purpose: two runs read the same "
                    "'what is already indexed', so the second replaces rows the first had just "
                    "written and decided to skip, and two embedders at once multiply the peak "
                    "memory on the host doing the work.\n"
                    "Wait for that run to finish and try again. If you have confirmed the two "
                    f"runs cannot touch the same files, set {ENV_ALLOW_CONCURRENT}=1."
                )
            time.sleep(_POLL_SECONDS)
        try:
            yield True
        finally:
            with suppress(Exception):
                conn.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (key,))
    finally:
        with suppress(Exception):
            conn.close()
