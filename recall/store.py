from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import datetime
from typing import TypeVar
from ipaddress import ip_address
from urllib.parse import unquote, urlsplit

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from recall.observability import METRICS, get_logger
from recall.types import Chunk, ScoredChunk

#: The built-in dev credentials shipped in the default DSN — safe only against a local database.
_DEFAULT_CREDS = ("recall", "recall")
#: "" covers a hostless/unix-socket DSN. Bracketed IPv6 is absent on purpose: urlsplit strips
#: the brackets. All of 127.0.0.0/8 is handled numerically by `_is_local_host`.
_LOCAL_HOSTS = ("", "localhost", "::1", "0.0.0.0", "host.docker.internal")


def _is_local_host(host: str) -> bool:
    """True when `host` cannot reach a shared database (loopback, unix socket, or unset)."""
    if host in _LOCAL_HOSTS or host.startswith(("/", "%2f")):  # %2f: percent-encoded socket dir
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def redacted_dsn(dsn: str) -> str:
    """`dsn` with any password removed — safe to print to a log or a systemd journal.

    A connection failure is exactly when an operator wants the DSN in the logs, and exactly when
    printing it verbatim would write the password to disk.
    """
    try:
        parts = urlsplit(dsn)
        if not parts.hostname:
            return "<dsn>"
        userinfo = f"{parts.username}:***@" if parts.password else (
            f"{parts.username}@" if parts.username else ""
        )
        port = f":{parts.port}" if parts.port else ""
        return f"{parts.scheme}://{userinfo}{parts.hostname}{port}{parts.path}"
    except ValueError:  # pragma: no cover - malformed URL
        return "<dsn>"

_T = TypeVar("_T")

_log = get_logger("store")


def warn_if_insecure_dsn(dsn: str) -> str | None:
    """Warn (to stderr) when the built-in ``recall:recall`` credentials target a NON-local host.

    A shared, well-known password is fine against localhost but a footgun the moment the DSN
    points at a real remote database. This warns loudly and returns the message; it never blocks
    execution (returns None when there is nothing to warn about).
    """
    try:
        parts = urlsplit(dsn)
    except ValueError:
        return None
    # unquote: urlsplit returns the RAW percent-encoded form, so "recal%6C" is the password
    # "recall" and must not slip past the comparison
    if (unquote(parts.username or ""), unquote(parts.password or "")) != _DEFAULT_CREDS:
        return None
    if _is_local_host((parts.hostname or "").lower()):
        return None
    msg = (
        f"recall: WARNING — using the default 'recall:recall' credentials against non-local host "
        f"{parts.hostname!r}. Set a strong password via RECALL_DSN before using a remote database."
    )
    _log.warning(msg)
    return msg


#: Tenant assigned to rows written before tenancy existed, and the default for a
#: single-tenant deployment — so an upgrade changes nothing for an existing install.
DEFAULT_TENANT = "default"
#: Postgres session variable the row-level-security policy reads. A custom GUC (it must contain
#: a dot) set per connection, so the policy compares against the connection's own tenant.
TENANT_GUC = "recall.tenant_id"

#: Opt-out for `require_secure_dsn`. Named so it cannot be set by accident, and so its presence
#: in a deploy is a visible, greppable decision rather than an oversight.
INSECURE_DSN_OPT_OUT = "RECALL_ALLOW_INSECURE_DSN"


def require_secure_dsn(dsn: str) -> None:
    """Raise unless `dsn` is safe to use unattended; the fail-closed form of the warning above.

    `warn_if_insecure_dsn` detects the built-in `recall:recall` credentials against a remote host
    and then RETURNS, so the process carries on talking to a shared database with a password
    published in this repository's README. A warning on stderr is not a control: under systemd it
    lands in a journal nobody reads, and the server comes up looking healthy.

    A server should therefore call this instead. The escape hatch is an explicit environment
    variable, because the legitimate case (a private network where the operator has genuinely
    accepted the risk) must be expressible — just not by default and not silently.
    """
    if os.environ.get(INSECURE_DSN_OPT_OUT):
        return
    if warn_if_insecure_dsn(dsn) is None:
        return
    raise PermissionError(
        f"refusing to start against {redacted_dsn(dsn)}: the default 'recall:recall' credentials "
        f"are published in this project's README and this DSN points at a non-local host. Set a "
        f"real password, or set {INSECURE_DSN_OPT_OUT}=1 to accept the risk deliberately."
    )


def _basename(file: str) -> str:
    """Basename of a root-relative (posix) file identifier."""
    return file.rsplit("/", 1)[-1]


def resolve_supersession(
    rows: list[tuple[str | None, str | None]],
) -> tuple[dict[str, str], frozenset[str]]:
    """Build the superseded -> superseding map from ``(file, supersedes)`` rows.

    ``file`` is a root-relative path; ``supersedes`` references its target by basename (the
    authoring convention). Three cases:

    - **Unambiguous** (exactly one indexed file bears that basename): resolve to its
      root-relative path. This is the fix for the original bug — a naive basename key would
      have collided with an unrelated same-named file in another directory.
    - **Dangling** (no indexed file bears that basename — the predecessor was never indexed,
      or was deleted): fall back to the raw basename as the key. There is nothing to
      disambiguate, so this cannot mis-map; dropping it would just as silently discard a valid
      supersession claim (e.g. a memo intentionally superseding a doc that was since removed).
    - **Ambiguous** (two or more indexed files share that basename): do not guess — a silent
      mis-map to the wrong file is worse than a broken chain, since we cannot tell which one the
      author meant. The candidates are returned in ``unresolved`` so the read path can fail
      closed on them; dropping the edge and saying nothing would leave a possibly-superseded
      memory looking perfectly healthy.

    Both keys and values in the mapping are root-relative paths (or a bare basename for the
    dangling case). ``unresolved`` holds root-relative paths.

    Pure and DB-free so the resolution rule can be unit-tested without a database.
    """
    files = [f for f, _ in rows if f]
    by_base: dict[str, list[str]] = {}
    for f in files:
        by_base.setdefault(_basename(f), []).append(f)
    mapping: dict[str, str] = {}
    unresolved: set[str] = set()
    for file, supersedes in rows:
        if not file or not supersedes:
            continue
        target_basename = _basename(supersedes)
        candidates = by_base.get(target_basename, [])
        if len(candidates) == 1:
            mapping[candidates[0]] = file
        elif len(candidates) == 0:
            mapping[target_basename] = file
        else:
            # Ambiguous: don't guess — but don't stay silent either. Dropping the edge alone
            # would leave the (possibly superseded) memories looking perfectly `ok`, which is
            # the same wrong answer the trust layer exists to prevent. Naming them lets the
            # read path fail closed and tell the operator what to fix.
            unresolved.update(candidates)
    return mapping, frozenset(unresolved)


class PgVectorStore:
    """The single, production-grade vector store: PostgreSQL + pgvector."""

    #: Errors that mean the connection itself is broken (dropped socket, server restart, idle
    #: timeout) — as opposed to a query/data error. These trigger a reconnect-and-retry.
    _CONN_ERRORS = (psycopg.OperationalError, psycopg.InterfaceError)

    #: Class-level default so `_pool` always exists, including on an instance built without
    #: __init__ (the store tests do this to exercise the retry logic against a fake connection).
    #: Single-connection mode is the default, so None is also the honest value.
    _pool = None

    def __init__(
        self,
        dsn: str,
        dim: int,
        table: str = "chunks",
        *,
        tenant: str = DEFAULT_TENANT,
        pool_size: int | None = None,
        statement_timeout_ms: int | None = None,
        connect_timeout_s: int | None = 10,
    ) -> None:
        """Open a store against `dsn`.

        `pool_size` selects the CONNECTION MODE, and the default (None) is deliberate:

        - **None — one long-lived connection.** Correct for a CLI or any single-threaded caller,
          and the mode whose reconnect semantics the rest of this class is built around. Sharing
          it across threads serialises them, and a reconnect swaps `self._conn` underneath a
          thread that is using it.
        - **an int — a connection pool.** What a server needs: each operation borrows its own
          connection, so concurrent callers actually proceed concurrently. Opt-in rather than
          default because a pool starts a background maintenance thread, which a one-shot CLI
          invocation should not pay for or have to shut down.

        `statement_timeout_ms` bounds every statement server-side. Without it a single runaway
        query occupies a connection until the process dies, with nothing to cancel it; that is
        the difference between a slow request and an exhausted pool.
        """
        # `dim` and `table` are interpolated directly into SQL — as a type modifier and an
        # identifier respectively — because Postgres cannot bind those as parameters. They
        # are therefore strictly validated here: this is the SQL-injection guard. Every
        # other value in this class is passed via psycopg bound parameters, never formatted.
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive int")
        if not table.isidentifier():
            raise ValueError("table must be a valid SQL identifier")
        if pool_size is not None and (not isinstance(pool_size, int) or pool_size < 1):
            raise ValueError("pool_size must be a positive int or None")
        if not isinstance(tenant, str) or not tenant:
            raise ValueError("tenant must be a non-empty str")
        self._dsn = dsn
        self._dim = dim
        self._table = table
        self._tenant = tenant
        self._statement_timeout_ms = statement_timeout_ms
        self._connect_timeout_s = connect_timeout_s
        #: (fingerprint, edges, unresolved) — see `supersession()`. The fingerprint is what
        #: makes the cache safe to reuse across processes.
        self._supersession_cache: tuple | None = None
        #: Count of full supersession scans actually performed (cache misses). Surfaced so a
        #: test can prove the cache still works, and so a rescan storm is visible as a metric.
        self._supersession_scans = 0
        self._closed = False
        self._pool = self._open_pool(pool_size) if pool_size else None
        self._conn = None if self._pool is not None else self._connect()

    def _connect_kwargs(self) -> dict:
        kw: dict = {"autocommit": True}
        if self._connect_timeout_s is not None:
            # Without this a dead host hangs the caller on the TCP handshake indefinitely.
            kw["connect_timeout"] = self._connect_timeout_s
        return kw

    def _open_pool(self, size: int):
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "pool_size requires the pool extra: pip install recall[pool]"
            ) from exc

        # `configure` runs on every connection the pool creates, not just the first — the vector
        # type registration is per-connection state, so a pool that skipped it would work until
        # it opened its second connection and then fail on a Vector parameter.
        pool = ConnectionPool(
            self._dsn,
            min_size=1,
            max_size=size,
            kwargs=self._connect_kwargs(),
            configure=self._prepare,
            open=False,
        )
        pool.open(wait=True, timeout=self._connect_timeout_s or 30)
        return pool

    def _prepare(self, conn: "psycopg.Connection") -> None:
        """Per-connection setup: extension, vector type registration, statement timeout."""
        # register_vector needs the `vector` type to already exist, so ensure the extension
        # is installed first — this makes a brand-new database work out of the box (the
        # README quickstart path). If this role lacks privilege to create it, fall through:
        # register_vector still succeeds when an admin has pre-installed the extension, and
        # fails with a clear "vector type not found" if it genuinely isn't there.
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except psycopg.Error:
            pass
        register_vector(conn)
        # Per-connection tenant for the RLS policy. Safe to set once at connection setup because
        # a store is bound to ONE tenant: the pool belongs to the store, so no connection is ever
        # shared between tenants. A server handling many tenants opens a store per tenant.
        conn.execute(f"SELECT set_config('{TENANT_GUC}', %s, false)", (self._tenant,))
        if self._statement_timeout_ms is not None:
            conn.execute(f"SET statement_timeout = {int(self._statement_timeout_ms)}")

    def _connect(self) -> "psycopg.Connection":
        """Open one autocommit connection and prepare it (extension + vector type registration)."""
        conn = psycopg.connect(self._dsn, **self._connect_kwargs())
        try:
            self._prepare(conn)  # same per-connection setup the pool applies via `configure`
        except Exception:
            conn.close()
            raise
        return conn

    def _reconnect(self) -> None:
        """Discard the (broken) connection and open a fresh, prepared one."""
        try:
            self._conn.close()
        except Exception:
            pass
        self._conn = self._connect()

    def _with_retry(self, op: Callable[["psycopg.Connection"], _T]) -> _T:
        """Run ``op(conn)``; on a broken-connection error, reconnect once and retry.

        A single long-lived autocommit connection can be severed by the server (idle timeout,
        restart, transient network blip). Rather than failing every subsequent call, transparently
        reconnect and retry the operation exactly once — a second failure propagates. Safe because
        every ``op`` here is a single self-contained statement or an atomic transaction that rolls
        back cleanly, so re-running it on a fresh connection cannot double-apply.

        The retry is deliberately narrow. ``OperationalError`` is NOT a synonym for "the
        connection is gone" — ``QueryCanceled`` (statement_timeout), ``DeadlockDetected`` and
        ``SerializationFailure`` are all subclasses raised on a perfectly LIVE connection.
        Retrying those re-runs the statement on a fresh session that no longer carries the
        setting which killed it, i.e. silently escapes the very guard that fired. So the retry
        additionally requires the connection to be observably dead.

        A reconnect is REPORTED to stderr: a silent one hides an outage behind a process that
        still looks healthy, which is how a dead dependency goes unnoticed for days.
        """
        if self._closed:
            raise RuntimeError("store is closed")
        if self._pool is not None:
            return self._with_retry_pooled(op)
        try:
            return op(self._conn)
        except self._CONN_ERRORS:
            # getattr: `broken` only exists from psycopg 3.2 and the declared floor is 3.1 —
            # without the default this except-block would raise AttributeError and mask the
            # original database error on an older install.
            if not (self._conn.closed or getattr(self._conn, "broken", False)):
                raise
            _log.warning("database connection lost — reconnecting")
            METRICS.increment("recall_db_reconnects_total")
            self._reconnect()
            return op(self._conn)

    def _with_retry_pooled(self, op: Callable[["psycopg.Connection"], _T]) -> _T:
        """Pooled variant: borrow a connection per operation, retry once on a dead one.

        The pool already replaces connections it knows are broken, but it can hand out one that
        died while idle and has not been probed yet, so the first use still fails. Retrying
        borrows a DIFFERENT connection — the pooled equivalent of reconnecting.

        The same narrow predicate as the single-connection path applies, and for the same
        reason: `QueryCanceled` from `statement_timeout` is an `OperationalError` raised on a
        perfectly live connection, and retrying it on a fresh session would silently escape the
        timeout that fired. So a retry additionally requires the connection to be observably
        dead. Nothing here is a `nonlocal` on shared state: each borrow is thread-confined,
        which is the whole point of the mode.
        """
        for attempt in (0, 1):
            with self._pool.connection() as conn:
                try:
                    return op(conn)
                except self._CONN_ERRORS:
                    dead = conn.closed or getattr(conn, "broken", False)
                    if not dead or attempt == 1:
                        raise
                    _log.warning("pooled database connection lost — retrying on another")
                    METRICS.increment("recall_db_reconnects_total", pooled="true")
        raise AssertionError("unreachable: the loop either returns or raises")  # pragma: no cover

    @property
    def table(self) -> str:
        return self._table

    def close(self) -> None:
        """Close the connection (or pool) for good.

        Sticky by design: without the flag, any later call would hit `_with_retry`'s reconnect
        and silently resurrect a connection nobody owns — a leak on first accidental reuse.
        """
        self._closed = True
        if self._pool is not None:
            self._pool.close()  # also stops the pool's background maintenance thread
        else:
            self._conn.close()

    def drop_table(self) -> None:
        """Drop this store's table if it exists.

        Exists so callers stop reaching into `store._conn` to do it (the eval harness, the
        calibration runner, the semantic linter and the test fixtures all did). That reach-through
        is not merely untidy: in pooled mode there IS no `_conn`, so every one of those call
        sites would raise `AttributeError` on a store configured for a server.
        """
        self._supersession_cache = None
        self._with_retry(lambda conn: conn.execute(f"DROP TABLE IF EXISTS {self._table}"))

    def __enter__(self) -> "PgVectorStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def ensure_schema(self) -> None:
        def _op(conn: "psycopg.Connection") -> None:
            t = self._table
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {t} (
                    tenant_id TEXT NOT NULL DEFAULT '{DEFAULT_TENANT}',
                    id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector({self._dim}),
                    indexed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
                    PRIMARY KEY (tenant_id, id)
                )
                """
            )
            self._migrate_to_tenanted(conn)
            actual_dim = conn.execute(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = %s::regclass AND attname = 'embedding'",
                (t,),
            ).fetchone()[0]
            if actual_dim > 0 and actual_dim != self._dim:
                raise ValueError(
                    f"table {t!r} has a vector({actual_dim}) embedding column but this store is "
                    f"configured for dim {self._dim} — use a matching embedder or a different "
                    f"table (drop and re-index for a clean slate)."
                )
            conn.execute(f"CREATE INDEX IF NOT EXISTS {t}_tsv_idx ON {t} USING GIN (tsv)")
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {t}_emb_idx ON {t} "
                f"USING hnsw (embedding vector_cosine_ops)"
            )
            # The vector and full-text indexes cover the two retrieval legs; these cover the
            # rest of the hot path, each of which was a sequential scan growing with the corpus:
            #   indexed_at — `newest_indexed_at()` runs a max() on EVERY search; a DESC index
            #                turns it into a one-row backward scan.
            #   source     — a source-filtered search cannot use HNSW without it, and
            #                replace_sources/delete_sources match `source = ANY(...)` per re-index.
            #   file       — the supersession map groups on metadata->>'file'.
            conn.execute(f"CREATE INDEX IF NOT EXISTS {t}_indexed_at_idx ON {t} (indexed_at DESC)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS {t}_source_idx ON {t} (source)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS {t}_file_idx ON {t} ((metadata->>'file'))")
            # Every hot-path predicate leads with tenant_id, so it leads the index too.
            conn.execute(f"CREATE INDEX IF NOT EXISTS {t}_tenant_idx ON {t} (tenant_id)")
            self._enable_rls(conn)

        self._with_retry(_op)

    def _migrate_to_tenanted(self, conn: "psycopg.Connection") -> None:
        """Add `tenant_id` to a table created before tenancy existed, idempotently.

        An existing deployment has `id` as the sole primary key. Chunk ids are derived from the
        file path, so two tenants indexing the same path produce the SAME id — with a single-column
        key, one tenant's re-index would overwrite the other's row. The key therefore has to become
        `(tenant_id, id)`, which means dropping the old constraint.

        Existing rows are assigned to `DEFAULT_TENANT`, so an upgrade is invisible to a
        single-tenant deployment: the store's default tenant is the same value.
        """
        t = self._table
        has_tenant = conn.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = %s AND column_name = 'tenant_id'",
            (t,),
        ).fetchone()
        if has_tenant:
            return
        conn.execute(
            f"ALTER TABLE {t} ADD COLUMN tenant_id TEXT NOT NULL DEFAULT '{DEFAULT_TENANT}'"
        )
        pkey = conn.execute(
            "SELECT conname FROM pg_constraint "
            "WHERE conrelid = %s::regclass AND contype = 'p'",
            (t,),
        ).fetchone()
        if pkey:
            conn.execute(f'ALTER TABLE {t} DROP CONSTRAINT "{pkey[0]}"')
        conn.execute(f"ALTER TABLE {t} ADD PRIMARY KEY (tenant_id, id)")

    def _enable_rls(self, conn: "psycopg.Connection") -> None:
        """Enforce tenant isolation in the DATABASE, not only in this class's WHERE clauses.

        Every query here already filters on `tenant_id`. That is the correctness mechanism and it
        works for any role. This is the CONTROL: a policy means a forgotten predicate — in future
        code, in a migration script, in someone's psql session — returns nothing instead of
        another tenant's memories.

        `FORCE` matters: without it the policy does not apply to the table's OWNER, which is
        usually the very role the application connects as.

        ⚠️ **A superuser bypasses RLS entirely, and so does a role with BYPASSRLS.** If the
        application connects as one (the default `docker-compose.yml` role IS a superuser), this
        policy is decoration and only the WHERE clauses are protecting you. See
        `check_rls_effective()`.
        """
        t = self._table
        conn.execute(f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY")
        conn.execute(f"ALTER TABLE {t} FORCE ROW LEVEL SECURITY")
        # CREATE POLICY has no IF NOT EXISTS before PG 15's ... it still doesn't; drop first.
        conn.execute(f"DROP POLICY IF EXISTS {t}_tenant_isolation ON {t}")
        conn.execute(
            f"CREATE POLICY {t}_tenant_isolation ON {t} "
            f"USING (tenant_id = current_setting('{TENANT_GUC}', true)) "
            f"WITH CHECK (tenant_id = current_setting('{TENANT_GUC}', true))"
        )

    def check_rls_effective(self) -> bool:
        """True when row-level security actually constrains THIS connection's role.

        Returns False for a superuser or a `BYPASSRLS` role — for whom the policy created above
        is inert. Exposed rather than merely documented because "we enabled RLS" is the kind of
        claim that gets believed without being true, and the difference is invisible until a
        tenant reads another tenant's memory.
        """
        row = self._with_retry(
            lambda conn: conn.execute(
                "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user"
            ).fetchone()
        )
        return not (row and row[0])

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        self._supersession_cache = None  # metadata may change; recompute on next read
        self._with_retry(lambda conn: self._upsert_in(conn, chunks, embeddings))
        return len(chunks)

    def _upsert_in(
        self, conn: "psycopg.Connection", chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        # NUL bytes cannot be stored in a PostgreSQL text column, and psycopg's own error names
        # neither the row nor the source — indexing a real 792-file corpus failed on TWO bytes in
        # ONE file with no indication of which. Fail here instead, pointing at the chunk. The
        # Indexer strips them before this (with a warning); this catches the direct-API caller.
        for c in chunks:
            if "\x00" in c.text or "\x00" in c.id or "\x00" in c.source:
                raise ValueError(
                    f"chunk {c.id!r} from source {c.source!r} contains a NUL (0x00) byte, which "
                    f"PostgreSQL text columns cannot store — strip it before upserting"
                )
        # One transaction for the whole batch: a mid-loop failure rolls the batch back
        # instead of leaving earlier rows committed (the connection is autocommit). When called
        # from replace_sources' outer transaction this becomes a savepoint (same commit).
        t = self._table
        # executemany, not a Python loop of execute(): psycopg3 pipelines it into one round
        # trip per batch instead of one per row. A full re-index is thousands of rows, and at
        # ~0.2-0.5ms of round-trip each that loop was seconds of pure latency.
        with conn.transaction(), conn.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {t}
                    (tenant_id, id, source, text, metadata, embedding, indexed_at)
                VALUES (%s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (tenant_id, id) DO UPDATE SET
                    source = EXCLUDED.source,
                    text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    indexed_at = now()
                """,
                [
                    (self._tenant, c.id, c.source, c.text, json.dumps(c.metadata), Vector(e))
                    for c, e in zip(chunks, embeddings)
                ],
            )

    def _rows_to_hits(self, rows: list[tuple]) -> list[ScoredChunk]:
        hits: list[ScoredChunk] = []
        for cid, source, text, metadata, indexed_at, score in rows:
            md = metadata if isinstance(metadata, dict) else json.loads(metadata)
            hits.append(
                ScoredChunk(
                    chunk=Chunk(id=cid, source=source, text=text, metadata=md),
                    score=float(score),
                    indexed_at=indexed_at,
                )
            )
        return hits

    def query_dense(
        self, vector: list[float], k: int, source: str | None = None
    ) -> list[ScoredChunk]:
        if k <= 0:
            raise ValueError("k must be a positive int")
        t = self._table
        where = "AND source = %(source)s" if source else ""
        sql = f"""
            SELECT id, source, text, metadata, indexed_at, 1 - (embedding <=> %(vec)s) AS score
            FROM {t}
            WHERE tenant_id = %(tenant)s {where}
            ORDER BY embedding <=> %(vec)s
            LIMIT %(k)s
        """
        params: dict = {"vec": Vector(vector), "k": k, "tenant": self._tenant}
        if source:
            params["source"] = source
        rows = self._with_retry(lambda conn: conn.execute(sql, params).fetchall())
        return self._rows_to_hits(rows)

    def query_sparse(
        self, text: str, k: int, source: str | None = None, vec: list[float] | None = None
    ) -> list[ScoredChunk]:
        """Full-text search. Ranking is always ts_rank; when `vec` is given, each hit's `score`
        is its true dense cosine against `vec` instead of the ts_rank value, so lexical-only
        hits are comparable with dense hits downstream."""
        if k <= 0:
            raise ValueError("k must be a positive int")
        t = self._table
        where = "AND source = %(source)s" if source else ""
        if vec is not None:
            # cosine only for the k ts_rank winners — computed in the SELECT list of the flat
            # query it would run for EVERY tsquery-matching row before the sort discards them
            sql = f"""
                SELECT id, source, text, metadata, indexed_at,
                       1 - (embedding <=> %(vec)s) AS score
                FROM (
                    SELECT id, source, text, metadata, indexed_at, embedding,
                           ts_rank(tsv, websearch_to_tsquery('english', %(q)s)) AS rank
                    FROM {t}
                    WHERE tenant_id = %(tenant)s
                      AND tsv @@ websearch_to_tsquery('english', %(q)s)
                    {where}
                    ORDER BY rank DESC
                    LIMIT %(k)s
                ) top_k
                ORDER BY rank DESC
            """
        else:
            sql = f"""
                SELECT id, source, text, metadata, indexed_at,
                       ts_rank(tsv, websearch_to_tsquery('english', %(q)s)) AS score
                FROM {t}
                WHERE tenant_id = %(tenant)s
                  AND tsv @@ websearch_to_tsquery('english', %(q)s)
                {where}
                ORDER BY score DESC
                LIMIT %(k)s
            """
        params: dict = {"q": text, "k": k, "tenant": self._tenant}
        if vec is not None:
            params["vec"] = Vector(vec)
        if source:
            params["source"] = source
        rows = self._with_retry(lambda conn: conn.execute(sql, params).fetchall())
        return self._rows_to_hits(rows)

    def replace_sources(
        self, sources: list[str], chunks: list[Chunk], embeddings: list[list[float]]
    ) -> int:
        """Atomically replace every row of `sources` with the given chunks.

        Delete + insert run in ONE transaction: a failure (or a concurrent reader) never
        observes the sources deleted without their replacement rows. Callers must compute
        `embeddings` BEFORE calling — an embedding failure then leaves the old rows intact.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        self._supersession_cache = None

        def _op(conn):
            with conn.transaction():
                if sources:
                    conn.execute(
                        f"DELETE FROM {self._table} "
                        f"WHERE tenant_id = %s AND source = ANY(%s)",
                        (self._tenant, sources),
                    )
                if chunks:
                    self._upsert_in(conn, chunks, embeddings)  # savepoint, same commit

        self._with_retry(_op)
        return len(chunks)

    def delete_sources(self, sources: list[str]) -> int:
        """Delete every chunk belonging to the given `source` values; returns rows removed.

        Standalone removal API (the Indexer uses the atomic `replace_sources` instead).
        """
        if not sources:
            return 0
        self._supersession_cache = None
        cur = self._with_retry(
            lambda conn: conn.execute(
                f"DELETE FROM {self._table} WHERE tenant_id = %s AND source = ANY(%s)",
                (self._tenant, sources),
            )
        )
        return cur.rowcount or 0

    def touch_files(self, files: list[str]) -> int:
        """Reset ``indexed_at`` to now() for every chunk whose metadata file name matches.

        A timestamp-only touch — text and embeddings are untouched by construction, so it is
        the honest way to simulate a re-sync (the eval's recency arm uses it; re-indexing
        identical text would also re-embed, which is only a no-op for deterministic
        embedders). Matches on the ``file`` metadata key (basename), so it works for nested
        corpora too. Returns rows updated.
        """
        if not files:
            return 0
        cur = self._with_retry(
            lambda conn: conn.execute(
                f"UPDATE {self._table} SET indexed_at = now() "
                f"WHERE tenant_id = %s AND metadata->>'file' = ANY(%s)",
                (self._tenant, files),
            )
        )
        return cur.rowcount or 0

    def supersession(self) -> tuple[dict[str, str], frozenset[str]]:
        """The supersession relation: ``(edges, unresolved)``.

        `edges` maps superseded file -> superseding file (both root-relative). `unresolved`
        names the files an edge pointed at but could not identify, so the read path can fail
        closed on them rather than serve them as healthy.


        The ``supersedes:`` frontmatter references its target by basename (the authoring
        convention), but files are identified by their root-relative path so same-named files in
        different directories cannot collide. This resolves each basename reference to the unique
        file that bears it: an AMBIGUOUS target (a basename shared by two files) is skipped
        rather than mis-mapped — the same refusal `recall lint` makes when it flags
        ``ambiguous-supersedes-target`` — so a stray sibling can never be silently marked
        superseded. A dangling target (no such basename in the corpus) is skipped too. The
        The result is cached, but the cache is VALIDATED against the table on every call rather
        than trusted. It previously was not: it was invalidated only by this instance's own
        writes, so a long-lived reader (an MCP server holds one store for its lifetime) never saw
        an edge written by a separate `recall index` run. It kept serving the superseded memory as
        `ok` until someone restarted the process — the trust layer returning exactly the wrong
        answer, silently, which is the failure it exists to prevent.

        Freshness is established by a cheap fingerprint — `(max(indexed_at), count(*))` for this
        tenant — and the expensive `DISTINCT` scan runs only when that fingerprint moves. Both
        halves are needed: `max(indexed_at)` alone cannot see a DELETE, and deleting a superseding
        document must stop its edge from applying, or the reader keeps demoting a memory that is
        current again.

        Measured on a 50k-row table: fingerprint ~12 ms, full scan ~80 ms. So this is cheaper than
        rescanning on every call and, unlike caching indefinitely, it is correct.

        Fingerprint and scan share ONE connection, so they cannot straddle a concurrent write and
        cache a result under a fingerprint that never described it.
        """

        def _op(conn: "psycopg.Connection"):
            fingerprint = conn.execute(
                f"SELECT max(indexed_at), count(*) FROM {self._table} WHERE tenant_id = %s",
                (self._tenant,),
            ).fetchone()
            cached = self._supersession_cache
            if cached is not None and cached[0] == fingerprint:
                return cached[1], cached[2]
            rows = conn.execute(
                f"""
                SELECT DISTINCT metadata->>'file' AS file, metadata->>'supersedes' AS supersedes
                FROM {self._table}
                WHERE tenant_id = %s AND metadata ? 'file'
                """,
                (self._tenant,),
            ).fetchall()
            self._supersession_scans += 1
            METRICS.increment("recall_supersession_scans_total")
            edges, unresolved = resolve_supersession(rows)
            self._supersession_cache = (fingerprint, edges, unresolved)
            return edges, unresolved

        edges, unresolved = self._with_retry(_op)
        return dict(edges), unresolved

    def source_content_hashes(self) -> dict[str, str]:
        """`{source: content_hash}` for this tenant — what the indexer compares against.

        One row per source: the hash is a property of the file, so every chunk of it carries the
        same value and `DISTINCT` collapses them. A source indexed before content hashing existed
        has no hash and is reported as `""`, which can never equal a real sha256 — so it is
        re-indexed once and then skipped like everything else.
        """
        rows = self._with_retry(
            lambda conn: conn.execute(
                f"SELECT DISTINCT source, coalesce(metadata->>'content_hash', '') "
                f"FROM {self._table} WHERE tenant_id = %s",
                (self._tenant,),
            ).fetchall()
        )
        return {source: content_hash for source, content_hash in rows}

    def supersession_map(self) -> dict[str, str]:
        """Resolvable supersession edges only — convenience view of `supersession`."""
        return self.supersession()[0]

    def newest_indexed_at(self) -> datetime | None:
        row = self._with_retry(
            lambda conn: conn.execute(
                f"SELECT max(indexed_at) FROM {self._table} WHERE tenant_id = %s",
                (self._tenant,),
            ).fetchone()
        )
        return row[0] if row else None

    def count(self) -> int:
        row = self._with_retry(
            lambda conn: conn.execute(
                f"SELECT count(*) FROM {self._table} WHERE tenant_id = %s",
                (self._tenant,),
            ).fetchone()
        )
        return int(row[0]) if row else 0
