from __future__ import annotations

import json
import sys
from collections.abc import Callable
from datetime import datetime
from typing import TypeVar
from ipaddress import ip_address
from urllib.parse import unquote, urlsplit

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

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
    print(msg, file=sys.stderr)
    return msg


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

    def __init__(self, dsn: str, dim: int, table: str = "chunks") -> None:
        # `dim` and `table` are interpolated directly into SQL — as a type modifier and an
        # identifier respectively — because Postgres cannot bind those as parameters. They
        # are therefore strictly validated here: this is the SQL-injection guard. Every
        # other value in this class is passed via psycopg bound parameters, never formatted.
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive int")
        if not table.isidentifier():
            raise ValueError("table must be a valid SQL identifier")
        self._dsn = dsn
        self._dim = dim
        self._table = table
        self._supersession_cache: tuple[dict[str, str], frozenset[str]] | None = None
        self._closed = False
        self._conn = self._connect()

    def _connect(self) -> "psycopg.Connection":
        """Open one autocommit connection and prepare it (extension + vector type registration)."""
        conn = psycopg.connect(self._dsn, autocommit=True)
        try:
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
        try:
            return op(self._conn)
        except self._CONN_ERRORS:
            # getattr: `broken` only exists from psycopg 3.2 and the declared floor is 3.1 —
            # without the default this except-block would raise AttributeError and mask the
            # original database error on an older install.
            if not (self._conn.closed or getattr(self._conn, "broken", False)):
                raise
            print("recall: database connection lost — reconnecting", file=sys.stderr)
            self._reconnect()
            return op(self._conn)

    @property
    def table(self) -> str:
        return self._table

    def close(self) -> None:
        """Close the connection for good.

        Sticky by design: without the flag, any later call would hit `_with_retry`'s reconnect
        and silently resurrect a connection nobody owns — a leak on first accidental reuse.
        """
        self._closed = True
        self._conn.close()

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
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector({self._dim}),
                    indexed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
                )
                """
            )
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

        self._with_retry(_op)

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        self._supersession_cache = None  # metadata may change; recompute on next read
        self._with_retry(lambda conn: self._upsert_in(conn, chunks, embeddings))
        return len(chunks)

    def _upsert_in(
        self, conn: "psycopg.Connection", chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        # One transaction for the whole batch: a mid-loop failure rolls the batch back
        # instead of leaving earlier rows committed (the connection is autocommit). When called
        # from replace_sources' outer transaction this becomes a savepoint (same commit).
        t = self._table
        with conn.transaction(), conn.cursor() as cur:
            for c, e in zip(chunks, embeddings):
                cur.execute(
                    f"""
                    INSERT INTO {t} (id, source, text, metadata, embedding, indexed_at)
                    VALUES (%s, %s, %s, %s, %s, now())
                    ON CONFLICT (id) DO UPDATE SET
                        source = EXCLUDED.source,
                        text = EXCLUDED.text,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding,
                        indexed_at = now()
                    """,
                    (c.id, c.source, c.text, json.dumps(c.metadata), Vector(e)),
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
        where = "WHERE source = %(source)s" if source else ""
        sql = f"""
            SELECT id, source, text, metadata, indexed_at, 1 - (embedding <=> %(vec)s) AS score
            FROM {t}
            {where}
            ORDER BY embedding <=> %(vec)s
            LIMIT %(k)s
        """
        params: dict = {"vec": Vector(vector), "k": k}
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
                    WHERE tsv @@ websearch_to_tsquery('english', %(q)s)
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
                WHERE tsv @@ websearch_to_tsquery('english', %(q)s)
                {where}
                ORDER BY score DESC
                LIMIT %(k)s
            """
        params: dict = {"q": text, "k": k}
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
                    conn.execute(f"DELETE FROM {self._table} WHERE source = ANY(%s)", (sources,))
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
            lambda conn: conn.execute(f"DELETE FROM {self._table} WHERE source = ANY(%s)", (sources,))
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
                f"WHERE metadata->>'file' = ANY(%s)",
                (files,),
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
        result is cached per store instance and invalidated by this instance's own writes
        (upsert / delete_sources); a concurrent writer on another connection is not observed
        until a new store is opened.
        """
        if self._supersession_cache is None:
            rows = self._with_retry(
                lambda conn: conn.execute(
                    f"""
                    SELECT DISTINCT metadata->>'file' AS file, metadata->>'supersedes' AS supersedes
                    FROM {self._table}
                    WHERE metadata ? 'file'
                    """
                ).fetchall()
            )
            self._supersession_cache = resolve_supersession(rows)
        edges, unresolved = self._supersession_cache
        return dict(edges), unresolved

    def supersession_map(self) -> dict[str, str]:
        """Resolvable supersession edges only — convenience view of `supersession`."""
        return self.supersession()[0]

    def newest_indexed_at(self) -> datetime | None:
        row = self._with_retry(
            lambda conn: conn.execute(f"SELECT max(indexed_at) FROM {self._table}").fetchone()
        )
        return row[0] if row else None

    def count(self) -> int:
        row = self._with_retry(
            lambda conn: conn.execute(f"SELECT count(*) FROM {self._table}").fetchone()
        )
        return int(row[0]) if row else 0
