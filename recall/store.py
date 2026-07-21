from __future__ import annotations

import json
import sys
from datetime import datetime
from ipaddress import ip_address
from urllib.parse import unquote, urlsplit

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from recall.types import Chunk, ScoredChunk

#: The credentials the README/quickstart/docker-compose ship. Fine on a developer's laptop,
#: a liability the moment the same DSN is pointed at a real host.
_DEFAULT_USER = "recall"
_DEFAULT_PASSWORD = "recall"
#: "" covers a hostless/unix-socket DSN. Bracketed IPv6 is absent on purpose: urlsplit
#: strips the brackets, so "[::1]" could never match. Numeric loopback is handled by
#: `_is_local_host` via the ip_address check, which covers all of 127.0.0.0/8.
_LOCAL_HOSTS = {"localhost", "::1", "", "host.docker.internal", "0.0.0.0"}


def _is_local_host(host: str) -> bool:
    """True when `host` cannot reach a shared database (loopback, unix socket, or unset)."""
    if host in _LOCAL_HOSTS or host.startswith(("/", "%2f")):  # %2f: percent-encoded socket dir
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def redacted_dsn(dsn: str) -> str:
    """`dsn` with any password removed — safe to print to a log or a journal.

    A connection failure is exactly when an operator wants the DSN in the logs, and exactly
    when printing it verbatim would write the password to disk.
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


def insecure_default_credentials(dsn: str) -> str | None:
    """Warning text if `dsn` ships the documented dev credentials at a non-local host, else None.

    A copy-pasted quickstart DSN that reaches a shared database is a weak-credentials
    deployment; say so loudly rather than connecting silently. Only URL-form DSNs are
    inspected — a keyword/value DSN is the caller's own construction, not a copied default.
    """
    try:
        parts = urlsplit(dsn)
        if not parts.scheme.startswith("postgres"):
            return None
        host = (parts.hostname or "").lower()
        # unquote: urlsplit returns the RAW percent-encoded form, so "recal%6C" is the
        # password "recall" and must not slip past the comparison
        if (
            unquote(parts.username or "") != _DEFAULT_USER
            or unquote(parts.password or "") != _DEFAULT_PASSWORD
        ):
            return None
        if _is_local_host(host):
            return None
    except ValueError:  # pragma: no cover - malformed URL
        return None
    return (
        f"recall: DSN uses the default quickstart credentials (recall:recall) against a "
        f"non-local host ({host}). These are published in the README and docker-compose — "
        f"create a dedicated role with its own password before using this database."
    )


class PgVectorStore:
    """The single, production-grade vector store: PostgreSQL + pgvector."""

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
        self._in_transaction = False
        self._closed = False
        warning = insecure_default_credentials(dsn)
        if warning:
            print(warning, file=sys.stderr)
        self._conn = self._open()

    def _open(self) -> psycopg.Connection:
        """Open a connection with the `vector` type installed and its adapters registered."""
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
            register_vector(conn)  # per-connection: a reconnect must redo this
        except Exception:
            conn.close()
            raise
        return conn

    def _execute(self, sql: str, params: object = None):
        """Run one statement, reconnecting ONCE if the connection has actually died.

        The MCP server opens a single connection for its whole lifetime, so without this a
        single DB restart or network blip would poison every later tool call until the server
        itself is restarted.

        The retry is deliberately narrow. `OperationalError` is NOT a synonym for "the
        connection is gone" — `QueryCanceled` (statement_timeout), `DeadlockDetected` and
        `SerializationFailure` are all subclasses raised on a perfectly live connection.
        Retrying those would re-run the statement on a fresh session that no longer carries
        the setting which killed it, i.e. silently escape the very guard that fired. So the
        retry requires the connection to be observably dead (`closed`/`broken`), and it never
        runs inside an explicit transaction, where re-running one statement of a half-applied
        batch would break the atomicity `replace_sources` depends on.

        A reconnect is REPORTED to stderr: a silent one hides an outage behind a process that
        still looks healthy, which is how a dead dependency goes unnoticed for days.
        """
        if self._closed:
            raise RuntimeError("store is closed")
        try:
            return self._conn.execute(sql, params) if params is not None else self._conn.execute(sql)
        except (psycopg.OperationalError, psycopg.InterfaceError):
            # getattr: `broken` only exists from psycopg 3.2, and the declared floor is 3.1 —
            # without the default this except-block would raise AttributeError and mask the
            # original database error on an older install.
            dead = self._conn.closed or getattr(self._conn, "broken", False)
            if self._in_transaction or not dead:
                raise
            print("recall: database connection lost — reconnecting", file=sys.stderr)
            try:
                self._conn.close()
            except Exception:  # pragma: no cover - already-dead connection
                pass
            self._conn = self._open()
            return self._conn.execute(sql, params) if params is not None else self._conn.execute(sql)

    @property
    def table(self) -> str:
        return self._table

    def close(self) -> None:
        """Close the connection for good.

        Sticky by design: without the flag, any later call would hit `_execute`'s reconnect
        and silently resurrect a connection nobody owns — a leak on first accidental reuse.
        """
        self._closed = True
        self._conn.close()

    def __enter__(self) -> "PgVectorStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def ensure_schema(self) -> None:
        t = self._table
        self._execute("CREATE EXTENSION IF NOT EXISTS vector")
        self._execute(
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
        actual_dim = self._execute(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = %s::regclass AND attname = 'embedding'",
            (t,),
        ).fetchone()[0]
        if actual_dim > 0 and actual_dim != self._dim:
            raise ValueError(
                f"table {t!r} has a vector({actual_dim}) embedding column but this store is "
                f"configured for dim {self._dim} — use a matching embedder or a different table "
                f"(drop and re-index for a clean slate)."
            )
        self._execute(f"CREATE INDEX IF NOT EXISTS {t}_tsv_idx ON {t} USING GIN (tsv)")
        self._execute(
            f"CREATE INDEX IF NOT EXISTS {t}_emb_idx ON {t} USING hnsw (embedding vector_cosine_ops)"
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        self._supersession_cache = None  # metadata may change; recompute on next read
        t = self._table
        # One transaction for the whole batch: a mid-loop failure rolls the batch back
        # instead of leaving earlier rows committed (the connection is autocommit).
        # `_in_transaction` disables the reconnect-and-retry in `_execute` for the duration:
        # a retry on a fresh connection would silently break this atomicity.
        outer, self._in_transaction = self._in_transaction, True
        try:
            with self._conn.transaction(), self._conn.cursor() as cur:
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
        finally:
            self._in_transaction = outer
        self._supersession_cache = None  # again, post-commit: a read may have repopulated it
        return len(chunks)

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
        rows = self._execute(sql, params).fetchall()
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
        rows = self._execute(sql, params).fetchall()
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
        outer, self._in_transaction = self._in_transaction, True
        try:
            with self._conn.transaction():
                if sources:
                    self._conn.execute(
                        f"DELETE FROM {self._table} WHERE source = ANY(%s)", (sources,)
                    )
                if chunks:
                    self.upsert(chunks, embeddings)  # nested transaction -> savepoint, same commit
        finally:
            self._in_transaction = outer
        self._supersession_cache = None  # again, post-commit: a read may have repopulated it
        return len(chunks)

    def delete_sources(self, sources: list[str]) -> int:
        """Delete every chunk belonging to the given `source` values; returns rows removed.

        Standalone removal API (the Indexer uses the atomic `replace_sources` instead).
        """
        if not sources:
            return 0
        self._supersession_cache = None
        cur = self._execute(
            f"DELETE FROM {self._table} WHERE source = ANY(%s)", (sources,)
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
        cur = self._execute(
            f"UPDATE {self._table} SET indexed_at = now() "
            f"WHERE metadata->>'file' = ANY(%s)",
            (files,),
        )
        return cur.rowcount or 0

    def supersession(self) -> tuple[dict[str, str], frozenset[str]]:
        """The supersession relation: ``(edges, unresolved)``.

        `edges` maps superseded file -> superseding file, read from `supersedes` chunk
        metadata. `unresolved` holds basenames whose edge could NOT be resolved and was
        therefore withheld.

        Why a resolution step exists at all: `supersedes:` names a BASENAME (the frontmatter
        convention — see `recall.lint`), while a corpus can hold that basename in several
        directories (``a/notes.md``, ``b/notes.md``). Keying the relation on the basename
        alone would silently attribute supersession to the wrong document at query time.
        Keying it on the full path instead is not an option either: the author wrote a
        basename and no path information exists to resolve it against. So an edge whose
        superseded or superseding basename is carried by more than one source is DROPPED and
        its superseded endpoint reported in `unresolved` — the trust layer then refuses to
        serve that memory as ``ok`` instead of guessing. `recall lint` reports the same
        condition at write time as ``ambiguous-supersedes-target``, where it can be fixed.

        If several documents claim to supersede the same file, the last row wins (no defined
        order) — declare a single successor per file for deterministic behavior. The result is
        cached per store instance and invalidated by this instance's own writes (upsert /
        delete_sources); a concurrent writer on another connection is not observed until a
        new store is opened.
        """
        if self._supersession_cache is None:
            rows = self._execute(
                f"""
                SELECT DISTINCT metadata->>'supersedes' AS old, metadata->>'file' AS new
                FROM {self._table}
                WHERE metadata ? 'supersedes'
                """
            ).fetchall()
            declared = {old: new for old, new in rows if old and new}
            ambiguous = set()
            if declared:
                # A basename carried by more than one source cannot identify one document.
                # Scoped to the basenames an edge actually names (usually a handful) instead
                # of grouping the whole table: the unscoped form is a full scan paid on every
                # cold cache — i.e. every CLI invocation and every post-reindex search.
                names = list(set(declared) | set(declared.values()))
                ambiguous = {
                    r[0]
                    for r in self._execute(
                        f"""
                        SELECT metadata->>'file'
                        FROM {self._table}
                        WHERE metadata->>'file' = ANY(%s)
                        GROUP BY 1
                        HAVING count(DISTINCT source) > 1
                        """,
                        (names,),
                    ).fetchall()
                    if r[0]
                }
            edges, unresolved = {}, set()
            for old, new in declared.items():
                if old in ambiguous or new in ambiguous:
                    unresolved.add(old)
                else:
                    edges[old] = new
            self._supersession_cache = (edges, frozenset(unresolved))
        edges, unresolved = self._supersession_cache
        return dict(edges), unresolved

    def supersession_map(self) -> dict[str, str]:
        """Resolvable supersession edges only (superseded file -> superseding file).

        Convenience view of `supersession`; edges dropped for basename ambiguity are absent.
        """
        return self.supersession()[0]

    def newest_indexed_at(self) -> datetime | None:
        row = self._execute(f"SELECT max(indexed_at) FROM {self._table}").fetchone()
        return row[0] if row else None

    def count(self) -> int:
        row = self._execute(f"SELECT count(*) FROM {self._table}").fetchone()
        return int(row[0]) if row else 0
