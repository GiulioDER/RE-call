from __future__ import annotations

import json
import sys
from datetime import datetime
from urllib.parse import urlsplit

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from recall.types import Chunk, ScoredChunk

#: The built-in dev credentials shipped in the default DSN — safe only against a local database.
_DEFAULT_CREDS = ("recall", "recall")
_LOCAL_HOSTS = ("", "localhost", "127.0.0.1", "::1")


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
    if (parts.username, parts.password) != _DEFAULT_CREDS:
        return None
    if (parts.hostname or "").lower() in _LOCAL_HOSTS:
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


def resolve_supersession(rows: list[tuple[str | None, str | None]]) -> dict[str, str]:
    """Build the superseded -> superseding map from ``(file, supersedes)`` rows.

    ``file`` is a root-relative path; ``supersedes`` references its target by basename (the
    authoring convention). Each reference is resolved to the UNIQUE file bearing that basename;
    an ambiguous basename (shared by two files) or a dangling one (no such file) is skipped
    rather than mis-mapped. Both keys and values in the result are root-relative paths.

    Pure and DB-free so the resolution rule can be unit-tested without a database.
    """
    files = [f for f, _ in rows if f]
    by_base: dict[str, list[str]] = {}
    for f in files:
        by_base.setdefault(_basename(f), []).append(f)
    mapping: dict[str, str] = {}
    for file, supersedes in rows:
        if not file or not supersedes:
            continue
        candidates = by_base.get(_basename(supersedes), [])
        if len(candidates) == 1:  # unambiguous: resolve; else skip (ambiguous/dangling)
            mapping[candidates[0]] = file
    return mapping


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
        self._supersession_cache: dict[str, str] | None = None
        self._conn = psycopg.connect(dsn, autocommit=True)
        try:
            # register_vector needs the `vector` type to already exist, so ensure the extension
            # is installed first — this makes a brand-new database work out of the box (the
            # README quickstart path). If this role lacks privilege to create it, fall through:
            # register_vector still succeeds when an admin has pre-installed the extension, and
            # fails with a clear "vector type not found" if it genuinely isn't there.
            try:
                self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except psycopg.Error:
                pass
            register_vector(self._conn)
        except Exception:
            self._conn.close()
            raise

    @property
    def table(self) -> str:
        return self._table

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PgVectorStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def ensure_schema(self) -> None:
        t = self._table
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self._conn.execute(
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
        actual_dim = self._conn.execute(
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
        self._conn.execute(f"CREATE INDEX IF NOT EXISTS {t}_tsv_idx ON {t} USING GIN (tsv)")
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS {t}_emb_idx ON {t} USING hnsw (embedding vector_cosine_ops)"
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        self._supersession_cache = None  # metadata may change; recompute on next read
        t = self._table
        # One transaction for the whole batch: a mid-loop failure rolls the batch back
        # instead of leaving earlier rows committed (the connection is autocommit).
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
        rows = self._conn.execute(sql, params).fetchall()
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
        rows = self._conn.execute(sql, params).fetchall()
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
        with self._conn.transaction():
            if sources:
                self._conn.execute(
                    f"DELETE FROM {self._table} WHERE source = ANY(%s)", (sources,)
                )
            if chunks:
                self.upsert(chunks, embeddings)  # nested transaction -> savepoint, same commit
        return len(chunks)

    def delete_sources(self, sources: list[str]) -> int:
        """Delete every chunk belonging to the given `source` values; returns rows removed.

        Standalone removal API (the Indexer uses the atomic `replace_sources` instead).
        """
        if not sources:
            return 0
        self._supersession_cache = None
        cur = self._conn.execute(
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
        cur = self._conn.execute(
            f"UPDATE {self._table} SET indexed_at = now() "
            f"WHERE metadata->>'file' = ANY(%s)",
            (files,),
        )
        return cur.rowcount or 0

    def supersession_map(self) -> dict[str, str]:
        """Map of superseded file -> superseding file (both root-relative), from `supersedes`
        chunk metadata.

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
            rows = self._conn.execute(
                f"""
                SELECT DISTINCT metadata->>'file' AS file, metadata->>'supersedes' AS supersedes
                FROM {self._table}
                WHERE metadata ? 'file'
                """
            ).fetchall()
            self._supersession_cache = resolve_supersession(rows)
        return dict(self._supersession_cache)

    def newest_indexed_at(self) -> datetime | None:
        row = self._conn.execute(f"SELECT max(indexed_at) FROM {self._table}").fetchone()
        return row[0] if row else None

    def count(self) -> int:
        row = self._conn.execute(f"SELECT count(*) FROM {self._table}").fetchone()
        return int(row[0]) if row else 0
