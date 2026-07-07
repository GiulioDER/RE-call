from __future__ import annotations

import json
from datetime import datetime

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from recall.types import Chunk, ScoredChunk


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
        for cid, source, text, metadata, score in rows:
            md = metadata if isinstance(metadata, dict) else json.loads(metadata)
            hits.append(
                ScoredChunk(
                    chunk=Chunk(id=cid, source=source, text=text, metadata=md),
                    score=float(score),
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
            SELECT id, source, text, metadata, 1 - (embedding <=> %(vec)s) AS score
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
        self, text: str, k: int, source: str | None = None
    ) -> list[ScoredChunk]:
        if k <= 0:
            raise ValueError("k must be a positive int")
        t = self._table
        where = "AND source = %(source)s" if source else ""
        sql = f"""
            SELECT id, source, text, metadata,
                   ts_rank(tsv, websearch_to_tsquery('english', %(q)s)) AS score
            FROM {t}
            WHERE tsv @@ websearch_to_tsquery('english', %(q)s)
            {where}
            ORDER BY score DESC
            LIMIT %(k)s
        """
        params: dict = {"q": text, "k": k}
        if source:
            params["source"] = source
        rows = self._conn.execute(sql, params).fetchall()
        return self._rows_to_hits(rows)

    def newest_indexed_at(self) -> datetime | None:
        row = self._conn.execute(f"SELECT max(indexed_at) FROM {self._table}").fetchone()
        return row[0] if row else None

    def count(self) -> int:
        row = self._conn.execute(f"SELECT count(*) FROM {self._table}").fetchone()
        return int(row[0]) if row else 0
