"""Read only store view pinned to one tenant's active immutable generation."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import TYPE_CHECKING, Any

import psycopg
from pgvector import Vector

from recall.generations import NoActiveGeneration
from recall.store import (
    DEFAULT_TABLE,
    EdgeCandidates,
    PgVectorStore,
    resolve_supersession_candidates,
)
from recall.types import Chunk, ScoredChunk

if TYPE_CHECKING:
    from recall.calibration_v2 import CalibrationResolution


class ImmutableGenerationError(RuntimeError):
    pass


class GenerationStore(PgVectorStore):
    """The retrieval surface for v1, scoped to a request-consistent active generation."""

    def __init__(
        self,
        dsn: str,
        dim: int,
        *,
        tenant: str,
        migration_target: str = DEFAULT_TABLE,
        pool_size: int | None = None,
        statement_timeout_ms: int | None = None,
    ) -> None:
        super().__init__(
            dsn,
            dim,
            table="recall_chunks_v1",
            tenant=tenant,
            pool_size=pool_size,
            statement_timeout_ms=statement_timeout_ms,
        )
        self._migration_target = migration_target
        self._pinned_generation: ContextVar[str | None] = ContextVar(
            f"recall_generation_{uuid.uuid4().hex}", default=None
        )

    def check_schema(self) -> None:
        from recall.schema import check_schema

        self._with_retry(
            lambda conn: check_schema(conn, table=self._migration_target, dim=self._dim)
        )

    def ensure_schema(self) -> None:
        raise ImmutableGenerationError(
            "GenerationStore never migrates; run `recall schema apply` with the migration role"
        )

    def drop_table(self) -> None:
        raise ImmutableGenerationError("the shared generation table cannot be dropped by a store")

    def active_generation_id(self) -> str:
        row = self._with_retry(
            lambda conn: conn.execute(
                "SELECT s.active_generation_id FROM recall_tenant_state s "
                "JOIN recall_generations g ON g.tenant_id = s.tenant_id "
                "AND g.generation_id = s.active_generation_id "
                "WHERE s.tenant_id = %s AND g.state = 'active'",
                (self._tenant,),
            ).fetchone()
        )
        if not row or not row[0]:
            raise NoActiveGeneration(f"tenant {self._tenant!r} has no active generation")
        return str(row[0])

    @contextmanager
    def snapshot(self) -> Iterator[str]:
        """Pin all reads in this context to one atomically observed active pointer."""
        existing = self._pinned_generation.get()
        if existing is not None:
            yield existing
            return
        generation_id = self.active_generation_id()
        token = self._pinned_generation.set(generation_id)
        try:
            yield generation_id
        finally:
            self._pinned_generation.reset(token)

    def _generation_id(self) -> str:
        return self._pinned_generation.get() or self.active_generation_id()

    @contextmanager
    def pin_generation(self, generation_id: str) -> Iterator[str]:
        """Administrative read view for calibrating one explicit immutable generation."""
        row = self._with_retry(
            lambda conn: conn.execute(
                "SELECT 1 FROM recall_generations WHERE tenant_id = %s "
                "AND generation_id = %s AND state IN ('ready', 'active', 'retired')",
                (self._tenant, generation_id),
            ).fetchone()
        )
        if row is None:
            raise NoActiveGeneration(
                f"tenant {self._tenant!r} has no calibratable generation {generation_id!r}"
            )
        token = self._pinned_generation.set(generation_id)
        try:
            yield generation_id
        finally:
            self._pinned_generation.reset(token)

    def generation_binding(self) -> dict[str, str]:
        generation_id = self._generation_id()
        row = self._with_retry(
            lambda conn: conn.execute(
                "SELECT pipeline_fingerprint, corpus_fingerprint FROM recall_generations "
                "WHERE tenant_id = %s AND generation_id = %s",
                (self._tenant, generation_id),
            ).fetchone()
        )
        if row is None:
            raise NoActiveGeneration(generation_id)
        return {
            "tenant_id": self._tenant,
            "generation_id": generation_id,
            "pipeline_fingerprint": str(row[0]),
            "corpus_fingerprint": str(row[1]),
        }

    def resolve_calibration(self) -> CalibrationResolution:
        from recall.calibration_v2 import CalibrationRepository

        return CalibrationRepository(self._dsn, self._tenant, actor="generation-search").resolve(
            self._generation_id()
        )

    @staticmethod
    def _generation_rows(rows: list[tuple[Any, ...]]) -> list[ScoredChunk]:
        hits: list[ScoredChunk] = []
        for chunk_id, source_uri, text, metadata, indexed_at, score in rows:
            value = metadata if isinstance(metadata, dict) else json.loads(metadata)
            hits.append(
                ScoredChunk(
                    chunk=Chunk(
                        id=str(chunk_id),
                        source=str(source_uri),
                        text=str(text),
                        metadata=value,
                    ),
                    score=float(score),
                    indexed_at=indexed_at,
                    first_indexed_at=indexed_at,
                )
            )
        return hits

    def query_dense(
        self, vector: list[float], k: int, source: str | None = None
    ) -> list[ScoredChunk]:
        if k <= 0:
            raise ValueError("k must be a positive int")
        generation_id = self._generation_id()
        source_filter = (
            "AND (metadata->>'file' = %(source)s OR source_uri = %(source)s)" if source else ""
        )
        sql = f"""
            SELECT chunk_id, source_uri, text, metadata, indexed_at,
                   1 - (embedding <=> %(vec)s) AS score
            FROM recall_chunks_v1
            WHERE tenant_id = %(tenant)s AND generation_id = %(generation)s {source_filter}
            ORDER BY embedding <=> %(vec)s
            LIMIT %(k)s
        """
        params: dict[str, Any] = {
            "vec": Vector(vector),
            "k": k,
            "tenant": self._tenant,
            "generation": generation_id,
        }
        if source:
            params["source"] = source
        ef_search, iterative_scan = self._hnsw_filtered_tuning()

        def _op(conn: psycopg.Connection) -> list[tuple[Any, ...]]:
            with conn.transaction():
                conn.execute(f"SET LOCAL hnsw.ef_search = {ef_search}")
                conn.execute(f"SET LOCAL hnsw.iterative_scan = {iterative_scan}")
                return conn.execute(sql, params).fetchall()

        return self._generation_rows(self._with_retry(_op))

    def query_sparse(
        self,
        text: str,
        k: int,
        source: str | None = None,
        vec: list[float] | None = None,
    ) -> list[ScoredChunk]:
        if k <= 0:
            raise ValueError("k must be a positive int")
        generation_id = self._generation_id()
        source_filter = (
            "AND (c.metadata->>'file' = %(source)s OR c.source_uri = %(source)s)" if source else ""
        )
        score = "1 - (embedding <=> %(vec)s)" if vec is not None else "rank"
        sql = f"""
            WITH config AS (
                SELECT pipeline_identity #>> '{{fts_configuration,language}}' AS language
                FROM recall_generations
                WHERE tenant_id = %(tenant)s AND generation_id = %(generation)s
            ), q AS (
                SELECT (
                    SELECT string_agg(quote_literal(lexeme), ' | ')
                    FROM config,
                         unnest(to_tsvector(config.language::regconfig, %(q)s))
                )::tsquery AS tsq
            ), top_k AS (
                SELECT c.chunk_id, c.source_uri, c.text, c.metadata, c.indexed_at,
                       c.embedding, ts_rank(c.tsv, q.tsq) AS rank
                FROM recall_chunks_v1 c, q
                WHERE c.tenant_id = %(tenant)s AND c.generation_id = %(generation)s
                  AND c.tsv @@ q.tsq {source_filter}
                ORDER BY rank DESC
                LIMIT %(k)s
            )
            SELECT chunk_id, source_uri, text, metadata, indexed_at, {score} AS score
            FROM top_k ORDER BY rank DESC
        """
        params: dict[str, Any] = {
            "q": text,
            "k": k,
            "tenant": self._tenant,
            "generation": generation_id,
        }
        if vec is not None:
            params["vec"] = Vector(vec)
        if source:
            params["source"] = source
        rows = self._with_retry(lambda conn: conn.execute(sql, params).fetchall())
        return self._generation_rows(rows)

    def newest_indexed_at(self) -> datetime | None:
        generation_id = self._generation_id()
        row = self._with_retry(
            lambda conn: conn.execute(
                "SELECT max(indexed_at) FROM recall_chunks_v1 "
                "WHERE tenant_id = %s AND generation_id = %s",
                (self._tenant, generation_id),
            ).fetchone()
        )
        return row[0] if row else None

    def count(self) -> int:
        generation_id = self._generation_id()
        row = self._with_retry(
            lambda conn: conn.execute(
                "SELECT count(*) FROM recall_chunks_v1 WHERE tenant_id = %s AND generation_id = %s",
                (self._tenant, generation_id),
            ).fetchone()
        )
        return int(row[0]) if row else 0

    def source_content_hashes(self) -> dict[str, str]:
        generation_id = self._generation_id()
        rows = self._with_retry(
            lambda conn: conn.execute(
                "SELECT DISTINCT source_uri, source_sha256 FROM recall_chunks_v1 "
                "WHERE tenant_id = %s AND generation_id = %s",
                (self._tenant, generation_id),
            ).fetchall()
        )
        return {str(source): str(digest) for source, digest in rows}

    def sources_in_any_generation(self) -> frozenset[str]:
        """Every source this tenant has indexed, across ALL generations.

        `source_content_hashes()` is scoped to one generation, so it answers "can I read this
        now", not "does this exist". Erasure needs the second question: a source that dropped
        out of the active generation must still be erasable, while one that was never indexed
        must still be refused as a typo, because forgetting it writes a permanent tombstone
        that bars that URI from every future build.
        """
        rows = self._with_retry(
            lambda conn: conn.execute(
                "SELECT DISTINCT source_uri FROM recall_chunks_v1 WHERE tenant_id = %s",
                (self._tenant,),
            ).fetchall()
        )
        return frozenset(str(row[0]) for row in rows)

    def sources_in_any_manifest(self) -> frozenset[str]:
        """Every source URI the tenant's generations were BUILT FROM.

        Chunk rows are not the corpus. An object that chunks to nothing (a frontmatter-only
        document, say) is built as `empty_objects` and writes no row, so asking
        `recall_chunks_v1` alone would call it a typo and refuse to erase it, leaving no
        tombstone and letting the next build re-ingest it the moment its content changes.
        The manifest is the record of what the corpus contains.
        """
        rows = self._with_retry(
            lambda conn: conn.execute(
                "SELECT manifest FROM recall_generations "
                "WHERE tenant_id = %s AND state != 'legacy_unverified'",
                (self._tenant,),
            ).fetchall()
        )
        uris: set[str] = set()
        for (manifest,) in rows:
            if not isinstance(manifest, Mapping):
                continue
            for entry in manifest.get("objects") or ():
                uri = entry.get("uri") if isinstance(entry, Mapping) else None
                if isinstance(uri, str):
                    uris.add(uri)
        return frozenset(uris)

    def supersession(self) -> tuple[dict[str, str], frozenset[str]]:
        edges, unresolved, _candidates = self.supersession_all()
        return edges, unresolved

    def supersession_all(
        self,
    ) -> tuple[dict[str, str], frozenset[str], EdgeCandidates]:
        generation_id = self._generation_id()
        rows = self._with_retry(
            lambda conn: conn.execute(
                "SELECT metadata->>'file', metadata->>'supersedes', min(indexed_at) "
                "FROM recall_chunks_v1 WHERE tenant_id = %s AND generation_id = %s "
                "AND metadata ? 'file' GROUP BY 1, 2 ORDER BY 1, 2",
                (self._tenant, generation_id),
            ).fetchall()
        )
        edges, unresolved, candidates = resolve_supersession_candidates(rows)
        return (
            dict(edges),
            unresolved,
            {target: list(claims) for target, claims in candidates.items()},
        )

    def sources_for_identifiers(self, identifiers: list[str]) -> dict[str, list[str]]:
        if not identifiers:
            return {}
        generation_id = self._generation_id()
        rows = self._with_retry(
            lambda conn: conn.execute(
                "SELECT DISTINCT source_uri, metadata->>'file' FROM recall_chunks_v1 "
                "WHERE tenant_id = %s AND generation_id = %s "
                "AND (metadata->>'file' = ANY(%s) OR source_uri = ANY(%s))",
                (self._tenant, generation_id, identifiers, identifiers),
            ).fetchall()
        )
        requested = set(identifiers)
        resolved: dict[str, list[str]] = {}
        for source, file in rows:
            for identifier in {file, source} & requested:
                resolved.setdefault(str(identifier), []).append(str(source))
        return resolved

    def iter_chunks(self, batch_size: int = 1000) -> Iterator[Chunk]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        generation_id = self._generation_id()
        with (
            self._borrowed() as conn,
            conn.cursor(name=f"recall_gen_{uuid.uuid4().hex[:12]}") as cur,
        ):
            cur.execute(
                "SELECT chunk_id, source_uri, text, metadata FROM recall_chunks_v1 "
                "WHERE tenant_id = %s AND generation_id = %s ORDER BY chunk_id",
                (self._tenant, generation_id),
            )
            while rows := cur.fetchmany(batch_size):
                for chunk_id, source, text, metadata in rows:
                    value = metadata if isinstance(metadata, dict) else json.loads(metadata)
                    yield Chunk(str(chunk_id), str(source), str(text), value)

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        raise ImmutableGenerationError("active generations are read only")

    def replace_sources(
        self, sources: list[str], chunks: list[Chunk], embeddings: list[list[float]]
    ) -> int:
        raise ImmutableGenerationError("active generations are read only")

    def delete_sources(self, sources: list[str]) -> int:
        from recall.generations import GenerationManager

        manager = GenerationManager(
            self._dsn,
            self._tenant,
            actor="generation-store-forget",
        )
        return sum(manager.forget(source).chunks_removed for source in sources)

    def touch_files(self, files: list[str]) -> int:
        raise ImmutableGenerationError("immutable generations cannot be touched")
