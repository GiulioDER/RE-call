"""Immutable blue green index generations and their lifecycle."""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from recall.embeddings import Embedder
from recall.frontmatter import parse_frontmatter, validity_bounds
from recall.lineage import GenerationState, IndexManifestV1, PipelineIdentity, canonical_json
from recall.manifest import S3ObjectReader
from recall.types import Chunk

Chunker = Callable[[str], list[str]]

DEFAULT_RETENTION_DAYS = 7
DEFAULT_RETAIN_PREVIOUS = 2
TEMPORARY_STORAGE_MULTIPLIER = 2.2


class GenerationError(RuntimeError):
    """A generation lifecycle invariant was violated."""


class GenerationNotFound(GenerationError):
    pass


class InvalidGenerationTransition(GenerationError):
    pass


class UnsafePromotion(GenerationError):
    pass


class NoActiveGeneration(GenerationError):
    pass


@dataclass(frozen=True)
class GenerationRecord:
    tenant_id: str
    generation_id: str
    state: GenerationState
    pipeline_fingerprint: str
    corpus_fingerprint: str
    manifest_digest: str
    corpus_version: str
    parent_generation_id: str | None
    failure_reason: str | None
    created_at: datetime
    activated_at: datetime | None
    retired_at: datetime | None


@dataclass(frozen=True)
class BuildStats:
    generation_id: str
    objects: int
    chunks: int
    reused_objects: int
    reused_chunks: int
    tombstoned_objects: int
    empty_objects: int


@dataclass(frozen=True)
class ValidationResult:
    generation_id: str
    sources: int
    chunks: int
    state: GenerationState


@dataclass(frozen=True)
class ErasureResult:
    source_uri: str
    generations: tuple[str, ...]
    chunks_removed: int
    event_id: str


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _failure(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:2000]


def _record(row: tuple[Any, ...]) -> GenerationRecord:
    return GenerationRecord(
        tenant_id=str(row[0]),
        generation_id=str(row[1]),
        state=GenerationState(str(row[2])),
        pipeline_fingerprint=str(row[3]),
        corpus_fingerprint=str(row[4]),
        manifest_digest=str(row[5]),
        corpus_version=str(row[6]),
        parent_generation_id=str(row[7]) if row[7] else None,
        failure_reason=str(row[8]) if row[8] else None,
        created_at=row[9],
        activated_at=row[10],
        retired_at=row[11],
    )


_GENERATION_COLUMNS = (
    "tenant_id, generation_id, state, pipeline_fingerprint, corpus_fingerprint, "
    "manifest_digest, corpus_version, parent_generation_id, failure_reason, created_at, "
    "activated_at, retired_at"
)


class GenerationManager:
    """Tenant-scoped generation administration over the versioned v1 tables."""

    def __init__(
        self,
        dsn: str,
        tenant_id: str,
        *,
        actor: str = "recall-cli",
        environment: str | None = None,
    ) -> None:
        if not tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")
        self._dsn = dsn
        self.tenant_id = tenant_id
        self.actor = actor
        self.environment = (environment or os.environ.get("RECALL_ENV", "development")).lower()
        if self.environment not in {"development", "test", "production"}:
            raise ValueError("environment must be development, test, or production")

    @contextmanager
    def _connect(self) -> Iterator[psycopg.Connection]:
        with psycopg.connect(self._dsn, autocommit=True, connect_timeout=10) as conn:
            register_vector(conn)
            conn.execute(
                "SELECT set_config('recall.tenant_id', %s, false)", (self.tenant_id,)
            )
            yield conn

    def _audit(
        self,
        conn: psycopg.Connection,
        event_type: str,
        *,
        generation_id: str | None = None,
        source_uri: str | None = None,
        payload: Mapping[str, Any] | None = None,
        event_id: str | None = None,
    ) -> str:
        event_id = event_id or _new_id("evt")
        conn.execute(
            "INSERT INTO recall_audit_events "
            "(tenant_id, event_id, event_type, actor, generation_id, source_uri, payload) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (
                self.tenant_id,
                event_id,
                event_type,
                self.actor,
                generation_id,
                source_uri,
                Jsonb(dict(payload or {})),
            ),
        )
        return event_id

    def _require_generation(
        self, conn: psycopg.Connection, generation_id: str, *, lock: bool = False
    ) -> GenerationRecord:
        suffix = " FOR UPDATE" if lock else ""
        row = conn.execute(
            f"SELECT {_GENERATION_COLUMNS} FROM recall_generations "
            f"WHERE tenant_id = %s AND generation_id = %s{suffix}",
            (self.tenant_id, generation_id),
        ).fetchone()
        if row is None:
            raise GenerationNotFound(
                f"generation {generation_id!r} does not exist for tenant {self.tenant_id!r}"
            )
        return _record(row)

    def create(
        self,
        manifest: IndexManifestV1,
        pipeline: PipelineIdentity,
        *,
        allow_unverified: bool = False,
        generation_id: str | None = None,
    ) -> GenerationRecord:
        if manifest.tenant_id != self.tenant_id:
            raise GenerationError(
                f"manifest tenant {manifest.tenant_id!r} does not match authenticated tenant "
                f"{self.tenant_id!r}"
            )
        if self.environment == "production":
            pipeline.require_production_identity()
            if allow_unverified:
                raise GenerationError("allow_unverified is unavailable in production")
        elif not pipeline.verified and not allow_unverified:
            raise GenerationError(
                "unverified development pipeline requires allow_unverified=True explicitly"
            )
        generation_id = generation_id or _new_id("gen")
        with self._connect() as conn, conn.transaction():
            dimension = conn.execute(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = 'recall_chunks_v1'::regclass AND attname = 'embedding'"
            ).fetchone()
            actual_dimension = dimension[0] if dimension else None
            if actual_dimension != pipeline.embedder.dimension:
                raise GenerationError(
                    f"pipeline dimension {pipeline.embedder.dimension} does not match "
                    f"recall_chunks_v1 vector({actual_dimension})"
                )
            fts_language = pipeline.fts_configuration.get("language")
            fts_schema = pipeline.fts_configuration.get("schema_version")
            if not isinstance(fts_language, str) or fts_schema != 1:
                raise GenerationError(
                    "FTS configuration requires a language string and schema_version=1"
                )
            try:
                conn.execute("SELECT %s::regconfig", (fts_language,)).fetchone()
            except psycopg.errors.UndefinedObject as exc:
                raise GenerationError(
                    f"PostgreSQL text search configuration {fts_language!r} is absent"
                ) from exc
            state = conn.execute(
                "SELECT active_generation_id FROM recall_tenant_state "
                "WHERE tenant_id = %s FOR UPDATE",
                (self.tenant_id,),
            ).fetchone()
            parent = str(state[0]) if state and state[0] else None
            conn.execute(
                "INSERT INTO recall_generations "
                "(tenant_id, generation_id, state, pipeline_identity, pipeline_fingerprint, "
                "corpus_fingerprint, manifest, manifest_digest, corpus_version, "
                "parent_generation_id, created_by) "
                "VALUES (%s, %s, 'building', %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    self.tenant_id,
                    generation_id,
                    Jsonb(pipeline.to_dict()),
                    pipeline.fingerprint,
                    manifest.corpus_fingerprint,
                    Jsonb(manifest.to_dict()),
                    manifest.digest,
                    manifest.corpus_version,
                    parent,
                    self.actor,
                ),
            )
            conn.execute(
                "INSERT INTO recall_tenant_state (tenant_id) VALUES (%s) "
                "ON CONFLICT (tenant_id) DO NOTHING",
                (self.tenant_id,),
            )
            self._audit(
                conn,
                "generation_created",
                generation_id=generation_id,
                payload={
                    "pipeline_fingerprint": pipeline.fingerprint,
                    "manifest_digest": manifest.digest,
                    "verified_pipeline": pipeline.verified,
                },
            )
            return self._require_generation(conn, generation_id)

    def get(self, generation_id: str) -> GenerationRecord:
        with self._connect() as conn:
            return self._require_generation(conn, generation_id)

    def list_generations(self) -> tuple[GenerationRecord, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_GENERATION_COLUMNS} FROM recall_generations "
                "WHERE tenant_id = %s ORDER BY created_at DESC, generation_id",
                (self.tenant_id,),
            ).fetchall()
        return tuple(_record(row) for row in rows)

    def _load_identity(
        self, conn: psycopg.Connection, generation_id: str
    ) -> tuple[IndexManifestV1, PipelineIdentity]:
        row = conn.execute(
            "SELECT manifest, pipeline_identity FROM recall_generations "
            "WHERE tenant_id = %s AND generation_id = %s",
            (self.tenant_id, generation_id),
        ).fetchone()
        if row is None:
            raise GenerationNotFound(generation_id)
        manifest_raw = row[0]
        pipeline_raw = row[1]
        if not isinstance(manifest_raw, Mapping) or not isinstance(pipeline_raw, Mapping):
            raise GenerationError("stored generation identity is malformed")
        return IndexManifestV1.from_dict(manifest_raw), PipelineIdentity.from_dict(pipeline_raw)

    @staticmethod
    def _source_lock(conn: psycopg.Connection, tenant_id: str, source_uri: str) -> None:
        conn.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"{tenant_id}\x1f{source_uri}",),
        )

    def _is_tombstoned(self, conn: psycopg.Connection, source_uri: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM recall_source_tombstones "
            "WHERE tenant_id = %s AND source_uri = %s",
            (self.tenant_id, source_uri),
        ).fetchone()
        return row is not None

    def _reuse_source(
        self,
        conn: psycopg.Connection,
        generation_id: str,
        pipeline_fingerprint: str,
        source_uri: str,
        source_sha256: str,
        object_version_id: str,
    ) -> int:
        source = conn.execute(
            "SELECT c.generation_id FROM recall_chunks_v1 c "
            "JOIN recall_generations g "
            "ON g.tenant_id = c.tenant_id AND g.generation_id = c.generation_id "
            "WHERE c.tenant_id = %s AND c.source_uri = %s AND c.source_sha256 = %s "
            "AND g.pipeline_fingerprint = %s AND g.state IN ('active', 'ready', 'retired') "
            "ORDER BY g.activated_at DESC NULLS LAST, g.created_at DESC LIMIT 1",
            (self.tenant_id, source_uri, source_sha256, pipeline_fingerprint),
        ).fetchone()
        if source is None:
            return 0
        copied = conn.execute(
            "INSERT INTO recall_chunks_v1 "
            "(tenant_id, generation_id, chunk_id, source_uri, object_version_id, "
            "source_sha256, chunk_ordinal, text, metadata, embedding, indexed_at, tsv) "
            "SELECT tenant_id, %s, chunk_id, source_uri, %s, source_sha256, chunk_ordinal, "
            "text, metadata || jsonb_build_object('reused_from_generation', generation_id), "
            "embedding, clock_timestamp(), tsv FROM recall_chunks_v1 "
            "WHERE tenant_id = %s AND generation_id = %s AND source_uri = %s",
            (
                generation_id,
                object_version_id,
                self.tenant_id,
                str(source[0]),
                source_uri,
            ),
        )
        return copied.rowcount

    def _write_source(
        self,
        conn: psycopg.Connection,
        generation_id: str,
        source_uri: str,
        object_version_id: str,
        source_sha256: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        fts_language: str,
    ) -> int:
        if len(chunks) != len(embeddings):
            raise GenerationError("chunk and embedding counts do not match")
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            conn.execute(
                "INSERT INTO recall_chunks_v1 "
                "(tenant_id, generation_id, chunk_id, source_uri, object_version_id, "
                "source_sha256, chunk_ordinal, text, metadata, embedding, tsv) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                "to_tsvector(%s::regconfig, %s))",
                (
                    self.tenant_id,
                    generation_id,
                    chunk.id,
                    source_uri,
                    object_version_id,
                    source_sha256,
                    int(chunk.metadata["ord"]),
                    chunk.text,
                    Jsonb(chunk.metadata),
                    embedding,
                    fts_language,
                    chunk.text,
                ),
            )
        return len(chunks)

    def build(
        self,
        generation_id: str,
        reader: S3ObjectReader,
        embedder: Embedder,
        chunker: Chunker,
    ) -> BuildStats:
        chunks_written = reused_objects = reused_chunks = tombstoned = empty = 0
        indexed_sources: list[str] = []
        try:
            with self._connect() as conn:
                record = self._require_generation(conn, generation_id)
                if record.state != GenerationState.BUILDING:
                    raise InvalidGenerationTransition(
                        f"build requires building state, found {record.state.value}"
                    )
                manifest, pipeline = self._load_identity(conn, generation_id)
            if embedder.dim != pipeline.embedder.dimension:
                raise GenerationError(
                    f"embedder dimension {embedder.dim} does not match pipeline identity "
                    f"{pipeline.embedder.dimension}"
                )
            if embedder.name != pipeline.embedder.model:
                raise GenerationError(
                    f"embedder implementation {embedder.name!r} does not match pipeline model "
                    f"{pipeline.embedder.model!r}"
                )
            fts_language = pipeline.fts_configuration.get("language")
            if not isinstance(fts_language, str):
                raise GenerationError("pipeline FTS language is malformed")

            for entry in manifest.objects:
                verified = reader.fetch(entry)
                try:
                    text = verified.data.decode("utf-8-sig")
                except UnicodeDecodeError as exc:
                    raise GenerationError(f"{entry.uri} is not valid UTF-8 text") from exc
                text = text.replace("\x00", "")

                with self._connect() as conn, conn.transaction():
                    self._source_lock(conn, self.tenant_id, entry.uri)
                    if self._is_tombstoned(conn, entry.uri):
                        tombstoned += 1
                        continue
                    reused = self._reuse_source(
                        conn,
                        generation_id,
                        pipeline.fingerprint,
                        entry.uri,
                        entry.sha256,
                        entry.version_id,
                    )
                    if reused:
                        reused_objects += 1
                        reused_chunks += reused
                        chunks_written += reused
                        indexed_sources.append(entry.uri)
                        continue

                metadata: dict[str, Any] = {}
                body = text
                if entry.media_type in {"text/markdown", "text/x-markdown"}:
                    metadata, body = parse_frontmatter(text)
                    try:
                        validity_bounds(metadata)
                    except ValueError as exc:
                        raise GenerationError(f"{entry.uri}: {exc}") from exc
                pieces = chunker(body)
                if not pieces:
                    empty += 1
                    continue
                chunks: list[Chunk] = []
                for ordinal, piece in enumerate(pieces):
                    chunk_id = hashlib.sha256(
                        canonical_json(
                            {
                                "source_uri": entry.uri,
                                "source_sha256": entry.sha256,
                                "pipeline_fingerprint": pipeline.fingerprint,
                                "ordinal": ordinal,
                                "text": piece,
                            }
                        )
                    ).hexdigest()
                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            source=entry.uri,
                            text=piece,
                            metadata={
                                **metadata,
                                "file": PurePosixPath(entry.uri).name,
                                "ord": ordinal,
                                "content_hash": entry.sha256,
                                "object_version_id": entry.version_id,
                            },
                        )
                    )
                embeddings = embedder.embed([chunk.text for chunk in chunks])
                with self._connect() as conn, conn.transaction():
                    self._source_lock(conn, self.tenant_id, entry.uri)
                    if self._is_tombstoned(conn, entry.uri):
                        tombstoned += 1
                        continue
                    chunks_written += self._write_source(
                        conn,
                        generation_id,
                        entry.uri,
                        entry.version_id,
                        entry.sha256,
                        chunks,
                        embeddings,
                        fts_language,
                    )
                    indexed_sources.append(entry.uri)

            summary = {
                "objects": len(manifest.objects),
                "chunks": chunks_written,
                "reused_objects": reused_objects,
                "reused_chunks": reused_chunks,
                "tombstoned_objects": tombstoned,
                "empty_objects": empty,
                "indexed_sources": sorted(indexed_sources),
            }
            with self._connect() as conn, conn.transaction():
                current = self._require_generation(conn, generation_id, lock=True)
                if current.state != GenerationState.BUILDING:
                    raise InvalidGenerationTransition(
                        f"generation changed to {current.state.value} during build"
                    )
                conn.execute(
                    "UPDATE recall_generations SET state = 'validating', "
                    "validating_at = clock_timestamp(), validation_summary = %s "
                    "WHERE tenant_id = %s AND generation_id = %s",
                    (Jsonb(summary), self.tenant_id, generation_id),
                )
                self._audit(
                    conn, "generation_built", generation_id=generation_id, payload=summary
                )
            return BuildStats(
                generation_id,
                len(manifest.objects),
                chunks_written,
                reused_objects,
                reused_chunks,
                tombstoned,
                empty,
            )
        except BaseException as exc:
            try:
                self.fail(generation_id, _failure(exc))
            except InvalidGenerationTransition:
                pass
            raise

    def fail(self, generation_id: str, reason: str) -> None:
        with self._connect() as conn, conn.transaction():
            current = self._require_generation(conn, generation_id, lock=True)
            if current.state in {
                GenerationState.ACTIVE,
                GenerationState.RETIRED,
                GenerationState.READY,
                GenerationState.LEGACY_UNVERIFIED,
            }:
                raise InvalidGenerationTransition(
                    f"cannot fail generation in {current.state.value} state"
                )
            conn.execute(
                "UPDATE recall_generations SET state = 'failed', failure_reason = %s "
                "WHERE tenant_id = %s AND generation_id = %s",
                (reason[:2000], self.tenant_id, generation_id),
            )
            self._audit(
                conn,
                "generation_failed",
                generation_id=generation_id,
                payload={"reason": reason[:2000]},
            )

    def _validate(self, generation_id: str) -> ValidationResult:
        with self._connect() as conn, conn.transaction():
            current = self._require_generation(conn, generation_id, lock=True)
            if current.state != GenerationState.VALIDATING:
                raise InvalidGenerationTransition(
                    f"validate requires validating state, found {current.state.value}"
                )
            row = conn.execute(
                "SELECT manifest, validation_summary FROM recall_generations "
                "WHERE tenant_id = %s AND generation_id = %s",
                (self.tenant_id, generation_id),
            ).fetchone()
            if row is None or not isinstance(row[0], Mapping) or not isinstance(row[1], Mapping):
                raise GenerationError("generation has no build summary")
            manifest = IndexManifestV1.from_dict(row[0])
            expected = set(str(item) for item in row[1].get("indexed_sources", []))
            actual_rows = conn.execute(
                "SELECT source_uri, min(source_sha256), min(object_version_id), count(*) "
                "FROM recall_chunks_v1 WHERE tenant_id = %s AND generation_id = %s "
                "GROUP BY source_uri",
                (self.tenant_id, generation_id),
            ).fetchall()
            actual = {str(item[0]) for item in actual_rows}
            if actual != expected:
                raise GenerationError(
                    f"generation source mismatch: expected={sorted(expected)}, "
                    f"actual={sorted(actual)}"
                )
            by_uri = {item.uri: item for item in manifest.objects}
            for source_uri, source_hash, version_id, _count in actual_rows:
                entry = by_uri.get(str(source_uri))
                if entry is None or str(source_hash) != entry.sha256 or str(version_id) != entry.version_id:
                    raise GenerationError(f"generation lineage mismatch for {source_uri}")
            chunks = sum(int(item[3]) for item in actual_rows)
            summary = dict(row[1])
            summary.update({"validated_sources": len(actual), "validated_chunks": chunks})
            conn.execute(
                "UPDATE recall_generations SET state = 'ready', ready_at = clock_timestamp(), "
                "validation_summary = %s WHERE tenant_id = %s AND generation_id = %s",
                (Jsonb(summary), self.tenant_id, generation_id),
            )
            self._audit(
                conn,
                "generation_validated",
                generation_id=generation_id,
                payload={"sources": len(actual), "chunks": chunks},
            )
            return ValidationResult(generation_id, len(actual), chunks, GenerationState.READY)

    def validate(self, generation_id: str) -> ValidationResult:
        try:
            return self._validate(generation_id)
        except BaseException as exc:
            try:
                self.fail(generation_id, _failure(exc))
            except InvalidGenerationTransition:
                pass
            raise

    def promote(self, generation_id: str, *, unsafe_development: bool = False) -> None:
        if self.environment == "production":
            raise UnsafePromotion(
                "generation promotion is unavailable in production until certification gates land"
            )
        if not unsafe_development:
            raise UnsafePromotion("development promotion requires unsafe_development=True")
        with self._connect() as conn, conn.transaction():
            conn.execute(
                "INSERT INTO recall_tenant_state (tenant_id) VALUES (%s) "
                "ON CONFLICT (tenant_id) DO NOTHING",
                (self.tenant_id,),
            )
            state = conn.execute(
                "SELECT active_generation_id, previous_generation_id "
                "FROM recall_tenant_state WHERE tenant_id = %s FOR UPDATE",
                (self.tenant_id,),
            ).fetchone()
            target = self._require_generation(conn, generation_id, lock=True)
            if target.state != GenerationState.READY:
                raise InvalidGenerationTransition(
                    f"promotion requires ready state, found {target.state.value}"
                )
            active = str(state[0]) if state and state[0] else None
            if active:
                conn.execute(
                    "UPDATE recall_generations SET state = 'retired', "
                    "retired_at = clock_timestamp() "
                    "WHERE tenant_id = %s AND generation_id = %s AND state = 'active'",
                    (self.tenant_id, active),
                )
            conn.execute(
                "UPDATE recall_generations SET state = 'active', "
                "activated_at = clock_timestamp(), retired_at = NULL "
                "WHERE tenant_id = %s AND generation_id = %s",
                (self.tenant_id, generation_id),
            )
            conn.execute(
                "UPDATE recall_tenant_state SET active_generation_id = %s, "
                "previous_generation_id = %s, updated_at = clock_timestamp() "
                "WHERE tenant_id = %s",
                (generation_id, active, self.tenant_id),
            )
            self._audit(
                conn,
                "generation_promoted_unsafe_development",
                generation_id=generation_id,
                payload={"previous_generation_id": active},
            )

    def rollback(self) -> str:
        with self._connect() as conn, conn.transaction():
            state = conn.execute(
                "SELECT active_generation_id, previous_generation_id "
                "FROM recall_tenant_state WHERE tenant_id = %s FOR UPDATE",
                (self.tenant_id,),
            ).fetchone()
            if not state or not state[1]:
                raise NoActiveGeneration("no previous generation is available for rollback")
            active = str(state[0]) if state[0] else None
            previous = str(state[1])
            target = self._require_generation(conn, previous, lock=True)
            if target.state not in {GenerationState.RETIRED, GenerationState.READY}:
                raise InvalidGenerationTransition(
                    f"rollback target is {target.state.value}, expected retired or ready"
                )
            if active:
                conn.execute(
                    "UPDATE recall_generations SET state = 'ready', retired_at = NULL "
                    "WHERE tenant_id = %s AND generation_id = %s AND state = 'active'",
                    (self.tenant_id, active),
                )
            conn.execute(
                "UPDATE recall_generations SET state = 'active', "
                "activated_at = clock_timestamp(), retired_at = NULL "
                "WHERE tenant_id = %s AND generation_id = %s",
                (self.tenant_id, previous),
            )
            conn.execute(
                "UPDATE recall_tenant_state SET active_generation_id = %s, "
                "previous_generation_id = %s, updated_at = clock_timestamp() "
                "WHERE tenant_id = %s",
                (previous, active, self.tenant_id),
            )
            self._audit(
                conn,
                "generation_rolled_back",
                generation_id=previous,
                payload={"replaced_generation_id": active},
            )
            return previous

    def active_generation_id(self) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT active_generation_id FROM recall_tenant_state WHERE tenant_id = %s",
                (self.tenant_id,),
            ).fetchone()
        if not row or not row[0]:
            raise NoActiveGeneration(f"tenant {self.tenant_id!r} has no active generation")
        return str(row[0])

    def forget(self, source_uri: str) -> ErasureResult:
        if not source_uri:
            raise ValueError("source_uri must be non-empty")
        event_id = _new_id("evt")
        with self._connect() as conn, conn.transaction():
            self._source_lock(conn, self.tenant_id, source_uri)
            state = conn.execute(
                "SELECT active_generation_id, previous_generation_id "
                "FROM recall_tenant_state WHERE tenant_id = %s FOR UPDATE",
                (self.tenant_id,),
            ).fetchone()
            selected = {str(item) for item in (state or ()) if item}
            mutable = conn.execute(
                "SELECT generation_id FROM recall_generations "
                "WHERE tenant_id = %s AND state != 'legacy_unverified' "
                "FOR UPDATE",
                (self.tenant_id,),
            ).fetchall()
            selected.update(str(row[0]) for row in mutable)
            self._audit(
                conn,
                "source_forgotten",
                source_uri=source_uri,
                payload={"generation_ids": sorted(selected)},
                event_id=event_id,
            )
            conn.execute(
                "INSERT INTO recall_source_tombstones "
                "(tenant_id, source_uri, event_id, erased_at) "
                "VALUES (%s, %s, %s, clock_timestamp()) "
                "ON CONFLICT (tenant_id, source_uri) DO UPDATE SET "
                "event_id = EXCLUDED.event_id, erased_at = EXCLUDED.erased_at",
                (self.tenant_id, source_uri, event_id),
            )
            removed = 0
            if selected:
                result = conn.execute(
                    "DELETE FROM recall_chunks_v1 WHERE tenant_id = %s "
                    "AND generation_id = ANY(%s) AND source_uri = %s",
                    (self.tenant_id, list(selected), source_uri),
                )
                removed = result.rowcount
            return ErasureResult(source_uri, tuple(sorted(selected)), removed, event_id)

    def gc(
        self,
        *,
        now: datetime | None = None,
        retention_days: int = DEFAULT_RETENTION_DAYS,
        retain_previous: int = DEFAULT_RETAIN_PREVIOUS,
    ) -> tuple[str, ...]:
        if retention_days < 0 or retain_previous < 0:
            raise ValueError("retention_days and retain_previous cannot be negative")
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(days=retention_days)
        with self._connect() as conn, conn.transaction():
            state = conn.execute(
                "SELECT active_generation_id, previous_generation_id "
                "FROM recall_tenant_state WHERE tenant_id = %s FOR UPDATE",
                (self.tenant_id,),
            ).fetchone()
            protected = {str(item) for item in (state or ()) if item}
            rows = conn.execute(
                "SELECT generation_id, state, COALESCE(retired_at, created_at) AS age "
                "FROM recall_generations WHERE tenant_id = %s "
                "AND state IN ('retired', 'failed') "
                "ORDER BY COALESCE(retired_at, created_at) DESC",
                (self.tenant_id,),
            ).fetchall()
            previous = str(state[1]) if state and state[1] else None
            kept_retired = 1 if previous else 0
            delete: list[str] = []
            for generation_id, generation_state, age in rows:
                generation_id = str(generation_id)
                if generation_id in protected:
                    continue
                if generation_state == GenerationState.RETIRED.value and (
                    kept_retired < retain_previous
                ):
                    kept_retired += 1
                    continue
                if age >= cutoff:
                    continue
                delete.append(generation_id)
            if delete:
                conn.execute(
                    "DELETE FROM recall_generations WHERE tenant_id = %s "
                    "AND generation_id = ANY(%s)",
                    (self.tenant_id, delete),
                )
                self._audit(conn, "generation_gc", payload={"generation_ids": delete})
            return tuple(delete)
