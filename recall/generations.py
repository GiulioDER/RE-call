"""Immutable blue green index generations and their lifecycle."""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from recall.context import ContextPolicy, StructuredChunk, contextual_passages
from recall.document import parse_document
from recall.embeddings import Embedder, embed_passages, embedding_profile, embedding_profile_id
from recall.errors import RecallError
from recall.extraction import ExtractedDocument, chunk_extracted_document
from recall.frontmatter import legacy_pairing_differs, validity_bounds
from recall.lineage import (
    GenerationState,
    IndexManifestV1,
    PipelineIdentity,
    canonical_json,
    canonical_sha256,
)
from recall.manifest import ObjectReader
from recall.observability import METRICS
from recall.semantic_graph import GraphReadiness, SemanticGraphProjection, build_semantic_graph, write_semantic_graph
from recall.types import Chunk

Chunker = Callable[[str], list[str]]

DEFAULT_RETENTION_DAYS = 7
DEFAULT_RETAIN_PREVIOUS = 2
TEMPORARY_STORAGE_MULTIPLIER = 2.2
DEFAULT_TABLE_MAX_CHARS = 800
DEFAULT_TABLE_OVERLAP = 80


def _context_policy_for_pipeline(pipeline: PipelineIdentity) -> ContextPolicy:
    """Resolve and validate the passage context declared by an immutable pipeline."""
    mode = pipeline.embedder.context_mode
    if mode not in {"none", "document", "section", "neighbor"}:
        raise GenerationError(f"pipeline context mode is unsupported: {mode!r}")
    expected = "raw-v1" if mode == "none" else f"context-{mode}-v1"
    if pipeline.embedder.context_version != expected:
        raise GenerationError(
            f"pipeline context version {pipeline.embedder.context_version!r} does not match "
            f"context mode {mode!r}"
        )
    return ContextPolicy(mode=mode)


class GenerationError(RuntimeError, RecallError):
    """A generation lifecycle invariant was violated."""


class GenerationNotFound(GenerationError):
    pass


class InvalidGenerationTransition(GenerationError):
    pass


class UnsafePromotion(GenerationError):
    pass


class ConcurrentIngest(GenerationError):
    """Another upload holds this tenant's bounded ingest lock."""


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


#: The media types whose body is derived by `parse_frontmatter`. Anything else is chunked as it
#: arrived, so the frontmatter pairing rule cannot have moved its body.
_MARKDOWN_MEDIA_TYPES = frozenset({"text/markdown", "text/x-markdown"})
_BODY_RULE_VERSION_KEY = "body_rule_version"
_BODY_RULE_VERSION = "frontmatter-pairing-2026-08-11"


def _body_rule_changed(media_type: str, text: str) -> bool:
    """True when this object's body moved because the frontmatter pairing rule changed.

    A second freshness guard lives in this module and `recall.index`'s trigger does not reach
    it. `_reuse_source` copies an earlier generation's chunks whenever tenant, URI, object
    sha256 and pipeline fingerprint all match, and it returns BEFORE `parse_frontmatter` runs.
    None of those four terms moved when the pairing rule changed, and `PipelineIdentity` covers
    only the schema version, embedder, chunker and FTS configuration, so an object indexed
    before the fix would carry its truncated chunk set into every generation built after it:
    exactly the "stale forever" outcome the index side trigger exists to prevent.

    Narrow, but NOT symmetric with `recall.index._body_derivation_hash`, and the difference is
    worth stating rather than discovering. That one stores its perturbed fingerprint, so an
    affected file rebuilds once and is skipped afterwards. This is a pure function of the
    object's text with nowhere to record that the rebuild already happened, because reuse is
    keyed on tenant, URI, sha256 and pipeline fingerprint and none of those can carry a body
    rule term. The generation path therefore persists a marker in chunk metadata and requires it
    only for this affected subset, so the first post-fix rebuild runs once and later generations
    can reuse that repaired source.

    Accepted rather than fixed: the cost falls only on objects that actually contain the defect,
    and removing it means either a new term in `PipelineIdentity`, which re-embeds every corpus,
    or a body rule column threaded through `_write_source` and `_reuse_source`'s SELECT, which
    is a schema change to a reuse path. Recorded as a follow up.
    """
    return media_type in _MARKDOWN_MEDIA_TYPES and legacy_pairing_differs(text)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _failure(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:2000]


def _effective_corpus_fingerprint(
    manifest: IndexManifestV1, tombstoned_sources: set[str]
) -> str:
    excluded = sorted({entry.uri for entry in manifest.objects} & tombstoned_sources)
    if not excluded:
        return manifest.corpus_fingerprint
    return canonical_sha256(
        {
            "manifest_corpus_fingerprint": manifest.corpus_fingerprint,
            "excluded_sources": excluded,
        }
    )


def _semantic_graph_marker(graph: SemanticGraphProjection) -> dict[str, Any]:
    return {
        "graph_id": graph.graph_id,
        "graph_fingerprint": graph.fingerprint,
        "entity_count": len(graph.entities),
        "mention_count": len(graph.mentions),
        "relation_count": len(graph.relations),
        "diagnostic_count": len(graph.diagnostics),
        "diagnostics": [
            {
                "id": diagnostic.id,
                "kind": diagnostic.kind,
                "reference": diagnostic.reference,
                "message": diagnostic.message,
                "entity_ids": list(diagnostic.entity_ids),
                "relation_ids": list(diagnostic.relation_ids),
            }
            for diagnostic in graph.diagnostics
        ],
        "ready": True,
    }


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


def with_provenance(metadata: dict, provenance: dict) -> dict:
    """Frontmatter first, provenance LAST, and never mutating the input.

    Provenance overrides because a document must not be able to relabel its own origin: if
    frontmatter won, any indexed file could claim to come from another project and provenance would
    be an assertion made by the data rather than a record made by the builder.

    A copy, not an update in place: the caller reuses one `metadata` dict across every chunk of a
    document, so mutating it would accumulate across chunks and leak between documents.
    """
    if not provenance:
        return metadata
    return {**metadata, **provenance}


class GenerationManager:
    """Tenant-scoped generation administration over the versioned v1 tables."""

    def __init__(
        self,
        dsn: str,
        tenant_id: str,
        *,
        actor: str = "recall-cli",
        environment: str | None = None,
        serving_environment: str | None = None,
    ) -> None:
        if not tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")
        self._dsn = dsn
        self.tenant_id = tenant_id
        self.actor = actor
        self.environment = (environment or os.environ.get("RECALL_ENV", "development")).lower()
        if self.environment not in {"development", "test", "production"}:
            raise ValueError("environment must be development, test, or production")
        self.serving_environment = (serving_environment or self.environment).lower()
        if self.serving_environment not in {"development", "test", "production"}:
            raise ValueError("serving_environment must be development, test, or production")

    @property
    def certification_required(self) -> bool:
        return self.serving_environment == "production"

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
            manifest_sources = [entry.uri for entry in manifest.objects]
            tombstones = (
                {
                    str(row[0])
                    for row in conn.execute(
                        "SELECT source_uri FROM recall_source_tombstones "
                        "WHERE tenant_id = %s AND source_uri = ANY(%s)",
                        (self.tenant_id, manifest_sources),
                    ).fetchall()
                }
                if manifest_sources
                else set()
            )
            corpus_fingerprint = _effective_corpus_fingerprint(manifest, tombstones)
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
                    corpus_fingerprint,
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

    @contextmanager
    def tenant_ingest_lock(self, *, wait_seconds: float = 20.0) -> Iterator[None]:
        """Serialize manifest read through promotion for one tenant."""
        key = f"ingest\x1f{self.tenant_id}"
        with self._connect() as conn:
            deadline = time.monotonic() + wait_seconds
            while True:
                row = conn.execute(
                    "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))", (key,)
                ).fetchone()
                if row is not None and row[0]:
                    break
                if time.monotonic() >= deadline:
                    raise ConcurrentIngest(
                        f"another upload into tenant {self.tenant_id!r} is still running; "
                        "try again shortly"
                    )
                time.sleep(0.25)
            try:
                yield
            finally:
                with suppress(Exception):
                    conn.execute(
                        "SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (key,)
                    )

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

    def _scrub_sparse_rows(self, conn: psycopg.Connection, chunk_ids: list[str]) -> None:
        """Remove learned sparse sidecars for generation chunks in this transaction."""
        if not chunk_ids:
            return
        sidecar = conn.execute("SELECT to_regclass(%s)", ("recall_sparse_v1",)).fetchone()
        if not (sidecar and sidecar[0]):
            return
        conn.execute(
            "DELETE FROM recall_sparse_v1 "
            "WHERE tenant_id = %s AND chunk_table = %s AND id = ANY(%s)",
            (self.tenant_id, "recall_chunks_v1", chunk_ids),
        )

    def _reuse_source(
        self,
        conn: psycopg.Connection,
        generation_id: str,
        pipeline_fingerprint: str,
        source_uri: str,
        source_sha256: str,
        object_version_id: str,
        *,
        require_body_rule_version: str | None = None,
    ) -> int:
        source = conn.execute(
            "SELECT c.generation_id FROM recall_chunks_v1 c "
            "JOIN recall_generations g "
            "ON g.tenant_id = c.tenant_id AND g.generation_id = c.generation_id "
            "WHERE c.tenant_id = %s AND c.source_uri = %s AND c.source_sha256 = %s "
            "AND (%s OR c.metadata ->> %s = %s) "
            "AND g.pipeline_fingerprint = %s AND g.state IN ('active', 'ready', 'retired') "
            "ORDER BY g.activated_at DESC NULLS LAST, g.created_at DESC LIMIT 1",
            (
                self.tenant_id,
                source_uri,
                source_sha256,
                require_body_rule_version is None,
                _BODY_RULE_VERSION_KEY,
                require_body_rule_version,
                pipeline_fingerprint,
            ),
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
        reader: ObjectReader,
        embedder: Embedder,
        chunker: Chunker,
        provenance: dict | None = None,
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
            runtime_profile = embedding_profile(embedder)
            if pipeline.embedder.profile_id is not None:
                if embedding_profile_id(embedder) != pipeline.embedder.profile_id:
                    raise GenerationError(
                        f"embedder profile {embedding_profile_id(embedder)!r} does not match "
                        f"pipeline profile {pipeline.embedder.profile_id!r}"
                    )
                if runtime_profile.context_version != pipeline.embedder.context_version:
                    raise GenerationError(
                        f"embedder context {runtime_profile.context_version!r} does not match "
                        f"pipeline context {pipeline.embedder.context_version!r}"
                    )
            context_policy = _context_policy_for_pipeline(pipeline)
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
                    body_rule_changed = _body_rule_changed(entry.media_type, text)
                    reused = self._reuse_source(
                        conn,
                        generation_id,
                        pipeline.fingerprint,
                        entry.uri,
                        entry.sha256,
                        entry.version_id,
                        require_body_rule_version=(
                            _BODY_RULE_VERSION if body_rule_changed else None
                        ),
                    )
                    if reused:
                        reused_objects += 1
                        reused_chunks += reused
                        chunks_written += reused
                        indexed_sources.append(entry.uri)
                        continue

                metadata: dict[str, Any] = dict(verified.metadata)
                body = text
                if entry.media_type in _MARKDOWN_MEDIA_TYPES:
                    # Not optional. `recall index` is refused under RECALL_ENV=production
                    # (`recall/cli.py:1209`), so hooking only the index path would leave the one
                    # build path that runs in production reading derived blocks as evidence.
                    document = parse_document(text)
                    metadata = {**metadata, **document.meta}
                    body = document.human_body
                    try:
                        validity_bounds(metadata)
                    except ValueError as exc:
                        raise GenerationError(f"{entry.uri}: {exc}") from exc
                # Stamped here, after frontmatter has been read and before any chunk is built,
                # so every chunk of every document carries it and no document can override it.
                metadata = with_provenance(metadata, provenance or {})
                piece_metadata: list[dict[str, Any]] = []
                has_structured_tables = (
                    entry.media_type not in _MARKDOWN_MEDIA_TYPES
                    and bool(verified.blocks)
                    and any(block.kind == "table" for block in verified.blocks)
                )
                if has_structured_tables:
                    configuration = pipeline.chunker.configuration
                    max_chars = configuration.get("max_chars", DEFAULT_TABLE_MAX_CHARS)
                    overlap = configuration.get("overlap", DEFAULT_TABLE_OVERLAP)
                    if type(max_chars) is not int or type(overlap) is not int:
                        raise GenerationError("chunker configuration has non integer table bounds")
                    extracted_document = ExtractedDocument(
                        text,
                        str(metadata.get("media_type", entry.media_type)),
                        metadata,
                        verified.blocks,
                    )
                    typed_chunks = chunk_extracted_document(
                        extracted_document,
                        max_chars=max_chars,
                        overlap=overlap,
                    )
                    pieces = [piece for piece, _ in typed_chunks]
                    piece_metadata = [chunk_meta for _, chunk_meta in typed_chunks]
                else:
                    pieces = chunker(body)
                    piece_metadata = [{} for _ in pieces]
                if not pieces:
                    empty += 1
                    continue
                structured: list[StructuredChunk] = []
                embedding_texts = [piece for piece in pieces]
                if entry.media_type in _MARKDOWN_MEDIA_TYPES:
                    structured, embedding_texts = contextual_passages(
                        text,
                        body,
                        pieces,
                        entry.uri,
                        context_policy,
                    )
                chunks: list[Chunk] = []
                for ordinal, piece in enumerate(pieces):
                    structured_chunk = structured[ordinal] if structured else None
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
                                **piece_metadata[ordinal],
                                **(
                                    {_BODY_RULE_VERSION_KEY: _BODY_RULE_VERSION}
                                    if body_rule_changed
                                    else {}
                                ),
                                "file": PurePosixPath(entry.uri).name,
                                "ord": ordinal,
                                "content_hash": entry.sha256,
                                "object_version_id": entry.version_id,
                                "context_mode": context_policy.mode,
                                "context_version": pipeline.embedder.context_version,
                                "text_start": (
                                    structured_chunk.start if structured_chunk is not None else None
                                ),
                                "text_end": (
                                    structured_chunk.end if structured_chunk is not None else None
                                ),
                                "heading_hierarchy": (
                                    list(structured_chunk.headings)
                                    if structured_chunk is not None
                                    else []
                                ),
                                **(
                                    {"embedding_profile": pipeline.embedder.profile_id}
                                    if pipeline.embedder.profile_id is not None
                                    else {}
                                ),
                            },
                        )
                    )
                # PASSAGE encoding: these vectors are what a query is matched against. With an
                # asymmetric model the query encoder produces a different vector for the same
                # text, and a generation built with the wrong one is the right width, scores in
                # range, and silently retrieves worse. Falls back to `embed` for an embedder
                # that only implements the symmetric interface.
                embeddings = embed_passages(embedder, embedding_texts)
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

            with self._connect() as conn, conn.transaction():
                graph_started = time.perf_counter()
                current = self._require_generation(conn, generation_id)
                rows = conn.execute(
                    "SELECT chunk_id, source_uri, text, metadata FROM recall_chunks_v1 "
                    "WHERE tenant_id = %s AND generation_id = %s ORDER BY chunk_id",
                    (self.tenant_id, generation_id),
                ).fetchall()
                graph_chunks = [
                    Chunk(
                        str(row[0]),
                        str(row[1]),
                        str(row[2]),
                        row[3] if isinstance(row[3], dict) else {},
                    )
                    for row in rows
                ]
                try:
                    semantic_graph = build_semantic_graph(
                        graph_chunks,
                        tenant_id=self.tenant_id,
                        generation_id=generation_id,
                        pipeline_fingerprint=pipeline.fingerprint,
                        corpus_fingerprint=current.corpus_fingerprint,
                    )
                    write_semantic_graph(conn, semantic_graph)
                except BaseException:
                    METRICS.increment("recall_graph_build_failure_total")
                    METRICS.observe(
                        "recall_graph_latency_ms",
                        (time.perf_counter() - graph_started) * 1000.0,
                    )
                    raise
                METRICS.increment("recall_graph_build_total")
                METRICS.observe(
                    "recall_graph_latency_ms", (time.perf_counter() - graph_started) * 1000.0
                )
            summary = {
                "objects": len(manifest.objects),
                "chunks": chunks_written,
                "reused_objects": reused_objects,
                "reused_chunks": reused_chunks,
                "tombstoned_objects": tombstoned,
                "empty_objects": empty,
                "indexed_sources": sorted(indexed_sources),
                "semantic_graph": _semantic_graph_marker(semantic_graph),
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

    def abandon(self, generation_id: str, reason: str) -> None:
        """Move an unprotected READY generation to failed so GC can reclaim it."""
        with self._connect() as conn, conn.transaction():
            state = conn.execute(
                "SELECT active_generation_id, previous_generation_id "
                "FROM recall_tenant_state WHERE tenant_id = %s FOR UPDATE",
                (self.tenant_id,),
            ).fetchone()
            current = self._require_generation(conn, generation_id, lock=True)
            if current.state != GenerationState.READY:
                raise InvalidGenerationTransition(
                    f"abandon requires ready state, found {current.state.value}"
                )
            if generation_id in {str(item) for item in (state or ()) if item}:
                raise InvalidGenerationTransition(
                    f"cannot abandon protected generation {generation_id!r}"
                )
            conn.execute(
                "UPDATE recall_generations SET state = 'failed', failure_reason = %s "
                "WHERE tenant_id = %s AND generation_id = %s",
                (reason[:2000], self.tenant_id, generation_id),
            )
            self._audit(
                conn,
                "generation_abandoned",
                generation_id=generation_id,
                payload={"reason": reason[:2000]},
            )

    def calibration_status_for(
        self, generation_id: str, *, conn: psycopg.Connection | None = None
    ) -> str:
        from recall.calibration_v2 import CalibrationRepository

        repository = CalibrationRepository(self._dsn, self.tenant_id, actor=self.actor)
        try:
            if conn is None:
                resolution = repository.resolve(generation_id)
            else:
                # A failed SELECT aborts PostgreSQL's whole transaction. The status is
                # advisory during rollback, so isolate the read in a savepoint and keep the
                # recovery update usable when a partially migrated install makes it fail.
                with conn.transaction():
                    resolution = repository.resolve_within(conn, generation_id)
            return str(resolution.status.value)
        except Exception:  # noqa: BLE001
            return "unknown"

    def require_certified_for_production(
        self, generation_id: str, *, conn: psycopg.Connection | None = None
    ) -> None:
        from recall.calibration_v2 import CalibrationBindingError, CalibrationRepository, CalibrationStatus

        repository = CalibrationRepository(self._dsn, self.tenant_id, actor=self.actor)
        try:
            resolution = (
                repository.resolve_within(conn, generation_id)
                if conn is not None
                else repository.resolve(generation_id)
            )
        except CalibrationBindingError as exc:
            raise UnsafePromotion(
                f"generation {generation_id} cannot go live in production: {exc}"
            ) from exc
        if resolution.status is not CalibrationStatus.CERTIFIED:
            raise UnsafePromotion(
                f"generation {generation_id} cannot go live in production: calibration is "
                f"{resolution.status.value}; run `recall calibration calibrate`"
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
            graph_marker = summary.get("semantic_graph")
            if not isinstance(graph_marker, Mapping) or graph_marker.get("ready") is not True:
                raise GenerationError("generation semantic graph is not ready")
            graph_rows = conn.execute(
                "SELECT chunk_id, source_uri, text, metadata FROM recall_chunks_v1 "
                "WHERE tenant_id = %s AND generation_id = %s ORDER BY chunk_id",
                (self.tenant_id, generation_id),
            ).fetchall()
            graph = build_semantic_graph(
                [
                    Chunk(
                        str(item[0]),
                        str(item[1]),
                        str(item[2]),
                        item[3] if isinstance(item[3], dict) else {},
                    )
                    for item in graph_rows
                ],
                tenant_id=self.tenant_id,
                generation_id=generation_id,
                pipeline_fingerprint=current.pipeline_fingerprint,
                corpus_fingerprint=current.corpus_fingerprint,
            )
            if (
                graph_marker.get("graph_id") != graph.graph_id
                or graph_marker.get("graph_fingerprint") != graph.fingerprint
            ):
                raise GenerationError("generation semantic graph fingerprint mismatch")
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

    def rebuild_graph(self, generation_id: str) -> GraphReadiness:
        """Build only the deterministic semantic graph for an existing v1 generation."""
        with self._connect() as conn, conn.transaction():
            current = self._require_generation(conn, generation_id, lock=True)
            if current.state == GenerationState.LEGACY_UNVERIFIED:
                raise GenerationError(
                    "legacy_unverified generations have no v1 chunk projection; rebuild the "
                    "generation before building its semantic graph"
                )
            if current.state in {GenerationState.BUILDING, GenerationState.VALIDATING}:
                raise InvalidGenerationTransition(
                    f"graph rebuild requires a stable generation, found {current.state.value}"
                )
            rows = conn.execute(
                "SELECT chunk_id, source_uri, text, metadata FROM recall_chunks_v1 "
                "WHERE tenant_id = %s AND generation_id = %s ORDER BY chunk_id",
                (self.tenant_id, generation_id),
            ).fetchall()
            graph_started = time.perf_counter()
            try:
                graph = build_semantic_graph(
                    [
                        Chunk(
                            str(row[0]),
                            str(row[1]),
                            str(row[2]),
                            row[3] if isinstance(row[3], dict) else {},
                        )
                        for row in rows
                    ],
                    tenant_id=self.tenant_id,
                    generation_id=generation_id,
                    pipeline_fingerprint=current.pipeline_fingerprint,
                    corpus_fingerprint=current.corpus_fingerprint,
                )
                write_semantic_graph(conn, graph)
            except BaseException:
                METRICS.increment("recall_graph_build_failure_total")
                METRICS.observe(
                    "recall_graph_latency_ms",
                    (time.perf_counter() - graph_started) * 1000.0,
                )
                raise
            METRICS.increment("recall_graph_build_total")
            METRICS.observe(
                "recall_graph_latency_ms", (time.perf_counter() - graph_started) * 1000.0
            )
            summary_row = conn.execute(
                "SELECT validation_summary FROM recall_generations "
                "WHERE tenant_id = %s AND generation_id = %s",
                (self.tenant_id, generation_id),
            ).fetchone()
            summary = dict(summary_row[0]) if summary_row and isinstance(summary_row[0], Mapping) else {}
            summary["semantic_graph"] = _semantic_graph_marker(graph)
            conn.execute(
                "UPDATE recall_generations SET validation_summary = %s "
                "WHERE tenant_id = %s AND generation_id = %s",
                (Jsonb(summary), self.tenant_id, generation_id),
            )
            self._audit(
                conn,
                "generation_graph_rebuilt",
                generation_id=generation_id,
                payload=summary["semantic_graph"],
            )
            return graph.readiness()

    def promote(self, generation_id: str, *, unsafe_development: bool = False) -> None:
        if self.certification_required:
            if unsafe_development:
                raise UnsafePromotion(
                    "unsafe_development is unavailable for a tenant served under production"
                )
        elif not unsafe_development:
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
            if self.certification_required:
                self.require_certified_for_production(generation_id, conn=conn)
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

    def rollback(self, *, provisional_reason: str | None = None) -> str:
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
            status = self.calibration_status_for(previous, conn=conn)
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
                payload={
                    "replaced_generation_id": active,
                    "calibration_status": status,
                    "provisional_reason": (
                        provisional_reason or "rollback: incident recovery"
                        if status != "certified"
                        else None
                    ),
                },
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

    def servable_manifest(self) -> IndexManifestV1:
        """Return the newest READY or ACTIVE manifest for carry-forward builds."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT manifest FROM recall_generations "
                "WHERE tenant_id = %s AND state IN ('ready', 'active') "
                "ORDER BY created_at DESC, generation_id DESC LIMIT 1",
                (self.tenant_id,),
            ).fetchone()
        if row and isinstance(row[0], Mapping):
            return IndexManifestV1.from_dict(row[0])
        return self.active_manifest()

    def superseded_ready_generations(
        self, keep: str, *, corpus_version_prefix: str | None = None
    ) -> tuple[str, ...]:
        return tuple(
            record.generation_id
            for record in self.list_generations()
            if record.state is GenerationState.READY
            and record.generation_id != keep
            and (
                corpus_version_prefix is None
                or record.corpus_version.startswith(corpus_version_prefix)
            )
        )

    def active_manifest(self) -> IndexManifestV1:
        """Return the manifest for the active generation.

        Desktop uploads are staged in immutable, per-job directories. A new build must carry
        forward only the files that belong to the active corpus, plus the current job. Reading the
        whole tenant upload tree would also pick up files from failed or abandoned jobs.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT g.manifest "
                "FROM recall_tenant_state AS t "
                "JOIN recall_generations AS g "
                "  ON g.tenant_id = t.tenant_id AND g.generation_id = t.active_generation_id "
                "WHERE t.tenant_id = %s",
                (self.tenant_id,),
            ).fetchone()
        if not row or not isinstance(row[0], Mapping):
            raise NoActiveGeneration(f"tenant {self.tenant_id!r} has no active generation")
        return IndexManifestV1.from_dict(row[0])

    def forget(self, source_uri: str, *, legacy_table: str | None = None) -> ErasureResult:
        """Erase a source from every generation, and tombstone it against future builds.

        `legacy_table` additionally erases the adopted v0.8 rows, which migration 0008 leaves
        in place and which remain readable through the legacy API. Pass it whenever the caller
        knows the migration target, or the erasure is only partial.
        """
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
                "SELECT generation_id, manifest FROM recall_generations "
                "WHERE tenant_id = %s AND state != 'legacy_unverified' "
                "FOR UPDATE",
                (self.tenant_id,),
            ).fetchall()
            selected.update(str(row[0]) for row in mutable)
            manifests = {
                str(row[0]): IndexManifestV1.from_dict(row[1])
                for row in mutable
                if isinstance(row[1], Mapping)
            }
            self._audit(
                conn,
                "source_forgotten",
                source_uri=source_uri,
                payload={"generation_ids": sorted(selected)},
                event_id=event_id,
            )
            conn.execute(
                # DO NOTHING, not DO UPDATE: `erased_at` records WHEN an irreversible erasure
                # happened, and re-issuing the request does not move that moment. Updating it
                # let the recorded time of a right-to-erasure action drift forward every time
                # anyone repeated the call. The repeat is still recorded, as its own
                # `source_forgotten` audit event above, so nothing is lost by keeping the first.
                "INSERT INTO recall_source_tombstones "
                "(tenant_id, source_uri, event_id, erased_at) "
                "VALUES (%s, %s, %s, clock_timestamp()) "
                "ON CONFLICT (tenant_id, source_uri) DO NOTHING",
                (self.tenant_id, source_uri, event_id),
            )
            tombstones = {
                str(row[0])
                for row in conn.execute(
                    "SELECT source_uri FROM recall_source_tombstones WHERE tenant_id = %s",
                    (self.tenant_id,),
                ).fetchall()
            }
            for generation_id, manifest in manifests.items():
                conn.execute(
                    "UPDATE recall_generations SET corpus_fingerprint = %s "
                    "WHERE tenant_id = %s AND generation_id = %s",
                    (
                        _effective_corpus_fingerprint(manifest, tombstones),
                        self.tenant_id,
                        generation_id,
                    ),
                )
            removed = 0
            if selected:
                rows = conn.execute(
                    "DELETE FROM recall_chunks_v1 WHERE tenant_id = %s "
                    "AND generation_id = ANY(%s) AND source_uri = %s RETURNING chunk_id",
                    (self.tenant_id, list(selected), source_uri),
                ).fetchall()
                removed = len(rows)
                self._scrub_sparse_rows(conn, [str(row[0]) for row in rows])
                # Chunk foreign keys remove mentions and relation evidence. Relations and
                # entities are derived rows, so remove any that no longer have surviving support
                # and leave the generation marker mismatched until an explicit graph rebuild.
                conn.execute(
                    "DELETE FROM recall_graph_relations_v1 r "
                    "WHERE r.tenant_id = %s AND r.generation_id = ANY(%s) "
                    "AND NOT EXISTS (SELECT 1 FROM recall_graph_relation_evidence_v1 e "
                    "WHERE e.tenant_id = r.tenant_id AND e.generation_id = r.generation_id "
                    "AND e.relation_id = r.relation_id)",
                    (self.tenant_id, list(selected)),
                )
                conn.execute(
                    "DELETE FROM recall_graph_entities_v1 e "
                    "WHERE e.tenant_id = %s AND e.generation_id = ANY(%s) "
                    "AND NOT EXISTS (SELECT 1 FROM recall_graph_mentions_v1 m "
                    "WHERE m.tenant_id = e.tenant_id AND m.generation_id = e.generation_id "
                    "AND m.entity_id = e.entity_id) "
                    "AND NOT EXISTS (SELECT 1 FROM recall_graph_relations_v1 r "
                    "WHERE r.tenant_id = e.tenant_id AND r.generation_id = e.generation_id "
                    "AND (r.subject_id = e.entity_id OR r.object_id = e.entity_id))",
                    (self.tenant_id, list(selected)),
                )
            if legacy_table is not None:
                # Migration 0008 adopts a v0.8 install's rows in place: they stay in the legacy
                # table and remain readable through the legacy API. Erasing only
                # recall_chunks_v1 would write the tombstone and leave the actual data on disk,
                # which on a right-to-erasure path is worse than refusing outright. Same
                # transaction, so the tombstone and the deletion cannot diverge.
                if not legacy_table.isidentifier():
                    raise ValueError("legacy_table must be a valid SQL identifier")
                exists = conn.execute(
                    "SELECT to_regclass(%s) IS NOT NULL", (legacy_table,)
                ).fetchone()
                if exists and exists[0]:
                    legacy_rows = conn.execute(
                        psycopg.sql.SQL(
                            "DELETE FROM {} WHERE tenant_id = %s AND source = %s RETURNING id"
                        ).format(psycopg.sql.Identifier(legacy_table)),
                        (self.tenant_id, source_uri),
                    ).fetchall()
                    removed += len(legacy_rows)
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
                chunk_rows = conn.execute(
                    "SELECT chunk_id FROM recall_chunks_v1 "
                    "WHERE tenant_id = %s AND generation_id = ANY(%s)",
                    (self.tenant_id, delete),
                ).fetchall()
                conn.execute(
                    "DELETE FROM recall_generations WHERE tenant_id = %s "
                    "AND generation_id = ANY(%s)",
                    (self.tenant_id, delete),
                )
                self._scrub_sparse_rows(conn, [str(row[0]) for row in chunk_rows])
                self._audit(conn, "generation_gc", payload={"generation_ids": delete})
            return tuple(delete)
