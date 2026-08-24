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

from recall.document import parse_document
from recall.embeddings import Embedder, embed_passages
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
from recall.errors import RecallError

Chunker = Callable[[str], list[str]]

DEFAULT_RETENTION_DAYS = 7
DEFAULT_RETAIN_PREVIOUS = 2
TEMPORARY_STORAGE_MULTIPLIER = 2.2
DEFAULT_TABLE_MAX_CHARS = 800
DEFAULT_TABLE_OVERLAP = 80


class GenerationError(RuntimeError, RecallError):
    """A generation lifecycle invariant was violated."""


class GenerationNotFound(GenerationError):
    pass


class InvalidGenerationTransition(GenerationError):
    pass


class UnsafePromotion(GenerationError):
    pass


class ConcurrentIngest(GenerationError):
    """Another ingest holds this tenant's lock.

    Raised rather than waited out, because the caller is an interactive upload and an unbounded
    wait inside a tool call is indistinguishable from a hang. A retry is cheap; a lost upload that
    reported success is not.
    """


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


def _scrub_sparse_rows(
    conn: psycopg.Connection, tenant_id: str, chunk_table: str, chunk_ids: list[str]
) -> int:
    """Delete the learned-sparse sidecar rows for chunks an erasure just removed.

    A deliberate duplicate of `PgVectorStore._scrub_sparse_rows` rather than an import: this
    module must not depend on the store (see the store helper for the full rationale). Same
    contract: same transaction as the chunk delete, every profile's rows die together, and
    the table name is a bound VALUE against the sidecar's `chunk_table` column.
    """
    if not chunk_ids:
        return 0
    sidecar = conn.execute("SELECT to_regclass('recall_sparse_v1')").fetchone()
    if not (sidecar and sidecar[0]):
        return 0
    return (
        conn.execute(
            "DELETE FROM recall_sparse_v1 "
            "WHERE tenant_id = %s AND chunk_table = %s AND id = ANY(%s)",
            (tenant_id, chunk_table, chunk_ids),
        ).rowcount
        or 0
    )


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
        """`environment` is where the BUILD runs; `serving_environment` is where the tenant is READ.

        ⚠️ **They are the same value almost everywhere and different on the path that matters.** The
        certification gate keys on the serving one, because certification is a claim about what a
        query will get back, not about which process built it.

        The wizard is the case that forced this apart, and it is the ordinary install rather than a
        corner: it builds every corpus under `environment="development"`, because a production build
        demands a verifiable embedder identity that a bundled FastEmbed model does not have, and
        then writes `RECALL_ENV=production` into the MCP server block that serves those same
        tenants. So before this parameter existed, the gate ran on **no tenant the wizard
        creates** — every real first install took the ungated branch and was then served as
        production. The wizard's own pipeline refuses to promote an uncertified generation, so
        nothing shipped uncertified; but the gate and the pipeline check were two independent
        implementations of one rule, and only one of them was tested as the thing that holds it.

        `serving_environment` defaults to `environment`, so every existing caller is unchanged.
        """
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
            raise ValueError(
                "serving_environment must be development, test, or production"
            )

    @property
    def certification_required(self) -> bool:
        """Whether promotion here must present a CERTIFIED calibration.

        Reads the SERVING environment, not the build one. `promote` and every caller that decides
        whether to pass `unsafe_development` consult this rather than comparing environments
        themselves, so the two cannot drift apart.
        """
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
        """Serialise the whole ingest for ONE tenant: read the base, build, promote, reclaim.

        ⛔ **Session-scoped, not `pg_advisory_xact_lock`.** The sibling `_source_lock` uses the
        transaction-scoped form, which is right for the single statement it guards and useless
        here: an ingest spans several transactions on several connections, and a transaction lock
        is released at the first commit, which is before the interesting part even starts.

        ⛔ **Why this exists at all.** Nothing serialised `servable_manifest()` through `promote`,
        so two uploads to one tenant both seeded from the same base manifest M, built M+A and M+B
        respectively, and whichever promoted second retired the other. The loser's files were
        absent from the live corpus while its tool call had already reported success — a silent
        loss with a success message on top, which is the worst shape this codebase has. Nothing
        counted it, because nothing had vanished: `_vanished_note` reports 0 and is right to.

        A second, narrower hazard closes with it: `_release_superseded` selects READY generations
        by prefix, and a concurrent upload sits READY between `validate()` and `promote()`. It was
        therefore a candidate for abandonment by the other upload, which would destroy a build that
        had already succeeded and then fail its promote with `InvalidGenerationTransition`.

        ⚠️ **Refuses after `wait_seconds` instead of waiting.** The caller is an interactive
        upload. A bounded wait absorbs the ordinary case of two clicks in quick succession; beyond
        that, saying "another upload is in progress, try again" is honest and a hang is not.

        The lock is held on its OWN connection, so it outlives each step's transaction, and it is
        released by closing that connection even if the process dies mid-ingest — a session lock
        does not survive its backend.
        """
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
                        f"another upload into tenant {self.tenant_id!r} is still running. "
                        "Uploads to one tenant are serialised, because two at once would each "
                        "build from the same starting point and the second to finish would "
                        "silently drop the first one's files. Wait for it to finish and try again."
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
                            },
                        )
                    )
                # PASSAGE encoding: these vectors are what a query is matched against. With an
                # asymmetric model the query encoder produces a different vector for the same
                # text, and a generation built with the wrong one is the right width, scores in
                # range, and silently retrieves worse. Falls back to `embed` for an embedder
                # that only implements the symmetric interface.
                embeddings = embed_passages(embedder, [chunk.text for chunk in chunks])
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
        """Mark a READY generation failed, so `gc` can reclaim its rows.

        `fail` refuses READY, ACTIVE, RETIRED and LEGACY_UNVERIFIED, and that is right for the
        states where failing something would destroy what is serving. But it left READY with no
        exit at all, and READY is where a generation lands when it built and validated and was
        then not promoted. `gc` collects only `retired` and `failed`, and `recall_chunks_v1`
        cascades from `recall_generations`, so such a generation holds a full copy of the corpus's
        chunk rows that no supported path could ever reclaim. The installation wizard reaches that
        state twice per run whenever certification falls short, which is the ordinary outcome of a
        first install, so without this the database grows without bound on the success path.

        READY only. This is a reclaim route, not a second way to fail something, and widening
        `fail` instead would have removed the protection ACTIVE and RETIRED actually need.
        """
        with self._connect() as conn, conn.transaction():
            # ⚠️ Lock ORDER matters, and it is tenant state first, then the generation row. That is
            # the order `promote` and `gc` both take, and taking the two the other way round is a
            # deadlock: `promote` would hold tenant state waiting for the generation row while this
            # held the generation row waiting for tenant state. The first version of this method
            # had them reversed.
            #
            # The generation a rollback would return to must survive. `promote` moves the outgoing
            # generation to RETIRED, so a READY generation is normally neither, but the read is
            # cheap and the alternative is destroying the only thing `rollback` can restore.
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
            protected = {str(item) for item in (state or ()) if item}
            if generation_id in protected:
                raise InvalidGenerationTransition(
                    f"cannot abandon generation {generation_id!r}: it is the tenant's active or "
                    "previous generation"
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
        """Make one ready generation live, gated on how the tenant is SERVED.

        See `certification_required`: the gate follows the serving environment, so a build that runs
        under `development` in order to use an unverifiable embedder is still gated when its output
        will be read under `RECALL_ENV=production`. That is the wizard, which is to say every
        install this project ships.
        """
        if self.certification_required:
            # ⛔ **`unsafe_development` does not open this door.** Refused here rather than after
            # the certification check, so a caller cannot learn that the flag would otherwise have
            # worked. The production path has exactly one way through: a certified calibration.
            if unsafe_development:
                raise UnsafePromotion(
                    "unsafe_development is unavailable for a tenant served under production; "
                    "promotion there requires a published, certified calibration bound to this "
                    "generation"
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
            # ⚠️ **After the existence and state checks, deliberately.** Run first, the gate answered
            # a missing generation with `CalibrationBindingError: generation ... does not exist` and
            # a mid-build one with the same, replacing `GenerationNotFound` and
            # `InvalidGenerationTransition` — three different problems arriving as one unrelated
            # exception type. Measured before this was moved. Ordering the cheap, specific checks
            # first keeps each failure named by the thing that actually failed.
            if self.certification_required:
                # On THIS connection, inside THIS transaction. See the method's docstring: the
                # generation row is already locked here, so the check and the activation cannot be
                # separated by a concurrent fingerprint rewrite.
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
            # Two paths reach here and they are not the same event. Recording a certified
            # production promotion as `..._unsafe_development` would make the audit trail say the
            # opposite of what happened, and the audit trail is the artefact that outlives everyone
            # who remembers the code.
            self._audit(
                conn,
                (
                    "generation_promoted_certified"
                    if self.certification_required
                    else "generation_promoted_unsafe_development"
                ),
                generation_id=generation_id,
                payload={"previous_generation_id": active},
            )

    def rollback(self, *, provisional_reason: str | None = None) -> str:
        """Return the tenant to its previous generation. **Never refuses on certification grounds.**

        ⛔ **This refused in production for one release, and that was wrong.**
        `docs/UNCALIBRATED_FIRST_RUN_DESIGN.md` section 6 decision 2 had already settled the
        question, in bold: "`rollback(*, provisional_reason: str | None = None)`, and it never
        refuses on certification grounds ... Rollback is the incident path. A gate that blocks
        recovery precisely when recovery is needed is worse than serving a `provisional` answer
        that says so on the wire, and an operator facing a bad generation will route around a
        refusal in a way nobody audits. **Refusing here would trade a visible degradation for an
        invisible workaround.**" The gate was added without reading that section.

        Three ways the refusal bit, each found independently in the audit that caught this:

        * **`forget()` bricked it permanently.** It rewrites `corpus_fingerprint` on every
          generation of the tenant, so after one erasure request every calibration resolves STALE
          and there was no target left to return to — ever.
        * **Upgrading bricked it.** Production `promote` refused outright before this release, so
          every generation an existing install is serving was promoted under `development` and has
          no published calibration. Upgrading would have removed rollback from every one of them.
        * There is no override to reach for, which is exactly the "invisible workaround" the
          decision predicted: the remaining routes are a mid-incident recalibration that must
          certify, or flipping `RECALL_ENV`, which silently changes five other policies.

        **The invariant F2 is about survives**, because it was never "only certified generations go
        live". It was "no generation becomes active without the operator being told what they are
        activating". A rollback to an uncertified target is recorded as such: the audit event
        carries the resolved calibration status and the reason, so the downgrade is visible rather
        than prevented. `docs/UNCALIBRATED_FIRST_RUN_DESIGN.md`: "Prevented, no; hidden, never."

        `promote` keeps its gate. Promotion is the planned path and has somewhere to go back to;
        rollback is what you reach for when it does not.
        """
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
            # ⛔ Deliberately NOT gated. See the docstring: this is the incident path, and the
            # status is reported into the audit event below rather than used to refuse.
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
                    # ⚠️ The downgrade is REPORTED, which is the whole of decision 2's bargain:
                    # "Prevented, no; hidden, never." A rollback onto an uncertified target is
                    # allowed and leaves a record saying exactly what was activated and why.
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
        """The manifest a NEW build should carry forward: the newest generation worth continuing.

        ⛔ **`active_manifest` is the wrong base once a build can finish without being activated.**
        A desktop upload whose promotion is refused leaves its generation READY, never active, so
        `active_generation_id` does not advance. The next upload then seeds from the OLD active
        manifest and silently contains none of the previous upload's files: two READY generations,
        neither holding the whole corpus, and the message from the first told the user to certify
        the one that will be superseded. Three auditors found this independently.

        So the base is the newest generation in a state that can still become active — READY or
        ACTIVE — falling back to the active one, and to an empty manifest when the tenant has
        neither. RETIRED and FAILED are excluded: continuing from a retired corpus would resurrect
        content that was deliberately rolled away from.
        """
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
        """READY generations for this tenant other than `keep`, newest first.

        The reclaim list. A READY generation holds a full copy of the corpus's chunk rows and `gc`
        collects only `retired` and `failed`, so one left behind per refused upload grows the
        database without bound — the leak `abandon` was written to close, documented in its own
        docstring, and reintroduced by a path that returns success instead of raising.

        ⛔ **`corpus_version_prefix` is not optional in spirit, and omitting it destroys other
        people's corpora.** Without it this selects on STATE ALONE, and `abandon` protects only the
        tenant's active and previous generations — so a generation that was built and validated but
        deliberately NEVER promoted is fair game. That is precisely what a degraded wizard install
        leaves behind: `recall/wizard/pipeline.py` skips `promote` when the corpus did not certify,
        on purpose, so an operator can promote it later. One refused desktop upload would then
        abandon it, `gc` would collect it, and the chunk rows would cascade away.

        Caller-supplied rather than inferred, because only the caller knows which generations are
        its own to reclaim. A caller that cannot name its own provenance should not be reclaiming.
        """
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
                _scrub_sparse_rows(
                    conn, self.tenant_id, "recall_chunks_v1", [row[0] for row in rows]
                )
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
                    _scrub_sparse_rows(
                        conn, self.tenant_id, legacy_table, [row[0] for row in legacy_rows]
                    )
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
                # The chunk rows die by ON DELETE CASCADE, which reaches no _scrub_sparse_rows,
                # so the learned-sparse sidecar would orphan term-weight rows (partially
                # reconstructable content) for every chunk_id unique to a collected generation.
                # Anti-join scrub in the same transaction: delete sidecar rows for this table
                # whose chunk_id no longer exists in ANY surviving generation. `_reuse_source`
                # shares a chunk_id across generations, so a chunk still held by the active
                # generation still exists here and its row is correctly kept. Guarded, like the
                # other scrubs, so a pre-0012 install has nothing to fail against.
                sidecar = conn.execute("SELECT to_regclass('recall_sparse_v1')").fetchone()
                if sidecar and sidecar[0]:
                    conn.execute(
                        "DELETE FROM recall_sparse_v1 s "
                        "WHERE s.tenant_id = %s AND s.chunk_table = 'recall_chunks_v1' "
                        "AND NOT EXISTS ("
                        "  SELECT 1 FROM recall_chunks_v1 c "
                        "  WHERE c.tenant_id = s.tenant_id AND c.chunk_id = s.id)",
                        (self.tenant_id,),
                    )
                self._audit(conn, "generation_gc", payload={"generation_ids": delete})
            return tuple(delete)
