"""Persistence adapter for hosted raw and compiled memories."""

from __future__ import annotations

from collections.abc import Sequence
from collections import Counter
from dataclasses import replace
import hashlib
import json
from typing import Any, Protocol

from recall.embeddings import Embedder, embed_passages
from recall.sparse import SparseEncoderProtocol
from recall.store import SPARSE_TABLE, PgVectorStore
from recall.types import Chunk
from recall_aml.compiler import StoredCodingRecord
from recall_aml.models import CodingMemoryRecord
from recall_aml.identity import atomic_view_tenant, graph_tenant, specialist_tenant
from recall_aml.multimodal import media_tenant, multimodal_tenant


class Repository(Protocol):
    def acquire_request_lock(self, tenant: str, request_id: str) -> Any: ...
    def release_request_lock(self, handle: Any) -> None: ...
    def get_receipt(self, tenant: str, request_id: str, fingerprint: str) -> str | None: ...
    def record_receipt(
        self, tenant: str, request_id: str, fingerprint: str, result: str
    ) -> None: ...
    def prior_records(
        self, tenant: str, source: str, *, graph_sidecar: bool = False
    ) -> list[StoredCodingRecord]: ...
    def persist(self, tenant: str, chunks: Sequence[Chunk]) -> int: ...
    def persist_graph(self, tenant: str, chunks: Sequence[Chunk]) -> int: ...
    def persist_media(self, tenant: str, chunks: Sequence[Chunk]) -> int: ...
    def persist_multimodal(
        self, tenant: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> int: ...
    def persist_multimodal_bundle(
        self,
        tenant: str,
        primary_chunks: Sequence[Chunk],
        specialist_profile: str | None,
        media_chunks: Sequence[Chunk],
        multimodal_chunks: Sequence[Chunk],
        multimodal_vectors: Sequence[Sequence[float]],
    ) -> int: ...
    def persist_specialist(
        self, tenant: str, embedding_profile: str, chunks: Sequence[Chunk]
    ) -> int: ...
    def specialist_store(self, tenant: str, embedding_profile: str) -> PgVectorStore: ...
    def persist_atomic_views(
        self, tenant: str, embedding_profile: str | None, chunks: Sequence[Chunk]
    ) -> int: ...
    def atomic_view_store(self, scope_tenant: str) -> PgVectorStore: ...
    def media_store(self, tenant: str) -> PgVectorStore: ...
    def multimodal_store(self, tenant: str) -> PgVectorStore: ...
    def graph_store(self, tenant: str) -> PgVectorStore: ...
    def verify_sparse_coverage(self, tenant: str) -> dict[str, object]: ...
    def backfill_sparse(self, tenant: str) -> dict[str, object]: ...
    def tenant_store(self, tenant: str) -> PgVectorStore: ...
    def corpus_status(self, tenant: str) -> dict[str, object]: ...
    def graph_corpus_status(self, tenant: str) -> dict[str, object]: ...
    def health(self) -> dict[str, object]: ...
    def delete_tenant(self, tenant: str) -> int: ...


class PgHostedRepository:
    distributed_locks = True

    def __init__(
        self,
        base_store: PgVectorStore,
        embedder: Embedder,
        sparse_encoder: SparseEncoderProtocol | None = None,
        specialist_embedders: dict[str, Embedder] | None = None,
    ) -> None:
        self._base_store = base_store
        self._embedder = embedder
        self._sparse_encoder = sparse_encoder
        self._specialist_embedders = dict(specialist_embedders or {})

    def tenant_store(self, tenant: str) -> PgVectorStore:
        return self._base_store.for_tenant(tenant)

    def media_store(self, tenant: str) -> PgVectorStore:
        return self.tenant_store(media_tenant(tenant))

    def multimodal_store(self, tenant: str) -> PgVectorStore:
        return self.tenant_store(multimodal_tenant(tenant))

    def graph_store(self, tenant: str) -> PgVectorStore:
        return self.tenant_store(graph_tenant(tenant))

    def specialist_store(self, tenant: str, embedding_profile: str) -> PgVectorStore:
        if embedding_profile not in self._specialist_embedders:
            raise RuntimeError(f"specialist embedder is not configured: {embedding_profile}")
        return self.tenant_store(specialist_tenant(tenant, embedding_profile))

    def atomic_view_store(self, scope_tenant: str) -> PgVectorStore:
        """The atomic views of one retrieval scope, isolated from the corpus they rescue."""
        return self.tenant_store(atomic_view_tenant(scope_tenant))

    def persist_atomic_views(
        self, tenant: str, embedding_profile: str | None, chunks: Sequence[Chunk]
    ) -> int:
        """Embed one request's atomic views with its scope's embedder, in its own namespace.

        ``embedding_profile`` None is the primary scope; a profile is that Context specialist.
        The views of one request are one embedding call, so a contextual embedder sees them as
        one document, as it sees the request's windows in ``persist_specialist``.
        """
        materialized = list(chunks)
        if not materialized:
            return 0
        if embedding_profile is None:
            embedder = self._embedder
            scope = tenant
        else:
            specialist = self._specialist_embedders.get(embedding_profile)
            if specialist is None:
                raise RuntimeError(f"specialist embedder is not configured: {embedding_profile}")
            embedder = specialist
            scope = specialist_tenant(tenant, embedding_profile)
            materialized = [
                replace(chunk, metadata={**chunk.metadata, "embedding_profile": embedding_profile})
                for chunk in materialized
            ]
        vectors = embed_passages(embedder, [chunk.text for chunk in materialized])
        return self.atomic_view_store(scope).upsert(materialized, vectors)

    def acquire_request_lock(self, tenant: str, request_id: str) -> Any:
        guard = self.tenant_store(tenant).operation_lock("hosted_add_v1:" + request_id)
        guard.__enter__()
        return guard

    @staticmethod
    def release_request_lock(handle: Any) -> None:
        handle.__exit__(None, None, None)

    def get_receipt(self, tenant: str, request_id: str, fingerprint: str) -> str | None:
        return self.tenant_store(tenant).get_operation_receipt(
            request_id, operation="hosted_add_v1", request_fingerprint=fingerprint
        )

    def record_receipt(self, tenant: str, request_id: str, fingerprint: str, result: str) -> None:
        self.tenant_store(tenant).record_operation_receipt(
            request_id,
            result,
            operation="hosted_add_v1",
            request_fingerprint=fingerprint,
        )

    def prior_records(
        self, tenant: str, source: str, *, graph_sidecar: bool = False
    ) -> list[StoredCodingRecord]:
        records: list[StoredCodingRecord] = []
        store = self.graph_store(tenant) if graph_sidecar else self.tenant_store(tenant)
        for chunk in store.chunks_for_source(source):
            payload = chunk.metadata.get("coding_record")
            if chunk.metadata.get("record_type") != "compiled" or not isinstance(payload, dict):
                continue
            try:
                records.append(
                    StoredCodingRecord(chunk.id, CodingMemoryRecord.model_validate(payload))
                )
            except ValueError:
                continue
        return records

    def persist(self, tenant: str, chunks: Sequence[Chunk]) -> int:
        materialized = list(chunks)
        vectors = embed_passages(self._embedder, [chunk.text for chunk in materialized])
        store = self.tenant_store(tenant)
        written = store.upsert(materialized, vectors)
        if self._sparse_encoder is not None and materialized:
            sparse_vectors = self._sparse_encoder.encode([chunk.text for chunk in materialized])
            if len(sparse_vectors) != len(materialized):
                raise RuntimeError("learned sparse encoder returned the wrong number of vectors")
            if any(not vector for vector in sparse_vectors):
                raise RuntimeError("learned sparse encoder returned an empty passage vector")
            store.upsert_sparse(
                self._sparse_encoder.profile.profile_id,
                {
                    chunk.id: vector
                    for chunk, vector in zip(materialized, sparse_vectors, strict=True)
                },
            )
            self.verify_sparse_coverage(tenant)
        return written

    def persist_graph(self, tenant: str, chunks: Sequence[Chunk]) -> int:
        """Persist derived records in an isolated tenant so raw membership cannot drift."""
        materialized = list(chunks)
        if not materialized:
            return 0
        vectors = embed_passages(self._embedder, [chunk.text for chunk in materialized])
        return self.graph_store(tenant).upsert(materialized, vectors)

    def persist_media(self, tenant: str, chunks: Sequence[Chunk]) -> int:
        """Store exact media once without paying for a meaningless text embedding."""
        materialized = list(chunks)
        if not materialized:
            return 0
        zeros = [[0.0] * self._embedder.dim for _ in materialized]
        return self.media_store(tenant).upsert(materialized, zeros)

    def persist_multimodal(
        self,
        tenant: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        materialized = list(chunks)
        materialized_vectors = [list(vector) for vector in vectors]
        if len(materialized) != len(materialized_vectors):
            raise ValueError("multimodal chunks and vectors must have equal length")
        return self.multimodal_store(tenant).upsert(materialized, materialized_vectors)

    def persist_multimodal_bundle(
        self,
        tenant: str,
        primary_chunks: Sequence[Chunk],
        specialist_profile: str | None,
        media_chunks: Sequence[Chunk],
        multimodal_chunks: Sequence[Chunk],
        multimodal_vectors: Sequence[Sequence[float]],
    ) -> int:
        """Persist all multimodal namespaces in one PostgreSQL transaction.

        The physical table is shared by logical tenants. Switching the transaction-local RLS
        tenant before each upsert lets the primary, specialist, media, and native-vector rows
        share one commit boundary without weakening row-level security.
        """
        primary = list(primary_chunks)
        media = list(media_chunks)
        native = list(multimodal_chunks)
        native_vectors = [list(vector) for vector in multimodal_vectors]
        if len(native) != len(native_vectors):
            raise ValueError("multimodal chunks and vectors must have equal length")
        specialist: list[Chunk] = []
        specialist_vectors: list[list[float]] = []
        if specialist_profile is not None:
            embedder = self._specialist_embedders.get(specialist_profile)
            if embedder is None:
                raise RuntimeError(f"specialist embedder is not configured: {specialist_profile}")
            specialist = [
                replace(chunk, metadata={**chunk.metadata, "embedding_profile": specialist_profile})
                for chunk in primary
            ]
            specialist_vectors = embed_passages(embedder, [chunk.text for chunk in specialist])
        primary_vectors = embed_passages(self._embedder, [chunk.text for chunk in primary])
        media_vectors = [[0.0] * self._embedder.dim for _ in media]
        table = self._base_store._table

        def _write(conn: Any, physical_tenant: str, chunks: list[Chunk], vectors: list[list[float]]) -> int:
            if not chunks:
                return 0
            conn.execute("SELECT set_config('recall.tenant_id', %s, true)", (physical_tenant,))
            tenant_store = object.__new__(type(self._base_store))
            tenant_store.__dict__.update(self._base_store.__dict__)
            tenant_store._tenant = physical_tenant
            tenant_store._upsert_in(conn, chunks, vectors)
            return len(chunks)

        def _op(conn: Any) -> int:
            with conn.transaction():
                written = _write(conn, tenant, primary, primary_vectors)
                if specialist_profile is not None:
                    written += _write(
                        conn,
                        specialist_tenant(tenant, specialist_profile),
                        specialist,
                        specialist_vectors,
                    )
                written += _write(conn, media_tenant(tenant), media, media_vectors)
                written += _write(conn, multimodal_tenant(tenant), native, native_vectors)
                return written

        del table  # keep the validated shared-table invariant explicit for static checkers
        return self._base_store._with_retry(_op)

    def persist_specialist(
        self,
        tenant: str,
        embedding_profile: str,
        chunks: Sequence[Chunk],
    ) -> int:
        """Embed identical logical chunks into one isolated specialist namespace."""
        embedder = self._specialist_embedders.get(embedding_profile)
        if embedder is None:
            raise RuntimeError(f"specialist embedder is not configured: {embedding_profile}")
        materialized = list(chunks)
        vectors = embed_passages(embedder, [chunk.text for chunk in materialized])
        specialist_chunks = [
            replace(chunk, metadata={**chunk.metadata, "embedding_profile": embedding_profile})
            for chunk in materialized
        ]
        return self.specialist_store(tenant, embedding_profile).upsert(specialist_chunks, vectors)

    def verify_sparse_coverage(self, tenant: str) -> dict[str, object]:
        if self._sparse_encoder is None:
            raise RuntimeError("learned sparse coverage requested without an encoder")
        store = self.tenant_store(tenant)
        dense_count = store.count()
        sparse_count = store.sparse_row_count(self._sparse_encoder.profile.profile_id)
        if sparse_count != dense_count:
            raise RuntimeError(
                f"learned sparse coverage is incomplete: dense={dense_count} sparse={sparse_count}"
            )
        return {
            "sparse_ready": True,
            "sparse_profile": self._sparse_encoder.profile.profile_id,
            "sparse_chunk_count": sparse_count,
        }

    def backfill_sparse(self, tenant: str) -> dict[str, object]:
        """Build only the SPLADE sidecar for an existing dense corpus."""
        if self._sparse_encoder is None:
            raise RuntimeError("learned sparse backfill requested without an encoder")
        store = self.tenant_store(tenant)
        profile_id = self._sparse_encoder.profile.profile_id
        dense_count = store.count()
        if dense_count == 0:
            raise RuntimeError("learned sparse backfill requires an existing dense corpus")
        if store.sparse_row_count(profile_id) == dense_count:
            return self.verify_sparse_coverage(tenant)
        batch: list[Chunk] = []
        for chunk in store.iter_chunks(batch_size=64):
            batch.append(chunk)
            if len(batch) == 64:
                self._persist_sparse_batch(store, profile_id, batch)
                batch = []
        if batch:
            self._persist_sparse_batch(store, profile_id, batch)
        return self.verify_sparse_coverage(tenant)

    def _persist_sparse_batch(
        self, store: PgVectorStore, profile_id: str, chunks: Sequence[Chunk]
    ) -> None:
        assert self._sparse_encoder is not None
        vectors = self._sparse_encoder.encode([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks) or any(not vector for vector in vectors):
            raise RuntimeError("learned sparse backfill returned invalid passage vectors")
        store.upsert_sparse(
            profile_id,
            {chunk.id: vector for chunk, vector in zip(chunks, vectors, strict=True)},
        )

    def health(self) -> dict[str, object]:
        self._base_store.check_schema()
        detail: dict[str, object] = {
            "database_ready": True,
            "generation_id": self._base_store.generation_id,
        }
        if self._sparse_encoder is not None:
            detail["sparse_profile"] = self._sparse_encoder.profile.profile_id
            detail["sparse_device"] = str(getattr(self._sparse_encoder, "device", "unknown"))
        return detail

    def corpus_status(self, tenant: str) -> dict[str, object]:
        return describe_corpus(self.tenant_store(tenant))

    def graph_corpus_status(self, tenant: str) -> dict[str, object]:
        return describe_corpus(self.graph_store(tenant))

    def delete_tenant(self, tenant: str) -> int:
        specialists = [specialist_tenant(tenant, profile) for profile in self._specialist_embedders]
        physical_tenants = [
            tenant,
            media_tenant(tenant),
            multimodal_tenant(tenant),
            graph_tenant(tenant),
            *specialists,
            atomic_view_tenant(tenant),
            *(atomic_view_tenant(scope) for scope in specialists),
        ]
        table = self._base_store._table

        def _op(conn: Any) -> int:
            deleted = 0
            with conn.transaction():
                for physical_tenant in physical_tenants:
                    conn.execute("SELECT set_config('recall.tenant_id', %s, true)", (physical_tenant,))
                    ids = [
                        str(row[0])
                        for row in conn.execute(
                            f"SELECT id FROM {table} WHERE tenant_id = %s",  # noqa: S608
                            (physical_tenant,),
                        ).fetchall()
                    ]
                    if ids and conn.execute("SELECT to_regclass(%s)", (SPARSE_TABLE,)).fetchone()[0]:
                        conn.execute(
                            f"DELETE FROM {SPARSE_TABLE} "  # noqa: S608
                            "WHERE tenant_id = %s AND chunk_table = %s AND id = ANY(%s)",
                            (physical_tenant, table, ids),
                        )
                    deleted += int(
                        conn.execute(
                            f"DELETE FROM {table} WHERE tenant_id = %s",  # noqa: S608
                            (physical_tenant,),
                        ).rowcount
                        or 0
                    )
                    if physical_tenant == tenant:
                        conn.execute(
                            "DELETE FROM recall_idempotency_receipts WHERE tenant_id = %s",
                            (physical_tenant,),
                        )
            return deleted

        return self._base_store._with_retry(_op)


_ELIGIBLE_GRAPH_RELATIONS = frozenset(
    {"supports", "references", "depends_on", "caused", "supersedes"}
)


def describe_corpus(store: PgVectorStore) -> dict[str, object]:
    """Return a deterministic text-side identity for one Hosted tenant."""
    chunk_digests: list[str] = []
    raw_chunk_digests: list[str] = []
    compiled_chunk_digests: list[str] = []
    compiled_kind_counts: Counter[str] = Counter()
    compiler_profile_counts: Counter[str] = Counter()
    raw_count = 0
    compiled_count = 0
    source_sessions: set[str] = set()
    authored_relations = 0
    eligible_relations = 0
    for chunk in store.iter_chunks(batch_size=256):
        metadata = chunk.metadata
        record_type = str(metadata.get("record_type", ""))
        raw_count += int(record_type == "raw")
        compiled_count += int(record_type == "compiled")
        session = metadata.get("source_session_id")
        if session:
            source_sessions.add(str(session))
        graph = metadata.get("recall_graph")
        if isinstance(graph, dict):
            relations = graph.get("relations", [])
            if isinstance(relations, list):
                for relation in relations:
                    if not isinstance(relation, dict):
                        continue
                    authored_relations += 1
                    eligible_relations += int(
                        str(relation.get("relation", "")) in _ELIGIBLE_GRAPH_RELATIONS
                    )
            dependencies = graph.get("depends_on", [])
            if isinstance(dependencies, list):
                authored_relations += len(dependencies)
                eligible_relations += len(dependencies)
        serialized = json.dumps(
            {
                "id": chunk.id,
                "source": chunk.source,
                "text": chunk.text,
                "metadata": metadata,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        digest = hashlib.sha256(serialized).hexdigest()
        chunk_digests.append(digest)
        if record_type == "raw":
            raw_chunk_digests.append(digest)
        elif record_type == "compiled":
            compiled_chunk_digests.append(digest)
            compiled_kind_counts[str(metadata.get("kind", ""))] += 1
            compiler_profile_counts[str(metadata.get("compiler_profile", ""))] += 1
    chunk_digests.sort()
    raw_chunk_digests.sort()
    compiled_chunk_digests.sort()
    corpus_sha256 = hashlib.sha256("\n".join(chunk_digests).encode("ascii")).hexdigest()
    raw_corpus_sha256 = hashlib.sha256(
        "\n".join(raw_chunk_digests).encode("ascii")
    ).hexdigest()
    compiled_corpus_sha256 = hashlib.sha256(
        "\n".join(compiled_chunk_digests).encode("ascii")
    ).hexdigest()
    relation_counter = getattr(store, "authored_graph_relation_count", None)
    store_relation_count = int(relation_counter()) if callable(relation_counter) else 0
    return {
        "generation_id": store.generation_id,
        "chunk_count": len(chunk_digests),
        "raw_chunk_count": raw_count,
        "compiled_chunk_count": compiled_count,
        "source_session_count": len(source_sessions),
        "authored_relation_count": authored_relations,
        "eligible_relation_count": eligible_relations,
        "store_relation_count": store_relation_count,
        "corpus_sha256": corpus_sha256,
        "raw_corpus_sha256": raw_corpus_sha256,
        "compiled_corpus_sha256": compiled_corpus_sha256,
        "compiled_kind_counts": dict(sorted(compiled_kind_counts.items())),
        "compiler_profile_counts": dict(sorted(compiler_profile_counts.items())),
    }
