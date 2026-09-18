"""Persistence adapter for hosted raw and compiled memories."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from typing import Any, Protocol

from recall.embeddings import Embedder, embed_passages
from recall.sparse import SparseEncoderProtocol
from recall.store import PgVectorStore
from recall.types import Chunk
from recall_aml.compiler import StoredCodingRecord
from recall_aml.models import CodingMemoryRecord


class Repository(Protocol):
    def acquire_request_lock(self, tenant: str, request_id: str) -> Any: ...
    def release_request_lock(self, handle: Any) -> None: ...
    def get_receipt(self, tenant: str, request_id: str, fingerprint: str) -> str | None: ...
    def record_receipt(
        self, tenant: str, request_id: str, fingerprint: str, result: str
    ) -> None: ...
    def prior_records(self, tenant: str, source: str) -> list[StoredCodingRecord]: ...
    def persist(self, tenant: str, chunks: Sequence[Chunk]) -> int: ...
    def verify_sparse_coverage(self, tenant: str) -> dict[str, object]: ...
    def backfill_sparse(self, tenant: str) -> dict[str, object]: ...
    def tenant_store(self, tenant: str) -> PgVectorStore: ...
    def corpus_status(self, tenant: str) -> dict[str, object]: ...
    def health(self) -> dict[str, object]: ...
    def delete_tenant(self, tenant: str) -> int: ...


class PgHostedRepository:
    def __init__(
        self,
        base_store: PgVectorStore,
        embedder: Embedder,
        sparse_encoder: SparseEncoderProtocol | None = None,
    ) -> None:
        self._base_store = base_store
        self._embedder = embedder
        self._sparse_encoder = sparse_encoder

    def tenant_store(self, tenant: str) -> PgVectorStore:
        return self._base_store.for_tenant(tenant)

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

    def prior_records(self, tenant: str, source: str) -> list[StoredCodingRecord]:
        records: list[StoredCodingRecord] = []
        for chunk in self.tenant_store(tenant).chunks_for_source(source):
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

    def delete_tenant(self, tenant: str) -> int:
        return self.tenant_store(tenant).delete_tenant_data()


_ELIGIBLE_GRAPH_RELATIONS = frozenset(
    {"supports", "references", "depends_on", "caused", "supersedes"}
)


def describe_corpus(store: PgVectorStore) -> dict[str, object]:
    """Return a deterministic text-side identity for one Hosted tenant."""
    chunk_digests: list[str] = []
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
        chunk_digests.append(hashlib.sha256(serialized).hexdigest())
    chunk_digests.sort()
    corpus_sha256 = hashlib.sha256("\n".join(chunk_digests).encode("ascii")).hexdigest()
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
    }
