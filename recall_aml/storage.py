"""Persistence adapter for hosted raw and compiled memories."""

from __future__ import annotations

from collections.abc import Sequence
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
    def tenant_store(self, tenant: str) -> PgVectorStore: ...
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
                "learned sparse coverage is incomplete: "
                f"dense={dense_count} sparse={sparse_count}"
            )
        return {
            "sparse_ready": True,
            "sparse_profile": self._sparse_encoder.profile.profile_id,
            "sparse_chunk_count": sparse_count,
        }

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

    def delete_tenant(self, tenant: str) -> int:
        return self.tenant_store(tenant).delete_tenant_data()
