"""Persistence adapter for hosted raw and compiled memories."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from recall.embeddings import Embedder, embed_passages
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
    def tenant_store(self, tenant: str) -> PgVectorStore: ...
    def health(self) -> dict[str, object]: ...
    def delete_tenant(self, tenant: str) -> int: ...


class PgHostedRepository:
    def __init__(self, base_store: PgVectorStore, embedder: Embedder) -> None:
        self._base_store = base_store
        self._embedder = embedder

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
        return self.tenant_store(tenant).upsert(materialized, vectors)

    def health(self) -> dict[str, object]:
        self._base_store.check_schema()
        return {
            "database_ready": True,
            "generation_id": self._base_store.generation_id,
        }

    def delete_tenant(self, tenant: str) -> int:
        return self.tenant_store(tenant).delete_tenant_data()
