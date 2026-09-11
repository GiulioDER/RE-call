"""Hosted Add, Search, and deletion orchestration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import timezone
import logging
import time
from typing import Any

from recall.types import Chunk
from recall_aml.compiler import Compiler, deterministic_extract
from recall_aml.identity import canonical_digest, session_digest, tenant_for
from recall_aml.models import (
    AddRequest,
    AddResponse,
    CodingMemoryRecord,
    SearchRequest,
    SearchResponse,
)
from recall_aml.retrieval import HostedRetriever, pack_evidence, render_full_evidence
from recall_aml.storage import Repository
from recall_aml.variants import DEFAULT_VARIANT, HostedVariant, variant


log = logging.getLogger("recall_aml")
RAW_SEGMENT_CHARS = 6_000


@dataclass
class _RequestLock:
    lock: asyncio.Lock
    users: int = 0


def _source(session_id: str) -> str:
    return "aml://session/" + session_digest(session_id)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()


def build_chunks(request: AddRequest, records: list[CodingMemoryRecord]) -> list[Chunk]:
    source = _source(request.session_id)
    chunks: list[Chunk] = []
    for ordinal, message in enumerate(request.messages):
        segments = [
            message.content[offset : offset + RAW_SEGMENT_CHARS]
            for offset in range(0, len(message.content), RAW_SEGMENT_CHARS)
        ]
        for segment_index, content in enumerate(segments):
            payload = {
                "request_id": request.request_id,
                "ordinal": ordinal,
                "segment": segment_index,
                "role": message.role,
                "content": content,
                "timestamp": _iso(message.timestamp),
            }
            chunk_id = "raw_" + canonical_digest(payload)
            timestamp = _iso(message.timestamp)
            prefix = f"timestamp: {timestamp}\n" if timestamp else ""
            chunks.append(
                Chunk(
                    id=chunk_id,
                    source=source,
                    text=f"{prefix}role: {message.role}\ncontent: {content}",
                    metadata={
                        "record_type": "raw",
                        "kind": "raw",
                        "source_session_id": request.session_id,
                        "session_digest": session_digest(request.session_id),
                        "event_time": timestamp,
                        "embedding_profile": "voyage-4",
                        "retrieval_profile": "hosted-quality",
                        "ordinal": ordinal,
                        "segment": segment_index,
                        "segment_count": len(segments),
                        "file": f"{chunk_id}.md",
                    },
                )
            )
    for record in records[:8]:
        payload = record.model_dump(mode="json")
        chunk_id = "mem_" + canonical_digest(payload)
        chunks.append(
            Chunk(
                id=chunk_id,
                source=source,
                text=record.rendered(),
                metadata={
                    "record_type": "compiled",
                    "kind": record.kind,
                    "source_session_id": request.session_id,
                    "session_digest": session_digest(request.session_id),
                    "event_time": _iso(record.event_time),
                    "embedding_profile": "voyage-4",
                    "retrieval_profile": "hosted-quality",
                    "supersedes": list(record.supersedes),
                    "file": f"{chunk_id}.md",
                    "coding_record": payload,
                },
            )
        )
    return chunks


class HostedService:
    def __init__(
        self,
        repository: Repository,
        compiler: Compiler | None,
        retriever: HostedRetriever,
        *,
        context_chars: int = 7_000,
        model_clients_ready: bool = True,
        behavior: HostedVariant | None = None,
    ) -> None:
        self._repository = repository
        self._compiler = compiler
        self._retriever = retriever
        self._context_chars = context_chars
        self._model_clients_ready = model_clients_ready
        self._behavior = behavior or variant(DEFAULT_VARIANT)
        if (self._behavior.compiler or self._behavior.facets) and compiler is None:
            raise ValueError(f"{self._behavior.name} requires a compiler client")
        self._lock_guard = asyncio.Lock()
        self._add_locks: dict[tuple[str, str], _RequestLock] = {}

    async def _request_lock(
        self, tenant: str, request_id: str
    ) -> tuple[tuple[str, str], _RequestLock]:
        key = (tenant, canonical_digest(request_id))
        async with self._lock_guard:
            entry = self._add_locks.setdefault(key, _RequestLock(asyncio.Lock()))
            entry.users += 1
            return key, entry

    async def add(self, request: AddRequest) -> AddResponse:
        tenant = tenant_for(request.user_id)
        fingerprint = canonical_digest(request.model_dump(mode="json"))
        lock_key, entry = await self._request_lock(tenant, request.request_id)
        started = time.perf_counter()
        fallback = False
        try:
            async with entry.lock:
                receipt = await asyncio.to_thread(
                    self._repository.get_receipt,
                    tenant,
                    request.request_id,
                    fingerprint,
                )
                if receipt is not None:
                    return AddResponse.model_validate_json(receipt)
                records: list[CodingMemoryRecord] = []
                if self._behavior.compiler:
                    prior = await asyncio.to_thread(
                        self._repository.prior_records, tenant, _source(request.session_id)
                    )
                    try:
                        assert self._compiler is not None
                        records = await asyncio.to_thread(
                            self._compiler.compile,
                            request.messages,
                            request.session_id,
                            prior,
                        )
                        if not records:
                            raise ValueError("compiler returned no supported records")
                    except Exception:  # BROAD-CATCH: mandatory searchable fallback
                        fallback = True
                        records = deterministic_extract(request.messages, request.session_id)
                chunks = build_chunks(request, records)
                await asyncio.to_thread(self._repository.persist, tenant, chunks)
                response = AddResponse(
                    request_id=request.request_id,
                    user_id=request.user_id,
                    session_id=request.session_id,
                    raw_count=sum(
                        chunk.metadata.get("record_type") == "raw" for chunk in chunks
                    ),
                    compiled_count=len(records),
                    compiler_fallback=fallback,
                )
                encoded = response.model_dump_json()
                await asyncio.to_thread(
                    self._repository.record_receipt,
                    tenant,
                    request.request_id,
                    fingerprint,
                    encoded,
                )
                return response
        finally:
            elapsed = (time.perf_counter() - started) * 1_000
            log.info(
                "hosted_add_complete",
                extra={
                    "tenant_digest": tenant.removeprefix("aml_")[:16],
                    "request_digest": canonical_digest(request.request_id)[:16],
                    "message_count": len(request.messages),
                    "latency_ms": round(elapsed, 3),
                    "compiler_fallback": fallback,
                },
            )
            async with self._lock_guard:
                entry.users -= 1
                if entry.users == 0 and self._add_locks.get(lock_key) is entry:
                    self._add_locks.pop(lock_key)

    async def search(self, request: SearchRequest) -> SearchResponse:
        tenant = tenant_for(request.user_id)
        started = time.perf_counter()
        facet_fallback = False
        reranker_fallback = False
        try:
            facets: list[str] = []
            if self._behavior.facets:
                try:
                    assert self._compiler is not None
                    facets = await asyncio.to_thread(
                        self._compiler.facets,
                        request.query,
                        {"choices": request.options or []},
                    )
                except Exception:  # BROAD-CATCH: original query remains a complete fallback
                    facet_fallback = True
            store = self._repository.tenant_store(tenant)
            run = await asyncio.to_thread(
                self._retriever.search,
                store,
                request.query,
                facets,
                rerank=self._behavior.reranker,
            )
            reranker_fallback = run.reranker_fallback
            if self._behavior.pack:
                items = pack_evidence(
                    run.hits,
                    request.query,
                    top_k=request.top_k,
                    char_budget=self._behavior.context_chars or self._context_chars,
                    superseded_ids=run.superseded_ids,
                )
            else:
                items = render_full_evidence(
                    run.hits,
                    request.query,
                    top_k=request.top_k,
                    superseded_ids=run.superseded_ids,
                )
            return SearchResponse(data=items)

        finally:
            log.info(
                "hosted_search_complete",
                extra={
                    "tenant_digest": tenant.removeprefix("aml_")[:16],
                    "query_digest": canonical_digest(request.query)[:16],
                    "latency_ms": round((time.perf_counter() - started) * 1_000, 3),
                    "facet_fallback": facet_fallback,
                    "reranker_fallback": reranker_fallback,
                },
            )

    @property
    def variant_name(self) -> str:
        return self._behavior.name

    async def health(self) -> dict[str, object]:
        if not self._model_clients_ready:
            raise RuntimeError("mandatory model clients are not ready")
        return await asyncio.to_thread(self._repository.health)

    async def delete_user(self, user_id: str) -> int:
        return await asyncio.to_thread(self._repository.delete_tenant, tenant_for(user_id))
