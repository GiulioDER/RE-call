"""Hosted Add, Search, and deletion orchestration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import time
from typing import Any

from recall.entailment import (
    DEFAULT_QNLI_MODEL,
    DEFAULT_QNLI_REVISION,
    EntailmentJudge,
)
from recall.types import Chunk
from recall_aml.compiler import Compiler, QueryPlan, deterministic_extract
from recall_aml.config import (
    EMBEDDING_PROFILE,
    RERANK_MODEL,
    RERANK_PRICE_USD_PER_MILLION_TOKENS,
    RETRIEVAL_PROFILE,
)
from recall_aml.identity import canonical_digest, session_digest, tenant_for
from recall_aml.models import (
    AddRequest,
    AddResponse,
    CodingMemoryRecord,
    Message,
    SearchRequest,
    SearchResponse,
    TaskType,
)
from recall_aml.retrieval import HostedRetriever, pack_evidence, render_full_evidence
from recall_aml.storage import Repository
from recall_aml.variants import DEFAULT_VARIANT, HostedVariant, variant


log = logging.getLogger("recall_aml")
RAW_SEGMENT_CHARS = 4_500
POSTGRES_NUL_REPLACEMENT = "\u2400"


@dataclass
class _RequestLock:
    lock: asyncio.Lock
    users: int = 0


def _source(session_id: str) -> str:
    return "aml://session/" + session_digest(session_id)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()


def _normalize_messages(messages: list[Message]) -> tuple[list[Message], int]:
    """Replace PostgreSQL's unrepresentable NUL while preserving character offsets.

    JSON strings may legally contain U+0000, but PostgreSQL text values may not.  U+2400 is a
    single, visible code point, so every later character keeps the same ordinal and compiler
    evidence spans remain mechanically checkable against the normalized message.  The original
    request still owns idempotency fingerprinting; normalization is only the persisted view.
    """
    count = sum(message.content.count("\x00") for message in messages)
    if not count:
        return messages, 0
    return [
        message.model_copy(
            update={"content": message.content.replace("\x00", POSTGRES_NUL_REPLACEMENT)}
        )
        for message in messages
    ], count


def _replace_postgres_nul(value: Any) -> tuple[Any, int]:
    """Replace NUL recursively in a model payload headed for PostgreSQL."""
    if isinstance(value, str):
        count = value.count("\x00")
        return value.replace("\x00", POSTGRES_NUL_REPLACEMENT), count
    if isinstance(value, list):
        normalized: list[Any] = []
        count = 0
        for item in value:
            normalized_item, item_count = _replace_postgres_nul(item)
            normalized.append(normalized_item)
            count += item_count
        return normalized, count
    if isinstance(value, dict):
        normalized_dict: dict[Any, Any] = {}
        count = 0
        for key, item in value.items():
            normalized_item, item_count = _replace_postgres_nul(item)
            normalized_dict[key] = normalized_item
            count += item_count
        return normalized_dict, count
    return value, 0


def _normalize_records(
    records: list[CodingMemoryRecord],
) -> tuple[list[CodingMemoryRecord], int]:
    """Make accepted provider output safe for both rendered text and JSONB metadata."""
    normalized: list[CodingMemoryRecord] = []
    count = 0
    for record in records:
        payload, record_count = _replace_postgres_nul(record.model_dump(mode="python"))
        normalized.append(CodingMemoryRecord.model_validate(payload))
        count += record_count
    return normalized, count


def build_chunks(
    request: AddRequest,
    records: list[CodingMemoryRecord],
    *,
    include_raw: bool = True,
    source_nul_replacements: int = 0,
) -> list[Chunk]:
    source = _source(request.session_id)
    chunks: list[Chunk] = []
    for ordinal, message in enumerate(request.messages):
        starts = list(range(0, len(message.content), RAW_SEGMENT_CHARS))
        for segment_index, char_start in enumerate(starts):
            char_end = min(char_start + RAW_SEGMENT_CHARS, len(message.content))
            content = message.content[char_start:char_end]
            payload = {
                "request_id": request.request_id,
                "ordinal": ordinal,
                "segment": segment_index,
                "char_start": char_start,
                "char_end": char_end,
                "role": message.role,
                "content": content,
                "timestamp": _iso(message.timestamp),
            }
            chunk_id = "raw_" + canonical_digest(payload)
            timestamp = _iso(message.timestamp)
            prefix = f"timestamp: {timestamp}\n" if timestamp else ""
            if include_raw:
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
                            "embedding_profile": EMBEDDING_PROFILE,
                            "retrieval_profile": RETRIEVAL_PROFILE,
                            "ordinal": ordinal,
                            "segment": segment_index,
                            "segment_count": len(starts),
                            "char_start": char_start,
                            "char_end": char_end,
                            "source_nul_replacements": source_nul_replacements,
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
                    "embedding_profile": EMBEDDING_PROFILE,
                    "retrieval_profile": RETRIEVAL_PROFILE,
                    "supersedes": list(record.supersedes),
                    "evidence_spans": [
                        span.model_dump(mode="json") for span in record.evidence_spans
                    ],
                    "source_nul_replacements": source_nul_replacements,
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
        entailment_judge: EntailmentJudge | None = None,
    ) -> None:
        self._repository = repository
        self._compiler = compiler
        self._retriever = retriever
        self._context_chars = context_chars
        self._model_clients_ready = model_clients_ready
        self._behavior = behavior or variant(DEFAULT_VARIANT)
        self._entailment_judge = entailment_judge
        self._reranker_model = self._behavior.reranker_model or RERANK_MODEL.split(":", 1)[1]
        if (self._behavior.compiler or self._behavior.facets) and compiler is None:
            raise ValueError(f"{self._behavior.name} requires a compiler client")
        if self._behavior.entailment and entailment_judge is None:
            raise ValueError(f"{self._behavior.name} requires an entailment judge")
        self._lock_guard = asyncio.Lock()
        self._add_locks: dict[tuple[str, str], _RequestLock] = {}
        self._corpus_status_cache: dict[str, dict[str, object]] = {}

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
                handle = await asyncio.to_thread(
                    self._repository.acquire_request_lock, tenant, request.request_id
                )
                try:
                    response = await self._add_once(request, tenant, fingerprint)
                    fallback = response.compiler_fallback
                    return response
                finally:
                    await asyncio.to_thread(self._repository.release_request_lock, handle)
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

    async def _add_once(self, request: AddRequest, tenant: str, fingerprint: str) -> AddResponse:
        receipt = await asyncio.to_thread(
            self._repository.get_receipt,
            tenant,
            request.request_id,
            fingerprint,
        )
        if receipt is not None:
            return AddResponse.model_validate_json(receipt)
        normalized_messages, nul_replacements = _normalize_messages(request.messages)
        normalized_request = request.model_copy(update={"messages": normalized_messages})
        if nul_replacements:
            log.info(
                "hosted_add_normalized_nul count=%d session_digest=%s",
                nul_replacements,
                session_digest(request.session_id)[:16],
            )
        fallback = False
        records: list[CodingMemoryRecord] = []
        if self._behavior.compiler:
            prior = await asyncio.to_thread(
                self._repository.prior_records, tenant, _source(request.session_id)
            )
            try:
                assert self._compiler is not None
                records = await asyncio.to_thread(
                    self._compiler.compile,
                    normalized_messages,
                    request.session_id,
                    prior,
                )
                if not records:
                    raise ValueError("compiler returned no supported records")
            except Exception:  # BROAD-CATCH: mandatory searchable fallback
                fallback = True
                records = deterministic_extract(normalized_messages, request.session_id)
        records, compiler_nul_replacements = _normalize_records(records)
        if compiler_nul_replacements:
            log.info(
                "hosted_add_normalized_compiler_nul count=%d session_digest=%s",
                compiler_nul_replacements,
                session_digest(request.session_id)[:16],
            )
        chunks = build_chunks(
            normalized_request,
            records,
            include_raw=self._behavior.raw,
            source_nul_replacements=nul_replacements,
        )
        await asyncio.to_thread(self._repository.persist, tenant, chunks)
        self._corpus_status_cache.pop(tenant, None)
        response = AddResponse(
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
            raw_count=sum(chunk.metadata.get("record_type") == "raw" for chunk in chunks),
            compiled_count=len(records),
            compiler_fallback=fallback,
        )
        await asyncio.to_thread(
            self._repository.record_receipt,
            tenant,
            request.request_id,
            fingerprint,
            response.model_dump_json(),
        )
        return response

    async def search(self, request: SearchRequest) -> SearchResponse:
        tenant = tenant_for(request.user_id)
        started = time.perf_counter()
        facet_fallback = False
        reranker_fallback = False
        entailment_attempted = False
        entailment_completed = False
        entailment_input_count = 0
        entailment_output_count = 0
        entailment_accepted_count = 0
        entailment_ms = 0.0
        run = None
        try:
            facets: list[str] = []
            task_type: TaskType = "unknown"
            if self._behavior.facets:
                try:
                    assert self._compiler is not None
                    options = {"choices": request.options or []}
                    if self._behavior.task_conditioned:
                        plan = await asyncio.to_thread(self._compiler.plan, request.query, options)
                        if not isinstance(plan, QueryPlan):
                            raise TypeError("query planner returned an invalid plan")
                        facets = plan.facets
                        task_type = plan.task_type
                    else:
                        facets = await asyncio.to_thread(
                            self._compiler.facets, request.query, options
                        )
                except Exception:  # BROAD-CATCH: original query remains a complete fallback
                    facet_fallback = True
            store = self._repository.tenant_store(tenant)
            if self._behavior.learned_sparse:
                await asyncio.to_thread(self._repository.verify_sparse_coverage, tenant)
            run = await asyncio.to_thread(
                self._retriever.search,
                store,
                request.query,
                facets,
                rerank=self._behavior.reranker,
                learned_sparse=self._behavior.learned_sparse,
            )
            reranker_fallback = run.reranker_fallback
            hits = run.hits
            if self._behavior.entailment:
                entailment_attempted = True
                entailment_input_count = len(hits)
                entailment_started = time.perf_counter()
                assert self._entailment_judge is not None
                decisions = await asyncio.to_thread(
                    self._entailment_judge.judge,
                    request.query,
                    [hit.chunk.text for hit in hits],
                )
                entailment_ms = (time.perf_counter() - entailment_started) * 1_000
                entailment_output_count = len(decisions)
                if entailment_output_count != entailment_input_count:
                    raise RuntimeError(
                        "entailment judge returned "
                        f"{entailment_output_count} decisions for "
                        f"{entailment_input_count} candidates"
                    )
                hits = [hit for hit, accepted in zip(hits, decisions) if accepted]
                entailment_accepted_count = len(hits)
                entailment_completed = True
            corpus = await self._corpus_status(tenant)
            if self._behavior.pack:
                items = pack_evidence(
                    hits,
                    request.query,
                    top_k=request.top_k,
                    char_budget=self._behavior.context_chars or self._context_chars,
                    superseded_ids=run.superseded_ids,
                    task_type=task_type,
                )
            else:
                items = render_full_evidence(
                    hits,
                    request.query,
                    top_k=request.top_k,
                    superseded_ids=run.superseded_ids,
                )
            return SearchResponse(
                data=items,
                facet_fallback=facet_fallback,
                reranker_fallback=reranker_fallback,
                task_type=task_type,
                reranker_attempted=run.reranker_attempted,
                reranker_completed=run.reranker_completed,
                reranker_provider="voyage" if run.reranker_attempted else "none",
                reranker_model=(
                    self._reranker_model if run.reranker_attempted else "none"
                ),
                candidate_input_count=run.candidate_input_count,
                candidate_output_count=run.candidate_output_count,
                candidate_permutation_valid=run.candidate_permutation_valid,
                top_10_order_changed=run.top_10_order_changed,
                top_10_membership_changed=run.top_10_membership_changed,
                top_100_order_changed=run.top_100_order_changed,
                top_100_membership_changed=run.top_100_membership_changed,
                candidate_character_count=run.candidate_character_count,
                rerank_ms=run.rerank_ms,
                estimated_reranker_cost_usd=(
                    (
                        run.candidate_character_count
                        + run.query_character_count * run.candidate_input_count
                    )
                    / 4
                    / 1_000_000
                    * RERANK_PRICE_USD_PER_MILLION_TOKENS
                    if run.reranker_attempted
                    else 0.0
                ),
                entailment_attempted=entailment_attempted,
                entailment_completed=entailment_completed,
                entailment_provider="sentence-transformers" if entailment_attempted else "none",
                entailment_model=DEFAULT_QNLI_MODEL if entailment_attempted else "none",
                entailment_revision=DEFAULT_QNLI_REVISION if entailment_attempted else "none",
                entailment_threshold=0.5 if entailment_attempted else 0.0,
                entailment_input_count=entailment_input_count,
                entailment_output_count=entailment_output_count,
                entailment_accepted_count=entailment_accepted_count,
                entailment_rejected_count=(
                    entailment_input_count - entailment_accepted_count
                ),
                entailment_ms=entailment_ms,
                generation_id=str(corpus["generation_id"]),
                corpus_sha256=str(corpus["corpus_sha256"]),
            )

        finally:
            log.info(
                "hosted_search_complete",
                extra={
                    "tenant_digest": tenant.removeprefix("aml_")[:16],
                    "query_digest": canonical_digest(request.query)[:16],
                    "latency_ms": round((time.perf_counter() - started) * 1_000, 3),
                    "facet_fallback": facet_fallback,
                    "reranker_fallback": reranker_fallback,
                    "task_type": task_type,
                    "reranker_attempted": bool(run and run.reranker_attempted),
                    "reranker_completed": bool(run and run.reranker_completed),
                    "candidate_input_count": run.candidate_input_count if run else 0,
                    "candidate_output_count": run.candidate_output_count if run else 0,
                    "candidate_permutation_valid": (
                        run.candidate_permutation_valid if run else False
                    ),
                    "candidate_character_count": run.candidate_character_count if run else 0,
                    "rerank_ms": round(run.rerank_ms, 3) if run else 0.0,
                    "entailment_attempted": entailment_attempted,
                    "entailment_completed": entailment_completed,
                    "entailment_input_count": entailment_input_count,
                    "entailment_output_count": entailment_output_count,
                    "entailment_accepted_count": entailment_accepted_count,
                    "entailment_rejected_count": (
                        entailment_input_count - entailment_accepted_count
                    ),
                    "entailment_ms": round(entailment_ms, 3),
                },
            )

    @property
    def variant_name(self) -> str:
        return self._behavior.name

    @property
    def reranker_model(self) -> str:
        return self._reranker_model

    @property
    def entailment_enabled(self) -> bool:
        return self._behavior.entailment

    async def health(self) -> dict[str, object]:
        if not self._model_clients_ready:
            raise RuntimeError("mandatory model clients are not ready")
        return await asyncio.to_thread(self._repository.health)

    async def delete_user(self, user_id: str) -> int:
        tenant = tenant_for(user_id)
        deleted = await asyncio.to_thread(self._repository.delete_tenant, tenant)
        self._corpus_status_cache.pop(tenant, None)
        return deleted

    async def _corpus_status(self, tenant: str) -> dict[str, object]:
        cached = self._corpus_status_cache.get(tenant)
        if cached is None:
            cached = await asyncio.to_thread(self._repository.corpus_status, tenant)
            self._corpus_status_cache[tenant] = cached
        return dict(cached)

    async def corpus_status(self, user_id: str) -> dict[str, object]:
        return await self._corpus_status(tenant_for(user_id))

    async def prepare_sparse_user(self, user_id: str) -> dict[str, object]:
        if not self._behavior.learned_sparse:
            raise ValueError(f"{self._behavior.name} has no learned sparse stage")
        return await asyncio.to_thread(self._repository.backfill_sparse, tenant_for(user_id))
