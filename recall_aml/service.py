"""Hosted Add, Search, and deletion orchestration."""

from __future__ import annotations

import asyncio
from bisect import bisect_left, bisect_right
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import logging
import time
from typing import Any

from recall.types import Chunk
from recall_aml.atomic_views import (
    ATOMIC_VIEW_PROFILE,
    BuildRefusal,
    build_view_chunks,
    view_query_width,
)
from recall_aml.code4 import BM25_PROFILE, word_windows
from recall_aml.compiler import (
    COMPILE_REQUEST_DIGEST,
    Compiler,
    QueryPlan,
    deterministic_extract,
)
from recall_aml.config import (
    EMBEDDING_PROFILE,
    RERANK_MODEL,
    RERANK_PRICE_USD_PER_MILLION_TOKENS,
    RETRIEVAL_PROFILE,
)
from recall_aml.identity import (
    canonical_digest,
    graph_tenant,
    session_digest,
    specialist_tenant,
    tenant_for,
)
from recall_aml.graph import attach_grounded_relations
from recall_aml.models import (
    AddRequest,
    AddResponse,
    CodingMemoryRecord,
    Message,
    SearchRequest,
    SearchResponse,
    TaskType,
)
from recall_aml.multimodal import (
    MULTIMODAL_EMBEDDING_MODEL,
    MULTIMODAL_EMBEDDING_PROFILE,
    MultimodalEmbedder,
    content_text,
    fuse_hits,
    is_multimodal,
    prepare_messages,
    render_preserved,
)
from recall_aml.retrieval import (
    AtomicRescueBinding,
    HostedRetriever,
    pack_evidence,
    render_full_evidence,
    render_multiview_evidence,
)
from recall_aml.storage import Repository
from recall_aml.specialists import (
    SPECIALIST_FUSION_PROFILE,
    SPECIALIST_ROUTER_PROFILE,
    route_query,
)
from recall_aml.variants import DEFAULT_VARIANT, HostedVariant, variant
from recall_aml.window_format import dated_items, looks_like_coding


log = logging.getLogger("recall_aml")
RAW_SEGMENT_CHARS = 4_500
POSTGRES_NUL_REPLACEMENT = "\u2400"
TENANT_MUTATION_LOCK_REQUEST_ID = "__hosted_tenant_mutation__"
MAX_CORPUS_STATUS_CACHE_ENTRIES = 1_024


@dataclass
class _RequestLock:
    lock: asyncio.Lock
    users: int = 0


@dataclass
class _EmbeddingPrefetch:
    """Vectors an Add computed beside its compile, for rows whose text the compile cannot change.

    ``raw_chunks`` are the request's raw windows exactly as ``build_chunks`` makes them without
    records, and each vector list is the answer to the very request the matching persist call
    would have sent later (same embedder, same texts, same order), so a row stores the same vector
    either way. A field left None was not computed or failed, and its persist call embeds inline
    as before, so a failed prefetch costs one wasted request and changes nothing else.
    """

    raw_chunks: list[Chunk]
    raw_vectors: list[list[float]] | None = None
    #: The views ``build_view_chunks`` made of ``raw_chunks``, or the refusal it raised; None when
    #: the variant builds no views or the builder failed some other way.
    views: list[Chunk] | BuildRefusal | None = None
    primary_view_vectors: list[list[float]] | None = None
    specialist_view_vectors: list[list[float]] | None = None


def _source(session_id: str) -> str:
    return "aml://session/" + session_digest(session_id)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()


def _utc_datetime(value: datetime) -> datetime:
    """Normalize aware and naive in process timestamps to one comparison zone."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _status_int(status: dict[str, object], key: str) -> int:
    value = status.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"corpus status {key} is not an integer")
    return value


def _normalize_messages(messages: list[Message]) -> tuple[list[Message], int]:
    """Replace PostgreSQL's unrepresentable NUL while preserving character offsets.

    JSON strings may legally contain U+0000, but PostgreSQL text values may not.  U+2400 is a
    single, visible code point, so every later character keeps the same ordinal and compiler
    evidence spans remain mechanically checkable against the normalized message.  The original
    request still owns idempotency fingerprinting; normalization is only the persisted view.
    """
    normalized: list[Message] = []
    count = 0
    for message in messages:
        payload, message_count = _replace_postgres_nul(message.model_dump(mode="python"))
        normalized.append(Message.model_validate(payload))
        count += message_count
    return (normalized, count) if count else (messages, 0)


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
    compiler_profile: str = "offset-v1",
    compiler_fallback: bool = False,
    embedding_profile: str = EMBEDDING_PROFILE,
    word_window_size: int | None = None,
    word_window_stride: int | None = None,
    content_only_windows: bool = False,
    stable_window_identity: bool = False,
    per_track_windows: bool = False,
) -> list[Chunk]:
    source = _source(request.session_id)
    chunks: list[Chunk] = []
    if per_track_windows:
        content_only_windows = looks_like_coding(request.messages)
    if word_window_size is not None:
        stride = word_window_stride or word_window_size
        rendered_messages = []
        message_word_ranges: list[tuple[int, int, int]] = []
        word_cursor = 0
        for ordinal, message in enumerate(request.messages):
            if not isinstance(message.content, str):
                raise TypeError("build_chunks requires text message content")
            if content_only_windows:
                rendered_messages.append(message.content)
            else:
                timestamp = _iso(message.timestamp)
                prefix = f"timestamp: {timestamp}\n" if timestamp else ""
                rendered_messages.append(
                    f"{prefix}role: {message.role}\ncontent: {message.content}"
                )
            word_count = len(rendered_messages[-1].split())
            message_word_ranges.append((ordinal, word_cursor, word_cursor + word_count))
            word_cursor += word_count
        session_text = (" " if content_only_windows else "\n").join(rendered_messages)
        windows = word_windows(session_text, size=word_window_size, stride=stride)
        event_times = [
            _utc_datetime(message.timestamp)
            for message in request.messages
            if message.timestamp is not None
        ]
        event_time = _iso(max(event_times)) if event_times else None
        # Message ranges are laid end to end, so their starts and ends are both nondecreasing.
        # Bisecting to the candidates keeps this linear in windows; scanning every message per
        # window was quadratic once an Add could carry thousands of messages.
        range_starts = [message_start for _, message_start, _ in message_word_ranges]
        range_ends = [message_end for _, _, message_end in message_word_ranges]
        for segment_index, content in enumerate(windows):
            word_start = segment_index * stride
            word_end = word_start + len(content.split())
            first = bisect_right(range_ends, word_start)
            stop = bisect_left(range_starts, word_end)
            message_ordinals = [
                ordinal
                for ordinal, message_start, message_end in message_word_ranges[first:stop]
                if message_start < word_end and message_end > word_start
            ]
            payload = (
                {
                    "source_session_id": request.session_id,
                    "segment": segment_index,
                    "content": content,
                }
                if stable_window_identity
                else {
                    "request_id": request.request_id,
                    "ordinal": 0,
                    "segment": segment_index,
                    "word_start": word_start,
                    "word_end": word_end,
                    "content": content,
                    "event_time": event_time,
                }
            )
            chunk_id = "raw_" + canonical_digest(payload)
            if include_raw:
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        source=source,
                        text=content,
                        metadata={
                            "record_type": "raw",
                            "kind": "raw",
                            "source_session_id": request.session_id,
                            "session_digest": session_digest(request.session_id),
                            "event_time": event_time,
                            "embedding_profile": embedding_profile,
                            "retrieval_profile": RETRIEVAL_PROFILE,
                            "ordinal": 0,
                            "segment": segment_index,
                            "segment_count": len(windows),
                            "word_start": word_start,
                            "word_end": word_end,
                            "message_ordinals": message_ordinals,
                            "word_window_size": word_window_size,
                            "word_window_stride": stride,
                            "lexical_profile": BM25_PROFILE,
                            "source_nul_replacements": source_nul_replacements,
                            "file": f"{chunk_id}.md",
                        },
                    )
                )
    else:
        for ordinal, message in enumerate(request.messages):
            if not isinstance(message.content, str):
                raise TypeError("build_chunks requires text message content")
            message_content = message.content
            starts = list(range(0, len(message_content), RAW_SEGMENT_CHARS))
            for segment_index, char_start in enumerate(starts):
                char_end = min(char_start + RAW_SEGMENT_CHARS, len(message_content))
                content = message_content[char_start:char_end]
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
                                "embedding_profile": embedding_profile,
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
    compiled_ids: set[str] = set()
    for record in records[:8]:
        payload = record.model_dump(mode="json")
        chunk_id = "mem_" + canonical_digest(payload)
        if chunk_id in compiled_ids:
            continue
        compiled_ids.add(chunk_id)
        chunks.append(
            Chunk(
                id=chunk_id,
                source=source,
                text=record.rendered(),
                metadata={
                    "record_type": "compiled",
                    "compiler_profile": compiler_profile,
                    "compiler_fallback": compiler_fallback,
                    "kind": record.kind,
                    "source_session_id": request.session_id,
                    "session_digest": session_digest(request.session_id),
                    "event_time": _iso(record.event_time),
                    "embedding_profile": embedding_profile,
                    "retrieval_profile": RETRIEVAL_PROFILE,
                    "supersedes": list(record.supersedes),
                    "evidence_spans": [
                        span.model_dump(mode="json") for span in record.evidence_spans
                    ],
                    "source_nul_replacements": source_nul_replacements,
                    "file": f"{chunk_id}.md",
                    "coding_record": payload,
                    "entities": list(record.entities),
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
        multimodal_embedder: MultimodalEmbedder | None = None,
        specialist_retrievers: dict[str, HostedRetriever] | None = None,
    ) -> None:
        self._repository = repository
        self._compiler = compiler
        self._retriever = retriever
        self._context_chars = context_chars
        self._model_clients_ready = model_clients_ready
        self._behavior = behavior or variant(DEFAULT_VARIANT)
        self._multimodal_embedder = multimodal_embedder
        self._specialist_retrievers = dict(specialist_retrievers or {})
        if (self._behavior.compiler or self._behavior.facets) and compiler is None:
            raise ValueError(f"{self._behavior.name} requires a compiler client")
        if self._behavior.multimodal_native and multimodal_embedder is None:
            raise ValueError(f"{self._behavior.name} requires a multimodal embedder")
        if (
            self._behavior.context_specialist
            and self._behavior.context_embedding_profile not in self._specialist_retrievers
        ):
            raise ValueError(f"{self._behavior.name} requires a Context specialist retriever")
        self._lock_guard = asyncio.Lock()
        self._add_locks: dict[tuple[str, str], _RequestLock] = {}
        self._corpus_status_cache: dict[str, dict[str, object]] = {}
        #: One in-flight status computation per tenant, shared by every Search that misses the
        #: cache while it runs. Invalidation removes the entry, so a Search that starts after a
        #: write never joins a computation that may have read the rows before it.
        self._corpus_status_flights: dict[str, asyncio.Future[dict[str, object]]] = {}
        local_locks = getattr(repository, "_hosted_async_tenant_locks", None)
        if local_locks is None:
            local_locks = {}
            try:
                setattr(repository, "_hosted_async_tenant_locks", local_locks)
            except (AttributeError, TypeError):
                # Immutable test doubles still need to support constructor-only
                # and failure-path tests, even though they cannot share locks.
                pass
        self._local_tenant_locks: dict[str, asyncio.Lock] = local_locks

    async def _request_lock(
        self, tenant: str, request_id: str
    ) -> tuple[tuple[str, str], _RequestLock]:
        key = (tenant, canonical_digest(request_id))
        async with self._lock_guard:
            entry = self._add_locks.setdefault(key, _RequestLock(asyncio.Lock()))
            entry.users += 1
            return key, entry

    def _invalidate_corpus_status(self, tenant: str, *, deleted: bool = False) -> None:
        """Invalidate raw and configured specialist status for one logical tenant.

        Also forgets any status computation still in flight for those tenants (so it cannot cache
        what it read before this write), and tells every retriever's Search caches which physical
        tenants were written (``deleted``: removed outright).
        """
        scopes = [tenant]
        if self._behavior.context_specialist:
            scopes.append(specialist_tenant(tenant, self._behavior.context_embedding_profile))
        for scope in scopes:
            self._corpus_status_cache.pop(scope, None)
            self._corpus_status_flights.pop(scope, None)
        physical = {
            tenant,
            graph_tenant(tenant),
            *(specialist_tenant(tenant, profile) for profile in self._specialist_retrievers),
            *scopes,
        }
        for retriever in (self._retriever, *self._specialist_retrievers.values()):
            invalidate = getattr(retriever, "invalidate_tenants", None)
            if callable(invalidate):
                invalidate(physical, drop=deleted)

    async def add(self, request: AddRequest) -> AddResponse:
        tenant = tenant_for(request.user_id)
        fingerprint = canonical_digest(request.model_dump(mode="json"))
        distributed = getattr(self._repository, "distributed_locks", False)
        tenant_handle = (
            await asyncio.to_thread(
                self._repository.acquire_request_lock, tenant, TENANT_MUTATION_LOCK_REQUEST_ID
            )
            if distributed
            else None
        )
        local_lock = None
        if not distributed:
            local_lock = self._local_tenant_locks.setdefault(tenant, asyncio.Lock())
            await local_lock.acquire()
        try:
            lock_key, entry = await self._request_lock(tenant, request.request_id)
            started = time.perf_counter()
            fallback = False
            try:
                async with entry.lock:
                    # The tenant-wide advisory lock is the cross-process mutation boundary.
                    # Acquiring a second blocking database lock here can exhaust the executor
                    # when many duplicate requests wait on the same tenant lock.
                    response = await self._add_once(request, tenant, fingerprint)
                    fallback = response.compiler_fallback
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
        finally:
            if local_lock is not None:
                local_lock.release()
            if tenant_handle is not None:
                await asyncio.to_thread(self._repository.release_request_lock, tenant_handle)

    async def _add_once(self, request: AddRequest, tenant: str, fingerprint: str) -> AddResponse:
        receipt = await asyncio.to_thread(
            self._repository.get_receipt,
            tenant,
            request.request_id,
            fingerprint,
        )
        if receipt is not None:
            return AddResponse.model_validate_json(receipt)
        if not request.messages:
            # Every message was blank and was dropped at validation. There is nothing to store,
            # and refusing would fail the sample permanently, so the Add is durable and empty.
            response = AddResponse(
                request_id=request.request_id,
                user_id=request.user_id,
                session_id=request.session_id,
                raw_count=0,
                compiled_count=0,
                compiler_fallback=False,
            )
            await asyncio.to_thread(
                self._repository.record_receipt,
                tenant,
                request.request_id,
                fingerprint,
                response.model_dump_json(),
            )
            return response
        normalized_messages, nul_replacements = await asyncio.to_thread(
            _normalize_messages, request.messages
        )
        normalized_request = request.model_copy(update={"messages": normalized_messages})
        if nul_replacements:
            log.info(
                "hosted_add_normalized_nul count=%d session_digest=%s",
                nul_replacements,
                session_digest(request.session_id)[:16],
            )
        has_multimodal = any(is_multimodal(message.content) for message in normalized_messages)
        if has_multimodal or (
            self._behavior.multimodal_preserve
            and not self._behavior.context_specialist
        ):
            if self._behavior.multimodal_preserve:
                prepared = prepare_messages(
                    normalized_messages,
                    request_id=request.request_id,
                    session_id=request.session_id,
                    source=_source(request.session_id),
                    source_nul_replacements=nul_replacements,
                )
                vectors: list[list[float]] = []
                if self._behavior.multimodal_native:
                    assert self._multimodal_embedder is not None
                    vectors = await asyncio.to_thread(
                        self._multimodal_embedder.embed_documents,
                        prepared.voyage_inputs,
                    )
                bundle_writer = getattr(self._repository, "persist_multimodal_bundle", None)
                if bundle_writer is not None:
                    await asyncio.to_thread(
                        bundle_writer,
                        tenant,
                        prepared.primary_chunks,
                        self._behavior.context_embedding_profile
                        if self._behavior.context_specialist
                        else None,
                        prepared.media_chunks,
                        prepared.vector_chunks if self._behavior.multimodal_native else [],
                        vectors if self._behavior.multimodal_native else [],
                    )
                else:
                    await asyncio.to_thread(self._repository.persist, tenant, prepared.primary_chunks)
                    if self._behavior.context_specialist:
                        await asyncio.to_thread(
                            self._repository.persist_specialist,
                            tenant,
                            self._behavior.context_embedding_profile,
                            prepared.primary_chunks,
                        )
                    await asyncio.to_thread(self._repository.persist_media, tenant, prepared.media_chunks)
                    if self._behavior.multimodal_native:
                        await asyncio.to_thread(
                            self._repository.persist_multimodal,
                            tenant,
                            prepared.vector_chunks,
                            vectors,
                        )
                chunks = prepared.primary_chunks
            else:
                text_messages = [
                    message.model_copy(update={"content": content_text(message.content)})
                    for message in normalized_messages
                ]
                text_request = normalized_request.model_copy(update={"messages": text_messages})
                chunks = await asyncio.to_thread(
                    self._build_chunks, text_request, [], nul_replacements
                )
                await asyncio.to_thread(self._repository.persist, tenant, chunks)
            self._invalidate_corpus_status(tenant)
            response = AddResponse(
                request_id=request.request_id,
                user_id=request.user_id,
                session_id=request.session_id,
                raw_count=sum(
                    chunk.metadata.get("record_type") == "raw" for chunk in chunks
                ),
                compiled_count=0,
                compiler_fallback=False,
            )
            await asyncio.to_thread(
                self._repository.record_receipt,
                tenant,
                request.request_id,
                fingerprint,
                response.model_dump_json(),
            )
            return response
        prefetch_task = self._start_embedding_prefetch(normalized_request, nul_replacements)
        try:
            return await self._compile_and_persist(
                request,
                tenant,
                fingerprint,
                normalized_request,
                normalized_messages,
                nul_replacements,
                prefetch_task,
            )
        finally:
            if prefetch_task is not None:
                # It never raises; waiting keeps its thread's work inside this Add.
                await asyncio.wait({prefetch_task})

    async def _compile_and_persist(
        self,
        request: AddRequest,
        tenant: str,
        fingerprint: str,
        normalized_request: AddRequest,
        normalized_messages: list[Message],
        nul_replacements: int,
        prefetch_task: asyncio.Task[_EmbeddingPrefetch | None] | None,
    ) -> AddResponse:
        fallback = False
        records: list[CodingMemoryRecord] = []
        if self._behavior.compiler:
            prior = await asyncio.to_thread(
                self._repository.prior_records,
                tenant,
                _source(request.session_id),
                graph_sidecar=self._behavior.graph_sidecar,
            )
            try:
                assert self._compiler is not None
                compile_method = (
                    self._compiler.compile_anchored_v3
                    if self._behavior.anchor_compiler_version == 3
                    else self._compiler.compile_anchored
                    if self._behavior.anchor_compiler
                    else self._compiler.compile
                )
                digest_token = COMPILE_REQUEST_DIGEST.set(
                    canonical_digest(request.request_id)[:16]
                )
                try:
                    records = await asyncio.to_thread(
                        compile_method, normalized_messages, request.session_id, prior
                    )
                finally:
                    COMPILE_REQUEST_DIGEST.reset(digest_token)
                if not records:
                    raise ValueError("compiler returned no supported records")
            except Exception as exc:  # BROAD-CATCH: mandatory searchable fallback
                fallback = True
                # Why this Add kept no compiled record, joined to it by request_digest. The class
                # only: a message can carry provider text, and the journal carries no content.
                fallback_fields: dict[str, object] = {
                    "request_digest": canonical_digest(request.request_id)[:16],
                    "error_class": type(exc).__name__,
                }
                # The SDK raises one class, APIStatusError, for a 402 and several other
                # statuses, so the class alone cannot say the credit ran out.
                http_status = getattr(exc, "status_code", None)
                if isinstance(http_status, int) and not isinstance(http_status, bool):
                    fallback_fields["http_status"] = http_status
                log.info("hosted_compiler_fallback", extra=fallback_fields)
                # A variant that drops fallback records must not build them: the extractor can
                # raise on valid input (a first message whose leading 300 characters are all
                # whitespace fails `require_substance`), and that raise was a permanent 422 for
                # records this variant throws away anyway.
                records = (
                    []
                    if self._behavior.drop_compiler_fallback
                    else deterministic_extract(normalized_messages, request.session_id)
                )
        if fallback and self._behavior.drop_compiler_fallback:
            records = []
        if self._behavior.compiled_kinds is not None:
            records = [
                record for record in records if record.kind in self._behavior.compiled_kinds
            ]
        records, compiler_nul_replacements = _normalize_records(records)
        if compiler_nul_replacements:
            log.info(
                "hosted_add_normalized_compiler_nul count=%d session_digest=%s",
                compiler_nul_replacements,
                session_digest(request.session_id)[:16],
            )
        chunks = await asyncio.to_thread(
            self._build_chunks,
            normalized_request,
            records,
            nul_replacements,
            compiler_profile=(
                "deterministic-fallback"
                if fallback
                else f"anchor-v{self._behavior.anchor_compiler_version}"
                if self._behavior.anchor_compiler
                else "offset-v1"
            ),
            compiler_fallback=fallback,
        )
        prefetch = await self._prefetched(prefetch_task, chunks)
        if self._behavior.graph_sidecar:
            chunks = await asyncio.to_thread(attach_grounded_relations, normalized_request, chunks)
            raw_chunks = [
                chunk for chunk in chunks if chunk.metadata.get("record_type") == "raw"
            ]
            graph_chunks = [
                chunk for chunk in chunks if chunk.metadata.get("record_type") == "compiled"
            ]
            if prefetch is not None and prefetch.raw_vectors is not None:
                await asyncio.to_thread(
                    self._repository.persist, tenant, raw_chunks, prefetch.raw_vectors
                )
            else:
                await asyncio.to_thread(self._repository.persist, tenant, raw_chunks)
            await asyncio.to_thread(self._repository.persist_graph, tenant, graph_chunks)
        else:
            await asyncio.to_thread(self._repository.persist, tenant, chunks)
        if self._behavior.context_specialist:
            await asyncio.to_thread(
                self._repository.persist_specialist,
                tenant,
                self._behavior.context_embedding_profile,
                chunks,
            )
        if self._behavior.atomic_views_at_add:
            await self._persist_atomic_views(tenant, chunks, prefetch)
        self._invalidate_corpus_status(tenant)
        response = AddResponse(
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
            raw_count=sum(chunk.metadata.get("record_type") == "raw" for chunk in chunks),
            compiled_count=sum(
                chunk.metadata.get("record_type") == "compiled" for chunk in chunks
            ),
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

    def _build_chunks(
        self,
        request: AddRequest,
        records: list[CodingMemoryRecord],
        nul_replacements: int,
        *,
        compiler_profile: str = "offset-v1",
        compiler_fallback: bool = False,
    ) -> list[Chunk]:
        """``build_chunks`` with this variant's window settings; CPU work, run off the loop."""
        return build_chunks(
            request,
            records,
            include_raw=self._behavior.raw,
            source_nul_replacements=nul_replacements,
            compiler_profile=compiler_profile,
            compiler_fallback=compiler_fallback,
            embedding_profile=self._behavior.embedding_profile,
            word_window_size=self._behavior.word_window_size,
            word_window_stride=self._behavior.word_window_stride,
            content_only_windows=self._behavior.content_only_windows,
            stable_window_identity=self._behavior.stable_window_order,
            per_track_windows=self._behavior.per_track_windows,
        )

    def _start_embedding_prefetch(
        self, normalized_request: AddRequest, nul_replacements: int
    ) -> asyncio.Task[_EmbeddingPrefetch | None] | None:
        """Start embedding, beside the compile, the rows whose text the compile cannot change.

        After the compile an Add made five sequential embedding calls. Three of them depend on
        nothing the compile produces: the raw windows persisted alone next to the graph sidecar,
        and the atomic views in both scopes, which are built from the raw windows only. Their
        requests are sent while the compile runs instead of after it. The persist calls, their
        order and their rows are unchanged; only where the vectors come from moves.
        """
        embed_texts = getattr(self._repository, "embed_texts", None)
        wants_raw = self._behavior.graph_sidecar and self._behavior.raw
        wants_views = self._behavior.atomic_views_at_add
        if not callable(embed_texts) or not (wants_raw or wants_views):
            return None
        context_profile = (
            self._behavior.context_embedding_profile if self._behavior.context_specialist else None
        )

        def compute() -> _EmbeddingPrefetch | None:
            try:
                raw_chunks = [
                    chunk
                    for chunk in self._build_chunks(normalized_request, [], nul_replacements)
                    if chunk.metadata.get("record_type") == "raw"
                ]
            except Exception:  # BROAD-CATCH: the Add builds its own chunks and raises there
                return None
            prefetch = _EmbeddingPrefetch(raw_chunks=raw_chunks)
            try:
                if wants_raw and raw_chunks:
                    prefetch.raw_vectors = embed_texts(None, [c.text for c in raw_chunks])
                if not wants_views:
                    return prefetch
                try:
                    views = build_view_chunks(raw_chunks)
                except BuildRefusal as refusal:
                    prefetch.views = refusal
                    return prefetch
                prefetch.views = views
                if not views:
                    return prefetch
                texts = [view.text for view in views]
                prefetch.primary_view_vectors = embed_texts(None, texts)
                if context_profile is not None:
                    prefetch.specialist_view_vectors = embed_texts(context_profile, texts)
            except Exception as exc:  # BROAD-CATCH: the persist calls embed inline as before
                log.info(
                    "hosted_add_embedding_prefetch_failed",
                    extra={"error_class": type(exc).__name__},
                )
            return prefetch

        return asyncio.create_task(asyncio.to_thread(compute))

    @staticmethod
    async def _prefetched(
        task: asyncio.Task[_EmbeddingPrefetch | None] | None, chunks: list[Chunk]
    ) -> _EmbeddingPrefetch | None:
        """The prefetch, only when its raw windows are exactly the ones this Add stores."""
        if task is None:
            return None
        prefetch = await task
        if prefetch is None:
            return None
        raw_chunks = [chunk for chunk in chunks if chunk.metadata.get("record_type") == "raw"]
        if raw_chunks != prefetch.raw_chunks:
            log.warning("hosted_add_embedding_prefetch_mismatch")
            return None
        return prefetch

    async def _persist_atomic_views(
        self, tenant: str, chunks: list[Chunk], prefetch: _EmbeddingPrefetch | None = None
    ) -> None:
        """Persist this request's atomic views in every scope a Search can read, before 200.

        An embedding or database failure raises and fails the Add, exactly as a failed window or
        specialist write does: no receipt is recorded, so the platform's retry rebuilds the same
        rows by content. A ``BuildRefusal`` is different: it is deterministic, so every retry
        would fail identically and the request would sink the job. It is logged and the request
        is served without views, which only means the rescue cannot pick its windows.
        """
        built = prefetch.views if prefetch is not None else None
        try:
            if isinstance(built, BuildRefusal):
                raise built
            views = (
                built
                if built is not None
                else await asyncio.to_thread(
                    build_view_chunks,
                    [chunk for chunk in chunks if chunk.metadata.get("record_type") == "raw"],
                )
            )
        except BuildRefusal as refusal:
            log.warning(
                "hosted_add_atomic_views_refused reason=%s tenant_digest=%s",
                refusal,
                tenant.removeprefix("aml_")[:16],
            )
            return
        if not views:
            return
        primary = [
            replace(
                view,
                metadata={**view.metadata, "embedding_profile": self._behavior.embedding_profile},
            )
            for view in views
        ]
        primary_vectors = (
            prefetch.primary_view_vectors if prefetch is not None and built is not None else None
        )
        specialist_vectors = (
            prefetch.specialist_view_vectors
            if prefetch is not None and built is not None
            else None
        )
        if primary_vectors is not None:
            await asyncio.to_thread(
                self._repository.persist_atomic_views, tenant, None, primary, primary_vectors
            )
        else:
            await asyncio.to_thread(self._repository.persist_atomic_views, tenant, None, primary)
        if self._behavior.context_specialist:
            if specialist_vectors is not None:
                await asyncio.to_thread(
                    self._repository.persist_atomic_views,
                    tenant,
                    self._behavior.context_embedding_profile,
                    views,
                    specialist_vectors,
                )
            else:
                await asyncio.to_thread(
                    self._repository.persist_atomic_views,
                    tenant,
                    self._behavior.context_embedding_profile,
                    views,
                )

    async def search(self, request: SearchRequest) -> SearchResponse:
        tenant = tenant_for(request.user_id)
        query_text = content_text(request.query)
        started = time.perf_counter()
        facet_fallback = False
        reranker_fallback = False
        run = None
        specialist_route = route_query(request.query)
        specialist_profile = self._behavior.embedding_profile
        try:
            facets: list[str] = []
            task_type: TaskType = "unknown"
            if self._behavior.facets:
                try:
                    assert self._compiler is not None
                    options = {"choices": request.options or []}
                    if self._behavior.task_conditioned:
                        plan = await asyncio.to_thread(self._compiler.plan, query_text, options)
                        if not isinstance(plan, QueryPlan):
                            raise TypeError("query planner returned an invalid plan")
                        facets = plan.facets
                        task_type = plan.task_type
                    else:
                        facets = await asyncio.to_thread(
                            self._compiler.facets, query_text, options
                        )
                except Exception:  # BROAD-CATCH: original query remains a complete fallback
                    facet_fallback = True
            corpus = await self._corpus_status(tenant)
            corpus_scope = tenant
            store = self._repository.tenant_store(tenant)
            retriever = self._retriever
            if specialist_route == "context" and self._behavior.context_specialist:
                specialist_profile = self._behavior.context_embedding_profile
                store = self._repository.specialist_store(tenant, specialist_profile)
                retriever = self._specialist_retrievers[specialist_profile]
                corpus = await self._corpus_status(
                    specialist_tenant(tenant, specialist_profile)
                )
                corpus_scope = specialist_tenant(tenant, specialist_profile)
            if self._behavior.learned_sparse:
                await asyncio.to_thread(self._repository.verify_sparse_coverage, tenant)
            run = await asyncio.to_thread(
                retriever.search,
                store,
                query_text,
                facets,
                rerank=self._behavior.reranker,
                learned_sparse=self._behavior.learned_sparse,
                code_aware=self._behavior.code_aware,
                canonical_bm25=self._behavior.canonical_bm25,
                exact_dense=self._behavior.exact_dense,
                # Code4's frozen tie-breaker is defined only for its raw, word-windowed
                # corpus. Context4 stores the same logical memories but also admits
                # compiler records, which deliberately have no Code4 window segment.
                # Applying the Code4 order there turns a valid conversational Search
                # into a public 422 instead of preserving the Context4 retrieval path.
                stable_window_order=(
                    self._behavior.stable_window_order and specialist_route == "code"
                ),
                atomic_rescue=self._atomic_rescue_binding(corpus, corpus_scope),
            )
            if self._behavior.graph_sidecar:
                try:
                    # The sidecar embeds with the primary retriever. When that is also the
                    # retriever this Search used (the Code route), its query vector is the one the
                    # sidecar would compute, so it is passed rather than embedded again; the
                    # Context route embedded with another model, and the sidecar embeds its own.
                    run = await asyncio.to_thread(
                        self._retriever.apply_graph_sidecar,
                        self._repository.graph_store(tenant),
                        query_text,
                        run,
                        query_vector=run.query_vector if retriever is self._retriever else None,
                    )
                except Exception:  # BROAD-CATCH: byte-identical raw ranking is mandatory fallback
                    run = self._retriever.graph_fallback(run)
            reranker_fallback = run.reranker_fallback
            if self._behavior.multimodal_native and (
                not self._behavior.context_specialist
                or specialist_route == "multimodal"
            ):
                assert self._multimodal_embedder is not None
                visual_vector = await asyncio.to_thread(
                    self._multimodal_embedder.embed_query, request.query
                )
                visual_hits = await asyncio.to_thread(
                    self._repository.multimodal_store(tenant).query_dense,
                    visual_vector,
                    100,
                )
                run.hits[:] = fuse_hits(run.hits, visual_hits)
            if self._behavior.multimodal_preserve and (
                not self._behavior.context_specialist
                or specialist_route == "multimodal"
            ):
                parent_ids = list(
                    dict.fromkeys(
                        str(hit.chunk.metadata.get("multimodal_parent_id", hit.chunk.id))
                        for hit in run.hits
                    )
                )
                primary_by_id = await asyncio.to_thread(store.chunks_by_ids, parent_ids)
                media_ids: list[str] = []
                for parent in primary_by_id.values():
                    manifest = parent.metadata.get("multimodal_manifest", [])
                    if not isinstance(manifest, list):
                        continue
                    media_ids.extend(
                        str(entry["media_id"])
                        for entry in manifest
                        if isinstance(entry, dict)
                        and entry.get("type") == "image_ref"
                        and isinstance(entry.get("media_id"), str)
                    )
                media_by_id = await asyncio.to_thread(
                    self._repository.media_store(tenant).chunks_by_ids,
                    list(dict.fromkeys(media_ids)),
                )
                items = render_preserved(
                    run.hits,
                    primary_by_id=primary_by_id,
                    media_by_id=media_by_id,
                    top_k=request.top_k,
                )
            elif self._behavior.pack:
                items = pack_evidence(
                    run.hits,
                    query_text,
                    top_k=request.top_k,
                    char_budget=self._behavior.context_chars or self._context_chars,
                    superseded_ids=run.superseded_ids,
                    task_type=task_type,
                )
            elif self._behavior.raw_rescue_tail:
                items = render_multiview_evidence(
                    run.hits,
                    query_text,
                    top_k=request.top_k,
                    superseded_ids=run.superseded_ids,
                )
            else:
                items = render_full_evidence(
                    run.hits,
                    query_text,
                    top_k=request.top_k,
                    superseded_ids=run.superseded_ids,
                )
            if self._behavior.dated_search_content:
                items = dated_items(items)
            return SearchResponse(
                data=items,
                facet_fallback=facet_fallback,
                reranker_fallback=reranker_fallback,
                task_type=task_type,
                specialist_route=specialist_route,
                specialist_embedding_profile=(
                    MULTIMODAL_EMBEDDING_PROFILE
                    if specialist_route == "multimodal" and self._behavior.multimodal_native
                    else specialist_profile
                ),
                reranker_attempted=run.reranker_attempted,
                reranker_completed=run.reranker_completed,
                reranker_provider="voyage" if run.reranker_attempted else "none",
                reranker_model=(
                    RERANK_MODEL.split(":", 1)[1] if run.reranker_attempted else "none"
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
                generation_id=str(corpus["generation_id"]),
                corpus_sha256=str(corpus["corpus_sha256"]),
                code_aware_attempted=run.code_aware_attempted,
                code_aware_fallback=run.code_aware_fallback,
                code_profile=run.code_profile,
                code_rrf_weight=run.code_rrf_weight,
                code_query_token_count=run.code_query_token_count,
                code_match_candidate_count=run.code_match_candidate_count,
                code_top_10_order_changed=run.code_top_10_order_changed,
                code_top_10_membership_changed=run.code_top_10_membership_changed,
                code_top_100_order_changed=run.code_top_100_order_changed,
                code_top_100_membership_changed=run.code_top_100_membership_changed,
                neighbour_seed_limit=run.neighbour_seed_limit,
                neighbour_seed_count=run.neighbour_seed_count,
                neighbour_activated_seed_count=run.neighbour_activated_seed_count,
                neighbour_ineligible_seed_count=run.neighbour_ineligible_seed_count,
                neighbour_restored_count=run.neighbour_restored_count,
                neighbour_invalid_count=run.neighbour_invalid_count,
                code_duplicate_output_count=run.code_duplicate_output_count,
                graph_attempted=run.graph_attempted,
                graph_fallback=run.graph_fallback,
                graph_profile=run.graph_profile,
                graph_relation_hits=run.graph_relation_hits,
                graph_candidate_count=run.graph_candidate_count,
                graph_promoted_count=run.graph_promoted_count,
                graph_invalid_relation_count=run.graph_invalid_relation_count,
                graph_top_10_order_changed=run.graph_top_10_order_changed,
                graph_top_100_membership_changed=run.graph_top_100_membership_changed,
                atomic_rescue_attempted=run.atomic_rescue_attempted,
                atomic_rescue_active=run.atomic_rescue_active,
                atomic_rescue_fallback=run.atomic_rescue_fallback,
                atomic_rescue_candidate_available=run.atomic_rescue_candidate_available,
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
                    "specialist_route": specialist_route,
                    "specialist_embedding_profile": specialist_profile,
                    "reranker_attempted": bool(run and run.reranker_attempted),
                    "reranker_completed": bool(run and run.reranker_completed),
                    "candidate_input_count": run.candidate_input_count if run else 0,
                    "candidate_output_count": run.candidate_output_count if run else 0,
                    "candidate_permutation_valid": (
                        run.candidate_permutation_valid if run else False
                    ),
                    "candidate_character_count": run.candidate_character_count if run else 0,
                    "rerank_ms": round(run.rerank_ms, 3) if run else 0.0,
                    "code_aware_attempted": bool(run and run.code_aware_attempted),
                    "code_profile": run.code_profile if run else "none",
                    "code_query_token_count": run.code_query_token_count if run else 0,
                    "code_match_candidate_count": run.code_match_candidate_count if run else 0,
                    "neighbour_seed_count": run.neighbour_seed_count if run else 0,
                    "neighbour_restored_count": run.neighbour_restored_count if run else 0,
                    "graph_attempted": bool(run and run.graph_attempted),
                    "graph_fallback": bool(run and run.graph_fallback),
                    "graph_relation_hits": run.graph_relation_hits if run else 0,
                    "graph_candidate_count": run.graph_candidate_count if run else 0,
                    "graph_promoted_count": run.graph_promoted_count if run else 0,
                    "graph_invalid_relation_count": (
                        run.graph_invalid_relation_count if run else 0
                    ),
                    "atomic_rescue_attempted": bool(run and run.atomic_rescue_attempted),
                    "atomic_rescue_active": bool(run and run.atomic_rescue_active),
                    "atomic_rescue_fallback": bool(run and run.atomic_rescue_fallback),
                    "atomic_rescue_candidate_available": bool(
                        run and run.atomic_rescue_candidate_available
                    ),
                    "bm25_cache": run.bm25_cache if run else "none",
                    "bm25_ms": round(run.bm25_ms, 3) if run else 0.0,
                },
            )

    @property
    def variant_name(self) -> str:
        return self._behavior.name

    @property
    def embedding_profile(self) -> str:
        return self._behavior.embedding_profile

    @property
    def lexical_profile(self) -> str:
        return BM25_PROFILE if self._behavior.canonical_bm25 else "postgres-english-tsvector"

    @property
    def word_window_size(self) -> int | None:
        return self._behavior.word_window_size

    @property
    def word_window_stride(self) -> int | None:
        return self._behavior.word_window_stride

    @property
    def exact_dense(self) -> bool:
        return self._behavior.exact_dense

    @property
    def ordering_profile(self) -> str:
        return (
            "source-session-c-collation-segment-v1"
            if self._behavior.stable_window_order
            else "chunk-id-v1"
        )

    @property
    def window_renderer_profile(self) -> str:
        if self._behavior.per_track_windows:
            return "per-track-coding-content-only-v1"
        return (
            "message-content-only-v1"
            if self._behavior.content_only_windows
            else "timestamp-role-content-v1"
        )

    @property
    def anchor_prior_records(self) -> str:
        return self._behavior.anchor_prior_records

    @property
    def anchor_compile_max_payload_chars(self) -> int | None:
        return self._behavior.anchor_compile_max_payload_chars

    @property
    def search_content_profile(self) -> str:
        return "created-at-header-v1" if self._behavior.dated_search_content else "content-v1"

    @property
    def active_components(self) -> dict[str, bool]:
        components = {
            "compiler": self._behavior.compiler,
            "facets": self._behavior.facets,
            "reranker": self._behavior.reranker,
            "learned_sparse": self._behavior.learned_sparse,
            "code_aware": self._behavior.code_aware,
            "graph_sidecar": self._behavior.graph_sidecar,
            "multimodal_native": self._behavior.multimodal_native,
            "canonical_bm25": self._behavior.canonical_bm25,
            "exact_dense": self._behavior.exact_dense,
        }
        if self._behavior.atomic_rescue:
            components["atomic_rescue"] = True
        return components

    def _atomic_rescue_binding(
        self, corpus: dict[str, object], scope_id: str
    ) -> AtomicRescueBinding | None:
        if not self._behavior.atomic_rescue:
            return None
        mode, placement = self.atomic_rescue_mode, self.atomic_rescue_placement
        if self._behavior.atomic_views_at_add:
            assert self._behavior.word_window_size is not None
            return AtomicRescueBinding(
                mode=mode,
                artifact_root="",
                scope_id=scope_id,
                generation_id=str(corpus["generation_id"]),
                calibration_id="",
                pipeline_fingerprint=ATOMIC_VIEW_PROFILE,
                corpus_fingerprint=str(corpus["corpus_sha256"]),
                placement=placement,
                view_store=self._repository.atomic_view_store(scope_id),
                view_query_k=view_query_width(self._behavior.word_window_size),
            )
        return AtomicRescueBinding(
            mode=mode,
            artifact_root=os.environ.get("RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT", "").strip(),
            scope_id=scope_id,
            generation_id=str(corpus["generation_id"]),
            calibration_id=os.environ.get(
                "RECALL_AML_ATOMIC_RESCUE_CALIBRATION_ID", ""
            ).strip(),
            pipeline_fingerprint=os.environ.get(
                "RECALL_AML_ATOMIC_RESCUE_PIPELINE_FINGERPRINT", ""
            ).strip(),
            corpus_fingerprint=str(corpus["corpus_sha256"]),
            placement=placement,
        )

    @property
    def atomic_rescue_mode(self) -> str:
        """The effective mode: the environment when set, else the variant's default."""
        configured = os.environ.get("RECALL_ATOMIC_RESCUE_MODE", "").strip().lower()
        return configured or self._behavior.atomic_rescue_default_mode

    @property
    def atomic_rescue_placement(self) -> str:
        configured = os.environ.get("RECALL_ATOMIC_RESCUE_PLACEMENT", "").strip().lower()
        return configured or self._behavior.atomic_rescue_default_placement

    @property
    def atomic_rescue_profile(self) -> dict[str, object]:
        """What the atomic stage will do, as ``/version`` reports it."""
        if not self._behavior.atomic_rescue:
            return {"enabled": False}
        return {
            "enabled": True,
            "source": "add-time-views" if self._behavior.atomic_views_at_add else "file-artifact",
            "view_profile": ATOMIC_VIEW_PROFILE if self._behavior.atomic_views_at_add else "none",
            "mode": self.atomic_rescue_mode,
            "placement": self.atomic_rescue_placement,
        }

    @property
    def compiled_kinds(self) -> list[str]:
        return sorted(self._behavior.compiled_kinds or ())

    @property
    def drops_compiler_fallback(self) -> bool:
        return self._behavior.drop_compiler_fallback

    @property
    def multimodal_preserve(self) -> bool:
        return self._behavior.multimodal_preserve

    @property
    def multimodal_native(self) -> bool:
        return self._behavior.multimodal_native

    @property
    def multimodal_embedding_profile(self) -> str:
        return MULTIMODAL_EMBEDDING_PROFILE if self._behavior.multimodal_native else "none"

    @property
    def multimodal_embedding_model(self) -> str:
        return MULTIMODAL_EMBEDDING_MODEL if self._behavior.multimodal_native else "none"

    @property
    def graph_sidecar(self) -> bool:
        return self._behavior.graph_sidecar

    @property
    def context_specialist(self) -> bool:
        return self._behavior.context_specialist

    @property
    def context_embedding_profile(self) -> str:
        return (
            self._behavior.context_embedding_profile
            if self._behavior.context_specialist
            else "none"
        )

    @property
    def specialist_router_profile(self) -> str:
        return SPECIALIST_ROUTER_PROFILE if self._behavior.context_specialist else "none"

    @property
    def specialist_fusion_profile(self) -> str:
        return SPECIALIST_FUSION_PROFILE if self._behavior.context_specialist else "none"

    async def health(self) -> dict[str, object]:
        if not self._model_clients_ready:
            raise RuntimeError("mandatory model clients are not ready")
        return await asyncio.to_thread(self._repository.health)

    async def delete_user(self, user_id: str) -> int:
        tenant = tenant_for(user_id)
        tenant_handle = await asyncio.to_thread(
            self._repository.acquire_request_lock, tenant, TENANT_MUTATION_LOCK_REQUEST_ID
        )
        try:
            return await asyncio.to_thread(self._repository.delete_tenant, tenant)
        finally:
            self._invalidate_corpus_status(tenant, deleted=True)
            await asyncio.to_thread(self._repository.release_request_lock, tenant_handle)

    async def _corpus_status(self, tenant: str) -> dict[str, object]:
        cached = self._corpus_status_cache.get(tenant)
        if cached is not None:
            return dict(cached)
        flight = self._corpus_status_flights.get(tenant)
        if flight is None:
            flight = asyncio.ensure_future(self._compute_corpus_status(tenant))
            self._corpus_status_flights[tenant] = flight
            flight.add_done_callback(
                lambda done, tenant=tenant: self._settle_corpus_status(tenant, done)
            )
        # Shielded: one waiter being cancelled must not cancel the computation the others share.
        return dict(await asyncio.shield(flight))

    def _settle_corpus_status(
        self, tenant: str, flight: asyncio.Future[dict[str, object]]
    ) -> None:
        """Cache a finished computation, unless a write invalidated it while it ran."""
        if self._corpus_status_flights.get(tenant) is not flight:
            if not flight.cancelled():
                flight.exception()  # retrieved, so an unawaited failure is not reported twice
            return
        del self._corpus_status_flights[tenant]
        if flight.cancelled() or flight.exception() is not None:
            return
        if len(self._corpus_status_cache) >= MAX_CORPUS_STATUS_CACHE_ENTRIES:
            self._corpus_status_cache.pop(next(iter(self._corpus_status_cache)))
        self._corpus_status_cache[tenant] = flight.result()

    async def _compute_corpus_status(self, tenant: str) -> dict[str, object]:
        cached = await asyncio.to_thread(self._repository.corpus_status, tenant)
        if self._behavior.graph_sidecar:
            try:
                graph = await asyncio.to_thread(
                    self._repository.graph_corpus_status, tenant
                )
                cached["chunk_count"] = _status_int(
                    cached, "chunk_count"
                ) + _status_int(graph, "chunk_count")
                cached["compiled_chunk_count"] = _status_int(
                    graph, "compiled_chunk_count"
                )
                cached["source_session_count"] = max(
                    _status_int(cached, "source_session_count"),
                    _status_int(graph, "source_session_count"),
                )
                cached["authored_relation_count"] = _status_int(
                    cached, "authored_relation_count"
                ) + _status_int(graph, "authored_relation_count")
                cached["eligible_relation_count"] = _status_int(
                    cached, "eligible_relation_count"
                ) + _status_int(graph, "eligible_relation_count")
                cached["store_relation_count"] = _status_int(
                    cached, "store_relation_count"
                ) + _status_int(graph, "store_relation_count")
                cached["compiled_corpus_sha256"] = graph[
                    "compiled_corpus_sha256"
                ]
                cached["compiled_kind_counts"] = graph["compiled_kind_counts"]
                cached["compiler_profile_counts"] = graph[
                    "compiler_profile_counts"
                ]
                cached["graph_sidecar_chunk_count"] = graph["chunk_count"]
                cached["graph_corpus_sha256"] = graph["corpus_sha256"]
                cached["corpus_sha256"] = canonical_digest(
                    [cached["raw_corpus_sha256"], graph["corpus_sha256"]]
                )
                cached["graph_status"] = "ready"
            except Exception:  # BROAD-CATCH: Search must retain the raw corpus identity
                cached["graph_status"] = "unavailable"
        return cached

    async def corpus_status(self, user_id: str) -> dict[str, object]:
        return await self._corpus_status(tenant_for(user_id))

    async def prepare_sparse_user(self, user_id: str) -> dict[str, object]:
        if not self._behavior.learned_sparse:
            raise ValueError(f"{self._behavior.name} has no learned sparse stage")
        tenant = tenant_for(user_id)
        handle = await asyncio.to_thread(
            self._repository.acquire_request_lock, tenant, TENANT_MUTATION_LOCK_REQUEST_ID
        )
        try:
            return await asyncio.to_thread(self._repository.backfill_sparse, tenant)
        finally:
            await asyncio.to_thread(self._repository.release_request_lock, handle)
