"""One compiled graph build: compile each uncached source, embed the records, resolve edges.

A credit stop, an exhausted budget, or the same refusal on three sources in a row (a configuration
the provider will refuse every time: an unknown model, an unsupported parameter) stops the build
and returns no artifact, so nothing half-built can serve. What was already paid for stays in the
cache, and the next build resumes from there.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from recall.compiled_graph.anchors import (
    SourceAnchor,
    build_source_anchors,
    compile_payload,
    encode_stored_data,
    fit_payload,
)
from recall.compiled_graph.artifact import ArtifactRecord, CompiledGraphArtifact
from recall.compiled_graph.budget import Budget
from recall.compiled_graph.cache import CompileCache, cache_key
from recall.compiled_graph.promotion import record_edges
from recall.compiled_graph.provider import (
    BudgetExhausted,
    CreditStop,
    PayloadTooLarge,
    ProviderConfig,
    SourceRejected,
    Transport,
    compile_source,
)
from recall.compiled_graph.records import CompiledRecord, prompt_digest, validate_reply
from recall.embedding_core import Embedder
from recall.types import Chunk

#: The AML compiler's first payload budget (``ANCHOR_PAYLOAD_BUDGET_CHARS``).
PAYLOAD_BUDGET_CHARS = 300_000
#: A refusal with the same status on this many sources in a row is a configuration error.
REPEATED_REJECTION_STOP = 3

Stop = Literal["", "budget", "credit", "provider"]


@dataclass(frozen=True)
class SourceInput:
    """One source as the indexer chunked it: ``body`` is the exact text its chunk offsets index."""

    source: str
    body: str
    chunks: tuple[Chunk, ...]


@dataclass
class BuildReport:
    calls: int = 0
    spent_usd: float = 0.0
    sources_compiled: int = 0
    sources_cached: int = 0
    sources_rejected: int = 0
    edgeless_chunks: int = 0
    stopped: Stop = ""
    rejections: list[str] = field(default_factory=list)


def _record_text(record: CompiledRecord | ArtifactRecord) -> str:
    return "\n".join(value for value in record.fields if value)


def _compile(
    source: SourceInput,
    anchors: Sequence[SourceAnchor],
    config: ProviderConfig,
    transport: Transport,
    budget: Budget,
    report: BuildReport,
    sleep: Callable[[float], None],
) -> list[CompiledRecord]:
    payload = compile_payload(source.source, anchors)
    payload = fit_payload(payload, PAYLOAD_BUDGET_CHARS) or payload
    try:
        report.calls += 1
        reply = compile_source(config, transport, payload, budget, sleep=sleep)
    except PayloadTooLarge:
        refit = fit_payload(payload, len(encode_stored_data(payload)) // 2)
        if refit is None:
            raise SourceRejected("the source does not fit the provider's context", 413, repeatable=True) from None
        payload = refit
        report.calls += 1
        try:
            reply = compile_source(config, transport, payload, budget, sleep=sleep)
        except PayloadTooLarge:
            raise SourceRejected("the source does not fit the provider's context", 413, repeatable=True) from None
    sent = [anchor["id"] for anchor in payload["anchors"]]
    try:
        records, _ = validate_reply(reply.content, anchors, sent_anchor_ids=sent)
    except ValueError as exc:
        raise SourceRejected("the reply carried no record list", 200, repeatable=True) from exc
    return records


def _commit_always(sources: Sequence[str], action: Callable[[], None]) -> bool:
    action()
    return True


def build_compiled_graph(
    sources: Sequence[SourceInput],
    embedder: Embedder,
    config: ProviderConfig,
    transport: Transport,
    cache: CompileCache,
    budget: Budget,
    *,
    lineage: Mapping[str, str],
    commit: Callable[[Sequence[str], Callable[[], None]], bool] = _commit_always,
    known_vectors: Mapping[str, np.ndarray] | None = None,
    retry_rejected: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[CompiledGraphArtifact | None, BuildReport]:
    """Build the artifact for ``lineage``; None, with ``report.stopped`` set, when it stopped.

    Every cache write goes through ``commit(sources, action)``, which runs ``action`` only while
    ``sources`` are still held, ordered against a forget's erasure, and says whether it ran: a
    source forgotten while its compile was in flight must not come back into the cache. A refusal
    the same request would get again (``SourceRejected.repeatable``) is cached too, and not sent
    again unless ``retry_rejected``. ``known_vectors`` holds earlier embeddings of record texts
    under the same embedder, which are reused rather than paid for again.
    """
    report = BuildReport()
    accepted: list[tuple[SourceInput, CompiledRecord]] = []
    last_status: int | None = None
    repeats = 0
    for source in sources:
        anchors = build_source_anchors(source.source, source.body)
        if not anchors:
            continue
        key = cache_key(config, source.source, hashlib.sha256(source.body.encode("utf-8")).hexdigest())
        records = cache.get(key)
        if records is not None:
            report.sources_cached += 1
        elif not retry_rejected and (refused := cache.rejection(key)) is not None:
            report.sources_rejected += 1
            report.rejections.append(f"{source.source}: {refused} (cached)")
            continue
        else:
            try:
                records = _compile(source, anchors, config, transport, budget, report, sleep)
            except CreditStop:
                report.stopped = "credit"
            except BudgetExhausted:
                report.stopped = "budget"
            except SourceRejected as exc:
                report.sources_rejected += 1
                report.rejections.append(f"{source.source}: {exc}")
                if exc.repeatable:
                    reason = str(exc)
                    commit([source.source], lambda: cache.put_rejection(key, source.source, reason))
                repeats = repeats + 1 if exc.status is not None and exc.status == last_status else 1
                last_status = exc.status
                if repeats >= REPEATED_REJECTION_STOP:
                    report.stopped = "provider"
                else:
                    continue
            if report.stopped:
                report.spent_usd = budget.spent_usd
                return None, report
            assert records is not None
            repeats, last_status = 0, None
            report.sources_compiled += 1
            compiled = records
            if not commit([source.source], lambda: cache.put(key, source.source, compiled)):
                continue
        if records:
            report.edgeless_chunks += sum(
                1 for chunk in source.chunks if chunk.metadata.get("text_start") is None
            )
        for record in records:
            accepted.append((source, record))
    report.spent_usd = budget.spent_usd
    artifact_records: list[ArtifactRecord] = []
    for index, (source, record) in enumerate(accepted):
        edges = record_edges(record, source.chunks)
        linked = [chunk for chunk in source.chunks if chunk.id in edges]
        mentions = tuple(
            (chunk.id, entity) for chunk in linked for entity in record.entities if entity in chunk.text
        )
        artifact_records.append(
            ArtifactRecord(
                key=f"{source.source}#{index}",
                source=source.source,
                kind=record.kind,
                fields=record.fields,
                entities=record.entities,
                edges=edges,
                mentions=mentions,
            )
        )
    vectors: np.ndarray = np.zeros((0, embedder.dim), dtype=np.float32)
    if artifact_records:
        known = known_vectors or {}
        texts = [_record_text(r) for r in artifact_records]
        missing = sorted({text for text in texts if text not in known})
        fresh: dict[str, np.ndarray] = (
            dict(zip(missing, np.asarray(embedder.embed(missing), dtype=np.float32))) if missing else {}
        )
        raw: np.ndarray = np.asarray(
            [known[text] if text in known else fresh[text] for text in texts], dtype=np.float32
        )
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        vectors = raw / np.where(norms == 0, 1.0, norms)
    full_lineage = {**lineage, "model": config.model, "prompt": prompt_digest()}
    return CompiledGraphArtifact(lineage=full_lineage, records=tuple(artifact_records), vectors=vectors), report
