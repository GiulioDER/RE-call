"""Run RE-call on EnterpriseRAG-Bench and write the leaderboard answer JSONL.

EnterpriseRAG-Bench expects one JSON object per question:

    {"question_id": "qst_0001", "answer": "...", "document_ids": ["dsid_..."]}

This runner is deliberately lightweight. It reads local release files, indexes documents into
RE-call with the original EnterpriseRAG document id preserved in metadata, retrieves document ids
per question, then writes an answer file plus a run manifest. Generation is optional: the default
`extractive` mode spends no model calls and is useful for validating ingestion, retrieval, and file
format before paying for a full answer run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import zipfile
from io import TextIOWrapper
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

from recall._env import load_dotenv
from recall.cache import EmbeddingCache, embed_query_with_cache
from recall.embeddings import Embedder, embed_query, embedding_profile_id, resolve_embedder
from recall.guards import DEFAULT_GAP_THRESHOLD
from recall.index import chunk_text
from recall.retriever import SPARSE_BACKENDS, HybridRetriever
from recall.reasoning_expansion import (
    ExpansionProposal,
    ExpansionReport,
    ExpansionRequest,
    EXPANSION_PROMPT_DIGEST,
    MAX_EXPANSION_EVIDENCE_CHARS,
    MAX_EXPANSION_EVIDENCE_ITEMS,
    MAX_EXPANSION_QUERIES,
    OpenAIExpansionProvider,
    resolve_expansion_provider,
)
from recall.store import PgVectorStore
from recall.types import Chunk, RetrievalDiagnostics, ScoredChunk
from benchmarks.artifact_contract import load_published_artifact

DEFAULT_TABLE = "bench_enterprise_rag_chunks"
DEFAULT_TENANT = "enterprise-rag"
DEFAULT_K = 8
DEFAULT_CANDIDATE_K = 80
DEFAULT_MAX_CHARS = 3500
DEFAULT_BATCH_CHUNKS = 256
DEFAULT_CHUNK_CHARS = 800
DEFAULT_CHUNK_OVERLAP = 80
DEFAULT_RERANK_DOCUMENT_CHARS = 4_000
MAX_RETRIEVAL_CAPTURES = 10
DEFAULT_MODEL = "openai/gpt-4o"
DEFAULT_SPLADE_MODEL = "prithivida/Splade_PP_en_v1"
DEFAULT_VOYAGE_RERANKER = "rerank-2.5"
DEFAULT_DSN = "postgresql://recall:recall@localhost:5432/recall"
TOP_CONFIG_K = 8
TOP_CONFIG_CANDIDATE_K = 200
TOP_CONFIG_BATCH_CHUNKS = 32
TOP_CONFIG_MAX_CHARS = 12_000
TOP_CONFIG_CHUNK_CHARS = 12_000
TOP_CONFIG_CHUNK_OVERLAP = 200
TOP_CONFIG_EMBEDDER = "voyage:voyage-4-large"
TOP_CONFIG_SPARSE_BACKEND = "both"
TOP_CONFIG_RERANKER = f"voyage:{DEFAULT_VOYAGE_RERANKER}"
DOC_ID_RE = re.compile(r"(dsid_[A-Za-z0-9]+)")
RUNTIME_EXCLUDED_QUESTION_FIELDS = frozenset(
    {
        "expected_doc_ids",
        "answer_facts",
        "answer",
        "answers",
        "gold_answer",
        "expected_answer",
        "competitor_answers",
    }
)


@dataclass(frozen=True)
class EnterpriseDoc:
    doc_id: str
    source_type: str
    title: str
    content: str


@dataclass(frozen=True)
class EnterpriseQuestion:
    question_id: str
    question: str
    raw: Mapping[str, Any]


def _json_rows(path: Path) -> Iterator[Mapping[str, Any]]:
    """Yield JSON objects from a `.json`, `.jsonl`, or `.zip` release artifact."""
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.suffix.lower() in {".json", ".jsonl", ".zip"}:
                yield from _json_rows(child)
        return
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            for name in sorted(zf.namelist()):
                if name.endswith("/") or Path(name).suffix.lower() not in {".json", ".jsonl"}:
                    continue
                with zf.open(name) as handle:
                    if name.endswith(".jsonl"):
                        for i, line in enumerate(
                            TextIOWrapper(handle, encoding="utf-8"), start=1
                        ):
                            if not line.strip():
                                continue
                            payload = json.loads(line)
                            if not isinstance(payload, Mapping):
                                raise ValueError(
                                    f"{path}!{name}:{i}: expected a JSON object"
                                )
                            yield payload
                    else:
                        text = handle.read().decode("utf-8")
                        yield from _json_text_rows(text, label=f"{path}!{name}")
        return
    yield from _json_text_rows(path.read_text(encoding="utf-8"), label=str(path))


def _json_text_rows(text: str, *, label: str) -> Iterator[Mapping[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return
    if stripped.startswith("["):
        payload = json.loads(stripped)
        if not isinstance(payload, list):
            raise ValueError(f"{label}: expected a JSON array")
        for item in payload:
            if isinstance(item, Mapping):
                yield item
        return
    if stripped.startswith("{") and "\n" not in stripped:
        payload = json.loads(stripped)
        if isinstance(payload, Mapping):
            yield payload
            return
        raise ValueError(f"{label}: expected a JSON object")
    for i, line in enumerate(stripped.splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{label}:{i}: expected a JSON object")
        yield payload


def load_documents(paths: Sequence[Path], *, limit: int | None = None) -> Iterator[EnterpriseDoc]:
    seen: set[str] = set()
    yielded = 0
    for path in paths:
        for doc in _document_rows(path):
            if doc.doc_id in seen:
                continue
            seen.add(doc.doc_id)
            yield doc
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def _document_rows(path: Path) -> Iterator[EnterpriseDoc]:
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.suffix.lower() in {".txt", ".json", ".jsonl", ".zip"}:
                yield from _document_rows(child)
        return
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = sorted(name for name in zf.namelist() if not name.endswith("/"))
            txt_names = [name for name in names if Path(name).suffix.lower() == ".txt"]
            if txt_names:
                for name in txt_names:
                    with zf.open(name) as handle:
                        content = handle.read().decode("utf-8", errors="replace")
                    yield _doc_from_text_file(name, content)
                return
        for row in _json_rows(path):
            yield _doc_from_row(row)
        return
    if path.suffix.lower() == ".txt":
        yield _doc_from_text_file(str(path), path.read_text(encoding="utf-8", errors="replace"))
        return
    for row in _json_rows(path):
        yield _doc_from_row(row)


def _doc_from_text_file(name: str, content: str) -> EnterpriseDoc:
    path = Path(name)
    match = DOC_ID_RE.search(path.name)
    if not match:
        raise ValueError(f"{name}: expected EnterpriseRAG document filename containing dsid_...")
    doc_id = match.group(1)
    source_type = path.parts[0] if len(path.parts) > 1 else "unknown"
    stem = path.stem
    title = stem.split("__", 1)[1] if "__" in stem else stem
    title = title.replace("-", " ").replace("_", " ").strip() or doc_id
    return EnterpriseDoc(
        doc_id=doc_id,
        source_type=source_type,
        title=title,
        content=content.replace("\x00", ""),
    )


def _doc_from_row(row: Mapping[str, Any]) -> EnterpriseDoc:
    doc_id = _required_text(row, "doc_id", "document_id", "id")
    source_type = _text(row.get("source_type") or row.get("source") or "unknown")
    title = _text(row.get("title") or row.get("name") or doc_id)
    content = _required_text(row, "content", "text", "body")
    return EnterpriseDoc(doc_id=doc_id, source_type=source_type, title=title, content=content)


def load_questions(
    path: Path,
    *,
    limit: int | None = None,
    question_types: Collection[str] | None = None,
    question_ids: Collection[str] | None = None,
) -> list[EnterpriseQuestion]:
    selected_types = {str(value).strip().lower() for value in (question_types or ()) if str(value).strip()}
    selected_ids = {str(value).strip() for value in (question_ids or ()) if str(value).strip()}
    questions: list[EnterpriseQuestion] = []
    for row in _json_rows(path):
        runtime_raw = {
            key: value
            for key, value in row.items()
            if key not in RUNTIME_EXCLUDED_QUESTION_FIELDS
        }
        question = EnterpriseQuestion(
            question_id=_required_text(row, "question_id", "id"),
            question=_required_text(row, "question", "query"),
            raw=runtime_raw,
        )
        if selected_types and _text(row.get("question_type")).lower() not in selected_types:
            continue
        if selected_ids and question.question_id not in selected_ids:
            continue
        questions.append(question)
        if limit is not None and len(questions) >= limit:
            break
    return questions


def _required_text(row: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = row.get(name)
        text = _text(value)
        if text:
            return text
    raise ValueError(f"row lacks required field among {names}: {row}")


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def doc_chunks(
    doc: EnterpriseDoc,
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    body = f"Title: {doc.title}\nSource type: {doc.source_type}\n\n{doc.content}"
    parts = chunk_text(body, max_chars=chunk_chars, overlap=chunk_overlap)
    return [
        Chunk(
            id=f"{doc.doc_id}#{i:04d}",
            source=doc.doc_id,
            text=part,
            metadata={
                "doc_id": doc.doc_id,
                "source_type": doc.source_type,
                "title": doc.title,
                "file": doc.doc_id,
            },
        )
        for i, part in enumerate(parts)
    ]


def index_documents(
    store: PgVectorStore,
    embedder: Embedder,
    docs: Iterable[EnterpriseDoc],
    *,
    batch_chunks: int = DEFAULT_BATCH_CHUNKS,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    reset: bool = False,
    skip_indexed_sources: bool = False,
) -> dict[str, int]:
    if chunk_chars < 1:
        raise ValueError("chunk_chars must be >= 1")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if reset:
        sources = sorted({chunk.source for chunk in store.iter_chunks()})
        if sources:
            store.delete_sources(sources)
    existing_sources: set[str] = set()
    if skip_indexed_sources:
        existing_sources = {chunk.source for chunk in store.iter_chunks()}
        print(f"index resume existing_sources={len(existing_sources)}", flush=True)
    batch: list[Chunk] = []
    indexed_docs = 0
    skipped_docs = 0
    indexed_chunks = 0
    for doc in docs:
        if doc.doc_id in existing_sources:
            skipped_docs += 1
            continue
        indexed_docs += 1
        batch.extend(doc_chunks(doc, chunk_chars=chunk_chars, chunk_overlap=chunk_overlap))
        if len(batch) >= batch_chunks:
            indexed_chunks += _write_batch(store, embedder, batch)
            batch = []
    if batch:
        indexed_chunks += _write_batch(store, embedder, batch)
    store.analyze_if_stale(indexed_chunks)
    return {"documents": indexed_docs, "chunks": indexed_chunks, "skipped_documents": skipped_docs}


def backfill_sparse(
    store: PgVectorStore,
    *,
    model: str,
    accept_noncommercial_license: bool,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    from recall.sparse import (
        SpladeEncoder,
        assert_sparse_coverage,
        attribution_notice,
        backfill_learned_sparse,
        inspect_sparse_device,
        resolve_sparse_device,
    )

    device_report = inspect_sparse_device(device)
    resolved_device = resolve_sparse_device(device, report=device_report)
    print(
        "sparse device "
        f"requested={device_report.requested} resolved={resolved_device} "
        f"name={device_report.device_name or 'none'} "
        f"cuda={device_report.torch_cuda_build or 'none'}",
        flush=True,
    )
    encoder = SpladeEncoder.from_pretrained(
        model, accept_noncommercial_license=accept_noncommercial_license, device=resolved_device
    )
    last_reported = -1

    def progress(written: int) -> None:
        nonlocal last_reported
        if written == last_reported:
            return
        last_reported = written
        print(f"splade backfill written={written}", flush=True)

    result = backfill_learned_sparse(store, encoder, batch_size=batch_size, progress=progress)
    assert_sparse_coverage(store, encoder.profile.profile_id, empty_ids=result.empty_ids)
    return {
        "model": model,
        "profile_id": encoder.profile.profile_id,
        "device": resolved_device,
        "device_report": asdict(device_report),
        "written": result.written,
        "empty_ids": result.empty_ids,
        "attribution": attribution_notice(model),
    }


def build_sparse_encoder(
    sparse_backend: str,
    *,
    model: str,
    accept_noncommercial_license: bool,
    device: str,
) -> object | None:
    if sparse_backend not in ("splade", "both"):
        return None
    from recall.sparse import SpladeEncoder, resolve_sparse_device

    return SpladeEncoder.from_pretrained(
        model,
        accept_noncommercial_license=accept_noncommercial_license,
        device=resolve_sparse_device(device),
    )


def build_reranker(
    name: str,
    *,
    max_document_chars: int | None = None,
    rank_weight: float = 1.0,
) -> object | None:
    if not name or name == "none":
        return None
    if name.startswith("voyage:"):
        from benchmarks.voyage_rerank import BlendedVoyageReranker, VoyageReranker

        if not 0.0 <= rank_weight <= 1.0:
            raise ValueError("rank_weight must be between 0 and 1")
        if rank_weight == 1.0:
            return VoyageReranker(
                model=name[len("voyage:"):],
                max_document_chars=max_document_chars,
            )
        return BlendedVoyageReranker(
            model=name[len("voyage:"):],
            rank_weight=rank_weight,
            max_document_chars=max_document_chars,
        )
    if name == "local" or name.startswith("local:"):
        from benchmarks.systems import resolve_reranker
        from recall.rerank import Reranker

        return cast(Reranker | None, resolve_reranker(name))
    raise ValueError("unknown reranker; use none, local, local:<model>, or voyage:<model>")


def _write_batch(store: PgVectorStore, embedder: Embedder, chunks: list[Chunk]) -> int:
    embeddings = embedder.embed([chunk.text for chunk in chunks])
    return store.upsert(chunks, embeddings)


def retrieve_docs_with_diagnostics(
    store: PgVectorStore,
    embedder: Embedder,
    question: str,
    *,
    k: int,
    candidate_k: int,
    sparse_backend: str,
    sparse_encoder: object | None,
    reranker: object | None,
    gap_threshold: float,
) -> tuple[list[str], list[ScoredChunk], bool, RetrievalDiagnostics]:
    retriever = HybridRetriever(
        store,
        embedder,
        candidate_k=candidate_k,
        reranker=reranker,  # type: ignore[arg-type]
        gap_threshold=gap_threshold,
        sparse_backend=sparse_backend,
        sparse_encoder=sparse_encoder,
        retrieval_profile=f"enterprise-rag:{sparse_backend}",
    )
    result = retriever.search(question, k=k)
    ids: list[str] = []
    seen: set[str] = set()
    for hit in result.hits:
        doc_id = str(hit.chunk.metadata.get("doc_id") or hit.chunk.source).strip()
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            ids.append(doc_id)
    return ids, result.hits, result.gap_warning, result.diagnostics


def retrieve_docs(
    store: PgVectorStore,
    embedder: Embedder,
    question: str,
    *,
    k: int,
    candidate_k: int,
    sparse_backend: str,
    sparse_encoder: object | None,
    reranker: object | None,
    gap_threshold: float,
) -> tuple[list[str], list[ScoredChunk], bool]:
    ids, hits, gap_warning, _diagnostics = retrieve_docs_with_diagnostics(
        store,
        embedder,
        question,
        k=k,
        candidate_k=candidate_k,
        sparse_backend=sparse_backend,
        sparse_encoder=sparse_encoder,
        reranker=reranker,
        gap_threshold=gap_threshold,
    )
    return ids, hits, gap_warning


class QueryCachedEmbedder:
    """Preserve passage embedding behavior while caching repeated query vectors."""

    def __init__(self, inner: Embedder, cache: EmbeddingCache) -> None:
        self._inner = inner
        self._cache = cache

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def name(self) -> str:
        return self._inner.name

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return embed_query_with_cache(self._inner, text, self._cache)


def _merge_scored_hits(initial: Sequence[ScoredChunk], additions: Sequence[ScoredChunk]) -> list[ScoredChunk]:
    merged: dict[str, ScoredChunk] = {}
    for hit in (*initial, *additions):
        previous = merged.get(hit.chunk.id)
        if previous is None or hit.score > previous.score:
            merged[hit.chunk.id] = hit
    # Expansion is useful only if a strong newly found hit can displace a weak original hit
    # before the document-level k cutoff. Ties retain the first-pass ordering for stability.
    return sorted(merged.values(), key=lambda hit: hit.score, reverse=True)


def _expansion_evidence(hits: Sequence[ScoredChunk]) -> tuple[Mapping[str, object], ...]:
    evidence: list[Mapping[str, object]] = []
    used_chars = 0
    for hit in hits:
        if len(evidence) >= MAX_EXPANSION_EVIDENCE_ITEMS:
            break
        remaining = MAX_EXPANSION_EVIDENCE_CHARS - used_chars
        if remaining <= 0:
            break
        text = hit.chunk.text[:remaining]
        evidence.append(
            {"chunk_id": hit.chunk.id, "source": hit.chunk.source, "text": text}
        )
        used_chars += len(text)
    return tuple(evidence)


def _expansion_report_to_cache(
    report: ExpansionReport,
    *,
    context: Mapping[str, object],
    provider_metadata: Mapping[str, object],
) -> dict[str, object]:
    return {
        "context": dict(context),
        "provider_metadata": dict(provider_metadata),
        "proposals": [
            {
                "id": proposal.id,
                "mode": proposal.mode,
                "query": proposal.query,
                "rationale": proposal.rationale,
                "parent_chunk_ids": list(proposal.parent_chunk_ids),
            }
            for proposal in report.proposals
        ]
    }


def _expansion_report_from_cache(payload: Mapping[str, Any]) -> ExpansionReport:
    raw = payload.get("proposals", [])
    if not isinstance(raw, list):
        raise ValueError("reasoning cache proposals must be an array")
    if len(raw) > MAX_EXPANSION_QUERIES:
        raise ValueError("reasoning cache exceeds the expansion query limit")
    return ExpansionReport(
        proposals=tuple(
            ExpansionProposal(
                id=str(item["id"]),
                mode=cast(Any, item["mode"]),
                query=str(item["query"]),
                rationale=str(item.get("rationale", "")),
                parent_chunk_ids=tuple(str(value) for value in item.get("parent_chunk_ids", [])),
            )
            for item in raw
            if isinstance(item, Mapping)
        )
    )


def _expansion_cache_context(
    question: EnterpriseQuestion,
    store: PgVectorStore,
    *,
    k: int,
    candidate_k: int,
    sparse_backend: str,
    gap_threshold: float,
    arm: str,
    provider: OpenAIExpansionProvider,
) -> dict[str, object]:
    provider_identity = provider.provider_metadata().to_dict()
    return {
        "question_sha256": hashlib.sha256(question.question.encode("utf-8")).hexdigest(),
        "tenant": store.tenant,
        "table": store.table,
        "generation_id": store.generation_id,
        "k": k,
        "candidate_k": candidate_k,
        "sparse_backend": sparse_backend,
        "gap_threshold": gap_threshold,
        "arm": arm,
        "provider_id": provider_identity["provider_id"],
        "model_id": provider_identity["model_id"],
        "model_revision": provider_identity["model_revision"],
        "prompt_digest": EXPANSION_PROMPT_DIGEST,
    }


def _expansion_cache_matches(
    payload: Mapping[str, Any], context: Mapping[str, object]
) -> bool:
    cached_context = payload.get("context")
    return isinstance(cached_context, Mapping) and dict(cached_context) == dict(context)


def load_reasoning_cache(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("reasoning cache must be a JSON object")
    return payload


def write_reasoning_cache(path: Path | None, cache: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def expand_retrieval_hits(
    question: EnterpriseQuestion,
    store: PgVectorStore,
    embedder: Embedder,
    *,
    initial_hits: Sequence[ScoredChunk],
    initial_gap_warning: bool,
    k: int,
    candidate_k: int,
    sparse_backend: str,
    sparse_encoder: object | None,
    reranker: object | None,
    gap_threshold: float,
    arm: str,
    provider: OpenAIExpansionProvider | None,
    expansion_cache: dict[str, Any] | None,
) -> tuple[list[str], list[ScoredChunk], dict[str, Any]]:
    """Run one of the preregistered retrieval expansion arms."""

    if arm == "none":
        return _doc_ids_from_hits(initial_hits, k=k), list(initial_hits), {
            "arm": arm,
            "passes": 1,
            "expanded": False,
        }

    hits = list(initial_hits)
    queries: list[str] = []
    fallback_reason: str | None = None
    model_metadata: dict[str, object] | None = None
    passes = 1
    depth_gap_warning: bool | None = None
    cache_context = (
        _expansion_cache_context(
            question,
            store,
            k=k,
            candidate_k=candidate_k,
            sparse_backend=sparse_backend,
            gap_threshold=gap_threshold,
            arm=arm,
            provider=provider,
        )
        if provider is not None
        else None
    )

    if arm in {"depth", "closed_loop"}:
        _, depth_hits, depth_gap_warning = retrieve_docs(
            store,
            embedder,
            question.question,
            k=min(candidate_k, max(k + 1, k * 2)),
            candidate_k=candidate_k,
            sparse_backend=sparse_backend,
            sparse_encoder=sparse_encoder,
            reranker=reranker,
            gap_threshold=gap_threshold,
        )
        hits = _merge_scored_hits(hits, depth_hits)
        passes = 2
        queries.append(question.question)

    if arm == "closed_loop" and not depth_gap_warning and depth_hits:
        return _doc_ids_from_hits(hits, k=k), hits, {
            "arm": arm,
            "passes": passes,
            "expanded": len(hits) > len(initial_hits),
            "queries": queries,
            "fallback_reason": None,
            "provider_skipped_reason": "depth_resolved",
            "model": None,
        }

    if arm in {"cheap", "closed_loop"}:
        if provider is None:
            fallback_reason = "cheap_expansion_provider_unavailable"
        else:
            expansion_request = ExpansionRequest(
                query=question.question,
                tenant_id="enterprise-rag",
                generation_id=None,
                evidence=_expansion_evidence(hits),
                gap_reason=(
                    "retrieval_gap"
                    if initial_gap_warning or not initial_hits
                    else "assess_evidence_completeness"
                ),
            )
            try:
                cached = expansion_cache.get(question.question_id) if expansion_cache else None
                if cache_context is not None and isinstance(cached, Mapping) and _expansion_cache_matches(
                    cached, cache_context
                ):
                    report = _expansion_report_from_cache(cached)
                else:
                    report = provider(expansion_request)
                    if expansion_cache is not None:
                        expansion_cache[question.question_id] = _expansion_report_to_cache(
                            report,
                            context=cache_context or {},
                            provider_metadata=provider.provider_metadata().to_dict(),
                        )
                if not isinstance(report, ExpansionReport):
                    raise TypeError("cheap provider returned an invalid report")
                proposals = sorted(
                    report.proposals,
                    key=lambda proposal: {"depth": 0, "rewrite": 1, "decompose": 2}[proposal.mode],
                )
                for proposal in proposals:
                    if proposal.mode == "depth" and arm == "closed_loop":
                        continue
                    _, proposal_hits, _ = retrieve_docs(
                        store,
                        embedder,
                        proposal.query,
                        k=k,
                        candidate_k=candidate_k,
                        sparse_backend=sparse_backend,
                        sparse_encoder=sparse_encoder,
                        reranker=reranker,
                        gap_threshold=gap_threshold,
                    )
                    hits = _merge_scored_hits(hits, proposal_hits)
                    queries.append(proposal.query)
                passes += int(bool(proposals))
                model_metadata = provider.provider_metadata().to_dict()
            except Exception as exc:
                fallback_reason = type(exc).__name__

    return _doc_ids_from_hits(hits, k=k), hits, {
        "arm": arm,
        "passes": passes,
        "expanded": len(hits) > len(initial_hits),
        "queries": queries,
        "fallback_reason": fallback_reason,
        "model": model_metadata,
    }


def extractive_answer(question: str, hits: Sequence[ScoredChunk], *, max_chars: int) -> str:
    if not hits:
        return "I could not find enough information in the retrieved enterprise documents to answer."
    parts = [f"Question: {question}", "Retrieved evidence:"]
    used = 0
    for hit in hits:
        doc_id = str(hit.chunk.metadata.get("doc_id") or hit.chunk.source)
        title = str(hit.chunk.metadata.get("title") or doc_id)
        snippet = " ".join(hit.chunk.text.split())
        remaining = max_chars - used
        if remaining <= 0:
            break
        snippet = snippet[:remaining]
        used += len(snippet)
        parts.append(f"[{doc_id}] {title}: {snippet}")
    parts.append(
        "Answer: The retrieved documents above contain the relevant evidence. "
        "Use the cited document_ids for evaluation."
    )
    return "\n\n".join(parts)


def generated_answer(
    question: str,
    hits: Sequence[ScoredChunk],
    *,
    model: str,
    api_key: str,
    max_chars: int,
    question_type: str | None = None,
    answer_policy: str = "baseline",
) -> str:
    from benchmarks.llm import OpenRouterLLM

    evidence: list[str] = []
    used = 0
    for hit in hits:
        doc_id = str(hit.chunk.metadata.get("doc_id") or hit.chunk.source)
        title = str(hit.chunk.metadata.get("title") or doc_id)
        source_type = str(hit.chunk.metadata.get("source_type") or "unknown")
        text = " ".join(hit.chunk.text.split())
        remaining = max_chars - used
        if remaining <= 0:
            break
        text = text[:remaining]
        used += len(text)
        evidence.append(f"Document {doc_id}\nSource type: {source_type}\nTitle: {title}\n{text}")
    system = (
        "Answer the enterprise question using only the provided documents. "
        "Return the shortest complete answer that preserves exact facts, quantities, dates, names, "
        "identifiers, and conditions. If multiple documents conflict, prefer the most specific and "
        "direct evidence; mention a conflict only when it changes the answer. If the documents do "
        "not support an answer, say that the available documents do not contain the answer. Do not "
        "invent facts. Do not include inline citations, because document_ids are submitted "
        "separately. Reason through conflicts, completeness, and missing evidence before giving "
        "the final answer, but return only the final answer."
    )
    if answer_policy == "category_aware":
        system += "\n\n" + category_answer_policy(question_type)
    elif answer_policy != "baseline":
        raise ValueError(f"unknown answer policy: {answer_policy}")
    type_line = f"Question type: {question_type}\n" if question_type else ""
    user = f"{type_line}Question:\n{question}\n\nDocuments:\n\n" + "\n\n".join(evidence)
    return OpenRouterLLM(model=model, api_key=api_key).complete(system, user)


def category_answer_policy(question_type: str | None) -> str:
    """Return the bounded reader policy for an official benchmark category.

    This is deliberately an answer-side policy. It does not use gold document ids or answer
    facts, and it is therefore safe to evaluate as a system behavior rather than as an oracle.
    """

    policies = {
        "project_related": (
            "This is a project-related multi-document question. Build one coherent answer from "
            "all relevant documents. Track each requested entity, decision, dependency, and "
            "causal step separately, then merge them without dropping a supported detail. "
            "Distinguish facts from different source systems and do not fill gaps from general "
            "knowledge."
        ),
        "completeness": (
            "This is a completeness question. Treat the requested result as a checklist or set. "
            "Search the provided documents for every distinct item, count, comparison, or fact "
            "needed by the question. Before answering, verify that every part of the checklist "
            "is supported and do not stop after finding only the most salient document."
        ),
        "conflicting_info": (
            "This is a conflicting-information question. Compare the relevant documents before "
            "answering. Use explicit dates, version markers, status, and source specificity to "
            "separate superseded information from the current answer. Mention the older or "
            "contradictory value when it is necessary to explain the change, and never silently "
            "blend incompatible values."
        ),
        "constrained": (
            "This is a constrained question. Treat every qualifier in the question as mandatory. "
            "Filter out documents that match the topic but fail even one condition, and report "
            "only facts supported after all constraints are applied."
        ),
        "intra_document_reasoning": (
            "This question requires joining evidence from different parts of a document. Check "
            "the whole supplied evidence for exceptions, conditions, and later qualifications "
            "before deciding what the document supports."
        ),
        "high_level": (
            "This is a high-level synthesis question. Separate directly supported company-wide "
            "patterns from individual examples, and state the scope of the evidence instead of "
            "inventing a precise number or universal conclusion."
        ),
    }
    return policies.get(
        (question_type or "").strip().lower(),
        "First identify the exact facts requested, then verify each fact against the supplied evidence before answering.",
    )


def answer_hits(
    hits: Sequence[ScoredChunk],
    document_ids: Sequence[str],
    *,
    question_type: str | None,
    answer_policy: str = "baseline",
) -> list[ScoredChunk]:
    """Keep answer context and submitted document ids aligned for the experimental policy."""

    if answer_policy == "baseline":
        return list(hits)
    if answer_policy != "category_aware":
        raise ValueError(f"unknown answer policy: {answer_policy}")

    allowed = {str(value) for value in document_ids}
    if not allowed:
        return list(hits)
    selected: list[ScoredChunk] = []
    seen_documents: set[str] = set()
    # Conflict questions benefit from one representative chunk per submitted document so the
    # reader compares distinct sources instead of repeating the same stale/current claim. The
    # other multi-evidence categories need all available chunks: completeness and project answers
    # often distribute separate requested facts across several chunks of one document.
    deduplicate_by_document = question_type == "conflicting_info"
    for hit in hits:
        doc_id = str(hit.chunk.metadata.get("doc_id") or hit.chunk.source)
        if doc_id not in allowed:
            continue
        if deduplicate_by_document and doc_id in seen_documents:
            continue
        seen_documents.add(doc_id)
        selected.append(hit)
    return selected or list(hits)


def write_answers(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    overwrite: bool,
) -> int:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_answer_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{i}: expected a JSON object")
            rows.append(payload)
    return rows


def write_answers_stream(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    overwrite: bool,
    resume: bool,
) -> tuple[int, list[Mapping[str, Any]]]:
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if path.exists() and not overwrite and not resume:
        raise FileExistsError(f"{path} exists. Pass --overwrite or --resume.")
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else "w"
    written_rows: list[Mapping[str, Any]] = []
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        for row in rows:
            public_row = {key: value for key, value in row.items() if not key.startswith("_")}
            handle.write(json.dumps(public_row, ensure_ascii=False) + "\n")
            handle.flush()
            written_rows.append(row)
            print(f"answered {public_row.get('question_id')}", flush=True)
    return len(written_rows), written_rows


def retrieval_capture_summary(captures: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not captures:
        return {
            "count": 0,
            "stable": None,
            "mean_document_jaccard": None,
            "mean_latency_ms": None,
            "p95_latency_ms": None,
            "mean_stage_ms": {},
            "calls": {},
        }
    document_sets = [
        {str(value) for value in capture.get("document_ids", ())}
        for capture in captures
    ]
    jaccard_sum = 0.0
    pair_count = 0
    for index, left in enumerate(document_sets):
        for right in document_sets[index + 1 :]:
            union = left | right
            jaccard_sum += len(left & right) / len(union) if union else 1.0
            pair_count += 1
    latencies = sorted(
        float(capture["latency_ms"])
        for capture in captures
        if isinstance(capture.get("latency_ms"), (int, float))
    )
    stage_totals: dict[str, float] = {}
    call_totals: dict[str, int] = {}
    for capture in captures:
        stage_ms = capture.get("stage_ms")
        if isinstance(stage_ms, Mapping):
            for name, value in stage_ms.items():
                if isinstance(value, (int, float)):
                    stage_totals[str(name)] = stage_totals.get(str(name), 0.0) + float(value)
        calls = capture.get("calls")
        if isinstance(calls, Mapping):
            for name, value in calls.items():
                if isinstance(value, int):
                    call_totals[str(name)] = call_totals.get(str(name), 0) + value
    p95_index = min(len(latencies) - 1, max(0, int(len(latencies) * 0.95 + 0.999) - 1))
    return {
        "count": len(captures),
        "stable": len({tuple(sorted(values)) for values in document_sets}) == 1,
        "mean_document_jaccard": jaccard_sum / pair_count if pair_count else 1.0,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "p95_latency_ms": latencies[p95_index] if latencies else None,
        "mean_stage_ms": (
            {name: value / len(captures) for name, value in sorted(stage_totals.items())}
            if stage_totals
            else {}
        ),
        "calls": call_totals,
    }


def runtime_telemetry(rows: Sequence[Mapping[str, Any]], *, answer_mode: str) -> dict[str, Any]:
    """Summarize non gold runtime measurements for the run manifest."""

    captures = [
        row.get("_diagnostics", {}).get("captures")
        for row in rows
        if isinstance(row.get("_diagnostics"), Mapping)
    ]
    summaries = [item for item in captures if isinstance(item, Mapping)]
    latencies = [
        float(item["mean_latency_ms"])
        for item in summaries
        if isinstance(item.get("mean_latency_ms"), (int, float))
    ]
    calls: dict[str, int] = {}
    for item in summaries:
        raw_calls = item.get("calls")
        if isinstance(raw_calls, Mapping):
            for name, value in raw_calls.items():
                if isinstance(value, int):
                    calls[str(name)] = calls.get(str(name), 0) + value
    calls["answer_model"] = len(rows) if answer_mode == "openrouter" else 0
    return {
        "questions_with_telemetry": len(summaries),
        "mean_question_retrieval_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "p95_question_retrieval_latency_ms": (
            sorted(latencies)[min(len(latencies) - 1, max(0, int(len(latencies) * 0.95 + 0.999) - 1))]
            if latencies
            else None
        ),
        "calls": calls,
        "cost_usd": None,
        "cost_status": "not_available_for_retrieval_providers",
    }


def _combine_retrieval_diagnostics(
    first: RetrievalDiagnostics,
    second: RetrievalDiagnostics,
) -> RetrievalDiagnostics:
    """Combine the two retrieval passes used by selective depth."""

    stage_ms: dict[str, float] = {}
    for diagnostics in (first, second):
        for name, value in diagnostics.stage_ms.items():
            stage_ms[name] = stage_ms.get(name, 0.0) + float(value)
    return RetrievalDiagnostics(
        embedding_profile=second.embedding_profile,
        retrieval_profile=second.retrieval_profile,
        index_generation=second.index_generation,
        candidate_pool_size=max(first.candidate_pool_size, second.candidate_pool_size),
        reranking_ran=first.reranking_ran or second.reranking_ran,
        stage_ms=stage_ms,
    )


def _answers(
    questions: Sequence[EnterpriseQuestion],
    store: PgVectorStore,
    embedder: Embedder,
    *,
    k: int,
    candidate_k: int,
    mode: str,
    model: str,
    api_key: str | None,
    max_chars: int,
    sparse_backend: str,
    sparse_encoder: object | None,
    reranker: object | None,
    gap_threshold: float,
    reasoning_arm: str,
    expansion_provider: OpenAIExpansionProvider | None,
    expansion_cache: dict[str, Any] | None,
    answer_policy: str = "baseline",
    retrieval_captures: int = 1,
    adaptive_depth_feature: str | None = None,
    adaptive_depth_threshold: float | None = None,
    adaptive_depth_k: int | None = None,
) -> Iterator[Mapping[str, Any]]:
    if retrieval_captures < 1 or retrieval_captures > MAX_RETRIEVAL_CAPTURES:
        raise ValueError(
            f"retrieval_captures must be between 1 and {MAX_RETRIEVAL_CAPTURES}"
        )
    for question in questions:
        retrieval_k = max(k, adaptive_depth_k or k)
        capture_rows: list[dict[str, Any]] = []
        first_answer: tuple[
            list[str], list[ScoredChunk], bool, dict[str, Any], list[str], int,
            RetrievalDiagnostics,
    ] | None = None
        for capture in range(retrieval_captures):
            capture_started = time.perf_counter()
            doc_ids, hits, gap_warning, retrieval_diagnostics = retrieve_docs_with_diagnostics(
                store,
                embedder,
                question.question,
                k=k,
                candidate_k=candidate_k,
                sparse_backend=sparse_backend,
                sparse_encoder=sparse_encoder,
                reranker=reranker,
                gap_threshold=gap_threshold,
            )
            document_k, confidence_value, depth_expanded = adaptive_depth_choice(
                hits,
                base_k=k,
                expanded_k=retrieval_k,
                feature=adaptive_depth_feature,
                threshold=adaptive_depth_threshold,
            )
            if depth_expanded:
                doc_ids, hits, gap_warning, expanded_diagnostics = retrieve_docs_with_diagnostics(
                    store,
                    embedder,
                    question.question,
                    k=retrieval_k,
                    candidate_k=candidate_k,
                    sparse_backend=sparse_backend,
                    sparse_encoder=sparse_encoder,
                    reranker=reranker,
                    gap_threshold=gap_threshold,
                )
                retrieval_diagnostics = _combine_retrieval_diagnostics(
                    retrieval_diagnostics,
                    expanded_diagnostics,
                )
            selected_document_ids = _doc_ids_from_hits(hits, k=document_k)
            reasoning_doc_ids, reasoning_hits, reasoning_diagnostics = expand_retrieval_hits(
                question,
                store,
                embedder,
                initial_hits=hits,
                initial_gap_warning=gap_warning,
                k=document_k,
                candidate_k=candidate_k,
                sparse_backend=sparse_backend,
                sparse_encoder=sparse_encoder,
                reranker=reranker,
                gap_threshold=gap_threshold,
                arm=reasoning_arm,
                provider=expansion_provider,
                expansion_cache=expansion_cache,
            )
            if first_answer is None:
                first_answer = (
                    reasoning_doc_ids,
                    reasoning_hits,
                    gap_warning,
                    reasoning_diagnostics,
                    selected_document_ids,
                    len(hits),
                    retrieval_diagnostics,
                )
            stage_ms = dict(retrieval_diagnostics.stage_ms)
            retrieval_passes = 1 + int(depth_expanded)
            calls = {
                "embedding": retrieval_passes,
                "lexical": retrieval_passes * int(sparse_backend in ("lexical", "both")),
                "splade": retrieval_passes * int(sparse_backend in ("splade", "both")),
                "reranker": retrieval_passes * int(retrieval_diagnostics.reranking_ran),
            }
            capture_rows.append(
                {
                    "capture": capture + 1,
                    "initial_document_ids": selected_document_ids,
                    "initial_hit_count": len(hits),
                    "document_ids": reasoning_doc_ids,
                    "adaptive_depth": {
                        "feature": adaptive_depth_feature,
                        "threshold": adaptive_depth_threshold,
                        "value": confidence_value,
                        "expanded": depth_expanded,
                        "selected_k": document_k,
                    },
                    "expanded": bool(reasoning_diagnostics.get("expanded")),
                    "latency_ms": round((time.perf_counter() - capture_started) * 1000.0, 3),
                    "stage_ms": stage_ms,
                    "calls": calls,
                }
            )
        assert first_answer is not None
        (
            reasoning_doc_ids,
            reasoning_hits,
            gap_warning,
            reasoning_diagnostics,
            initial_document_ids,
            initial_hit_count,
            retrieval_diagnostics,
        ) = first_answer
        capture_diagnostics = retrieval_capture_summary(capture_rows)
        if mode == "openrouter":
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY is required for --answer-mode openrouter")
            answer_hits_for_model = answer_hits(
                reasoning_hits,
                reasoning_doc_ids,
                question_type=_question_type(question),
                answer_policy=answer_policy,
            )
            answer = generated_answer(
                question.question,
                answer_hits_for_model,
                model=model,
                api_key=api_key,
                max_chars=max_chars,
                question_type=_text(question.raw.get("question_type")),
                answer_policy=answer_policy,
            )
        else:
            answer = extractive_answer(question.question, reasoning_hits, max_chars=max_chars)
        yield {
            "question_id": question.question_id,
            "answer": answer,
            "document_ids": reasoning_doc_ids,
            "_diagnostics": {
                "gap_warning": gap_warning,
                "initial_document_ids": initial_document_ids,
                "initial_hit_count": initial_hit_count,
                "adaptive_depth": {
                    "feature": adaptive_depth_feature,
                    "threshold": adaptive_depth_threshold,
                    "k": adaptive_depth_k,
                },
                "reasoning": reasoning_diagnostics,
                "answer_policy": answer_policy,
                "answer_context_document_ids": [
                    str(hit.chunk.metadata.get("doc_id") or hit.chunk.source)
                    for hit in answer_hits_for_model
                ] if mode == "openrouter" else reasoning_doc_ids,
                "captures": capture_diagnostics,
                "retrieval_diagnostics": {
                    "embedding_profile": retrieval_diagnostics.embedding_profile,
                    "retrieval_profile": retrieval_diagnostics.retrieval_profile,
                    "index_generation": retrieval_diagnostics.index_generation,
                    "candidate_pool_size": retrieval_diagnostics.candidate_pool_size,
                    "reranking_ran": retrieval_diagnostics.reranking_ran,
                    "stage_ms": dict(retrieval_diagnostics.stage_ms),
                },
            },
        }


def public_answer_rows(rows: Iterable[Mapping[str, Any]]) -> Iterator[Mapping[str, Any]]:
    for row in rows:
        yield {key: value for key, value in row.items() if not key.startswith("_")}


def _parse_int_list(value: str) -> list[int]:
    items = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not items:
        raise ValueError("expected at least one integer")
    if any(item < 1 for item in items):
        raise ValueError("k values must be >= 1")
    return sorted(set(items))


def _parse_float_list(value: str) -> list[float]:
    items = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not items:
        raise ValueError("expected at least one float")
    return sorted(set(items))


def _doc_ids_from_hits(hits: Sequence[ScoredChunk], *, k: int) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        doc_id = str(hit.chunk.metadata.get("doc_id") or hit.chunk.source).strip()
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            ids.append(doc_id)
        if len(ids) >= k:
            break
    return ids


ADAPTIVE_DEPTH_FEATURES = frozenset({"max_dense_score", "eighth_hit_dense_score"})


def adaptive_depth_choice(
    hits: Sequence[ScoredChunk],
    *,
    base_k: int,
    expanded_k: int,
    feature: str | None,
    threshold: float | None,
) -> tuple[int, float | None, bool]:
    """Choose a document depth from non-gold dense-score confidence."""

    if feature is None or threshold is None:
        return base_k, None, False
    if feature not in ADAPTIVE_DEPTH_FEATURES:
        raise ValueError(f"unknown adaptive depth feature: {feature}")
    scores = [float(hit.score) for hit in hits if isinstance(hit.score, (int, float))]
    if feature == "max_dense_score":
        value = max(scores) if scores else None
    else:
        value = float(hits[7].score) if len(hits) >= 8 else None
    expanded = value is not None and value < threshold
    return (expanded_k if expanded else base_k), value, expanded


def _expected_docs(question: EnterpriseQuestion) -> set[str]:
    expected_raw = question.raw.get("expected_doc_ids")
    if not isinstance(expected_raw, list):
        return set()
    return {str(value) for value in expected_raw if str(value).strip()}


def _question_type(question: EnterpriseQuestion) -> str:
    return _text(question.raw.get("question_type")).lower()


def best_dense_score(store: PgVectorStore, embedder: Embedder, question: str) -> float | None:
    qvec = embed_query(embedder, question)
    hits = store.query_dense(qvec, k=1)
    return hits[0].score if hits else None


def retrieval_calibration(
    questions: Sequence[EnterpriseQuestion],
    store: PgVectorStore,
    embedder: Embedder,
    *,
    k_values: Sequence[int],
    threshold_values: Sequence[float],
    candidate_k: int,
    sparse_backend: str,
    sparse_encoder: object | None,
    reranker: object | None,
) -> dict[str, Any]:
    max_k = max(k_values)
    rows: list[dict[str, Any]] = []
    for question in questions:
        _, hits, _ = retrieve_docs(
            store,
            embedder,
            question.question,
            k=max_k,
            candidate_k=candidate_k,
            sparse_backend=sparse_backend,
            sparse_encoder=sparse_encoder,
            reranker=reranker,
            gap_threshold=DEFAULT_GAP_THRESHOLD,
        )
        rows.append(
            {
                "question_id": question.question_id,
                "question_type": _question_type(question),
                "expected_docs": sorted(_expected_docs(question)),
                "best_dense_score": best_dense_score(store, embedder, question.question),
                "doc_ids_by_k": {str(k): _doc_ids_from_hits(hits, k=k) for k in k_values},
            }
        )

    k_metrics: dict[str, Any] = {}
    for k in k_values:
        scored = 0
        total_expected = 0
        total_hit = 0
        exact = 0
        extra = 0
        no_doc_questions = 0
        no_doc_returned_docs = 0
        for row in rows:
            expected = set(row["expected_docs"])
            predicted = set(row["doc_ids_by_k"][str(k)])
            if expected:
                scored += 1
                total_expected += len(expected)
                hit_count = len(expected & predicted)
                total_hit += hit_count
                exact += int(hit_count == len(expected))
                extra += len(predicted - expected)
            elif row["question_type"] == "info_not_found":
                no_doc_questions += 1
                no_doc_returned_docs += len(predicted)
        k_metrics[str(k)] = {
            "questions_with_expected_docs": scored,
            "document_recall": (total_hit / total_expected) if total_expected else None,
            "exact_doc_set_coverage": (exact / scored) if scored else None,
            "invalid_extra_doc_upper_bound": extra,
            "mean_extra_docs_per_scored_question": (extra / scored) if scored else None,
            "info_not_found_questions": no_doc_questions,
            "mean_docs_returned_for_info_not_found": (
                no_doc_returned_docs / no_doc_questions if no_doc_questions else None
            ),
        }

    threshold_metrics: dict[str, Any] = {}
    answerable = [row for row in rows if row["expected_docs"]]
    unanswerable = [row for row in rows if row["question_type"] == "info_not_found"]
    for threshold in threshold_values:
        answerable_gaps = [
            row for row in answerable if row["best_dense_score"] is None or row["best_dense_score"] < threshold
        ]
        unanswerable_confident = [
            row
            for row in unanswerable
            if row["best_dense_score"] is not None and row["best_dense_score"] >= threshold
        ]
        threshold_metrics[str(threshold)] = {
            "answerable_questions": len(answerable),
            "false_gap_rate_on_answerable": (
                len(answerable_gaps) / len(answerable) if answerable else None
            ),
            "info_not_found_questions": len(unanswerable),
            "false_confident_rate_on_info_not_found": (
                len(unanswerable_confident) / len(unanswerable) if unanswerable else None
            ),
        }

    return {
        "k_values": list(k_values),
        "threshold_values": list(threshold_values),
        "candidate_k": candidate_k,
        "sparse_backend": sparse_backend,
        "reranker": type(reranker).__name__ if reranker is not None else None,
        "k_metrics": k_metrics,
        "threshold_metrics": threshold_metrics,
        "rows": rows,
    }


def retrieval_summary(
    questions: Sequence[EnterpriseQuestion], answer_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_id = {str(row["question_id"]): row for row in answer_rows}
    scored = 0
    total_expected = 0
    total_hit = 0
    exact = 0
    for question in questions:
        expected_raw = question.raw.get("expected_doc_ids")
        if not isinstance(expected_raw, list) or not expected_raw:
            continue
        row = by_id.get(question.question_id)
        if row is None:
            continue
        predicted_raw = row.get("document_ids")
        if not isinstance(predicted_raw, list):
            predicted_raw = []
        expected = {str(value) for value in expected_raw}
        predicted = {str(value) for value in predicted_raw}
        hits = len(expected & predicted)
        scored += 1
        total_expected += len(expected)
        total_hit += hits
        exact += int(hits == len(expected))
    return {
        "questions_with_expected_docs": scored,
        "document_recall": (total_hit / total_expected) if total_expected else None,
        "exact_doc_set_coverage": (exact / scored) if scored else None,
        "expected_docs": total_expected,
        "hit_expected_docs": total_hit,
    }


def reasoning_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    diagnostics = [
        row.get("_diagnostics", {}).get("reasoning", {})
        for row in rows
        if isinstance(row.get("_diagnostics"), Mapping)
        and isinstance(row.get("_diagnostics", {}).get("reasoning"), Mapping)
    ]
    if not diagnostics:
        return None
    expanded = sum(bool(item.get("expanded")) for item in diagnostics)
    fallbacks = sum(bool(item.get("fallback_reason")) for item in diagnostics)
    capture_blocks = [
        row.get("_diagnostics", {}).get("captures")
        for row in rows
        if isinstance(row.get("_diagnostics"), Mapping)
        and isinstance(row.get("_diagnostics", {}).get("captures"), Mapping)
    ]
    stable_captures = [
        bool(block.get("stable")) for block in capture_blocks if isinstance(block, Mapping)
    ]
    return {
        "rows": len(diagnostics),
        "expanded_rows": expanded,
        "expanded_rate": expanded / len(diagnostics),
        "fallback_rows": fallbacks,
        "fallback_rate": fallbacks / len(diagnostics),
        "passes_total": sum(int(item.get("passes", 1)) for item in diagnostics),
        "queries_total": sum(len(item.get("queries", ())) for item in diagnostics),
        "capture_stability_rate": (
            sum(stable_captures) / len(stable_captures) if stable_captures else None
        ),
        "model": next(
            (item.get("model") for item in diagnostics if item.get("model") is not None), None
        ),
    }


def reasoning_promotion_gate(metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    """Evaluate the cheap model promotion gate from independent judge metrics.

    The benchmark runner does not act as the answer judge. Missing judge signals therefore keep
    the gate pending, and no expensive reasoning model can be selected by omission.
    """

    if metrics is None:
        return {
            "status": "pending",
            "expensive_model_allowed": False,
            "missing": [
                "baseline_correctness",
                "candidate_correctness",
                "useful_expansion_precision",
                "stable_repeated_captures",
                "validation_failure_rate",
                "no_material_false_abstention",
                "info_not_found_correctness",
            ],
            "failed": [],
        }
    checks: dict[str, bool | None] = {}
    missing: list[str] = []
    failed: list[str] = []

    baseline = _optional_number(metrics.get("baseline_correctness"))
    candidate = _optional_number(metrics.get("candidate_correctness"))
    if baseline is None:
        missing.append("baseline_correctness")
    if candidate is None:
        missing.append("candidate_correctness")
    checks["correctness_delta_at_least_3_points"] = (
        None if baseline is None or candidate is None else candidate - baseline >= 0.03
    )

    for name in ("useful_expansion_precision", "stable_repeated_captures"):
        value = metrics.get(name)
        checks[name] = value if isinstance(value, bool) else None
        if checks[name] is None:
            missing.append(name)

    validation_failure_rate = _optional_number(metrics.get("validation_failure_rate"))
    checks["validation_failure_rate_at_most_5_percent"] = (
        None if validation_failure_rate is None else validation_failure_rate <= 0.05
    )
    if validation_failure_rate is None:
        missing.append("validation_failure_rate")

    false_abstention = metrics.get("no_material_false_abstention")
    checks["false_abstention_regression_absent"] = (
        false_abstention if isinstance(false_abstention, bool) else None
    )
    if checks["false_abstention_regression_absent"] is None:
        missing.append("no_material_false_abstention")

    info_not_found_correctness = _optional_number(metrics.get("info_not_found_correctness"))
    checks["info_not_found_correctness_at_least_90_percent"] = (
        None if info_not_found_correctness is None else info_not_found_correctness >= 0.90
    )
    if info_not_found_correctness is None:
        missing.append("info_not_found_correctness")

    failed.extend(name for name, passed in checks.items() if passed is False)
    status = "eligible" if not missing and not failed else "blocked" if failed else "pending"
    return {
        "status": status,
        "expensive_model_allowed": status == "eligible",
        "checks": checks,
        "missing": missing,
        "failed": failed,
        "thresholds": {
            "correctness_delta": 0.03,
            "validation_failure_rate": 0.05,
            "false_abstention_regression": "independent_judge_decision",
            "info_not_found_correctness": 0.90,
        },
    }


def _optional_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def load_promotion_metrics(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return load_published_artifact(path)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _dsn(args: argparse.Namespace) -> str:
    dsn = (
        args.dsn
        or os.environ.get("RECALL_DSN")
        or os.environ.get("RECALL_SERVING_DSN")
        or DEFAULT_DSN
    )
    return str(dsn)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.enterprise_rag")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--documents", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dsn")
    parser.add_argument("--pool-size", type=int)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--tenant", default=DEFAULT_TENANT)
    parser.add_argument("--embedder", default="fastembed")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--gap-threshold", type=float, default=DEFAULT_GAP_THRESHOLD)
    parser.add_argument("--batch-chunks", type=int, default=DEFAULT_BATCH_CHUNKS)
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=DEFAULT_CHUNK_CHARS,
        help=(
            "target characters per indexed chunk. EnterpriseRAG scores document ids, so the "
            "launch preset uses near-document chunks to avoid turning a 512k document corpus "
            "into millions of tiny vectors."
        ),
    )
    parser.add_argument(
        "--reasoning-cache",
        type=Path,
        help="JSON cache for cheap model expansion proposals, enabling replay without model calls",
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        help="SQLite cache for query embeddings, enabling repeated retrieval captures without re-embedding",
    )
    parser.add_argument(
        "--promotion-metrics",
        type=Path,
        help="JSON metrics from the independent judge used to evaluate cheap model promotion",
    )
    parser.add_argument(
        "--retrieval-captures",
        type=int,
        default=1,
        help="repeat each retrieval capture to quantify embedding and ranking variance",
    )
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--sparse-backend", choices=sorted(SPARSE_BACKENDS), default="lexical")
    parser.add_argument("--splade-model", default=DEFAULT_SPLADE_MODEL)
    parser.add_argument("--splade-batch-size", type=int, default=32)
    parser.add_argument("--sparse-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--backfill-splade", action="store_true")
    parser.add_argument("--accept-noncommercial-splade-license", action="store_true")
    parser.add_argument("--reranker", default="none")
    parser.add_argument(
        "--rerank-document-chars",
        type=int,
        default=DEFAULT_RERANK_DOCUMENT_CHARS,
        help="maximum characters per candidate document sent to the reranker",
    )
    parser.add_argument(
        "--reranker-rank-weight",
        type=float,
        default=1.0,
        help="Voyage rank contribution; 1.0 preserves pure reranking, lower values blend hybrid rank",
    )
    parser.add_argument(
        "--adaptive-depth-feature",
        choices=sorted(ADAPTIVE_DEPTH_FEATURES),
        help="non-gold confidence feature used to widen the submitted document depth",
    )
    parser.add_argument(
        "--adaptive-depth-threshold",
        type=float,
        help="expand to --adaptive-depth-k when the selected feature is below this value",
    )
    parser.add_argument(
        "--adaptive-depth-k",
        type=int,
        default=12,
        help="expanded document depth for the selective-depth arm",
    )
    parser.add_argument("--limit-docs", type=int)
    parser.add_argument("--limit-questions", type=int)
    parser.add_argument(
        "--question-types",
        help="comma-separated official question_type values to run, preserving release order",
    )
    parser.add_argument(
        "--question-ids-file",
        type=Path,
        help="newline-delimited question ids to run, preserving the file order from the release",
    )
    parser.add_argument("--reset-index", action="store_true")
    parser.add_argument(
        "--skip-indexed-sources",
        action="store_true",
        help="resume indexing by skipping documents whose source ids already exist in the table",
    )
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--answer-mode", choices=("extractive", "openrouter"), default="extractive")
    parser.add_argument(
        "--answer-policy",
        choices=("baseline", "category_aware"),
        default="baseline",
        help="answer-side evidence policy; category_aware is an opt-in experiment",
    )
    parser.add_argument(
        "--reasoning-arm",
        choices=("none", "depth", "cheap", "closed_loop"),
        default="none",
        help=(
            "retrieval expansion arm. cheap and closed_loop require "
            "RECALL_REASONING_EXPANSION=1, RECALL_REASONING_EXPANSION_MODEL, and "
            "RECALL_REASONING_API_KEY; the answer model remains --model"
        ),
    )
    parser.add_argument("--model", default=os.environ.get("ENTERPRISE_RAG_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--calibrate-retrieval-out", type=Path)
    parser.add_argument("--calibrate-k-values", default="3,5,8,10,12,15")
    parser.add_argument("--calibrate-thresholds", default="0.35,0.4,0.45,0.5,0.55,0.6")
    parser.add_argument(
        "--top-config",
        action="store_true",
        help=(
            "Apply the EnterpriseRAG launch preset: Voyage embeddings, lexical plus SPLADE, "
            "SPLADE backfill, Voyage reranker, gpt-4o answering, k=8, candidate_k=200, "
            "and a larger answer context."
        ),
    )
    return parser


def apply_top_config(args: argparse.Namespace) -> None:
    if not args.top_config:
        return
    args.embedder = TOP_CONFIG_EMBEDDER
    args.k = TOP_CONFIG_K
    args.candidate_k = TOP_CONFIG_CANDIDATE_K
    args.batch_chunks = TOP_CONFIG_BATCH_CHUNKS
    args.sparse_backend = TOP_CONFIG_SPARSE_BACKEND
    args.backfill_splade = True
    args.reranker = TOP_CONFIG_RERANKER
    args.answer_mode = "openrouter"
    args.model = DEFAULT_MODEL
    args.max_context_chars = TOP_CONFIG_MAX_CHARS
    args.chunk_chars = TOP_CONFIG_CHUNK_CHARS
    args.chunk_overlap = TOP_CONFIG_CHUNK_OVERLAP


def main(argv: Iterable[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    apply_top_config(args)
    if args.retrieval_captures < 1 or args.retrieval_captures > MAX_RETRIEVAL_CAPTURES:
        raise ValueError(
            f"--retrieval-captures must be between 1 and {MAX_RETRIEVAL_CAPTURES}"
        )
    started = datetime.now(UTC).isoformat()
    t0 = time.perf_counter()
    embedder = resolve_embedder(args.embedder)
    embedding_cache = EmbeddingCache(args.embedding_cache) if args.embedding_cache else None
    retrieval_embedder: Embedder = (
        QueryCachedEmbedder(embedder, embedding_cache) if embedding_cache is not None else embedder
    )
    if not 0.0 <= args.reranker_rank_weight <= 1.0:
        raise ValueError("--reranker-rank-weight must be between 0 and 1")
    if (args.adaptive_depth_feature is None) != (args.adaptive_depth_threshold is None):
        raise ValueError(
            "--adaptive-depth-feature and --adaptive-depth-threshold must be provided together"
        )
    if args.adaptive_depth_threshold is not None and not 0.0 <= args.adaptive_depth_threshold <= 1.0:
        raise ValueError("--adaptive-depth-threshold must be between 0 and 1")
    if args.adaptive_depth_k < args.k:
        raise ValueError("--adaptive-depth-k must be at least --k")
    reranker = build_reranker(
        args.reranker,
        max_document_chars=args.rerank_document_chars,
        rank_weight=args.reranker_rank_weight,
    )
    expansion_provider = (
        resolve_expansion_provider()
        if args.reasoning_arm in ("cheap", "closed_loop")
        else None
    )
    expansion_cache = load_reasoning_cache(args.reasoning_cache)
    if args.retrieval_captures > 1 and args.reasoning_arm in ("cheap", "closed_loop"):
        expansion_cache = expansion_cache or {}
    promotion_metrics = load_promotion_metrics(args.promotion_metrics)
    sparse_encoder: object | None = None
    question_types = (
        {item.strip().lower() for item in args.question_types.split(",") if item.strip()}
        if args.question_types
        else None
    )
    question_ids = (
        {
            line.strip()
            for line in args.question_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        if args.question_ids_file
        else None
    )
    questions = load_questions(
        args.questions,
        limit=args.limit_questions,
        question_types=question_types,
        question_ids=question_ids,
    )
    all_questions = list(questions)
    existing_answer_rows: list[dict[str, Any]] = []
    if args.resume:
        existing_answer_rows = read_answer_rows(args.out)
        existing_question_ids = {
            str(row.get("question_id")) for row in existing_answer_rows if row.get("question_id")
        }
        questions = [question for question in questions if question.question_id not in existing_question_ids]
        print(
            f"resume loaded existing_rows={len(existing_answer_rows)} "
            f"remaining_questions={len(questions)}",
            flush=True,
        )
    dsn = _dsn(args)
    stats = {"documents": 0, "chunks": 0}
    sparse_stats: dict[str, Any] | None = None
    retrieval_metrics: dict[str, Any] | None = None
    reasoning_metrics: dict[str, Any] | None = None
    calibration: dict[str, Any] | None = None
    new_answer_rows: list[Mapping[str, Any]] = []
    with PgVectorStore(
        dsn,
        dim=embedder.dim,
        table=args.table,
        tenant=args.tenant,
        pool_size=args.pool_size,
    ) as store:
        store.ensure_schema()
        if not args.skip_index:
            docs = load_documents(args.documents, limit=args.limit_docs)
            stats = index_documents(
                store,
                embedder,
                docs,
                batch_chunks=args.batch_chunks,
                chunk_chars=args.chunk_chars,
                chunk_overlap=args.chunk_overlap,
                reset=args.reset_index,
                skip_indexed_sources=args.skip_indexed_sources,
            )
        if args.backfill_splade:
            sparse_stats = backfill_sparse(
                store,
                model=args.splade_model,
                accept_noncommercial_license=args.accept_noncommercial_splade_license,
                batch_size=args.splade_batch_size,
                device=args.sparse_device,
            )
        if args.index_only:
            written = 0
        elif args.calibrate_retrieval_out:
            sparse_encoder = build_sparse_encoder(
                args.sparse_backend,
                model=args.splade_model,
                accept_noncommercial_license=args.accept_noncommercial_splade_license,
                device=args.sparse_device,
            )
            calibration = retrieval_calibration(
                questions,
                store,
                retrieval_embedder,
                k_values=_parse_int_list(args.calibrate_k_values),
                threshold_values=_parse_float_list(args.calibrate_thresholds),
                candidate_k=args.candidate_k,
                sparse_backend=args.sparse_backend,
                sparse_encoder=sparse_encoder,
                reranker=reranker,
            )
            args.calibrate_retrieval_out.parent.mkdir(parents=True, exist_ok=True)
            args.calibrate_retrieval_out.write_text(
                json.dumps(calibration, indent=2), encoding="utf-8"
            )
            written = 0
        else:
            sparse_encoder = build_sparse_encoder(
                args.sparse_backend,
                model=args.splade_model,
                accept_noncommercial_license=args.accept_noncommercial_splade_license,
                device=args.sparse_device,
            )
            written, new_answer_rows = write_answers_stream(
                args.out,
                _answers(
                    questions,
                    store,
                    retrieval_embedder,
                    k=args.k,
                    candidate_k=args.candidate_k,
                    mode=args.answer_mode,
                    model=args.model,
                    api_key=os.environ.get("OPENROUTER_API_KEY"),
                    max_chars=args.max_context_chars,
                    sparse_backend=args.sparse_backend,
                    sparse_encoder=sparse_encoder,
                    reranker=reranker,
                    gap_threshold=args.gap_threshold,
                    reasoning_arm=args.reasoning_arm,
                    expansion_provider=expansion_provider,
                    expansion_cache=expansion_cache,
                    answer_policy=args.answer_policy,
                    retrieval_captures=args.retrieval_captures,
                    adaptive_depth_feature=args.adaptive_depth_feature,
                    adaptive_depth_threshold=args.adaptive_depth_threshold,
                    adaptive_depth_k=(
                        args.adaptive_depth_k if args.adaptive_depth_feature is not None else None
                    ),
                ),
                overwrite=args.overwrite,
                resume=args.resume,
            )
            answer_rows = [*existing_answer_rows, *new_answer_rows]
            # Gold document ids are intentionally excluded from the runtime object. Retrieval
            # quality is scored only after this answer file exists, by the posthoc evaluator.
            retrieval_metrics = None
            reasoning_metrics = reasoning_summary(answer_rows)
            write_reasoning_cache(args.reasoning_cache, expansion_cache)
    if embedding_cache is not None:
        embedding_cache.close()
    manifest = {
        "benchmark": "EnterpriseRAG-Bench",
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(),
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "recall_revision": git_revision(Path(__file__).resolve().parents[1]),
        "recall_version": package_version("recall-rag"),
        "questions": {
            "path": str(args.questions),
            "sha256": sha256_file(args.questions),
            "selected_count": len(all_questions),
            "question_types": sorted(question_types) if question_types else None,
            "question_ids_file": (
                str(args.question_ids_file) if args.question_ids_file else None
            ),
            "question_ids_file_sha256": (
                sha256_file(args.question_ids_file) if args.question_ids_file else None
            ),
        },
        "documents": [{"path": str(path), "sha256": sha256_file(path)} for path in args.documents],
        "index": {
            "table": args.table,
            "tenant": args.tenant,
            "pool_size": args.pool_size,
            "reset": args.reset_index,
            "skipped": args.skip_index,
            **stats,
        },
        "retrieval": {
            "embedder": args.embedder,
            "embedding_profile": embedding_profile_id(embedder),
            "embedding_cache": str(args.embedding_cache) if args.embedding_cache else None,
            "k": args.k,
            "candidate_k": args.candidate_k,
            "gap_threshold": args.gap_threshold,
            "sparse_backend": args.sparse_backend,
            "splade_model": args.splade_model if args.sparse_backend in ("splade", "both") else None,
            "splade_backfill": sparse_stats,
            "reranker": args.reranker,
            "reranker_rank_weight": args.reranker_rank_weight,
            "adaptive_depth_feature": args.adaptive_depth_feature,
            "adaptive_depth_threshold": args.adaptive_depth_threshold,
            "adaptive_depth_k": (
                args.adaptive_depth_k if args.adaptive_depth_feature is not None else None
            ),
            "retrieval_captures": args.retrieval_captures,
            "gold_fields_used_at_runtime": False,
            "posthoc_metrics": "scripts/enterprise_rag_posthoc_metrics.py",
        },
        "answering": {
            "mode": args.answer_mode,
            "model": args.model if args.answer_mode == "openrouter" else None,
            "max_context_chars": args.max_context_chars,
            "answer_policy": args.answer_policy,
        },
        "reasoning": {
            "arm": args.reasoning_arm,
            "provider": (
                expansion_provider.provider_metadata().to_dict()
                if expansion_provider is not None
                else None
            ),
            "summary": reasoning_metrics,
            "cache": str(args.reasoning_cache) if args.reasoning_cache else None,
            "promotion": reasoning_promotion_gate(promotion_metrics),
        },
        "outputs": {
            "answers": str(args.out),
            "rows": len(existing_answer_rows) + written,
            "new_rows": written,
            "resumed_rows": len(existing_answer_rows),
        },
        "retrieval_metrics": retrieval_metrics,
        "runtime_telemetry": runtime_telemetry(
            [*existing_answer_rows, *new_answer_rows], answer_mode=args.answer_mode
        ),
        "calibration": {"path": str(args.calibrate_retrieval_out)} if calibration else None,
    }
    manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {written} answer row(s) to {args.out}")
    if args.calibrate_retrieval_out:
        print(f"wrote retrieval calibration to {args.calibrate_retrieval_out}")
    print(f"wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
