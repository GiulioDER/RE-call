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
import json
import os
import re
import subprocess
import time
import zipfile
from io import TextIOWrapper
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

from recall._env import load_dotenv
from recall.embeddings import Embedder, embed_query, embedding_profile_id, resolve_embedder
from recall.guards import DEFAULT_GAP_THRESHOLD
from recall.index import chunk_text
from recall.retriever import SPARSE_BACKENDS, HybridRetriever
from recall.store import PgVectorStore
from recall.types import Chunk, ScoredChunk, TrustedResult

DEFAULT_TABLE = "bench_enterprise_rag_chunks"
DEFAULT_TENANT = "enterprise-rag"
DEFAULT_K = 8
DEFAULT_CANDIDATE_K = 80
DEFAULT_MAX_CHARS = 3500
DEFAULT_BATCH_CHUNKS = 256
DEFAULT_CHUNK_CHARS = 800
DEFAULT_CHUNK_OVERLAP = 80
DEFAULT_RERANK_DOCUMENT_CHARS = 4_000
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


def load_questions(path: Path, *, limit: int | None = None) -> list[EnterpriseQuestion]:
    questions: list[EnterpriseQuestion] = []
    for row in _json_rows(path):
        questions.append(
            EnterpriseQuestion(
                question_id=_required_text(row, "question_id", "id"),
                question=_required_text(row, "question", "query"),
                raw=row,
            )
        )
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


def build_reranker(name: str, *, max_document_chars: int | None = None) -> object | None:
    if not name or name == "none":
        return None
    if name.startswith("voyage:"):
        from benchmarks.voyage_rerank import VoyageReranker

        return VoyageReranker(
            model=name[len("voyage:"):],
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
    return ids, result.hits, result.gap_warning


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


#: What a library-arm row emits when the evidence boundary declines to answer. Library authored,
#: fixed, and never interpolated, so no corpus byte reaches the submission through this path. It
#: exists because the submission format needs a string and `insufficient_evidence=true` yields
#: `answer=None`; the count is reported separately so an abstention is never mistaken for a
#: wrong answer.
LIBRARY_ABSTENTION = (
    "The retrieved enterprise documents do not contain enough information to answer this question."
)


#: Ceiling on rows the evidence boundary refused, as a FRACTION of library-arm rows. Above it
#: the arm is broken rather than selective, and a mean over the rows that happened to parse is a
#: mean over a different population. Pre-registered as B3 in
#: `results/enterprise_rag/PREREGISTRATION-library-parity.md`.
LIBRARY_MAX_VALIDATION_FAILURE_RATE = 0.05


def _new_library_tally() -> dict[str, Any]:
    return {
        "rows": 0,
        "validation_failures": 0,
        "insufficient_evidence": 0,
        "generator_invoked": 0,
        "citations_total": 0,
        "answered_rows": 0,
        "evidence_items": {},
        "validation_error_samples": [],
    }


def _tally_library_row(tally: dict[str, Any], diagnostics: Mapping[str, Any]) -> None:
    """Accumulate what the library arm actually did, so it can reach the manifest.

    Without this the pre-registered B3 and B6 are unmeasurable and J3 and J5 have nothing to read:
    `write_answers_stream` drops every `_`-prefixed key before writing, so the per-row diagnostics
    existed only in memory and died with the process.
    """
    tally["rows"] += 1
    if diagnostics.get("validation_error"):
        tally["validation_failures"] += 1
        if len(tally["validation_error_samples"]) < 10:
            tally["validation_error_samples"].append(str(diagnostics["validation_error"]))
        return
    if diagnostics.get("generator_invoked"):
        tally["generator_invoked"] += 1
    if diagnostics.get("insufficient_evidence"):
        tally["insufficient_evidence"] += 1
    else:
        tally["answered_rows"] += 1
        tally["citations_total"] += int(diagnostics.get("citations") or 0)
    items = diagnostics.get("evidence_items")
    if items is not None:
        key = str(items)
        tally["evidence_items"][key] = tally["evidence_items"].get(key, 0) + 1


def library_tally_summary(tally: Mapping[str, Any]) -> dict[str, Any]:
    """The manifest block, with the rates the pre-registration predicts computed here."""
    rows = int(tally.get("rows") or 0)
    answered = int(tally.get("answered_rows") or 0)
    failures = int(tally.get("validation_failures") or 0)
    return {
        **{key: tally[key] for key in tally if key != "validation_error_samples"},
        "validation_failure_rate": round(failures / rows, 6) if rows else None,
        "insufficient_evidence_rate": (
            round(int(tally.get("insufficient_evidence") or 0) / rows, 6) if rows else None
        ),
        "mean_citations_per_answered_row": (
            round(int(tally.get("citations_total") or 0) / answered, 4) if answered else None
        ),
        "validation_error_samples": tally.get("validation_error_samples", []),
    }


def refuse_a_broken_library_arm(tally: Mapping[str, Any]) -> None:
    """Raise `SystemExit` when the library arm failed too often to have measured anything.

    The pre-registered decision rule gives this precedence over every other row: above the
    ceiling the result is an APPARATUS FAILURE and no number may be published. Asserted here
    rather than left to a reader, because the alternative shape is the one this repository keeps
    finding: a run that wrote a full file of empty answers, printed a success line and exited 0.
    """
    rows = int(tally.get("rows") or 0)
    if not rows:
        return
    failures = int(tally.get("validation_failures") or 0)
    rate = failures / rows
    if rate > LIBRARY_MAX_VALIDATION_FAILURE_RATE:
        samples = "; ".join(tally.get("validation_error_samples", [])[:3])
        raise SystemExit(
            f"APPARATUS FAILURE: {failures} of {rows} library rows "
            f"({rate:.1%}) were refused by the evidence boundary, above the pre-registered "
            f"ceiling of {LIBRARY_MAX_VALIDATION_FAILURE_RATE:.0%}. Publish no number from this "
            f"run. First failures: {samples}"
        )


def _system_prompt_digest() -> str:
    """SHA-256 of the exact system prompt bytes this run used.

    A parity delta is only attributable while the prompt is fixed, and "the prompt is unchanged"
    is precisely the kind of claim that rots silently. `SYSTEM_PROMPT` is a module constant, so a
    later edit would move every arm's score with nothing in the artifact recording that anything
    moved. The digest is the record.
    """
    import hashlib

    from recall.evidence import SYSTEM_PROMPT

    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def trusted_result_from_hits(
    question: str,
    hits: Sequence[ScoredChunk],
    *,
    max_chars: int,
    gap_warning: bool = False,
) -> "TrustedResult":
    """Project retrieved hits into the library's trusted shape WITHOUT re-running the trust layer.

    ⚠️ This is a PARITY fixture, and the omission is the point. `recall.trust` applies a dense
    cosine floor, and `results/enterprise_rag/dense_floor_strat100.retrieval.json` measures at
    least 5.6% of questions below it, with `false_confident_rate_on_info_not_found = 1.0` at 0.50.
    Running it here would change abstention behaviour and the `info_not_found` category, which
    scores 100.0 and is a registered invariant. The parity arm changes the GENERATION leg and
    nothing else; measuring the trust layer is a separate arm with its own prediction.

    Every hit therefore enters as `verdict="ok"`, in retrieval order, with the same per-hit
    character budget the bespoke harness applies, so the evidence TEXT reaching the model is the
    same bytes in both arms and the delta cannot be a truncation difference.
    """
    from recall.types import Provenance, StalenessReport, TrustedHit, TrustedResult, Validity

    trusted: list[TrustedHit] = []
    used = 0
    for hit in hits:
        remaining = max_chars - used
        if remaining <= 0:
            break
        text = " ".join(hit.chunk.text.split())[:remaining]
        used += len(text)
        trusted.append(
            TrustedHit(
                chunk=Chunk(
                    id=hit.chunk.id,
                    source=hit.chunk.source,
                    text=text,
                    metadata=dict(hit.chunk.metadata),
                ),
                cosine=hit.score,
                confidence=hit.score,
                verdict="ok",
                provenance=Provenance(
                    source=hit.chunk.source,
                    file=str(hit.chunk.metadata.get("doc_id") or hit.chunk.source),
                    ord=None,
                    indexed_at=hit.indexed_at,
                ),
                validity=Validity(valid_from=None, valid_until=None, superseded_by=None),
            )
        )
    return TrustedResult(
        query=question,
        hits=trusted,
        abstained=not trusted,
        reason="no hits retrieved" if not trusted else "",
        gap_warning=gap_warning,
        staleness=StalenessReport(
            stale=False, newest_indexed_at=None, age=None, max_age=timedelta(days=3650)
        ),
    )


def library_answer(
    question: str,
    hits: Sequence[ScoredChunk],
    *,
    provider: Callable[[str, str], str],
    max_chars: int,
    max_items: int,
    gap_warning: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Answer through `recall.evidence`, the product's generation boundary.

    This is the whole point of the library arm: the submitted score has always measured a bespoke
    harness that imports neither `recall.evidence` nor `recall.trust` nor `recall.reasoning` and
    writes its own prompt. Here the prompt is `SYSTEM_PROMPT`, the framing is the delimited
    evidence payload, and the reply clears `validate_answer` before it counts.

    Returns the answer text and a diagnostics mapping. A validation failure is RECORDED rather
    than raised, because one malformed reply must not discard 499 paid rows; the caller refuses
    the whole run if the failure rate says the arm is broken rather than selective.
    """
    from recall.evidence import EvidencePolicy, EvidenceValidationError, generate_from_evidence

    result = trusted_result_from_hits(
        question, hits, max_chars=max_chars, gap_warning=gap_warning
    )
    try:
        generated = generate_from_evidence(
            result, provider, EvidencePolicy(max_items=max_items)
        )
    except EvidenceValidationError as exc:
        return "", {"validation_error": str(exc), "generator_invoked": True}

    envelope = generated.envelope
    diagnostics: dict[str, Any] = {
        "generator_invoked": generated.generator_invoked,
        "evidence_items": len(generated.evidence.items),
        "citations": len(envelope.citations),
        "insufficient_evidence": envelope.insufficient_evidence,
        "citations_normalized": generated.citations_normalized,
    }
    if envelope.insufficient_evidence or not envelope.answer:
        return LIBRARY_ABSTENTION, diagnostics
    return envelope.answer, diagnostics


def generated_answer(
    question: str,
    hits: Sequence[ScoredChunk],
    *,
    model: str,
    api_key: str,
    max_chars: int,
    question_type: str | None = None,
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
    type_line = f"Question type: {question_type}\n" if question_type else ""
    user = f"{type_line}Question:\n{question}\n\nDocuments:\n\n" + "\n\n".join(evidence)
    return OpenRouterLLM(model=model, api_key=api_key).complete(system, user)


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
    answer_provider: Callable[[str, str], str] | None = None,
    evidence_max_items: int | None = None,
    tally: dict[str, Any] | None = None,
) -> Iterator[Mapping[str, Any]]:
    for question in questions:
        doc_ids, hits, gap_warning = retrieve_docs(
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
        diagnostics: dict[str, Any] = {"gap_warning": gap_warning}
        if mode == "openrouter":
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY is required for --answer-mode openrouter")
            answer = generated_answer(
                question.question,
                hits,
                model=model,
                api_key=api_key,
                max_chars=max_chars,
                question_type=_text(question.raw.get("question_type")),
            )
        elif mode == "library":
            if answer_provider is None:
                raise RuntimeError(
                    "--answer-mode library needs an answer provider. Set RECALL_REASONING=1 "
                    "with RECALL_REASONING_API_KEY, or pass one in."
                )
            # ⚠️ `question_type` is NOT passed. The bespoke arm puts it in the prompt; the library
            # arm has no channel for it, because `render_evidence_prompt` takes the bundle alone.
            # Recorded here rather than quietly dropped: it is one of the enumerated differences
            # the parity delta is decomposed against.
            answer, library_diagnostics = library_answer(
                question.question,
                hits,
                provider=answer_provider,
                max_chars=max_chars,
                max_items=evidence_max_items if evidence_max_items is not None else k,
                gap_warning=gap_warning,
            )
            diagnostics.update(library_diagnostics)
            if tally is not None:
                _tally_library_row(tally, library_diagnostics)
        else:
            answer = extractive_answer(question.question, hits, max_chars=max_chars)
        yield {
            "question_id": question.question_id,
            "answer": answer,
            "document_ids": doc_ids,
            "_diagnostics": diagnostics,
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
    parser.add_argument("--limit-docs", type=int)
    parser.add_argument("--limit-questions", type=int)
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
    parser.add_argument(
        "--answer-mode",
        choices=("extractive", "openrouter", "library"),
        default="extractive",
        help="library routes generation through recall.evidence, the product boundary, "
             "instead of this file's bespoke prompt",
    )
    parser.add_argument(
        "--evidence-max-items",
        type=int,
        default=None,
        help="EvidencePolicy.max_items for --answer-mode library. Defaults to --k, which "
             "HOLDS THE ITEM COUNT CONSTANT against the bespoke arm; the shipped library "
             "default is 5 and using it would confound the prompt delta with a context "
             "size delta",
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
    if args.answer_mode not in ("extractive", "openrouter"):
        # `--top-config` pins the SUBMITTED configuration, whose answering arm is `openrouter`.
        # Overwriting an explicit `--answer-mode library` silently charged the operator for a
        # full run of the arm they were trying to replace, and the manifest then recorded
        # `"mode": "openrouter"` with no hint that a different one had been asked for.
        raise SystemExit(
            f"--top-config pins --answer-mode openrouter, but --answer-mode "
            f"{args.answer_mode} was given. Run the library arm without --top-config, "
            f"passing the top-config retrieval flags explicitly."
        )
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
    started = datetime.now(UTC).isoformat()
    t0 = time.perf_counter()
    embedder = resolve_embedder(args.embedder)
    reranker = build_reranker(args.reranker, max_document_chars=args.rerank_document_chars)
    answer_provider = None
    library_tally = _new_library_tally()
    sparse_encoder: object | None = None
    questions = load_questions(args.questions, limit=args.limit_questions)
    all_questions = list(questions)
    existing_answer_rows: list[dict[str, Any]] = []
    if args.resume:
        existing_answer_rows = read_answer_rows(args.out)
        # An empty answer is a FAILED row, not a finished one. `library_answer` returns ""
        # when the evidence boundary refused the reply, and treating that as done baked the
        # failure permanently into the submission: resuming skipped it and no rerun could
        # repair it without hand-editing the file.
        existing_question_ids = {
            str(row.get("question_id"))
            for row in existing_answer_rows
            if row.get("question_id") and str(row.get("answer") or "").strip()
        }
        retryable = len(existing_answer_rows) - len(existing_question_ids)
        existing_answer_rows = [
            row
            for row in existing_answer_rows
            if str(row.get("question_id")) in existing_question_ids
        ]
        if retryable:
            print(f"resume dropping {retryable} empty-answer row(s) for retry", flush=True)
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
    calibration: dict[str, Any] | None = None
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
                embedder,
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
            if args.answer_mode == "library":
                # Built HERE, not at startup: `--index-only` and `--calibrate-retrieval-out`
                # return before this point, and constructing the provider up there made an
                # indexing run refuse without an API key it would never have used.
                #
                # ⚠️ `RECALL_REASONING_MODEL` is overridden with `--model`. Without this the
                # library arm answered with `DEFAULT_ANSWER_MODEL` while the baseline it is
                # compared against used `--model`, so the "prompt delta" would have been a
                # prompt delta PLUS a model swap. That is precisely the confound this whole
                # parity exercise exists to avoid, and it was silent: the manifest recorded
                # `"model": null` for library runs.
                from recall.answer_provider import answer_provider_from_env

                answer_provider = answer_provider_from_env(
                    {**os.environ, "RECALL_REASONING_MODEL": args.model}
                )
            written, new_answer_rows = write_answers_stream(
                args.out,
                _answers(
                    questions,
                    store,
                    embedder,
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
                    answer_provider=answer_provider,
                    evidence_max_items=args.evidence_max_items,
                    tally=library_tally,
                ),
                overwrite=args.overwrite,
                resume=args.resume,
            )
            answer_rows = [*existing_answer_rows, *new_answer_rows]
            retrieval_metrics = retrieval_summary(all_questions, answer_rows)
            # AFTER the rows are on disk, so a refused run keeps its evidence, and BEFORE the
            # manifest, so a refused run has no manifest to be mistaken for a result.
            refuse_a_broken_library_arm(library_tally)
    manifest = {
        "benchmark": "EnterpriseRAG-Bench",
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(),
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "recall_revision": git_revision(Path(__file__).resolve().parents[1]),
        "recall_version": package_version("recall-rag"),
        "questions": {"path": str(args.questions), "sha256": sha256_file(args.questions)},
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
            "k": args.k,
            "candidate_k": args.candidate_k,
            "gap_threshold": args.gap_threshold,
            "sparse_backend": args.sparse_backend,
            "splade_model": args.splade_model if args.sparse_backend in ("splade", "both") else None,
            "splade_backfill": sparse_stats,
            "reranker": args.reranker,
        },
        "answering": {
            "mode": args.answer_mode,
            "model": args.model if args.answer_mode in ("openrouter", "library") else None,
            "max_context_chars": args.max_context_chars,
            # The library arm's identity and the knobs the parity comparison holds constant.
            # Recorded because two committed EnterpriseRAG summaries carry `evaluator_options:
            # null` and an empty judge block, so their configuration survives only in a filename.
            "library": None if answer_provider is None else {
                "provider_id": answer_provider.provider_metadata().provider_id,
                "model_id": answer_provider.model_id,
                "model_revision": answer_provider.revision,
                "base_url": answer_provider.base_url,
                "temperature": answer_provider.temperature,
                "max_tokens": answer_provider.max_tokens,
                "evidence_max_items": args.evidence_max_items
                if args.evidence_max_items is not None else args.k,
                "system_prompt_sha256": _system_prompt_digest(),
                "tally": library_tally_summary(library_tally),
            },
        },
        "outputs": {
            "answers": str(args.out),
            "rows": len(existing_answer_rows) + written,
            "new_rows": written,
            "resumed_rows": len(existing_answer_rows),
        },
        "retrieval_metrics": retrieval_metrics,
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
