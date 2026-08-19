"""Run the RE call ATM retrieve, rerank, answer pipeline.

This driver is deliberately separate from ``atm_bench.py``.  That file measures retrieval only.
This file adds bounded answer generation and writes checkpointed artifacts which the official ATM
evaluator can consume without transforming the predictions.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import sys
import time
from pathlib import Path
from collections.abc import Iterator
from typing import Any

_SOURCE_ROOT = str(Path(__file__).resolve().parents[1])
if _SOURCE_ROOT not in sys.path:
    sys.path.insert(0, _SOURCE_ROOT)

import requests

from benchmarks.atm_bench import build_memory_items, git_revision, load_questions, sha256
from recall.embeddings import embed_passages, embedding_profile_id, resolve_embedder
from recall.rerank import reranker_from_name
from recall.retriever import HybridRetriever
from recall.store import PgVectorStore
from recall.types import Chunk


DEFAULT_DSN = "postgresql://recall:recall@localhost:5432/recall"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_EMBEDDER = "voyage:voyage-4-large"
DEFAULT_RERANKER = "voyage:rerank-2.5"
DEFAULT_MODEL = "deepseek/deepseek-v4-pro"
DEFAULT_EVIDENCE_CHARS = 8192
DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_CANDIDATE_K = 25
DEFAULT_RETRIEVAL_K = 10


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("id"):
                raise ValueError(f"invalid JSONL row at {path}:{line_number}")
            rows[str(row["id"])] = row
    return rows


def _compact_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _evidence_text(hits: list[dict[str, Any]], max_chars: int) -> str:
    parts: list[str] = []
    used = 0
    for hit in hits:
        evidence_id = str(hit["id"])
        text = _compact_text(hit.get("text"))
        part = f"[{evidence_id}] {text}"
        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(part) > remaining:
            part = part[:remaining]
        parts.append(part)
        used += len(part)
    return "\n\n".join(parts)


def _message_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        pieces: list[str] = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                pieces.append(item["text"])
        return "".join(pieces).strip()
    return ""


def _retryable_status(status: int) -> bool:
    return status == 429 or status >= 500


def generate_answer(
    *,
    question: str,
    qtype: str | None,
    evidence: str,
    model: str,
    base_url: str,
    api_key: str,
    reasoning_effort: str,
    max_output_tokens: int,
    max_attempts: int,
) -> tuple[str, dict[str, int], str | None]:
    system = (
        "Answer the ATM Bench question using only the retrieved memory evidence. "
        "Return the shortest complete answer that preserves every requested fact, number, date, "
        "time, name, location, condition, and list member. Resolve conflicts by preferring the "
        "most direct and specific evidence. If the evidence does not support an answer, say that "
        "the available memory does not contain enough information. Do not invent facts. "
        "Reason silently and return only the final answer, without analysis or a preamble."
    )
    type_line = f"Question type: {qtype}\n" if qtype else ""
    user = f"{type_line}Question:\n{question}\n\nRetrieved memory:\n{evidence}"
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_output_tokens,
        "reasoning": {"effort": reasoning_effort},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: str | None = None
    for attempt in range(max_attempts):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            if _retryable_status(response.status_code):
                last_error = f"provider status {response.status_code}"
                if attempt + 1 < max_attempts:
                    time.sleep(min(30.0, 2.0**attempt))
                    continue
            response.raise_for_status()
            body = response.json()
            choices = body.get("choices") if isinstance(body, dict) else None
            if not isinstance(choices, list) or not choices:
                raise RuntimeError("provider returned no choices")
            message = choices[0].get("message", {})
            answer = _message_text(message.get("content") if isinstance(message, dict) else None)
            if not answer:
                raise RuntimeError("provider returned an empty answer")
            usage = body.get("usage", {}) if isinstance(body, dict) else {}
            usage_row = {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            }
            returned_model = body.get("model") if isinstance(body, dict) else None
            return answer, usage_row, str(returned_model) if returned_model else None
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = str(exc).splitlines()[0][:240]
            if attempt + 1 < max_attempts:
                time.sleep(min(30.0, 2.0**attempt))
                continue
            raise RuntimeError(f"answer generation failed after {max_attempts} attempts: {last_error}") from exc
    raise RuntimeError(f"answer generation failed: {last_error or 'unknown provider error'}")


@contextmanager
def _build_retriever(args: argparse.Namespace) -> Iterator[tuple[Any, Any, list[Chunk], int]]:
    memory = build_memory_items(args.image_file, args.video_file, args.email_file)
    embedder = resolve_embedder(args.embedder)
    chunks = [
        Chunk(
            id=evidence_id,
            source=evidence_id,
            text=text,
            metadata={**metadata, "evidence_id": evidence_id, "modality": modality},
        )
        for evidence_id, modality, text, metadata in memory
    ]
    dsn = args.dsn or os.environ.get("RECALL_DSN") or DEFAULT_DSN
    reranker = reranker_from_name(args.reranker)
    index_started = time.perf_counter()
    embeddings: list[list[float]] = []
    store = PgVectorStore(dsn, dim=embedder.dim, table=args.table, tenant=args.tenant)
    try:
        store.ensure_schema()
        if args.reuse_index:
            facts = store.readiness_facts()
            if int(facts["rows"]) != len(chunks):
                raise ValueError(
                    f"reused ATM index has {facts['rows']} rows, expected {len(chunks)}"
                )
        else:
            embeddings = embed_passages(embedder, [chunk.text for chunk in chunks])
            store.upsert(chunks, embeddings)
            store.analyze()
        index_ms = int((time.perf_counter() - index_started) * 1000)
        retriever = HybridRetriever(
            store,
            embedder,
            reranker=reranker,
            candidate_k=args.candidate_k,
            gap_threshold=args.gap_threshold,
            use_dense=True,
            use_sparse=True,
            sparse_backend="lexical",
            retrieval_profile="atm_voyage4_lexical_hybrid",
            index_generation="atm_2026_08_19_voyage4",
        )
        yield retriever, embedder, chunks, index_ms
    finally:
        store.close()


def run(args: argparse.Namespace) -> int:
    questions = load_questions(args.qa_file)
    memory = build_memory_items(args.image_file, args.video_file, args.email_file)
    memory_ids = {item[0] for item in memory}
    if args.dry_run:
        print(json.dumps({
            "questions": len(questions),
            "memory_items": len(memory),
            "embedder": args.embedder,
            "reranker": args.reranker,
            "candidate_k": args.candidate_k,
            "retrieval_k": args.retrieval_k,
            "answer_model": args.answer_model,
            "reasoning_effort": args.reasoning_effort,
            "max_output_tokens": args.max_output_tokens,
            "evidence_chars": args.evidence_chars,
            "official_judge": "gpt-5-mini",
        }, indent=2))
        return 0

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for answer generation")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    answers_path = args.out_dir / "answers.jsonl"
    retrieval_path = args.out_dir / "retrieval.jsonl"
    answers = _load_jsonl(answers_path)
    retrieval_rows = _load_jsonl(retrieval_path)
    with _build_retriever(args) as (retriever, embedder, chunks, index_ms):
        usage_total = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        returned_models: set[str] = set()
        errors: list[dict[str, str]] = []
        for position, question in enumerate(questions, start=1):
            question_id = question["id"]
            if question_id in answers:
                continue
            row = retrieval_rows.get(question_id)
            if row is None:
                result = retriever.search(question["question"], k=args.retrieval_k)
                hits = [
                    {
                        "id": hit.chunk.id,
                        "text": hit.chunk.text,
                        "score": float(hit.score),
                    }
                    for hit in result.hits
                ]
                row = {
                    "id": question_id,
                    "question": question["question"],
                    "qtype": question.get("qtype"),
                    "retrieval_ids": [hit["id"] for hit in hits],
                    "hits": hits,
                    "gap_warning": bool(result.gap_warning),
                    "reranking_ran": bool(result.diagnostics.reranking_ran),
                }
                if any(hit_id not in memory_ids for hit_id in row["retrieval_ids"]):
                    raise RuntimeError(f"retrieval returned an unknown memory id for {question_id}")
                _append_jsonl(retrieval_path, row)
                retrieval_rows[question_id] = row
            answer, usage, returned_model = generate_answer(
                question=question["question"],
                qtype=question.get("qtype"),
                evidence=_evidence_text(row["hits"], args.evidence_chars),
                model=args.answer_model,
                base_url=args.answer_base_url,
                api_key=api_key,
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=args.max_output_tokens,
                max_attempts=args.max_attempts,
            )
            answer_row = {"id": question_id, "answer": answer}
            _append_jsonl(answers_path, answer_row)
            answers[question_id] = answer_row
            usage_total["calls"] += 1
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage_total[key] += usage[key]
            if returned_model:
                returned_models.add(returned_model)
            print(f"answered {position}/{len(questions)}")

        manifest = {
        "benchmark": "ATM-Bench",
        "measurement": "retrieve_rerank_answer",
        "question_count": len(questions),
        "answer_count": len(answers),
        "corpus_items": len(chunks),
        "embedder": args.embedder,
        "embedding_profile": embedding_profile_id(embedder),
        "reranker": args.reranker,
        "sparse_backend": "lexical",
        "candidate_k": args.candidate_k,
        "retrieval_k": args.retrieval_k,
        "evidence_chars": args.evidence_chars,
        "answer_model_requested": args.answer_model,
        "answer_models_returned": sorted(returned_models),
        "answer_base_url": args.answer_base_url,
        "reasoning": {
            "enabled": True,
            "requested_effort": args.reasoning_effort,
            "effective_deepseek_effort": "high" if args.reasoning_effort == "medium" else None,
        },
        "max_output_tokens": args.max_output_tokens,
        "usage": usage_total,
        "errors": errors,
        "index_ms": index_ms,
        "table": args.table,
        "tenant": args.tenant,
        "git_revision": git_revision(),
        "data_sha256": {
            str(path): sha256(path)
            for path in (args.qa_file, args.image_file, args.video_file, args.email_file)
        },
        }
        _write_json(args.out_dir / "manifest.json", manifest)
    print(f"wrote {answers_path}")
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-file", type=Path, required=True)
    ap.add_argument("--image-file", type=Path, required=True)
    ap.add_argument("--video-file", type=Path, required=True)
    ap.add_argument("--email-file", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dsn")
    ap.add_argument("--table", default="bench_atm_voyage4_chunks")
    ap.add_argument("--tenant", default="atm-bench-voyage4-20260819")
    ap.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    ap.add_argument("--reranker", default=DEFAULT_RERANKER)
    ap.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    ap.add_argument("--retrieval-k", type=int, default=DEFAULT_RETRIEVAL_K)
    ap.add_argument("--gap-threshold", type=float, default=0.50)
    ap.add_argument("--answer-base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--answer-model", default=DEFAULT_MODEL)
    ap.add_argument("--reasoning-effort", choices=("none", "minimal", "low", "medium", "high"), default="medium")
    ap.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    ap.add_argument("--evidence-chars", type=int, default=DEFAULT_EVIDENCE_CHARS)
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--reuse-index", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
