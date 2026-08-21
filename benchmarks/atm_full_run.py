"""Run the RE call ATM retrieve, rerank, answer pipeline.

This driver is deliberately separate from ``atm_bench.py``.  That file measures retrieval only.
This file adds bounded answer generation and writes checkpointed artifacts which the official ATM
evaluator can consume without transforming the predictions.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from collections.abc import Iterator
from typing import Any

_SOURCE_ROOT = str(Path(__file__).resolve().parents[1])
if _SOURCE_ROOT not in sys.path:
    sys.path.insert(0, _SOURCE_ROOT)

from benchmarks.atm_bench import build_memory_items, git_revision, load_questions, sha256
from recall.embeddings import (
    VoyageEmbedder,
    embed_passages,
    embedding_profile_id,
    resolve_embedder,
)
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
DEFAULT_EMBEDDING_BATCH_SIZE = 32
DEFAULT_ANSWER_WORKERS = 1
DEFAULT_TRUNCATION_RETRY_MAX_OUTPUT_TOKENS = 8192


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


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


def _load_answer_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    for record_path in sorted(path.glob("*.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or not record.get("id"):
            raise ValueError(f"invalid answer record at {record_path}")
        question_id = str(record["id"])
        if question_id in records:
            raise ValueError(f"duplicate answer record for {question_id}")
        usage = record.get("usage")
        if not isinstance(usage, dict) or any(
            not isinstance(usage.get(key), int) or usage[key] < 0
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        ):
            raise ValueError(f"invalid usage in answer record {record_path}")
        if not isinstance(record.get("answer"), str) or not record["answer"].strip():
            raise ValueError(f"invalid answer in answer record {record_path}")
        records[question_id] = record
    return records


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


class OutputCeilingReached(RuntimeError):
    """The provider completed only because the configured output ceiling was reached."""

    def __init__(
        self,
        message: str,
        *,
        usage: dict[str, int],
        returned_model: str | None,
    ) -> None:
        super().__init__(message)
        self.usage = usage
        self.returned_model = returned_model


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
        "You are a QA assistant. Use ONLY the provided evidence to answer. "
        "If the evidence is insufficient, answer 'Unknown'. Respond with only the answer. "
        "If the question asks to recall or list items (photos/emails/videos), respond with the "
        "corresponding evidence IDs only, comma-separated, with no extra text."
    )
    user = (
        f"Question: {question}\n\n"
        f"Evidence:\n{evidence}\n\n"
        "Provide the answer based solely on the evidence."
    )
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
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", prefix="recall-curl-", suffix=".conf", delete=False
            ) as config_handle:
                config_path = Path(config_handle.name)
                config_handle.write(f'url = "{url}"\n')
                config_handle.write(f'header = "Authorization: Bearer {api_key}"\n')
                config_handle.write('header = "Content-Type: application/json"\n')
                config_handle.write('header = "Connection: close"\n')
                config_handle.write('silent\nshow-error\nmax-time = 180\nconnect-timeout = 30\n')
                config_handle.write('write-out = "\\n__HTTP_STATUS__:%{http_code}"\n')
            try:
                completed = subprocess.run(
                    ["curl", "--config", str(config_path), "--data-binary", "@-"],
                    input=json.dumps(payload, ensure_ascii=False),
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=195,
                )
            finally:
                config_path.unlink(missing_ok=True)
            output, marker, status_text = completed.stdout.rpartition("\n__HTTP_STATUS__:")
            if not marker:
                raise RuntimeError(
                    f"provider transport failed with curl exit {completed.returncode}"
                )
            status = int(status_text.strip() or "0")
            if _retryable_status(status):
                last_error = f"provider status {status}"
                if attempt + 1 < max_attempts:
                    time.sleep(min(30.0, 2.0**attempt))
                    continue
            if completed.returncode != 0 or status >= 400:
                detail = completed.stderr.strip().splitlines()[0][:160] if completed.stderr else ""
                raise RuntimeError(
                    f"provider request failed with status {status}"
                    + (f": {detail}" if detail else "")
                )
            body = json.loads(output)
            choices = body.get("choices") if isinstance(body, dict) else None
            if not isinstance(choices, list) or not choices:
                raise RuntimeError("provider returned no choices")
            choice = choices[0]
            if not isinstance(choice, dict):
                raise RuntimeError("provider returned an invalid choice")
            usage = body.get("usage", {}) if isinstance(body, dict) else {}
            usage_row = {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            }
            returned_model = body.get("model") if isinstance(body, dict) else None
            returned_model = str(returned_model) if returned_model else None
            if choice.get("finish_reason") == "length":
                raise OutputCeilingReached(
                    "provider answer reached the configured output ceiling",
                    usage=usage_row,
                    returned_model=returned_model,
                )
            message = choice.get("message", {})
            answer = _message_text(message.get("content") if isinstance(message, dict) else None)
            if not answer:
                raise RuntimeError("provider returned an empty answer")
            return answer, usage_row, returned_model
        except OutputCeilingReached:
            raise
        except (subprocess.TimeoutExpired, ValueError, RuntimeError) as exc:
            last_error = str(exc).splitlines()[0][:240]
            if attempt + 1 < max_attempts:
                time.sleep(min(30.0, 2.0**attempt))
                continue
            raise RuntimeError(f"answer generation failed after {max_attempts} attempts: {last_error}") from exc
    raise RuntimeError(f"answer generation failed: {last_error or 'unknown provider error'}")


@contextmanager
def _build_retriever(args: argparse.Namespace) -> Iterator[tuple[Any, Any, list[Chunk], int]]:
    memory = build_memory_items(args.image_file, args.video_file, args.email_file)
    if args.embedder == "voyage" or args.embedder.startswith("voyage:"):
        voyage_model = args.embedder[len("voyage:") :] if args.embedder.startswith("voyage:") else "voyage-3"
        embedder = VoyageEmbedder(model=voyage_model, batch_size=args.embedding_batch_size)
    else:
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
            "embedding_batch_size": args.embedding_batch_size,
            "retrieval_k": args.retrieval_k,
            "answer_model": args.answer_model,
            "reasoning_effort": args.reasoning_effort,
            "max_output_tokens": args.max_output_tokens,
            "answer_workers": args.answer_workers,
            "evidence_chars": args.evidence_chars,
            "official_judge": "gpt-5-mini",
        }, indent=2))
        return 0

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for answer generation")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    answers_path = args.out_dir / "answers.jsonl"
    answer_records_path = args.out_dir / "answer_records"
    retrieval_path = args.out_dir / "retrieval.jsonl"
    usage_checkpoint_path = args.out_dir / "usage_checkpoint.json"
    answers = _load_jsonl(answers_path)
    answer_records = _load_answer_records(answer_records_path)
    orphan_answers = set(answers) - set(answer_records)
    if orphan_answers:
        raise RuntimeError(
            "answers.jsonl contains answers without atomic usage records; "
            "start a new output directory instead of guessing token usage: "
            + ", ".join(sorted(orphan_answers)[:5])
        )
    for question_id, record in answer_records.items():
        answer_row = {"id": question_id, "answer": record["answer"]}
        existing = answers.get(question_id)
        if existing is not None and existing.get("answer") != answer_row["answer"]:
            raise RuntimeError(f"answer record disagrees with answers.jsonl for {question_id}")
        if existing is None:
            _append_jsonl(answers_path, answer_row)
        answers[question_id] = answer_row
    retrieval_rows = _load_jsonl(retrieval_path)
    previous_manifest: dict[str, Any] = {}
    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.exists():
        loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded_manifest, dict):
            previous_manifest = loaded_manifest
    with _build_retriever(args) as (retriever, embedder, chunks, index_ms):
        source_commit_history = {
            commit.strip()
            for commit in os.environ.get("RECALL_PRIOR_SOURCE_COMMITS", "").split(",")
            if commit.strip()
        }
        source_commit_history.add(git_revision())
        usage_total = {
            key: 0
            for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens")
        }
        returned_models: set[str] = set()
        for record in answer_records.values():
            usage_total["calls"] += int(record.get("provider_call_count", 1))
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage_total[key] += int(record["usage"][key])
            if record.get("returned_model"):
                returned_models.add(str(record["returned_model"]))
        errors: list[dict[str, str]] = []
        pending: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
        for position, question in enumerate(questions, start=1):
            question_id = question["id"]
            if question_id in answer_records:
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
                    "max_dense_score": result.diagnostics.max_dense_score,
                    "gap_warning": bool(result.gap_warning),
                    "reranking_ran": bool(result.diagnostics.reranking_ran),
                }
                if any(hit_id not in memory_ids for hit_id in row["retrieval_ids"]):
                    raise RuntimeError(f"retrieval returned an unknown memory id for {question_id}")
                _append_jsonl(retrieval_path, row)
                retrieval_rows[question_id] = row
            pending.append((position, question, row))

        def answer_one(
            item: tuple[int, dict[str, Any], dict[str, Any]],
        ) -> tuple[int, str, dict[str, Any], dict[str, int], str | None, dict[str, Any]]:
            position, question, row = item
            generation_kwargs = {
                "question": question["question"],
                "qtype": question.get("qtype"),
                "evidence": _evidence_text(row["hits"], args.evidence_chars),
                "model": args.answer_model,
                "base_url": args.answer_base_url,
                "api_key": api_key,
                "reasoning_effort": args.reasoning_effort,
                "max_attempts": args.max_attempts,
            }
            used_truncation_retry = False
            generation_max_output_tokens = args.max_output_tokens
            generation_attempts: list[dict[str, Any]] = []
            try:
                answer, usage, returned_model = generate_answer(
                    **generation_kwargs,
                    max_output_tokens=args.max_output_tokens,
                )
                generation_attempts.append(
                    {
                        "max_output_tokens": args.max_output_tokens,
                        "finish_reason": "stop",
                        "usage": usage,
                    }
                )
            except OutputCeilingReached as exc:
                generation_attempts.append(
                    {
                        "max_output_tokens": args.max_output_tokens,
                        "finish_reason": "length",
                        "usage": exc.usage,
                    }
                )
                retry_limit = args.truncation_retry_max_output_tokens
                if retry_limit <= args.max_output_tokens:
                    raise
                used_truncation_retry = True
                generation_max_output_tokens = retry_limit
                answer, usage, returned_model = generate_answer(
                    **generation_kwargs,
                    max_output_tokens=retry_limit,
                )
                generation_attempts.append(
                    {
                        "max_output_tokens": retry_limit,
                        "finish_reason": "stop",
                        "usage": usage,
                    }
                )
            combined_usage = {
                key: sum(int(attempt["usage"][key]) for attempt in generation_attempts)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
            return (
                position,
                question["id"],
                {"id": question["id"], "answer": answer},
                combined_usage,
                returned_model,
                {
                    "requested_max_output_tokens": args.max_output_tokens,
                    "generation_max_output_tokens": generation_max_output_tokens,
                    "used_truncation_retry": used_truncation_retry,
                    "provider_call_count": len(generation_attempts),
                    "generation_attempts": generation_attempts,
                },
            )

        completed = len(answers)
        truncation_retry_count = sum(
            1 for record in answer_records.values() if record.get("used_truncation_retry")
        )

        def accept_answer(
            result: tuple[int, str, dict[str, Any], dict[str, int], str | None, dict[str, Any]]
        ) -> None:
            nonlocal completed, truncation_retry_count
            position, question_id, answer_row, usage, returned_model, generation_meta = result
            record = {
                "position": position,
                "id": question_id,
                "source_commit": git_revision(),
                "answer": answer_row["answer"],
                "answer_sha256": hashlib.sha256(
                    answer_row["answer"].encode("utf-8")
                ).hexdigest(),
                "usage": usage,
                "returned_model": returned_model,
                **generation_meta,
            }
            _write_json(answer_records_path / f"{position:04d}.json", record)
            answer_records[question_id] = record
            _append_jsonl(answers_path, answer_row)
            answers[question_id] = answer_row
            usage_total["calls"] += int(generation_meta.get("provider_call_count", 1))
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage_total[key] += usage[key]
            if returned_model:
                returned_models.add(returned_model)
            if generation_meta.get("used_truncation_retry"):
                truncation_retry_count += 1
            completed += 1
            _write_json(
                usage_checkpoint_path,
                {
                    "benchmark": "ATM-Bench",
                    "question_count": len(questions),
                    "answer_count": len(answers),
                    "usage": usage_total,
                    "answer_models_returned": sorted(returned_models),
                    "git_revision": git_revision(),
                },
            )
            print(f"answered {completed}/{len(questions)} question_position={position}", flush=True)

        if args.answer_workers == 1:
            for item in pending:
                accept_answer(answer_one(item))
        else:
            with ThreadPoolExecutor(max_workers=args.answer_workers) as executor:
                futures = [executor.submit(answer_one, item) for item in pending]
                for future in as_completed(futures):
                    accept_answer(future.result())

        if len(answer_records) != len(questions):
            raise RuntimeError(
                f"answer record count {len(answer_records)} does not match "
                f"question count {len(questions)}"
            )
        _write_jsonl(
            answers_path,
            [
                {"id": question["id"], "answer": answer_records[question["id"]]["answer"]}
                for question in questions
            ],
        )
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
        "embedding_batch_size": args.embedding_batch_size,
        "retrieval_k": args.retrieval_k,
        "evidence_chars": args.evidence_chars,
        "answer_model_requested": args.answer_model,
        "answer_models_returned": sorted(returned_models),
        "answer_base_url": args.answer_base_url,
        "answer_workers": args.answer_workers,
        "reasoning": {
            "enabled": True,
            "requested_effort": args.reasoning_effort,
            "effective_deepseek_effort": "high" if args.reasoning_effort == "medium" else None,
        },
        "max_output_tokens": args.max_output_tokens,
        "truncation_retry_max_output_tokens": args.truncation_retry_max_output_tokens,
        "truncation_retry_count": truncation_retry_count,
        "usage": usage_total,
        "usage_record_count": len(answer_records),
        "usage_source": "atomic per-answer records",
        "answer_records": answer_records_path.name,
        "usage_checkpoint": usage_checkpoint_path.name,
        "errors": errors,
        "index_ms": index_ms,
        "table": args.table,
        "tenant": args.tenant,
        "git_revision": git_revision(),
        "source_commit_history": sorted(source_commit_history),
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
    ap.add_argument("--embedding-batch-size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE)
    ap.add_argument("--retrieval-k", type=int, default=DEFAULT_RETRIEVAL_K)
    ap.add_argument("--gap-threshold", type=float, default=0.50)
    ap.add_argument("--answer-base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--answer-model", default=DEFAULT_MODEL)
    ap.add_argument("--reasoning-effort", choices=("none", "minimal", "low", "medium", "high"), default="medium")
    ap.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    ap.add_argument(
        "--truncation-retry-max-output-tokens",
        type=int,
        default=DEFAULT_TRUNCATION_RETRY_MAX_OUTPUT_TOKENS,
    )
    ap.add_argument("--evidence-chars", type=int, default=DEFAULT_EVIDENCE_CHARS)
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--answer-workers", type=_positive_int, default=DEFAULT_ANSWER_WORKERS)
    ap.add_argument("--reuse-index", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
