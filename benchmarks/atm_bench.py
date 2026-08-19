"""Run the no credit ATM Bench retrieval evaluation for RE call.

This runner deliberately stops before answer generation. It emits both the official MMRag
item level recall shape and additional question level and complete evidence diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

from recall.embeddings import embed_passages, embedding_profile_id, resolve_embedder
from recall.rerank import reranker_from_name
from recall.retriever import HybridRetriever
from recall.store import PgVectorStore
from recall.types import Chunk


DEFAULT_DSN = "postgresql://recall:recall@localhost:5432/recall"
K_VALUES = (1, 5, 10, 25, 50, 100)


class NativeSentenceTransformerEmbedder:
    """Minimal mean pooled transformer adapter for the official MiniLM model.

    The installed sentence transformers wrapper cannot import on this Windows runtime because
    its optional SciPy dependency is blocked by application policy. This adapter uses the same
    Hugging Face model and the model card pooling contract without importing SciPy.
    """

    name = "st:sentence-transformers/all-MiniLM-L6-v2"
    dim = 384

    def __init__(self, model_name: str, batch_size: int = 64) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("the native ATM MiniLM adapter requires torch and transformers") from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        self._batch_size = batch_size

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        with self._torch.no_grad():
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start : start + self._batch_size]
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                output = self._model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(output.size()).float()
                pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                normalized = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.extend(normalized.cpu().tolist())
        return [[float(value) for value in vector] for vector in vectors]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _value(item: dict[str, Any], key: str) -> str:
    value = item.get(key, "")
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(part) for part in value)
    return str(value)


def format_media(item: dict[str, Any], evidence_id: str, modality: str) -> str:
    fields = (
        ("ID", evidence_id),
        ("Type", modality),
        ("Timestamp", _value(item, "timestamp")),
        ("Location", _value(item, "location_name")),
        ("Short Caption", _value(item, "short_caption")),
        ("Caption", _value(item, "caption")),
        ("OCR", _value(item, "ocr_text")),
        ("Tags", _value(item, "tags")),
    )
    return "\n".join(f"{label}: {value}" for label, value in fields)


def format_email(item: dict[str, Any]) -> str:
    fields = (
        ("ID", _value(item, "id")),
        ("Timestamp", _value(item, "timestamp")),
        ("Summary", _value(item, "short_summary")),
        ("Detail", _value(item, "detail")),
    )
    return "\n".join(f"{label}: {value}" for label, value in fields)


def build_memory_items(
    image_file: Path, video_file: Path, email_file: Path
) -> list[tuple[str, str, str, dict[str, Any]]]:
    items: list[tuple[str, str, str, dict[str, Any]]] = []
    seen: set[str] = set()

    emails = load_json(email_file)
    images = load_json(image_file)
    videos = load_json(video_file)
    if not all(isinstance(rows, list) for rows in (emails, images, videos)):
        raise ValueError("ATM memory files must contain JSON lists")

    def add(evidence_id: str, modality: str, text: str, metadata: dict[str, Any]) -> None:
        if not evidence_id:
            return
        if evidence_id in seen:
            raise ValueError(f"duplicate ATM evidence ID: {evidence_id}")
        seen.add(evidence_id)
        items.append((evidence_id, modality, text, metadata))

    for item in emails:
        evidence_id = _value(item, "id")
        add(evidence_id, "email", format_email(item), {"modality": "email"})
    for item in images:
        raw_path = _value(item, "image_path")
        evidence_id = Path(raw_path).stem
        add(evidence_id, "image", format_media(item, evidence_id, "image"), {"modality": "image"})
    for item in videos:
        raw_path = _value(item, "video_path")
        evidence_id = Path(raw_path).stem
        add(evidence_id, "video", format_media(item, evidence_id, "video"), {"modality": "video"})
    return items


def load_questions(path: Path) -> list[dict[str, Any]]:
    rows = load_json(path)
    if isinstance(rows, dict) and isinstance(rows.get("qas"), list):
        rows = rows["qas"]
    if not isinstance(rows, list):
        raise ValueError("ATM question file must be a JSON list or an object with qas")
    questions: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("id") or not row.get("question"):
            raise ValueError("ATM question rows require id and question")
        evidence_ids = row.get("evidence_ids", [])
        if not isinstance(evidence_ids, list):
            raise ValueError(f"question {row['id']} has invalid evidence_ids")
        questions.append(
            {
                "id": str(row["id"]),
                "question": str(row["question"]),
                "qtype": row.get("qtype"),
                "evidence_ids": list(dict.fromkeys(str(value) for value in evidence_ids if value)),
            }
        )
    return questions


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def is_development_question(question_id: str) -> bool:
    value = int(hashlib.sha256(question_id.encode("utf-8")).hexdigest(), 16)
    return value % 10 < 7


def metrics_for(retrieved_ids: list[str], gold_ids: list[str]) -> dict[str, Any]:
    gold = set(gold_ids)
    item_recall: dict[str, float] = {}
    question_hit: dict[str, float] = {}
    complete: dict[str, float] = {}
    for k in K_VALUES:
        top = set(retrieved_ids[:k])
        item_recall[f"R@{k}"] = len(gold & top) / len(gold) if gold else 0.0
        question_hit[f"Recall@{k}"] = 1.0 if gold & top else 0.0
        complete[f"Recall@{k}GT"] = 1.0 if gold and gold.issubset(top) else 0.0
    return {
        "retrieval_recall": item_recall,
        "question_hit": question_hit,
        "complete_evidence": complete,
    }


def summarize(details: list[dict[str, Any]], key: str) -> dict[str, float]:
    return {
        metric: statistics.fmean(float(row[key][metric]) for row in details)
        for metric in details[0][key]
    } if details else {}


def run(args: argparse.Namespace) -> dict[str, Any]:
    questions = load_questions(args.qa_file)
    if args.question_split == "development":
        questions = [question for question in questions if is_development_question(question["id"])]
    elif args.question_split == "holdout":
        questions = [question for question in questions if not is_development_question(question["id"])]
    memory = build_memory_items(args.image_file, args.video_file, args.email_file)
    if args.embedder == "st:sentence-transformers/all-MiniLM-L6-v2":
        embedder: Any = NativeSentenceTransformerEmbedder(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
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

    table = args.table
    tenant = args.tenant
    dsn = args.dsn or os.environ.get("RECALL_DSN") or DEFAULT_DSN
    reranker = None if args.reranker == "none" else reranker_from_name(args.reranker)
    sparse_encoder = None
    sparse_profile_id = None
    sparse_attribution = None
    if args.sparse_backend in ("splade", "both"):
        from recall.sparse import SpladeEncoder, attribution_notice

        sparse_encoder = SpladeEncoder.from_pretrained(
            args.sparse_model,
            accept_noncommercial_license=True,
            device=args.sparse_device,
        )
        sparse_profile_id = sparse_encoder.profile.profile_id
        sparse_attribution = attribution_notice(args.sparse_model)
    index_started = time.perf_counter()
    embeddings: list[list[float]] = []
    if not args.reuse_index:
        embeddings = embed_passages(embedder, [chunk.text for chunk in chunks])
    with PgVectorStore(dsn, dim=embedder.dim, table=table, tenant=tenant) as store:
        store.ensure_schema()
        if args.reuse_index:
            facts = store.readiness_facts()
            if int(facts["rows"]) != len(chunks):
                raise ValueError(
                    f"reused ATM index has {facts['rows']} rows, expected {len(chunks)}"
                )
        else:
            store.upsert(chunks, embeddings)
            store.analyze()
        if sparse_encoder is not None:
            from recall.sparse import backfill_learned_sparse

            if args.reuse_index and store.sparse_row_count(sparse_profile_id) == len(chunks):
                pass
            else:
                backfill_learned_sparse(store, sparse_encoder, batch_size=args.sparse_batch_size)
        index_ms = (time.perf_counter() - index_started) * 1000.0
        arms = {
            "dense": HybridRetriever(
                store,
                embedder,
                reranker=reranker,
                candidate_k=args.candidate_k,
                gap_threshold=args.gap_threshold,
                use_dense=True,
                use_sparse=False,
                retrieval_profile="atm_dense",
                index_generation="atm_2026_08_19",
            ),
            "hybrid": HybridRetriever(
                store,
                embedder,
                reranker=reranker,
                candidate_k=args.candidate_k,
                gap_threshold=args.gap_threshold,
                use_dense=True,
                use_sparse=args.sparse_backend != "none",
                sparse_backend="lexical" if args.sparse_backend == "none" else args.sparse_backend,
                sparse_encoder=sparse_encoder,
                retrieval_profile=f"atm_{args.sparse_backend}",
                index_generation="atm_2026_08_19",
            ),
        }
        selected = [arm for arm in args.arms if arm in arms]
        if not selected:
            raise ValueError("at least one arm must be dense or hybrid")

        arm_results: dict[str, Any] = {}
        for arm_name in selected:
            retriever = arms[arm_name]
            details: list[dict[str, Any]] = []
            latencies: list[float] = []
            for question in questions:
                started = time.perf_counter()
                result = retriever.search(question["question"], k=max(K_VALUES))
                latency_ms = (time.perf_counter() - started) * 1000.0
                latencies.append(latency_ms)
                retrieved_ids = [hit.chunk.id for hit in result.hits]
                metric_values = metrics_for(retrieved_ids, question["evidence_ids"])
                details.append(
                    {
                        "id": question["id"],
                        "question": question["question"],
                        "qtype": question["qtype"],
                        "gt_evidence_ids": question["evidence_ids"],
                        "retrieval_ids": retrieved_ids,
                        "retrieval_scores": [float(hit.score) for hit in result.hits],
                        "gap_warning": bool(result.gap_warning),
                        "reranking_ran": bool(result.diagnostics.reranking_ran),
                        "latency_ms": round(latency_ms, 3),
                        **metric_values,
                    }
                )
            official_details = [
                {
                    "id": row["id"],
                    "question": row["question"],
                    "gt_evidence_ids": row["gt_evidence_ids"],
                    "retrieval_ids": row["retrieval_ids"],
                    "retrieval_scores": row["retrieval_scores"],
                    "retrieval_recall": row["retrieval_recall"],
                }
                for row in details
            ]
            arm_results[arm_name] = {
                "questions": len(details),
                "official_item_recall": summarize(details, "retrieval_recall"),
                "question_level_recall": summarize(details, "question_hit"),
                "complete_evidence_recall": summarize(details, "complete_evidence"),
                "latency_ms": {
                    "mean": statistics.fmean(latencies) if latencies else 0.0,
                    "p50": percentile(latencies, 0.50),
                    "p95": percentile(latencies, 0.95),
                },
                "details": details,
                "official_retrieval_recall_details": official_details,
            }

    manifest_paths = [args.qa_file, args.image_file, args.video_file, args.email_file]
    reasoning_enabled = os.environ.get("RECALL_REASONING", "").strip().lower() in {
        "1", "true", "yes", "on"
    }
    return {
        "benchmark": "ATM-Bench",
        "split": args.question_split,
        "measurement": "retrieval_only",
        "retrieval_max_k": max(K_VALUES),
        "candidate_k": args.candidate_k,
        "gap_threshold": args.gap_threshold,
        "embedder": args.embedder,
        "embedding_profile": embedding_profile_id(embedder),
        "reranker": args.reranker,
        "sparse_backend": args.sparse_backend,
        "sparse_model": args.sparse_model if sparse_encoder is not None else None,
        "sparse_profile_id": sparse_profile_id,
        "sparse_attribution": sparse_attribution,
        "corpus_items": len(chunks),
        "index_ms": round(index_ms, 3),
        "table": table,
        "tenant": tenant,
        "git_revision": git_revision(),
        "judge": {"used": False, "model": None, "reasoning_effort": None},
        "reasoning": {
            "enabled": reasoning_enabled,
            "model": os.environ.get("RECALL_REASONING_MODEL") if reasoning_enabled else None,
            "base_url": os.environ.get("RECALL_REASONING_BASE_URL") if reasoning_enabled else None,
            "wired_into_runner": False,
        },
        "data_sha256": {str(path): sha256(path) for path in manifest_paths},
        "arms": arm_results,
    }


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-file", type=Path, required=True)
    ap.add_argument("--image-file", type=Path, required=True)
    ap.add_argument("--video-file", type=Path, required=True)
    ap.add_argument("--email-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dsn")
    ap.add_argument("--table", default="bench_atm_bench_chunks")
    ap.add_argument("--tenant", default="atm-bench-20260819")
    ap.add_argument("--embedder", default="st:sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--candidate-k", type=int, default=200)
    ap.add_argument("--gap-threshold", type=float, default=0.50)
    ap.add_argument("--reranker", default="none")
    ap.add_argument("--sparse-backend", choices=("none", "lexical", "splade", "both"), default="lexical")
    ap.add_argument("--sparse-model", default="prithivida/Splade_PP_en_v1")
    ap.add_argument("--sparse-device", choices=("cpu", "cuda"), default=None)
    ap.add_argument("--sparse-batch-size", type=int, default=32)
    ap.add_argument("--question-split", choices=("all", "development", "holdout"), default="all")
    ap.add_argument(
        "--reuse-index",
        action="store_true",
        help="reuse the existing tenant index after verifying its row count",
    )
    ap.add_argument("--arms", nargs="+", choices=("dense", "hybrid"), default=["dense", "hybrid"])
    return ap


def main() -> int:
    args = parser().parse_args()
    payload = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for arm, result in payload["arms"].items():
        details_path = args.out.with_name(f"{args.out.stem}_{arm}_retrieval_recall_details.json")
        details_path.write_text(
            json.dumps(result["official_retrieval_recall_details"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"{arm}: item_R@10={result['official_item_recall']['R@10']:.4f} "
            f"question_Recall@10={result['question_level_recall']['Recall@10']:.4f} "
            f"complete_Recall@10GT={result['complete_evidence_recall']['Recall@10GT']:.4f} "
            f"p95_ms={result['latency_ms']['p95']:.2f}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
