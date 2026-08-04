from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from recall.embeddings import FastEmbedEmbedder
from recall.rerank import CrossEncoderReranker
from recall.retriever import HybridRetriever
from recall.store import PgVectorStore
from recall.types import Chunk

DOMAINS = ("clapnq", "cloud", "fiqa", "govt")
DOMAIN_ALIASES = {"ibmcloud": "cloud", **{domain: domain for domain in DOMAINS}}
EMBEDDER_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True)
class Arm:
    name: str
    query_mode: str
    candidate_k: int
    use_dense: bool = True
    use_sparse: bool = True
    rerank: bool = False
    role: str = "ablation"


# Frozen before the first evaluation. Do not reorder or change these arms after observing scores;
# add a separately named exploratory arm instead and record it as post-hoc.
ARMS = (
    Arm("recall_default_last", "last", 20, role="primary"),
    Arm("recall_default_recent3", "recent3", 20, role="secondary"),
    Arm("recall_rerank_last", "last", 100, rerank=True, role="secondary"),
    Arm("recall_rerank_recent3", "recent3", 100, rerank=True, role="competitive"),
    Arm("dense_last", "last", 100, use_sparse=False),
    Arm("sparse_last", "last", 100, use_dense=False),
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
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


def load_dsn(env_file: Path | None) -> str:
    if os.environ.get("RECALL_DSN"):
        return os.environ["RECALL_DSN"]
    if env_file is not None:
        try:
            from dotenv import dotenv_values
        except ImportError as exc:
            raise RuntimeError("--dsn-env-file requires python-dotenv") from exc
        values = dotenv_values(env_file)
        dsn = values.get("RECALL_DSN") or values.get("DATABASE_URL")
        if dsn:
            return str(dsn)
    raise RuntimeError("set RECALL_DSN or pass --dsn-env-file containing DATABASE_URL")


def corpus_zip(root: Path, domain: str) -> Path:
    return root / "corpora" / "passage_level" / f"{domain}.jsonl.zip"


def qrels_path(root: Path, domain: str) -> Path:
    return root / "mtragun-human" / "retrieval_tasks" / "qrels" / f"{domain}.tsv"


def tasks_path(root: Path) -> Path:
    return root / "mtragun-human" / "generation_tasks" / "reference.jsonl"


def iter_corpus(path: Path) -> Iterator[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if len(members) != 1:
            raise RuntimeError(f"expected one JSONL member in {path}, found {members}")
        with archive.open(members[0]) as raw:
            for line_no, line in enumerate(raw, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(f"invalid JSON in {path}:{line_no}: {exc}") from exc


def batched(items: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def table_name(prefix: str, domain: str) -> str:
    name = f"{prefix}_{domain}"
    if not name.isidentifier():
        raise ValueError(f"invalid table name {name!r}")
    return name


def index_domain(
    *, dsn: str, root: Path, domain: str, embedder: FastEmbedEmbedder,
    prefix: str, batch_size: int,
) -> dict[str, Any]:
    path = corpus_zip(root, domain)
    table = table_name(prefix, domain)
    started = time.perf_counter()
    with PgVectorStore(dsn, embedder.dim, table=table) as store:
        store.ensure_schema()
        existing = store.count()
        seen = 0
        written = 0
        for batch in batched(iter_corpus(path), batch_size):
            next_seen = seen + len(batch)
            if next_seen <= existing:
                seen = next_seen
                continue
            if seen < existing:
                batch = batch[existing - seen :]
                seen = existing
            chunks = []
            for item in batch:
                document_id = str(item.get("_id") or item.get("id"))
                if not document_id or document_id == "None":
                    raise RuntimeError(f"missing id in {path} after row {seen}")
                chunks.append(
                    Chunk(
                        id=document_id,
                        source=domain,
                        text=str(item.get("text") or "").replace("\x00", ""),
                        metadata={
                            "title": item.get("title"),
                            "url": item.get("url"),
                            "mtrag_collection": domain,
                            "document_id": document_id,
                        },
                    )
                )
            vectors = embedder.embed([chunk.text for chunk in chunks])
            store.upsert(chunks, vectors)
            written += len(chunks)
            seen += len(chunks)
            if seen % (batch_size * 10) == 0 or len(batch) < batch_size:
                print(
                    json.dumps(
                        {"event": "index_progress", "domain": domain, "rows": seen,
                         "written_this_run": written, "at": utc_now()}
                    ),
                    flush=True,
                )
        final_count = store.count()
        if final_count != seen:
            raise RuntimeError(
                f"{table}: stored {final_count} rows but corpus contains {seen}; "
                "refusing a partial/mixed index"
            )
        store.analyze()
    return {
        "domain": domain,
        "table": table,
        "rows": seen,
        "previous_rows": existing,
        "written_this_run": written,
        "elapsed_s": time.perf_counter() - started,
        "corpus_sha256": sha256_file(path),
    }


def load_tasks(root: Path) -> list[dict[str, Any]]:
    tasks = []
    with tasks_path(root).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            raw_domain = str(item.get("Collection", "")).lower()
            if raw_domain not in DOMAIN_ALIASES:
                raise RuntimeError(f"unknown Collection {raw_domain!r} at task line {line_no}")
            tasks.append(item)
    return tasks


def task_domain(task: dict[str, Any]) -> str:
    return DOMAIN_ALIASES[str(task["Collection"]).lower()]


def query_text(task: dict[str, Any], mode: str) -> str:
    turns = task["input"]
    user_turns = [str(turn["text"]).strip() for turn in turns if turn["speaker"] == "user"]
    if not user_turns:
        raise RuntimeError(f"task {task['task_id']} has no user turn")
    if mode == "last":
        return user_turns[-1]
    if mode == "recent3":
        return "\n".join(user_turns[-3:])
    raise ValueError(f"unknown query mode {mode!r}")


def load_qrels(root: Path) -> dict[str, dict[str, set[str]]]:
    all_qrels: dict[str, dict[str, set[str]]] = {}
    for domain in DOMAINS:
        domain_qrels: dict[str, set[str]] = {}
        with qrels_path(root, domain).open(encoding="utf-8") as handle:
            next(handle)
            for line in handle:
                query_id, corpus_id, score = line.rstrip("\n").split("\t")
                if int(score) > 0:
                    domain_qrels.setdefault(query_id, set()).add(corpus_id)
        all_qrels[domain] = domain_qrels
    return all_qrels


def ndcg_at(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, document_id in enumerate(ranked[:k])
        if document_id in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def recall_at(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def score_predictions(
    predictions: list[dict[str, Any]], qrels: dict[str, dict[str, set[str]]]
) -> dict[str, Any]:
    by_task = {item["task_id"]: item for item in predictions}
    ks = (1, 3, 5, 10)
    domain_rows: dict[str, Any] = {}
    # `dict.__or__` gives mypy no type context, so the merged comprehensions infer
    # `list[Never]`; `list[float]` here would be rejected by dict's invariance.
    all_values: dict[str, list[Any]] = (
        {f"nDCG@{k}": [] for k in ks} | {f"Recall@{k}": [] for k in ks}
    )
    per_query: dict[str, Any] = {}
    for domain in DOMAINS:
        values: dict[str, list[Any]] = (
            {f"nDCG@{k}": [] for k in ks} | {f"Recall@{k}": [] for k in ks}
        )
        for query_id, relevant in qrels[domain].items():
            if query_id not in by_task:
                raise RuntimeError(f"missing prediction for qrel query {query_id}")
            ranked = [ctx["document_id"] for ctx in by_task[query_id]["contexts"]]
            row = {}
            for k in ks:
                row[f"nDCG@{k}"] = ndcg_at(ranked, relevant, k)
                row[f"Recall@{k}"] = recall_at(ranked, relevant, k)
                values[f"nDCG@{k}"].append(row[f"nDCG@{k}"])
                values[f"Recall@{k}"].append(row[f"Recall@{k}"])
                all_values[f"nDCG@{k}"].append(row[f"nDCG@{k}"])
                all_values[f"Recall@{k}"].append(row[f"Recall@{k}"])
            per_query[query_id] = row
        domain_rows[domain] = {
            "count": len(qrels[domain]),
            **{
                metric: (sum(items) / len(items) if items else None)
                for metric, items in values.items()
            },
        }
    return {
        "overall": {
            "count": sum(len(rows) for rows in qrels.values()),
            **{metric: sum(items) / len(items) for metric, items in all_values.items()},
        },
        "domains": domain_rows,
        "per_query": per_query,
    }


def run_arm(
    *, arm: Arm, dsn: str, root: Path, output_dir: Path,
    embedder: FastEmbedEmbedder, prefix: str,
) -> dict[str, Any]:
    print(json.dumps({"event": "arm_start", "arm": arm.name, "at": utc_now()}), flush=True)
    started = time.perf_counter()
    reranker = CrossEncoderReranker() if arm.rerank else None
    stores = {
        domain: PgVectorStore(dsn, embedder.dim, table=table_name(prefix, domain))
        for domain in DOMAINS
    }
    retrievers = {
        domain: HybridRetriever(
            store,
            embedder,
            reranker=reranker,
            candidate_k=arm.candidate_k,
            use_dense=arm.use_dense,
            use_sparse=arm.use_sparse,
        )
        for domain, store in stores.items()
    }
    predictions = []
    latencies_ms = []
    gap_count = 0
    tasks = load_tasks(root)
    try:
        for position, task in enumerate(tasks, 1):
            domain = task_domain(task)
            query = query_text(task, arm.query_mode)
            query_started = time.perf_counter()
            result = retrievers[domain].search(query, k=10)
            latencies_ms.append((time.perf_counter() - query_started) * 1000.0)
            gap_count += int(result.gap_warning)
            contexts = []
            total = len(result.hits)
            for rank, hit in enumerate(result.hits):
                contexts.append(
                    {
                        "document_id": hit.chunk.id,
                        # The official evaluator reconstructs a run from scores, so this strictly
                        # descending rank score preserves RE-call's fused/reranked ordering.
                        "score": float(total - rank),
                        "cosine": hit.score,
                        "text": hit.chunk.text,
                        "title": hit.chunk.metadata.get("title"),
                        "source": domain,
                    }
                )
            predictions.append(
                {
                    "conversation_id": task["conversation_id"],
                    "task_id": task["task_id"],
                    "Collection": task["Collection"],
                    "input": task["input"],
                    "query_used": query,
                    "contexts": contexts,
                    "gap_warning": result.gap_warning,
                }
            )
            if position % 50 == 0:
                print(
                    json.dumps(
                        {"event": "query_progress", "arm": arm.name, "completed": position,
                         "total": len(tasks), "at": utc_now()}
                    ),
                    flush=True,
                )
    finally:
        for store in stores.values():
            store.close()

    prediction_path = output_dir / f"{arm.name}.predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for item in predictions:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    scores = score_predictions(predictions, load_qrels(root))
    ordered = sorted(latencies_ms)
    summary = {
        "arm": asdict(arm),
        "predictions": len(predictions),
        "gap_warnings": gap_count,
        "elapsed_s": time.perf_counter() - started,
        "latency_ms": {
            "mean": sum(latencies_ms) / len(latencies_ms),
            "p50": ordered[int(0.50 * (len(ordered) - 1))],
            "p95": ordered[int(0.95 * (len(ordered) - 1))],
        },
        "scores": scores,
        "prediction_sha256": sha256_file(prediction_path),
    }
    (output_dir / f"{arm.name}.metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"event": "arm_complete", "arm": arm.name,
             "nDCG@5": scores["overall"]["nDCG@5"], "at": utc_now()}
        ),
        flush=True,
    )
    return summary


def validate_release(root: Path) -> dict[str, Any]:
    missing = [
        path for path in
        [tasks_path(root), *(corpus_zip(root, d) for d in DOMAINS),
         *(qrels_path(root, d) for d in DOMAINS)]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing MTRAG release files: {missing}")
    tasks = load_tasks(root)
    task_ids = {task["task_id"] for task in tasks}
    qrels = load_qrels(root)
    qrel_ids = {query_id for rows in qrels.values() for query_id in rows}
    unknown = qrel_ids - task_ids
    if unknown:
        raise RuntimeError(f"{len(unknown)} qrel query ids are absent from reference tasks")
    by_domain = {domain: 0 for domain in DOMAINS}
    for task in tasks:
        by_domain[task_domain(task)] += 1
    return {
        "task_count": len(tasks),
        "tasks_by_domain": by_domain,
        "scored_query_count": len(qrel_ids),
        "unscored_query_count": len(task_ids - qrel_ids),
        "qrels_by_domain": {domain: len(rows) for domain, rows in qrels.items()},
        "input_sha256": {
            "tasks": sha256_file(tasks_path(root)),
            **{f"qrels_{domain}": sha256_file(qrels_path(root, domain)) for domain in DOMAINS},
            **{f"corpus_{domain}": sha256_file(corpus_zip(root, domain)) for domain in DOMAINS},
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mtrag-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dsn-env-file", type=Path)
    parser.add_argument("--table-prefix", default="recall_mtrag_bge_v1")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--phase", choices=("validate", "index", "evaluate", "all"), default="all")
    parser.add_argument(
        "--index-domains", nargs="+", choices=DOMAINS, default=list(DOMAINS),
        help="domains to index; evaluation always uses all four official domains",
    )
    parser.add_argument("--arms", nargs="*", choices=[arm.name for arm in ARMS])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.mtrag_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    release = validate_release(root)
    manifest = {
        "benchmark": "MTRAG-UN / MTRAGEval Task A (four-domain public release)",
        "started_at": utc_now(),
        "frozen_arms": [asdict(arm) for arm in ARMS],
        "release": release,
        "revisions": {
            "recall": git_revision(Path(__file__).resolve().parents[2]),
            "mtrag": git_revision(root),
            "adapter_sha256": sha256_file(Path(__file__)),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "packages": {
                name: package_version(name)
                for name in ("recall", "fastembed", "sentence-transformers", "torch", "psycopg")
            },
        },
        "embedder_model": EMBEDDER_MODEL,
        "phase": args.phase,
        "table_prefix": args.table_prefix,
        "batch_size": args.batch_size,
        "index_domains": args.index_domains,
    }
    (output_dir / "preregistered_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"event": "validated", **release, "at": utc_now()}), flush=True)
    if args.phase == "validate":
        return 0

    dsn = load_dsn(args.dsn_env_file)
    embedder = FastEmbedEmbedder(EMBEDDER_MODEL)
    if args.phase in ("index", "all"):
        index_results = []
        for domain in args.index_domains:
            index_results.append(
                index_domain(
                    dsn=dsn, root=root, domain=domain, embedder=embedder,
                    prefix=args.table_prefix, batch_size=args.batch_size,
                )
            )
        (output_dir / "index.json").write_text(
            json.dumps(index_results, indent=2), encoding="utf-8"
        )
    if args.phase in ("evaluate", "all"):
        selected = set(args.arms or [arm.name for arm in ARMS])
        summaries = []
        for arm in ARMS:
            if arm.name in selected:
                summaries.append(
                    run_arm(
                        arm=arm, dsn=dsn, root=root, output_dir=output_dir,
                        embedder=embedder, prefix=args.table_prefix,
                    )
                )
        (output_dir / "summary.json").write_text(
            json.dumps(summaries, indent=2), encoding="utf-8"
        )
    print(json.dumps({"event": "complete", "at": utc_now()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
