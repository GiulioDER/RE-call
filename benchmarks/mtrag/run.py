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

#: Default learned sparse checkpoint. apache-2.0; see recall.sparse.KNOWN_MODELS.
DEFAULT_SPARSE_MODEL = "prithivida/Splade_PP_en_v1"
EMBEDDER_MODEL = "BAAI/bge-small-en-v1.5"

#: How deep each query retrieves. 100, so Recall@100 is available; the top-10 ordering is
#: identical to a k=10 run because reranking happens over the fused pool BEFORE truncation.
RETRIEVAL_DEPTH = 100


@dataclass(frozen=True)
class Arm:
    name: str
    query_mode: str
    candidate_k: int
    use_dense: bool = True
    use_sparse: bool = True
    rerank: bool = False
    role: str = "ablation"
    sparse_backend: str = "lexical"

    def pool_bound(self) -> int:
        """The most distinct chunks fusion can hold for this arm, before truncation to k.

        Each enabled leg contributes at most `candidate_k` candidates and RRF unions them, so a
        metric cut DEEPER than this measures the POOL, not the ranking. `DEFAULT_CANDIDATE_K`
        already says that about the depth curve; this turns it into a number the results file
        carries, so nobody reads Recall@100 off an arm whose pool tops out at 40.
        """
        legs = 0
        if self.use_dense:
            legs += 1
        if self.use_sparse and self.sparse_backend in ("lexical", "both"):
            legs += 1
        if self.use_sparse and self.sparse_backend in ("splade", "both"):
            legs += 1
        return self.candidate_k * legs


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

#: Learned sparse arms, frozen 2026-08-06 BEFORE any score was observed.
#:
#: The question is single: does dense + SPLADE beat dense + ts_rank? Everything else is held
#: fixed, so the sparse leg's backend is the only thing that varies. `hybrid_both` is declared
#: here rather than added later precisely so that running it is pre-registered and not a
#: post-hoc rescue once the primary arm's number is known.
#:
#: All five use candidate_k=100, so every arm's pool bound is >= 100 and Recall@100 measures
#: retrieval depth rather than the pool. That is why they do NOT reuse the candidate_k=20
#: defaults above.
SPARSE_ARMS = (
    Arm("hybrid_lexical", "last", 100, role="control"),
    Arm("hybrid_splade", "last", 100, sparse_backend="splade", role="primary"),
    Arm("splade_only", "last", 100, use_dense=False, sparse_backend="splade"),
    Arm("hybrid_both", "last", 100, sparse_backend="both", role="secondary"),
    Arm("dense_only", "last", 100, use_sparse=False),
)

ALL_ARMS = ARMS + SPARSE_ARMS


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


#: Which judged set an evaluation scores against.
#:
#: ⛔ `test` is MTRAG-UN, the HELD-OUT set the official leaderboard scored. The archived
#: 2026-08-04 Task A baseline was run on it, which is why it is sealed: every further arm
#: comparison belongs on `dev` (MTRAG-human), or the held-out set stops being held out.
#: New arms default to dev for that reason, and choosing `test` has to be typed out.
SPLITS = ("dev", "test")

#: dev query modes are FILES, not computations. MTRAG-human ships the last user turn, the
#: full-conversation concatenation, and a GOLD human rewrite as three aligned files with the same
#: ids, so the rewriting ceiling is measurable with no LLM at all.
DEV_QUERY_FILES = {"last": "lastturn", "full": "questions", "rewrite": "rewrite"}


def qrels_path(root: Path, domain: str, split: str = "test") -> Path:
    if split == "dev":
        return root / "mtrag-human" / "retrieval_tasks" / domain / "qrels" / "dev.tsv"
    return root / "mtragun-human" / "retrieval_tasks" / "qrels" / f"{domain}.tsv"


def dev_tasks_path(root: Path, domain: str, query_mode: str) -> Path:
    suffix = DEV_QUERY_FILES.get(query_mode)
    if suffix is None:
        raise ValueError(
            f"dev split has no file for query mode {query_mode!r}; "
            f"available modes are {sorted(DEV_QUERY_FILES)}. "
            f"'recent3' is a TEST-split computation over a conversation and has no dev equivalent."
        )
    return root / "mtrag-human" / "retrieval_tasks" / domain / f"{domain}_{suffix}.jsonl"


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


def load_tasks(root: Path, split: str = "test", query_mode: str = "last") -> list[dict[str, Any]]:
    if split == "dev":
        return load_dev_tasks(root, query_mode)
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


def load_dev_tasks(root: Path, query_mode: str) -> list[dict[str, Any]]:
    """MTRAG-human tasks, normalised to the shape the evaluation loop already consumes.

    The dev files are per-domain `{"_id", "text"}` records, so the domain comes from WHICH FILE a
    row was read from rather than from a field inside it, and the query mode selects the file
    rather than slicing a conversation. Normalising here keeps the caller identical across splits
    instead of branching at every use site.
    """
    tasks: list[dict[str, Any]] = []
    for domain in DOMAINS:
        path = dev_tasks_path(root, domain, query_mode)
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if "_id" not in item or "text" not in item:
                    raise RuntimeError(f"{path}:{line_no} is missing _id or text")
                tasks.append(
                    {"task_id": str(item["_id"]), "_domain": domain, "_text": str(item["text"])}
                )
    return tasks


def strip_speaker(text: str) -> str:
    """Remove the '|user|: ' turn prefix MTRAG-human ships on every line.

    Byte-identical to `benchmarks/mtrag/probe/fix_retrieval_2x2.py` on `bench/mtrag-arm-r`, on
    purpose: the established dev baseline (nDCG@5 0.2849 / R@100 0.6865) was measured with THAT
    normalisation, and a merely similar one makes this run incomparable with it while looking
    fine.

    The prefix is not part of the question. Left in, the literal token reaches the embedder and
    the sparse encoder on every dev query and depresses the entire run, with nothing failing.
    Only a leading speaker tag is removed; a colon inside the question is content and survives.
    """
    return "\n".join(
        ln.split(":", 1)[1].strip() if ln.startswith("|") and ":" in ln else ln
        for ln in text.splitlines()
    ).strip()


def task_domain(task: dict[str, Any]) -> str:
    if "_domain" in task:
        return str(task["_domain"])
    return DOMAIN_ALIASES[str(task["Collection"]).lower()]


def query_text(task: dict[str, Any], mode: str) -> str:
    if "_text" in task:
        # dev: the mode already chose the FILE, so the only work left is dropping the speaker
        # tag the release prefixes to every turn.
        return strip_speaker(task["_text"])
    turns = task["input"]
    user_turns = [str(turn["text"]).strip() for turn in turns if turn["speaker"] == "user"]
    if not user_turns:
        raise RuntimeError(f"task {task['task_id']} has no user turn")
    if mode == "last":
        return user_turns[-1]
    if mode == "recent3":
        return "\n".join(user_turns[-3:])
    raise ValueError(f"unknown query mode {mode!r}")


def load_qrels(root: Path, split: str = "test") -> dict[str, dict[str, set[str]]]:
    all_qrels: dict[str, dict[str, set[str]]] = {}
    for domain in DOMAINS:
        domain_qrels: dict[str, set[str]] = {}
        with qrels_path(root, domain, split).open(encoding="utf-8") as handle:
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
    predictions: list[dict[str, Any]],
    qrels: dict[str, dict[str, set[str]]],
    pool_bound: int | None = None,
) -> dict[str, Any]:
    """Score predictions against the qrels, per query and aggregated.

    `overall` is a POOLED mean: every judged query contributes one value to one flat list, so a
    domain's influence is proportional to how many judged queries it has (on the four-domain
    release: clapnq 83, cloud 86, fiqa 58, govt 105). It is NOT the unweighted mean of the four
    `domains` figures, which is what `recall.promotion` calls a macro average. The two differ by
    1.5% to 6.6% on the 2026-08-04 baseline and they do not agree on every arm ordering, so the
    distinction is load bearing and the name has to say which one this is.

    `domains[d]` is the unweighted mean over domain `d`'s judged queries. Anyone who wants the
    macro figure can average those four; it is deliberately derivable rather than reported, so
    this function's output shape stays what the frozen run produced.
    """
    by_task = {item["task_id"]: item for item in predictions}
    # 100 is here because Recall@100 is the diagnostic that separates a COVERAGE problem from a
    # RANKING problem, and it costs nothing once retrieval already runs this deep. It is recorded,
    # not used as a gate.
    ks = (1, 3, 5, 10, 100)
    # A cutoff deeper than the fused pool measures the POOL, not the ranking. Naming the bounded
    # metrics keeps a pool-limited Recall@100 from rendering identically to a real one.
    bounded = [f"{m}@{k}" for k in ks for m in ("nDCG", "Recall")
               if pool_bound is not None and k > pool_bound]
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
        "pool_bound": pool_bound,
        "metrics_bounded_by_pool": bounded,
    }


def run_arm(
    *, arm: Arm, dsn: str, root: Path, output_dir: Path,
    embedder: FastEmbedEmbedder, prefix: str, split: str = "test",
    sparse_model: str | None = None, accept_noncommercial_license: bool = False,
) -> dict[str, Any]:
    print(
        json.dumps({"event": "arm_start", "arm": arm.name, "split": split, "at": utc_now()}),
        flush=True,
    )
    started = time.perf_counter()
    reranker = CrossEncoderReranker() if arm.rerank else None
    sparse_encoder = None
    if arm.sparse_backend in ("splade", "both"):
        # Imported HERE, not at module scope: torch and transformers are an optional extra, and a
        # lexical-only run must not require them to be installed at all.
        from recall.sparse import SpladeEncoder

        sparse_encoder = SpladeEncoder.from_pretrained(
            sparse_model or DEFAULT_SPARSE_MODEL,
            accept_noncommercial_license=accept_noncommercial_license,
        )
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
            sparse_backend=arm.sparse_backend,
            sparse_encoder=sparse_encoder,
        )
        for domain, store in stores.items()
    }
    predictions = []
    latencies_ms = []
    gap_count = 0
    tasks = load_tasks(root, split, arm.query_mode)
    try:
        for position, task in enumerate(tasks, 1):
            domain = task_domain(task)
            query = query_text(task, arm.query_mode)
            query_started = time.perf_counter()
            # Depth 100, not 10. The top-10 ordering is unchanged (reranking happens over the
            # fused pool BEFORE truncation), so every previously reported metric at k<=10 is
            # unaffected; what this buys is Recall@100 for free on the same pass.
            result = retrievers[domain].search(query, k=RETRIEVAL_DEPTH)
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
    scores = score_predictions(predictions, load_qrels(root, split), arm.pool_bound())
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


def validate_release(
    root: Path, split: str = "test", query_mode: str = "last"
) -> dict[str, Any]:
    """Describe and hash the release files for THIS split.

    ⚠️ This used to take only `root`, so every path helper it called fell back to its `"test"`
    default and it read the SEALED MTRAG-UN files no matter what `--split` said. `main()` writes
    `{"split": args.split, "release": validate_release(...)}` into one manifest, so a dev run
    emitted a manifest that said `dev` beside a provenance block describing the held-out set,
    down to the sha256 of files the run never opened. The SCORES were right the whole time
    (`run_arm` passes `split` to `load_qrels`); what was wrong was the record of what they were,
    which is the half nobody re-derives later.

    The pre-existing guard asserted `args.split == "dev"`, the argparse default. The flag was
    never the problem. Nothing downstream read it.
    """
    if split == "dev":
        task_files = {
            f"tasks_{domain}": dev_tasks_path(root, domain, query_mode) for domain in DOMAINS
        }
    else:
        task_files = {"tasks": tasks_path(root)}
    missing = [
        path for path in
        [*task_files.values(), *(corpus_zip(root, d) for d in DOMAINS),
         *(qrels_path(root, d, split) for d in DOMAINS)]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing MTRAG release files: {missing}")
    tasks = load_tasks(root, split, query_mode)
    task_ids = {task["task_id"] for task in tasks}
    qrels = load_qrels(root, split)
    qrel_ids = {query_id for rows in qrels.values() for query_id in rows}
    unknown = qrel_ids - task_ids
    if unknown:
        raise RuntimeError(f"{len(unknown)} qrel query ids are absent from reference tasks")
    by_domain = {domain: 0 for domain in DOMAINS}
    for task in tasks:
        by_domain[task_domain(task)] += 1
    return {
        "split": split,
        "query_mode_validated": query_mode,
        "task_count": len(tasks),
        "tasks_by_domain": by_domain,
        "scored_query_count": len(qrel_ids),
        "unscored_query_count": len(task_ids - qrel_ids),
        "qrels_by_domain": {domain: len(rows) for domain, rows in qrels.items()},
        "input_sha256": {
            **{name: sha256_file(path) for name, path in task_files.items()},
            **{
                f"qrels_{domain}": sha256_file(qrels_path(root, domain, split))
                for domain in DOMAINS
            },
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
    parser.add_argument("--arms", nargs="*", choices=[arm.name for arm in ALL_ARMS])
    parser.add_argument(
        "--split", choices=SPLITS, default="dev",
        help="dev = MTRAG-human (default). test = MTRAG-UN, the SEALED held-out set the "
             "leaderboard scored; the archived Task A baseline already used it, so a new arm "
             "comparison run there stops it being held out.",
    )
    parser.add_argument(
        "--sparse-model", default=None,
        help=f"learned sparse checkpoint (default {DEFAULT_SPARSE_MODEL})",
    )
    parser.add_argument(
        "--accept-noncommercial-license", action="store_true",
        help="required for naver/splade-v3 (cc-by-nc-sa-4.0)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.mtrag_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    release = validate_release(root, args.split)
    manifest = {
        "benchmark": "MTRAG-UN / MTRAGEval Task A (four-domain public release)",
        "started_at": utc_now(),
        "frozen_arms": [asdict(arm) for arm in ALL_ARMS],
        "split": args.split,
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
        for arm in ALL_ARMS:
            if arm.name in selected:
                summaries.append(
                    run_arm(
                        arm=arm, dsn=dsn, root=root, output_dir=output_dir,
                        embedder=embedder, prefix=args.table_prefix, split=args.split,
                        sparse_model=args.sparse_model,
                        accept_noncommercial_license=args.accept_noncommercial_license,
                    )
                )
        (output_dir / "summary.json").write_text(
            json.dumps(summaries, indent=2), encoding="utf-8"
        )
    print(json.dumps({"event": "complete", "at": utc_now()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
