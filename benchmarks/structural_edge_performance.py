"""Run the preregistered real retrieval comparison for benchmark structural edges.

This runner uses the production ``Indexer``, ``PgVectorStore``, and ``HybridRetriever`` path. It
does not use the lexical upper bound in ``structural_edge_effect.py``. It records a paired
baseline top ten against five semantic seeds plus at most five source-backed structural neighbors.
Answer generation is deliberately a separate stage: the resulting evidence contexts can be
replayed through the configured OpenRouter answer provider without changing retrieval scores.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import statistics
import tempfile
import time
from typing import Any, Iterable

PROTOCOL = "2026-09-11-structural-edge-performance-v1"
DEFAULT_DSN = "postgresql://recall:recall@localhost:5432/recall"
DEFAULT_SEED_K = 5
DEFAULT_CONTEXT_K = 10
_SAFE_RUN_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


def _repo_guard() -> None:
    """Refuse a benchmark that resolves imports from a different checkout."""
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.find_spec("recall")
    origin = Path(spec.origin).resolve() if spec and spec.origin else None
    if origin is None or root not in origin.parents:
        raise SystemExit(f"refusing to run: recall resolves to {origin}, not {root}")


_repo_guard()

from benchmarks.atm_bench import build_memory_items, load_questions, sha256  # noqa: E402
from benchmarks.atm_structural import build_memory_items_with_structural_edges  # noqa: E402
from recall.embeddings import embed_passages, resolve_embedder  # noqa: E402
from recall.eval.locomo import write_conversation_corpus  # noqa: E402
from recall.index import Indexer  # noqa: E402
from recall.retriever import DEFAULT_CANDIDATE_K, HybridRetriever  # noqa: E402
from recall.store import PgVectorStore  # noqa: E402
from recall.types import Chunk, ScoredChunk  # noqa: E402


def _run_id(value: str) -> str:
    if not _SAFE_RUN_ID.fullmatch(value):
        raise argparse.ArgumentTypeError("run id must be a safe ASCII identifier")
    return value


def _sha256_files(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path): sha256(path) for path in paths}


def _chunk_file(chunk: Chunk) -> str:
    value = chunk.metadata.get("file")
    return value if isinstance(value, str) and value else chunk.source


def _chunk_order(chunk: Chunk) -> tuple[int, str]:
    value = chunk.metadata.get("ord")
    return (value if isinstance(value, int) else 2**31, chunk.id)


def _relation_neighbors(
    chunks: Iterable[Chunk],
    *,
    relation_types: frozenset[str] | None = None,
) -> tuple[dict[str, tuple[tuple[str, str, str], ...]], dict[str, int]]:
    by_file: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_file[_chunk_file(chunk)].append(chunk)
    first_chunk = {
        file: min(rows, key=_chunk_order)
        for file, rows in by_file.items()
    }
    neighbors: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    for source_file, rows in by_file.items():
        for chunk in rows:
            graph = chunk.metadata.get("recall_graph")
            relations = graph.get("relations", ()) if isinstance(graph, dict) else ()
            if not isinstance(relations, list):
                continue
            for relation in relations:
                if not isinstance(relation, dict):
                    continue
                target = relation.get("object")
                edge_type = relation.get("structural_type")
                edge_key = relation.get("structural_key", "")
                if not isinstance(target, str) or target not in first_chunk:
                    continue
                if not isinstance(edge_type, str) or not edge_type:
                    continue
                if relation_types is not None and edge_type not in relation_types:
                    continue
                edge_key = edge_key if isinstance(edge_key, str) else str(edge_key)
                source_id = first_chunk[source_file].id
                target_id = first_chunk[target].id
                if source_id == target_id:
                    continue
                neighbors[source_id].append((target_id, edge_type, edge_key))
                neighbors[target_id].append((source_id, edge_type, edge_key))
                counts[edge_type] += 1
    return (
        {key: tuple(sorted(set(value), key=lambda item: (item[1], item[2], item[0]))) for key, value in neighbors.items()},
        dict(sorted(counts.items())),
    )


def _contexts(
    seed_hits: list[ScoredChunk],
    neighbors: dict[str, tuple[tuple[str, str, str], ...]],
    chunks_by_id: dict[str, Chunk],
    *,
    context_k: int,
    edge_budget: int,
    neighbor_order: str = "structural",
    retrieval_scores: dict[str, float] | None = None,
    direct_fallback: Iterable[ScoredChunk] = (),
    min_score: float | None = None,
) -> tuple[list[ScoredChunk], list[dict[str, str | float]]]:
    selected = list(seed_hits[:context_k])
    selected_ids = {hit.chunk.id for hit in selected}
    seed_ids = [hit.chunk.id for hit in seed_hits[:context_k]]
    additions: list[tuple[str, str, str]] = []
    for seed_id in seed_ids:
        additions.extend(
            (target_id, edge_type, edge_key)
            for target_id, edge_type, edge_key in neighbors.get(seed_id, ())
            if target_id not in selected_ids
        )
    unique: dict[str, tuple[str, str, str]] = {}
    for target_id, edge_type, edge_key in additions:
        unique.setdefault(target_id, (target_id, edge_type, edge_key))
    if neighbor_order == "retrieval":
        scores = retrieval_scores or {}
        ordered = sorted(
            unique.values(),
            key=lambda item: (
                0 if item[0] in scores else 1,
                -scores.get(item[0], 0.0),
                item[1],
                item[2],
                item[0],
            ),
        )
    else:
        ordered = sorted(unique.values(), key=lambda item: (item[1], item[2], item[0]))
    details: list[dict[str, str | float]] = []
    for target_id, edge_type, edge_key in ordered:
        if len(details) >= edge_budget or len(selected) >= context_k:
            break
        if min_score is not None and (retrieval_scores or {}).get(target_id, float("-inf")) < min_score:
            continue
        chunk = chunks_by_id.get(target_id)
        if chunk is None:
            continue
        selected.append(ScoredChunk(chunk=chunk, score=0.0, score_kind="structural"))
        selected_ids.add(target_id)
        details.append(
            {
                "chunk_id": target_id,
                "structural_type": edge_type,
                "structural_key": edge_key,
                "retrieval_score": (retrieval_scores or {}).get(target_id, 0.0),
            }
        )
    for hit in direct_fallback:
        if len(selected) >= context_k:
            break
        if hit.chunk.id in selected_ids:
            continue
        selected.append(hit)
        selected_ids.add(hit.chunk.id)
    return selected, details


def _ids_for_hits(hits: Iterable[ScoredChunk], *, dataset: str) -> list[str]:
    result: list[str] = []
    for hit in hits:
        value = hit.chunk.metadata.get("benchmark_id") or hit.chunk.metadata.get("file")
        if not isinstance(value, str) or not value:
            value = hit.chunk.id
        if dataset == "locomo":
            value = value.removesuffix(".md").replace("_", ":", 1)
        if value not in result:
            result.append(value)
    return result


def _context_payload(hits: Iterable[ScoredChunk], *, dataset: str) -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    seen: set[str] = set()
    for hit in hits:
        value = hit.chunk.metadata.get("benchmark_id") or hit.chunk.metadata.get("file")
        benchmark_id = value if isinstance(value, str) and value else hit.chunk.id
        if dataset == "locomo":
            benchmark_id = benchmark_id.removesuffix(".md").replace("_", ":", 1)
        if benchmark_id in seen:
            continue
        seen.add(benchmark_id)
        payload.append({"id": benchmark_id, "chunk_id": hit.chunk.id, "text": hit.chunk.text})
    return payload


def _metrics(ids: list[str], gold: list[str]) -> dict[str, float | None]:
    gold_set = set(gold)
    if not gold_set:
        return {"any_hit": None, "complete_hit": None, "mrr": None, "precision": None}
    rank = next((index + 1 for index, value in enumerate(ids) if value in gold_set), None)
    return {
        "any_hit": 1.0 if rank is not None else 0.0,
        "complete_hit": 1.0 if gold_set.issubset(ids) else 0.0,
        "mrr": 1.0 / rank if rank is not None else 0.0,
        "precision": len(gold_set.intersection(ids)) / len(ids) if ids else 0.0,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float | None:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return statistics.fmean(values) if values else None

    def p95(key: str) -> float | None:
        values = sorted(float(row[key]) for row in rows if row.get(key) is not None)
        if not values:
            return None
        return values[min(len(values) - 1, int(round((len(values) - 1) * 0.95)))]

    return {
        "questions": len(rows),
        "any_hit": mean("any_hit"),
        "complete_hit": mean("complete_hit"),
        "mrr": mean("mrr"),
        "precision": mean("precision"),
        "mean_context_items": mean("context_items"),
        "mean_added_items": mean("added_items"),
        "retrieval_latency_ms_mean": mean("retrieval_latency_ms"),
        "retrieval_latency_ms_p95": p95("retrieval_latency_ms"),
    }


def _arm_output(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("category") is not None:
            by_category[str(row["category"])].append(row)
    return {
        "summary": _aggregate(rows),
        "by_category": {key: _aggregate(value) for key, value in sorted(by_category.items())},
        "rows": rows,
    }


def _locomo_cases(data: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for conversation_index, sample in enumerate(data[:limit] if limit is not None else data):
        conversation = sample.get("conversation")
        if not isinstance(conversation, dict):
            continue
        for question_index, row in enumerate(sample.get("qa", ())):
            if not isinstance(row, dict) or row.get("category") not in {1, 2, 3, 4}:
                continue
            question = row.get("question")
            evidence = [value for value in row.get("evidence", ()) if isinstance(value, str)]
            if isinstance(question, str) and evidence:
                cases.append(
                    {
                        "id": f"conversation_{conversation_index}:qa_{question_index}",
                        "question": question,
                        "gold": evidence,
                        "category": row["category"],
                        "conversation_index": conversation_index,
                    }
                )
    return cases


def _parse_relation_types(value: str) -> frozenset[str] | None:
    if value.strip().casefold() in {"", "all"}:
        return None
    return frozenset(item.strip() for item in value.split(",") if item.strip())


def _configurations(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    relation_types = _parse_relation_types(args.relation_types)
    single = {
        "seed_k": args.seed_k,
        "edge_budget": args.edge_budget if args.edge_budget is not None else args.context_k - args.seed_k,
        "context_k": args.context_k,
        "retrieval_k": args.retrieval_k or args.context_k,
        "relation_types": relation_types,
        "neighbor_order": args.neighbor_order,
        "direct_fallback": False,
        "activation_categories": None,
        "graph_score_margin": None,
    }
    if args.selective_gate:
        activation_categories = (
            frozenset({args.selective_category})
            if args.selective_category is not None
            else None
        )
        existing = {
            **single,
            "seed_k": 8,
            "edge_budget": 2,
            "retrieval_k": 20,
            "neighbor_order": "retrieval",
            "direct_fallback": True,
            "graph_score_margin": 0.05,
            "activation_categories": activation_categories,
        }
        strict = {
            **existing,
            "graph_score_margin": args.selective_margin,
        }
        return [("selective_margin_005", existing), ("selective_margin_strict", strict)]
    if not args.sweep:
        return [("structural_edges", single)]
    return [
        ("current", {**single, "seed_k": 5, "edge_budget": 5, "retrieval_k": 20, "neighbor_order": "structural"}),
        ("more_direct", {**single, "seed_k": 8, "edge_budget": 2, "retrieval_k": 20, "neighbor_order": "structural"}),
        ("semantic_order", {**single, "seed_k": 5, "edge_budget": 5, "retrieval_k": 20, "neighbor_order": "retrieval"}),
        ("semantic_order_more_direct", {**single, "seed_k": 8, "edge_budget": 2, "retrieval_k": 20, "neighbor_order": "retrieval"}),
        ("semantic_order_most_direct", {**single, "seed_k": 9, "edge_budget": 1, "retrieval_k": 20, "neighbor_order": "retrieval"}),
        ("conversation_order_only", {**single, "seed_k": 5, "edge_budget": 5, "retrieval_k": 20, "relation_types": frozenset({"conversation_order"}), "neighbor_order": "structural"}),
        ("category_selective", {**single, "seed_k": 8, "edge_budget": 2, "retrieval_k": 20, "neighbor_order": "retrieval", "activation_categories": frozenset({3, 4})}),
    ]


def _configuration_output(config: dict[str, Any]) -> dict[str, Any]:
    return {
        **config,
        "relation_types": sorted(config["relation_types"]) if config["relation_types"] is not None else "all",
        "activation_categories": sorted(config["activation_categories"]) if config["activation_categories"] is not None else "all",
    }


def _run_locomo(args: argparse.Namespace) -> dict[str, Any]:
    data_path = args.data
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("LoCoMo input must be a JSON list")
    cases = _locomo_cases(data, args.conversations)
    embedder = resolve_embedder(args.embedder)
    configs = _configurations(args)
    baseline_rows: list[dict[str, Any]] = []
    treatment_rows: dict[str, list[dict[str, Any]]] = {name: [] for name, _config in configs}
    edge_counts: dict[str, int] = defaultdict(int)
    selected_data = data[: args.conversations] if args.conversations is not None else data
    for index, sample in enumerate(selected_data):
        conversation = sample.get("conversation")
        if not isinstance(conversation, dict):
            continue
        tenant = f"{args.run_id}-locomo-{index}"
        with PgVectorStore(args.dsn, dim=embedder.dim, table=args.table, tenant=tenant) as store:
            store.ensure_schema()
            corpus_dir = Path(tempfile.mkdtemp(prefix="structural-locomo-"))
            try:
                write_conversation_corpus(conversation, corpus_dir)
                Indexer(store, embedder).index_path(corpus_dir)
                chunks = list(store.iter_chunks())
                chunks_by_id = {chunk.id: chunk for chunk in chunks}
                _all_neighbors, counts = _relation_neighbors(chunks)
                for key, value in counts.items():
                    edge_counts[key] += value
                neighbors_by_types: dict[frozenset[str] | None, dict[str, tuple[tuple[str, str, str], ...]]] = {}
                retriever = HybridRetriever(store, embedder, candidate_k=args.candidate_k)
                max_retrieval_k = max(config["retrieval_k"] for _name, config in configs)
                for case in (item for item in cases if item["conversation_index"] == index):
                    started = time.perf_counter()
                    full = retriever.search(case["question"], k=max_retrieval_k)
                    retrieval_ms = (time.perf_counter() - started) * 1000.0
                    scored_hits = list(full.hits)
                    score_by_id = {hit.chunk.id: hit.score for hit in scored_hits}
                    baseline_hits = scored_hits[: args.context_k]
                    baseline_ids = _ids_for_hits(baseline_hits, dataset="locomo")
                    baseline_context = _context_payload(baseline_hits, dataset="locomo")
                    base = {
                        "id": case["id"],
                        "question": case["question"],
                        "category": case["category"],
                        "gold": case["gold"],
                        "retrieval_latency_ms": round(retrieval_ms, 3),
                        "context_items": len(baseline_ids),
                        "added_items": 0,
                    }
                    baseline_rows.append({**base, "arm": "baseline", **_metrics(baseline_ids, case["gold"]), "context_ids": baseline_ids, "context": baseline_context, "additions": []})
                    for name, config in configs:
                        active_categories = config["activation_categories"]
                        if active_categories is not None and case["category"] not in active_categories:
                            additions: list[dict[str, Any]] = []
                            treatment_hits = baseline_hits
                        else:
                            relation_types = config["relation_types"]
                            if relation_types not in neighbors_by_types:
                                neighbors_by_types[relation_types], _ = _relation_neighbors(chunks, relation_types=relation_types)
                            direct_tail = scored_hits[config["seed_k"] : config["context_k"]]
                            direct_tail_score = min((hit.score for hit in direct_tail), default=None)
                            graph_score_margin = config.get("graph_score_margin")
                            treatment_hits, additions = _contexts(
                                scored_hits[: config["seed_k"]],
                                neighbors_by_types[relation_types],
                                chunks_by_id,
                                context_k=config["context_k"],
                                edge_budget=config["edge_budget"],
                                neighbor_order=config["neighbor_order"],
                                retrieval_scores=score_by_id,
                                direct_fallback=scored_hits[config["seed_k"]: config["context_k"]] if config["direct_fallback"] else (),
                                min_score=(direct_tail_score + graph_score_margin) if direct_tail_score is not None and graph_score_margin is not None else None,
                            )
                        treatment_ids = _ids_for_hits(treatment_hits, dataset="locomo")
                        treatment_rows[name].append({**base, "arm": name, "context_items": len(treatment_ids), "added_items": len(additions), "context_ids": treatment_ids, "context": _context_payload(treatment_hits, dataset="locomo"), "additions": additions, **_metrics(treatment_ids, case["gold"])})
            finally:
                for path in sorted(corpus_dir.rglob("*"), reverse=True):
                    if path.is_file():
                        path.unlink()
                corpus_dir.rmdir()
            store.delete_sources(sorted({chunk.source for chunk in chunks}))
    return {
        "dataset": "locomo",
        "data_sha256": _sha256_files([data_path]),
        "conversations": len(selected_data),
        "cases": len(baseline_rows),
        "edge_counts": dict(sorted(edge_counts.items())),
        "configuration_set": {name: _configuration_output(config) for name, config in configs},
        "arms": {"baseline": _arm_output(baseline_rows)}
        | {name: _arm_output(rows) for name, rows in treatment_rows.items()},
    }


def _run_atm(args: argparse.Namespace) -> dict[str, Any]:
    paths = [args.qa_file, args.image_file, args.video_file, args.email_file]
    if not all(path.exists() for path in paths):
        missing = [str(path) for path in paths if not path.exists()]
        return {"dataset": "atm", "status": "not_run", "missing_inputs": missing}
    questions = load_questions(args.qa_file)
    base = build_memory_items(args.image_file, args.video_file, args.email_file)
    treatment = build_memory_items_with_structural_edges(args.image_file, args.video_file, args.email_file)
    graph = {
        evidence_id: metadata.get("recall_graph", {"schema_version": 1, "relations": []})
        for evidence_id, _modality, _text, metadata in treatment
    }
    edge_counts: dict[str, int] = defaultdict(int)
    for value in graph.values():
        for relation in value.get("relations", ()):
            edge_counts[str(relation["structural_type"])] += 1
    embedder = resolve_embedder(args.embedder)
    rows_by_arm: dict[str, list[dict[str, Any]]] = {"baseline": [], "structural_edges": []}
    with PgVectorStore(args.dsn, dim=embedder.dim, table=args.table, tenant=f"{args.run_id}-atm") as store:
        store.ensure_schema()
        chunks = [
            Chunk(id=evidence_id, source=evidence_id, text=text, metadata={**metadata, "file": evidence_id, "benchmark_id": evidence_id, "recall_graph": graph.get(evidence_id, {"schema_version": 1, "relations": []})})
            for evidence_id, modality, text, metadata in treatment
        ]
        store.upsert(chunks, embed_passages(embedder, [chunk.text for chunk in chunks]))
        store.analyze()
        stored = list(store.iter_chunks())
        chunks_by_id = {chunk.id: chunk for chunk in stored}
        neighbors, _counts = _relation_neighbors(
            stored,
            relation_types=_parse_relation_types(args.relation_types),
        )
        retriever = HybridRetriever(store, embedder, candidate_k=args.candidate_k)
        for question in questions:
            started = time.perf_counter()
            full = retriever.search(question["question"], k=args.context_k)
            baseline_hits = list(full.hits[: args.context_k])
            treatment_hits, additions = _contexts(
                baseline_hits[: args.seed_k],
                neighbors,
                chunks_by_id,
                context_k=args.context_k,
                edge_budget=args.edge_budget if args.edge_budget is not None else args.context_k - args.seed_k,
                neighbor_order=args.neighbor_order,
            )
            retrieval_ms = round((time.perf_counter() - started) * 1000.0, 3)
            for arm, hits, arm_additions in (("baseline", baseline_hits, []), ("structural_edges", treatment_hits, additions)):
                ids = _ids_for_hits(hits, dataset="atm")
                rows_by_arm[arm].append({"id": question["id"], "question": question["question"], "gold": question["evidence_ids"], "arm": arm, "context_ids": ids, "context": _context_payload(hits, dataset="atm"), "additions": arm_additions, "context_items": len(ids), "added_items": len(arm_additions), "retrieval_latency_ms": retrieval_ms, **_metrics(ids, question["evidence_ids"])})
        store.delete_sources(sorted({chunk.source for chunk in stored}))
    return {"dataset": "atm", "status": "measured", "data_sha256": _sha256_files(paths), "memory_items": len(base), "questions": len(questions), "edge_counts": dict(sorted(edge_counts.items())), "arms": {arm: _arm_output(rows) for arm, rows in rows_by_arm.items()}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("locomo", "atm"), required=True)
    parser.add_argument("--run-id", type=_run_id, default=datetime.now(timezone.utc).strftime("20260911T%H%M%SZ"))
    parser.add_argument("--dsn", default=os.environ.get("RECALL_DSN", DEFAULT_DSN))
    parser.add_argument("--table", default="structural_edge_performance")
    parser.add_argument("--embedder", default="fastembed")
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--seed-k", type=int, default=DEFAULT_SEED_K)
    parser.add_argument("--context-k", type=int, default=DEFAULT_CONTEXT_K)
    parser.add_argument("--edge-budget", type=int, default=None)
    parser.add_argument("--relation-types", default="all", help="comma-separated structural edge types, or all")
    parser.add_argument("--neighbor-order", choices=("structural", "retrieval"), default="structural")
    parser.add_argument("--retrieval-k", type=int, default=None)
    parser.add_argument("--sweep", action="store_true", help="run the preregistered configuration sweep")
    parser.add_argument("--selective-gate", action="store_true", help="compare unfiltered and score-gated 8 direct plus 2 graph arms")
    parser.add_argument("--selective-margin", type=float, default=0.10, help="strict selective graph score margin")
    parser.add_argument("--selective-category", type=int, choices=(1, 2, 3, 4), default=None, help="limit selective graph activation to one LOCOMO category")
    parser.add_argument("--data", type=Path, default=Path("locomo10.json"))
    parser.add_argument("--conversations", type=int, default=None)
    parser.add_argument("--qa-file", type=Path, default=Path("atm_questions.json"))
    parser.add_argument("--image-file", type=Path, default=Path("atm_images.json"))
    parser.add_argument("--video-file", type=Path, default=Path("atm_videos.json"))
    parser.add_argument("--email-file", type=Path, default=Path("atm_emails.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.seed_k < 1 or args.context_k < args.seed_k or args.candidate_k < 1:
        parser.error("candidate-k and seed-k must be positive, and context-k must be at least seed-k")
    if args.edge_budget is not None and args.edge_budget < 0:
        parser.error("edge-budget must not be negative")
    if args.retrieval_k is not None and args.retrieval_k < args.context_k:
        parser.error("retrieval-k must be at least context-k")
    result = _run_locomo(args) if args.dataset == "locomo" else _run_atm(args)
    artifact = {
        "protocol": PROTOCOL,
        "run_id": args.run_id,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": os.environ.get("RECALL_SOURCE_COMMIT"),
        "embedder": args.embedder,
        "candidate_k": args.candidate_k,
        "seed_k": args.seed_k,
        "context_k": args.context_k,
        "answer_stage": "separate OpenRouter replay required",
        **result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": args.dataset, "run_id": args.run_id, "out": str(args.out), "status": result.get("status", "measured"), "arms": {name: value.get("summary") for name, value in result.get("arms", {}).items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
