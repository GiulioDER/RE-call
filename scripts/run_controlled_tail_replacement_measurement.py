"""Measure the preregistered calibrated graph tail replacement arms on LoCoMo."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_SAFE_RUN_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
DEFAULT_DSN = "postgresql://recall:recall@localhost:5432/recall"
MARGIN = 0.05
CALIBRATION = {"embedder": "fastembed", "threshold": 0.50, "scale": 0.05}

sys.path.insert(0, str(ROOT))

from benchmarks.atm_bench import sha256  # noqa: E402
from benchmarks.structural_edge_performance import (  # noqa: E402
    _arm_output,
    _context_payload,
    _ids_for_hits,
    _metrics,
    _relation_neighbors,
)
from recall.calibration import Calibration  # noqa: E402
from recall.embeddings import resolve_embedder  # noqa: E402
from recall.eval.locomo import write_conversation_corpus  # noqa: E402
from recall.index import Indexer  # noqa: E402
from recall.retriever import DEFAULT_CANDIDATE_K, HybridRetriever  # noqa: E402
from recall.store import PgVectorStore  # noqa: E402
from recall.types import Chunk, ScoredChunk  # noqa: E402


def _run_id(value: str) -> str:
    if not _SAFE_RUN_ID.fullmatch(value):
        raise argparse.ArgumentTypeError("run id must be a safe ASCII identifier")
    return value


def _sha256(path: Path) -> str:
    return sha256(path)


def _locomo_cases(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for conversation_index, sample in enumerate(data):
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


def _confidence(calibration: Calibration, score: float) -> float:
    return calibration.confidence(float(score))


def _graph_candidates(
    seed_hits: list[ScoredChunk],
    neighbors: dict[str, tuple[tuple[str, str, str], ...]],
    chunks_by_id: dict[str, Chunk],
    direct_context_ids: set[str],
    retrieval_scores: dict[str, float],
) -> list[tuple[ScoredChunk, dict[str, str]]]:
    seed_ids = [hit.chunk.id for hit in seed_hits]
    unique: dict[str, tuple[str, str, str]] = {}
    for seed_id in seed_ids:
        for target_id, edge_type, edge_key in neighbors.get(seed_id, ()):
            if target_id in direct_context_ids:
                # The protected prefix and direct tail are not graph replacements.
                continue
            unique.setdefault(target_id, (target_id, edge_type, edge_key))
    ordered = sorted(
        unique.values(),
        key=lambda item: (
            0 if item[0] in retrieval_scores else 1,
            -retrieval_scores.get(item[0], 0.0),
            item[1],
            item[2],
            item[0],
        ),
    )
    result: list[tuple[ScoredChunk, dict[str, str]]] = []
    for target_id, edge_type, edge_key in ordered:
        chunk = chunks_by_id.get(target_id)
        if chunk is None:
            continue
        result.append(
            (
                ScoredChunk(
                    chunk=chunk,
                    score=retrieval_scores.get(target_id, 0.0),
                    score_kind="structural",
                ),
                {"chunk_id": target_id, "structural_type": edge_type, "structural_key": edge_key},
            )
        )
    return result


def _controlled_tail_context(
    scored_hits: list[ScoredChunk],
    neighbors: dict[str, tuple[tuple[str, str, str], ...]],
    chunks_by_id: dict[str, Chunk],
    *,
    context_k: int,
    seed_k: int,
    retrieval_scores: dict[str, float],
    calibration: Calibration,
) -> tuple[list[ScoredChunk], list[dict[str, str]], dict[str, int]]:
    direct_prefix = list(scored_hits[:seed_k])
    direct_tail = list(scored_hits[seed_k:context_k])
    direct_context_ids = {hit.chunk.id for hit in scored_hits[:context_k]}
    candidates = _graph_candidates(
        direct_prefix,
        neighbors,
        chunks_by_id,
        direct_context_ids,
        retrieval_scores,
    )
    selected = list(direct_prefix)
    additions: list[dict[str, str]] = []
    replacement_count = 0
    candidate_count = len(candidates)
    scored_count = sum(
        1 for candidate, _detail in candidates if candidate.chunk.id in retrieval_scores
    )
    tail_score = _confidence(calibration, direct_tail[0].score) if direct_tail else None
    if direct_tail and candidates:
        for candidate, detail in candidates:
            candidate_score = _confidence(calibration, candidate.score)
            if tail_score is not None and candidate_score > tail_score + MARGIN:
                selected.append(candidate)
                additions.append(detail)
                direct_tail = direct_tail[1:]
                replacement_count = 1
                break
    selected.extend(direct_tail)
    return (
        selected[:context_k],
        additions,
        {
            "graph_candidates": candidate_count,
            "graph_candidates_scored": scored_count,
            "tail_replacements": replacement_count,
        },
    )


def _base_row(case: dict[str, Any], retrieval_ms: float) -> dict[str, Any]:
    return {
        "id": case["id"],
        "question": case["question"],
        "category": case["category"],
        "gold": case["gold"],
        "retrieval_latency_ms": round(retrieval_ms, 3),
    }


def _append_row(
    rows: dict[str, list[dict[str, Any]]],
    arm: str,
    base: dict[str, Any],
    hits: list[ScoredChunk],
    additions: list[dict[str, str]],
) -> None:
    ids = _ids_for_hits(hits, dataset="locomo")
    rows[arm].append(
        {
            **base,
            "arm": arm,
            "context_items": len(ids),
            "added_items": len(additions),
            "context_ids": ids,
            "context": _context_payload(hits, dataset="locomo"),
            "additions": additions,
            **_metrics(ids, base["gold"]),
        }
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    data = json.loads(args.data.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("LoCoMo input must be a JSON list")
    cases = _locomo_cases(data)
    embedder = resolve_embedder(args.embedder)
    calibration = Calibration(**CALIBRATION)
    arms = (
        "baseline_k10",
        "tail_replacement_9_plus_1",
        "baseline_k5",
        "tail_replacement_4_plus_1",
        "established_8_plus_2",
    )
    rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
    edge_counts: dict[str, int] = defaultdict(int)
    selected_data = data[: args.conversations] if args.conversations is not None else data
    selected_indexes = {index for index, _sample in enumerate(selected_data)}
    for index, sample in enumerate(selected_data):
        conversation = sample.get("conversation")
        if not isinstance(conversation, dict):
            continue
        tenant = f"{args.run_id}-locomo-{index}"
        with PgVectorStore(args.dsn, dim=embedder.dim, table=args.table, tenant=tenant) as store:
            store.ensure_schema()
            corpus_dir = Path(tempfile.mkdtemp(prefix="controlled-tail-locomo-"))
            try:
                write_conversation_corpus(conversation, corpus_dir)
                Indexer(store, embedder).index_path(corpus_dir)
                chunks = list(store.iter_chunks())
                chunks_by_id = {chunk.id: chunk for chunk in chunks}
                neighbors, counts = _relation_neighbors(chunks)
                for key, value in counts.items():
                    edge_counts[key] += value
                retriever = HybridRetriever(store, embedder, candidate_k=args.candidate_k)
                for case in (item for item in cases if item["conversation_index"] == index):
                    started = time.perf_counter()
                    full = retriever.search(case["question"], k=20)
                    retrieval_ms = (time.perf_counter() - started) * 1000.0
                    scored_hits = list(full.hits)
                    score_by_id = {hit.chunk.id: hit.score for hit in scored_hits}
                    base = _base_row(case, retrieval_ms)
                    baseline10 = scored_hits[:10]
                    baseline5 = scored_hits[:5]
                    _append_row(rows, "baseline_k10", base, baseline10, [])
                    _append_row(rows, "baseline_k5", base, baseline5, [])
                    controlled10, additions10, diagnostics10 = _controlled_tail_context(
                        scored_hits,
                        neighbors,
                        chunks_by_id,
                        context_k=10,
                        seed_k=9,
                        retrieval_scores=score_by_id,
                        calibration=calibration,
                    )
                    _append_row(rows, "tail_replacement_9_plus_1", base, controlled10, additions10)
                    controlled5, additions5, diagnostics5 = _controlled_tail_context(
                        scored_hits,
                        neighbors,
                        chunks_by_id,
                        context_k=5,
                        seed_k=4,
                        retrieval_scores=score_by_id,
                        calibration=calibration,
                    )
                    _append_row(rows, "tail_replacement_4_plus_1", base, controlled5, additions5)
                    established = list(scored_hits[:8])
                    established_additions: list[dict[str, str]] = []
                    unique: dict[str, tuple[str, str, str]] = {}
                    for seed in established:
                        for target_id, edge_type, edge_key in neighbors.get(seed.chunk.id, ()):
                            if target_id not in {hit.chunk.id for hit in established}:
                                unique.setdefault(target_id, (target_id, edge_type, edge_key))
                    ordered = sorted(
                        unique.values(),
                        key=lambda item: (
                            0 if item[0] in score_by_id else 1,
                            -score_by_id.get(item[0], 0.0),
                            item[1],
                            item[2],
                            item[0],
                        ),
                    )
                    established_hits = list(established)
                    selected_ids = {hit.chunk.id for hit in established_hits}
                    for target_id, edge_type, edge_key in ordered:
                        if len(established_hits) >= 10:
                            break
                        if target_id in selected_ids or target_id not in chunks_by_id:
                            continue
                        established_hits.append(
                            ScoredChunk(
                                chunk=chunks_by_id[target_id],
                                score=score_by_id.get(target_id, 0.0),
                                score_kind="structural",
                            )
                        )
                        selected_ids.add(target_id)
                        established_additions.append(
                            {
                                "chunk_id": target_id,
                                "structural_type": edge_type,
                                "structural_key": edge_key,
                            }
                        )
                    _append_row(
                        rows, "established_8_plus_2", base, established_hits, established_additions
                    )
                    rows["tail_replacement_9_plus_1"][-1].update(diagnostics10)
                    rows["tail_replacement_4_plus_1"][-1].update(diagnostics5)
            finally:
                for path in sorted(corpus_dir.rglob("*"), reverse=True):
                    if path.is_file():
                        path.unlink()
                corpus_dir.rmdir()
            store.delete_sources(sorted({chunk.source for chunk in chunks}))
    return {
        "dataset": "locomo",
        "data_sha256": _sha256(args.data),
        "conversations": len(selected_data),
        "cases": len(rows["baseline_k10"]),
        "edge_counts": dict(sorted(edge_counts.items())),
        "calibration": CALIBRATION,
        "tail_replacement_margin": MARGIN,
        "arms": {arm: _arm_output(value) for arm, value in rows.items()},
        "skipped_conversation_indexes": sorted(set(range(len(data))) - selected_indexes),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=_run_id, required=True)
    parser.add_argument("--dsn", default=os.environ.get("RECALL_DSN", DEFAULT_DSN))
    parser.add_argument("--table", default="controlled_tail_replacement")
    parser.add_argument("--embedder", default="fastembed")
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--data", type=Path, default=Path("locomo10.json"))
    parser.add_argument("--conversations", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.candidate_k < 1:
        parser.error("candidate-k must be positive")
    result = _run(args)
    artifact = {
        "protocol": "2026-09-11-controlled-tail-replacement-v1",
        "preregistration": "docs/preregistrations/2026-09-11-controlled-tail-replacement.md",
        "run_id": args.run_id,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": os.environ.get("RECALL_SOURCE_COMMIT"),
        "embedder": args.embedder,
        "candidate_k": args.candidate_k,
        "answer_stage": "separate answer replay required",
        **result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "out": str(args.out),
                "cases": result["cases"],
                "arms": {name: value["summary"] for name, value in result["arms"].items()},
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
