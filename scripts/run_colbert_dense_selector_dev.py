"""Measure ColBERT MaxSim ordering over the original dense top 20."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.mtrag.rerank_offload import rerank_order  # noqa: E402
from scripts.run_anchor_boosted_query_dev import (  # noqa: E402
    _cutoff_counts,
    _identity,
    _minimum_rank,
    _normalize,
    selected_supported_anchors,
    validate_baseline,
)
from scripts.run_live_graph_performance_attribution import _initialize  # noqa: E402
from scripts.run_live_source_conditioned_admission import _audit  # noqa: E402
from scripts.run_live_source_conditioning_shadow import _call_query  # noqa: E402
from scripts.run_live_tty_graph_precision import TTYMCP, _command  # noqa: E402


CANDIDATE_CUTOFF = 20
EXPECTED_MODEL = "colbert-ir/colbertv2.0"
EXPECTED_ARM = "li_colbertv2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _rank(rows: Sequence[Mapping[str, object]], key: str) -> int | None:
    return next((index for index, row in enumerate(rows, 1) if bool(row[key])), None)


def dense_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Join dense rank and cosine onto the full audited candidate rows."""

    pool = list(_audit(payload, "source_admission_benchmark_audit")["items"])
    by_id = {str(item["chunk_id"]): item for item in pool}
    dense = sorted(
        _audit(payload, "retrieval_leg_benchmark_audit")["dense"],
        key=lambda item: int(item["rank"]),
    )
    return [
        {
            **by_id[str(item["chunk_id"])],
            "dense_rank": int(item["rank"]),
            "dense_score": float(item["cosine"]),
        }
        for item in dense
        if str(item["chunk_id"]) in by_id
    ]


def validate_collection(rows: list[dict[str, Any]]) -> None:
    answerable = [row for row in rows if row["expected_answerability"] == "answerable"]
    controls = [row for row in rows if row["expected_answerability"] == "unanswerable"]
    summary = {
        "eligible_answerable": sum(bool(row["eligible"]) for row in answerable),
        "eligible_controls": sum(bool(row["eligible"]) for row in controls),
        "original_dense_gold_source_by_cutoff": _cutoff_counts(
            answerable, "original_min_gold_rank"
        ),
        "original_dense_exact_span_by_cutoff": _cutoff_counts(
            answerable, "original_min_exact_rank"
        ),
    }
    validate_baseline(summary)


def write_offload_inputs(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, int]:
    """Write only eligible query and document text to the model scoring surface."""

    queries: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    pools: list[dict[str, object]] = []
    docs: dict[str, str] = {}
    for row in rows:
        if not row["eligible"]:
            continue
        qid = str(row["query_index"])
        queries.append({"qid": qid, "text": str(row["query"])})
        candidate_ids: list[str] = []
        for candidate in row["candidates"]:
            doc_id = str(candidate["chunk_id"])
            text = str(candidate["text"])
            if doc_id in docs and docs[doc_id] != text:
                raise RuntimeError(f"chunk {doc_id!r} has inconsistent text")
            docs[doc_id] = text
            candidate_ids.append(doc_id)
            pairs.append({"qid": qid, "doc_id": doc_id})
        pools.append(
            {
                "task_id": qid,
                "query": str(row["query"]),
                "candidates": candidate_ids,
            }
        )
    _jsonl(output_dir / "queries.jsonl", queries)
    _jsonl(
        output_dir / "docs.jsonl",
        [{"doc_id": doc_id, "text": text} for doc_id, text in docs.items()],
    )
    _jsonl(output_dir / "pairs.jsonl", pairs)
    _jsonl(output_dir / "pools" / "dense-top20.jsonl", pools)
    return {"queries": len(queries), "documents": len(docs), "pairs": len(pairs)}


def load_scores(path: Path) -> tuple[dict[str, object], dict[str, dict[str, float]]]:
    header: dict[str, object] | None = None
    scores: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("_header"):
                if header is not None:
                    raise RuntimeError("score artifact has more than one header")
                header = dict(row)
                continue
            if row.get("arm") != EXPECTED_ARM:
                raise RuntimeError("score row has the wrong late interaction arm")
            scores.setdefault(str(row["qid"]), {})[str(row["doc_id"])] = float(
                row["score"]
            )
    if header is None:
        raise RuntimeError("score artifact has no identity header")
    if header.get("arm") != EXPECTED_ARM or header.get("checkpoint") != EXPECTED_MODEL:
        raise RuntimeError("score artifact has the wrong model identity")
    if header.get("deployable") is not True or header.get("licence") != "mit":
        raise RuntimeError("score artifact does not establish a deployable MIT model")
    if not header.get("fastembed_version"):
        raise RuntimeError("score artifact does not record the FastEmbed version")
    return header, scores


def apply_scores(
    rows: list[dict[str, Any]], scores: Mapping[str, Mapping[str, float]]
) -> list[dict[str, Any]]:
    measured: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        candidates = [dict(candidate) for candidate in row["candidates"]]
        original_ids = [str(candidate["chunk_id"]) for candidate in candidates]
        if row["eligible"]:
            qid = str(row["query_index"])
            if qid not in scores:
                raise RuntimeError(f"eligible row {qid} has no ColBERT scores")
            missing = [doc_id for doc_id in original_ids if doc_id not in scores[qid]]
            extras = [doc_id for doc_id in scores[qid] if doc_id not in set(original_ids)]
            if missing or extras:
                raise RuntimeError(
                    f"row {qid} score membership mismatch: missing={missing[:3]} extras={extras[:3]}"
                )
            order = rerank_order(original_ids, dict(scores[qid]))
            by_id = {str(candidate["chunk_id"]): candidate for candidate in candidates}
            candidates = [by_id[doc_id] for doc_id in order]
            for candidate in candidates:
                candidate["maxsim_score"] = scores[qid][str(candidate["chunk_id"])]
        row["reranked_candidates"] = candidates
        reranked_ids = [str(candidate["chunk_id"]) for candidate in candidates]
        row["membership_preserved"] = sorted(reranked_ids) == sorted(original_ids)
        row["reranked_min_gold_rank"] = _rank(candidates, "gold_source")
        row["reranked_min_exact_rank"] = _rank(candidates, "exact_span")
        row["rank1_changed"] = bool(row["eligible"]) and reranked_ids[0] != original_ids[0]
        row["changed_rank1_exact"] = bool(row["rank1_changed"]) and bool(
            candidates[0]["exact_span"]
        )
        row["changed_rank1_gold"] = bool(row["rank1_changed"]) and bool(
            candidates[0]["gold_source"]
        )
        measured.append(row)
    return measured


def _rank_change(before: int | None, after: int | None) -> str:
    before_value = before if before is not None else CANDIDATE_CUTOFF + 1
    after_value = after if after is not None else CANDIDATE_CUTOFF + 1
    if after_value < before_value:
        return "improved"
    if after_value > before_value:
        return "worsened"
    return "tied"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["expected_answerability"] == "answerable"]
    controls = [row for row in rows if row["expected_answerability"] == "unanswerable"]
    changed = [row for row in answerable if row["rank1_changed"]]
    rank_changes = [
        _rank_change(row["original_min_exact_rank"], row["reranked_min_exact_rank"])
        for row in answerable
    ]
    summary: dict[str, Any] = {
        "empty_base_rows": len(rows),
        "answerable_rows": len(answerable),
        "control_rows": len(controls),
        "eligible_answerable": sum(bool(row["eligible"]) for row in answerable),
        "eligible_controls": sum(bool(row["eligible"]) for row in controls),
        "original_dense_gold_source_by_cutoff": _cutoff_counts(
            answerable, "original_min_gold_rank"
        ),
        "original_dense_exact_span_by_cutoff": _cutoff_counts(
            answerable, "original_min_exact_rank"
        ),
        "maxsim_gold_source_by_cutoff": _cutoff_counts(
            answerable, "reranked_min_gold_rank"
        ),
        "maxsim_exact_span_by_cutoff": _cutoff_counts(
            answerable, "reranked_min_exact_rank"
        ),
        "membership_preserved_rows": sum(bool(row["membership_preserved"]) for row in rows),
        "changed_rank1_rows": len(changed),
        "changed_rank1_exact_rows": sum(bool(row["changed_rank1_exact"]) for row in changed),
        "changed_rank1_gold_rows": sum(bool(row["changed_rank1_gold"]) for row in changed),
        "changed_rank1_exact_precision": (
            sum(bool(row["changed_rank1_exact"]) for row in changed) / len(changed)
            if changed
            else 0.0
        ),
        "changed_rank1_gold_precision": (
            sum(bool(row["changed_rank1_gold"]) for row in changed) / len(changed)
            if changed
            else 0.0
        ),
        "exact_rank_improved_rows": rank_changes.count("improved"),
        "exact_rank_tied_rows": rank_changes.count("tied"),
        "exact_rank_worsened_rows": rank_changes.count("worsened"),
    }
    validate_baseline(summary)
    if summary["membership_preserved_rows"] != len(rows):
        raise RuntimeError("MaxSim changed candidate membership")
    if summary["maxsim_gold_source_by_cutoff"]["20"] != 10:
        raise RuntimeError("MaxSim changed gold source top 20 reachability")
    if summary["maxsim_exact_span_by_cutoff"]["20"] != 10:
        raise RuntimeError("MaxSim changed exact span top 20 reachability")
    proceed = (
        summary["maxsim_exact_span_by_cutoff"]["1"] >= 5
        and summary["maxsim_exact_span_by_cutoff"]["5"] >= 5
        and summary["changed_rank1_exact_rows"] >= 2
        and summary["changed_rank1_exact_precision"] >= 0.50
        and summary["eligible_controls"] == 0
    )
    summary["decision"] = (
        "PROCEED_FRESH_COLBERT_SELECTOR_VALIDATION"
        if proceed
        else "STOP_GENERIC_RERANKING_ON_THIS_COHORT"
    )
    return summary


def collect(args: argparse.Namespace) -> None:
    queries = list(json.loads(args.query_pool.read_text(encoding="utf-8"))["queries"])
    holdout = json.loads(args.holdout_result.read_text(encoding="utf-8"))
    indexes = [
        int(row["query_index"])
        for row in holdout["rows"]
        if int(row["base_count"]) == 0
    ]
    if holdout.get("status") != "COMPLETE" or len(indexes) != 30:
        raise RuntimeError("development screen requires the completed 30 row cohort")

    command = _command(
        args.tenant,
        args.embedder,
        args.index_root,
        args.profile,
        "combined",
        "none",
        20260825,
        32,
        0.10,
        args.generation_id,
        benchmark_retrieval_leg_audit=True,
        benchmark_source_admission_audit=True,
        source_conditioning_mode="shadow",
        source_conditioning_artifact=str(args.artifact).replace("\\", "/"),
        source_conditioning_sample_rate=1.0,
        source_conditioning_policy="guarded_spare_slot",
    )
    rows: list[dict[str, Any]] = []
    client = TTYMCP(command, args.timeout)
    started = time.perf_counter()
    try:
        request_id = _initialize(client)
        for completed, query_index in enumerate(indexes, 1):
            query = queries[query_index]
            query_text = str(query["query"])
            payload = _call_query(client, request_id, query_text)
            request_id += 1
            _identity(payload, args)
            if _audit(payload, "source_conditioning_shadow").get("alpha008_chunk_hashes"):
                raise RuntimeError("development row no longer has an empty base")
            original_pool = list(_audit(payload, "source_admission_benchmark_audit")["items"])
            dense = dense_candidates(payload)[:CANDIDATE_CUTOFF]
            if len(dense) != CANDIDATE_CUTOFF:
                raise RuntimeError("original dense audit did not return 20 candidates")
            anchors = selected_supported_anchors(query_text, original_pool)
            answerable = query["expected_answerability"] == "answerable"
            gold = {str(value) for value in query.get("gold_sources", [])}
            span = _normalize(query["answer_span"])

            def exact(item: Mapping[str, object]) -> bool:
                return answerable and span in _normalize(item["text"])

            def gold_source(item: Mapping[str, object]) -> bool:
                return answerable and str(item["source"]) in gold

            candidates = [
                {
                    "chunk_id": str(item["chunk_id"]),
                    "source": str(item["source"]),
                    "text": str(item["text"]),
                    "dense_score": float(item["dense_score"]),
                    "gold_source": gold_source(item),
                    "exact_span": exact(item),
                }
                for item in dense
            ]
            rows.append(
                {
                    "query_index": query_index,
                    "query": query_text,
                    "expected_answerability": str(query["expected_answerability"]),
                    "anchors": list(anchors),
                    "eligible": len(anchors) == 3,
                    "original_min_gold_rank": _minimum_rank(dense, gold_source),
                    "original_min_exact_rank": _minimum_rank(dense, exact),
                    "candidates": candidates,
                }
            )
            print(f"ColBERT dense collection {completed}/{len(indexes)}", flush=True)
    finally:
        client.close()

    validate_collection(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = write_offload_inputs(rows, args.output_dir)
    collection = {
        "schema_version": 1,
        "protocol": "2026-09-15-colbert-maxsim-dense-top20-collection",
        "collected_at": datetime.now(UTC).isoformat(),
        "source_commit": os.environ.get("RECALL_SOURCE_COMMIT"),
        "preregistration_commit": os.environ.get("RECALL_POLICY_COMMIT"),
        "query_pool_sha256": _sha256(args.query_pool),
        "holdout_result_sha256": _sha256(args.holdout_result),
        "generation_id": args.generation_id,
        "calibration_id": args.calibration_id,
        "pipeline_fingerprint": args.pipeline_fingerprint,
        "corpus_fingerprint": args.corpus_fingerprint,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "offload_counts": counts,
        "rows": rows,
    }
    (args.output_dir / "collection.json").write_text(
        json.dumps(collection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({key: value for key, value in collection.items() if key != "rows"}))


def report(args: argparse.Namespace) -> None:
    collection = json.loads(args.collection.read_text(encoding="utf-8"))
    header, scores = load_scores(args.scores)
    rows = apply_scores(list(collection["rows"]), scores)
    summary = summarize(rows)
    result = {
        "schema_version": 1,
        "protocol": "2026-09-15-colbert-maxsim-dense-top20-development",
        "measured_at": datetime.now(UTC).isoformat(),
        "source_commit": collection.get("source_commit"),
        "preregistration_commit": collection.get("preregistration_commit"),
        "query_pool_sha256": collection["query_pool_sha256"],
        "holdout_result_sha256": collection["holdout_result_sha256"],
        "generation_id": collection["generation_id"],
        "calibration_id": collection["calibration_id"],
        "pipeline_fingerprint": collection["pipeline_fingerprint"],
        "corpus_fingerprint": collection["corpus_fingerprint"],
        "collection_sha256": _sha256(args.collection),
        "scores_sha256": _sha256(args.scores),
        "score_identity": header,
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--query-pool", type=Path, required=True)
    parser.add_argument("--holdout-result", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument("--pipeline-fingerprint", required=True)
    parser.add_argument("--corpus-fingerprint", required=True)
    parser.add_argument("--tenant", default="memory")
    parser.add_argument("--embedder", default="voyage-context:voyage-context-4")
    parser.add_argument("--index-root", default="/home/sentiment/recall-repos/memory")
    parser.add_argument("--profile", default="fast")
    parser.add_argument("--timeout", type=float, default=240)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    _common(collect_parser)
    collect_parser.add_argument("--output-dir", type=Path, required=True)
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--collection", type=Path, required=True)
    report_parser.add_argument("--scores", type=Path, required=True)
    report_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
