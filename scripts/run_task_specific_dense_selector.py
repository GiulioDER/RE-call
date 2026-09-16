"""Collect and evaluate a task specific selector over production dense pools."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recall.index import chunk_text  # noqa: E402
from scripts.run_live_graph_performance_attribution import _initialize  # noqa: E402
from scripts.run_live_tty_graph_precision import (  # noqa: E402
    TTYMCP,
    _command,
    _extract_payload,
)


POOL_SHA256 = "66ec82a058c9b06cf80314a1001779a1608e5144e097ace28b91664a48ede855"
AUDIT_SHA256 = "842e9bcbd279449222f773c20d27d6556cb3331ea388b6b51d3ca3788c055f18"
SPLIT_SEED = "extractive-selector-source-split-v1"
EXPECTED_SPLITS = {"train": 186, "validation": 33, "internal_test": 29}
CANDIDATE_COUNT = 20
NEGATIVES_PER_QUERY = 4
WHITESPACE = re.compile(r"\s+")


def _audit(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = (
        payload.get("diagnostics", {})
        .get("performance", {})
        .get("values", {})
        .get(name)
    )
    if not isinstance(value, dict):
        raise ValueError(f"{name} is missing")
    return value


def _call_query(client: TTYMCP, request_id: int, query: str) -> dict[str, Any]:
    response = client.call(
        request_id,
        "tools/call",
        {
            "name": "recall_reasoning_query",
            "arguments": {
                "query": query,
                "k": 5,
                "mode": "evidence_assembly",
                "max_steps": 12,
                "max_graph_nodes": 1,
                "max_evidence_tokens": 2048,
                "graph_expansion": "off",
            },
        },
    )
    payload = json.loads(_extract_payload(response))
    if not isinstance(payload, dict):
        raise ValueError("reasoning query payload must be an object")
    return payload


def dense_candidates(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize(value: object) -> str:
    return WHITESPACE.sub(" ", str(value)).strip()


def _split(source_sha256: str, seed: str = SPLIT_SEED) -> str:
    digest = hashlib.sha256(f"{seed}\0{source_sha256}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket <= 69:
        return "train"
    if bucket <= 84:
        return "validation"
    return "internal_test"


def prepare_rows(
    rows: Sequence[Mapping[str, Any]], *, enforce_expected_counts: bool = False
) -> list[dict[str, Any]]:
    """Remove controls and every repeated answer hash, then add the frozen source split."""

    answerable = [
        dict(row) for row in rows if row.get("expected_answerability") == "answerable"
    ]
    hashes = Counter(str(row.get("answer_span_sha256", "")) for row in answerable)
    prepared = []
    for row in answerable:
        if hashes[str(row.get("answer_span_sha256", ""))] != 1:
            continue
        row["split"] = _split(str(row["source_sha256"]))
        prepared.append(row)
    if enforce_expected_counts:
        observed = Counter(str(row["split"]) for row in prepared)
        if dict(observed) != EXPECTED_SPLITS:
            raise RuntimeError(
                f"deduplicated split mismatch: expected {EXPECTED_SPLITS}, got {dict(observed)}"
            )
    return prepared


def _cutoff_counts(rows: Sequence[Mapping[str, Any]], order_key: str, label: str) -> dict[str, int]:
    return {
        str(cutoff): sum(
            any(bool(candidate[label]) for candidate in _ordered_candidates(row, order_key)[:cutoff])
            for row in rows
        )
        for cutoff in (1, 3, 5, 10, 20)
    }


def _ordered_candidates(row: Mapping[str, Any], order_key: str) -> list[Mapping[str, Any]]:
    candidates = list(row["candidates"])
    if order_key == "dense_order":
        return candidates
    by_id = {str(candidate["chunk_id"]): candidate for candidate in candidates}
    return [by_id[str(chunk_id)] for chunk_id in row[order_key]]


def summarize_scored_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dense_exact = _cutoff_counts(rows, "dense_order", "exact_span")
    dense_gold = _cutoff_counts(rows, "dense_order", "gold_source")
    base_exact = _cutoff_counts(rows, "base_order", "exact_span")
    base_gold = _cutoff_counts(rows, "base_order", "gold_source")
    trained_exact = _cutoff_counts(rows, "trained_order", "exact_span")
    trained_gold = _cutoff_counts(rows, "trained_order", "gold_source")
    memberships_preserved = 0
    changed = 0
    exact_gains = 0
    exact_losses = 0
    gold_gains = 0
    gold_losses = 0
    changed_exact = 0
    changed_gold = 0
    pools_reordered_vs_base = 0
    for row in rows:
        candidates = list(row["candidates"])
        dense_ids = [str(candidate["chunk_id"]) for candidate in candidates]
        trained_ids = [str(value) for value in row["trained_order"]]
        base_ids = [str(value) for value in row["base_order"]]
        if sorted(dense_ids) == sorted(trained_ids) == sorted(base_ids):
            memberships_preserved += 1
        if trained_ids != base_ids:
            pools_reordered_vs_base += 1
        if not dense_ids or not trained_ids:
            continue
        if trained_ids[0] != dense_ids[0]:
            changed += 1
            by_id = {str(candidate["chunk_id"]): candidate for candidate in candidates}
            dense_first = by_id[dense_ids[0]]
            trained_first = by_id[trained_ids[0]]
            dense_first_exact = bool(dense_first["exact_span"])
            trained_first_exact = bool(trained_first["exact_span"])
            dense_first_gold = bool(dense_first["gold_source"])
            trained_first_gold = bool(trained_first["gold_source"])
            changed_exact += trained_first_exact
            changed_gold += trained_first_gold
            exact_gains += trained_first_exact and not dense_first_exact
            exact_losses += dense_first_exact and not trained_first_exact
            gold_gains += trained_first_gold and not dense_first_gold
            gold_losses += dense_first_gold and not trained_first_gold
    return {
        "rows": len(rows),
        "dense_exact_by_cutoff": dense_exact,
        "dense_gold_by_cutoff": dense_gold,
        "base_exact_by_cutoff": base_exact,
        "base_gold_by_cutoff": base_gold,
        "trained_exact_by_cutoff": trained_exact,
        "trained_gold_by_cutoff": trained_gold,
        "memberships_preserved": memberships_preserved,
        "pools_reordered_vs_base": pools_reordered_vs_base,
        "changed_rank1_rows": changed,
        "exact_rank1_gains": exact_gains,
        "exact_rank1_losses": exact_losses,
        "gold_rank1_gains": gold_gains,
        "gold_rank1_losses": gold_losses,
        "changed_rank1_exact_precision": changed_exact / changed if changed else 0.0,
        "changed_rank1_gold_precision": changed_gold / changed if changed else 0.0,
    }


def _resolve_source(source: str, roots: dict[str, Path]) -> Path:
    prefix, separator, relative = source.partition("/")
    if not separator or prefix not in roots or not relative:
        raise RuntimeError(f"unknown source root for {source!r}")
    root = roots[prefix].resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"source escapes root: {source!r}") from exc
    if not path.is_file():
        raise RuntimeError(f"source is unavailable: {source!r}")
    return path


def _positive_text(row: Mapping[str, Any], roots: dict[str, Path]) -> str:
    sources = list(row.get("gold_sources", []))
    if len(sources) != 1:
        raise RuntimeError(f"row {row.get('id')} does not have one gold source")
    path = _resolve_source(str(sources[0]), roots)
    if _sha256(path) != row.get("source_sha256"):
        raise RuntimeError(f"row {row.get('id')} source hash changed")
    chunks = [_normalize(chunk) for chunk in chunk_text(path.read_text(encoding="utf-8"))]
    ordinal = int(row["gold_ordinal"])
    if ordinal < 0 or ordinal >= len(chunks):
        raise RuntimeError(f"row {row.get('id')} has an invalid gold ordinal")
    positive = chunks[ordinal]
    if _normalize(row["answer_span"]) not in positive:
        raise RuntimeError(f"row {row.get('id')} positive does not contain its answer span")
    return positive


def _identity(payload: Mapping[str, Any], args: argparse.Namespace) -> None:
    observed = (
        payload.get("generation_id"),
        payload.get("calibration_id"),
        payload.get("pipeline_fingerprint"),
        payload.get("corpus_fingerprint"),
    )
    expected = (
        args.generation_id,
        args.calibration_id,
        args.pipeline_fingerprint,
        args.corpus_fingerprint,
    )
    if observed != expected:
        raise RuntimeError(f"serving lineage mismatch: expected {expected}, got {observed}")


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _collection_payload(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    *,
    started: float,
    status: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": "2026-09-16-task-specific-dense-hard-negative-selector-collection",
        "status": status,
        "collected_at": datetime.now(UTC).isoformat(),
        "source_commit": os.environ.get("RECALL_SOURCE_COMMIT"),
        "preregistration_commit": os.environ.get("RECALL_POLICY_COMMIT"),
        "benchmark_remote_code_root": os.environ.get("RECALL_BENCHMARK_REMOTE_CODE_ROOT"),
        "query_pool_sha256": _sha256(args.query_pool),
        "audit_sha256": _sha256(args.audit),
        "generation_id": args.generation_id,
        "calibration_id": args.calibration_id,
        "pipeline_fingerprint": args.pipeline_fingerprint,
        "corpus_fingerprint": args.corpus_fingerprint,
        "embedding_profile": "voyage-context-4-v1",
        "retrieval_profile": args.profile,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "rows": rows,
    }


def collect(args: argparse.Namespace) -> None:
    if _sha256(args.query_pool) != POOL_SHA256:
        raise RuntimeError("query pool hash mismatch")
    if _sha256(args.audit) != AUDIT_SHA256:
        raise RuntimeError("audit hash mismatch")
    pool = json.loads(args.query_pool.read_text(encoding="utf-8"))
    queries = prepare_rows(list(pool["queries"]), enforce_expected_counts=True)
    roots = dict(args.source_root)
    command = _command(
        args.tenant,
        args.embedder,
        args.index_root,
        args.profile,
        "combined",
        "none",
        20260916,
        32,
        0.10,
        args.generation_id,
        benchmark_retrieval_leg_audit=True,
        benchmark_source_admission_audit=True,
        source_conditioning_mode="off",
    )
    rows: list[dict[str, Any]] = []
    if args.output.exists():
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        identity = (
            prior.get("query_pool_sha256"),
            prior.get("audit_sha256"),
            prior.get("generation_id"),
        )
        expected = (POOL_SHA256, AUDIT_SHA256, args.generation_id)
        if identity != expected:
            raise RuntimeError(f"collection resume mismatch: expected {expected}, got {identity}")
        rows = list(prior.get("rows", []))
    started = time.perf_counter()
    client = TTYMCP(command, args.timeout)
    try:
        request_id = _initialize(client)
        for completed, query in enumerate(queries[len(rows) :], start=len(rows) + 1):
            payload = _call_query(client, request_id, str(query["query"]))
            request_id += 1
            _identity(payload, args)
            dense = dense_candidates(payload)[:CANDIDATE_COUNT]
            ids = [str(candidate["chunk_id"]) for candidate in dense]
            if len(dense) != CANDIDATE_COUNT or len(set(ids)) != CANDIDATE_COUNT:
                raise RuntimeError(f"row {query['id']} does not have 20 unique dense candidates")
            answer_span = _normalize(query["answer_span"])
            gold_sources = {str(value) for value in query.get("gold_sources", [])}
            candidates = [
                {
                    "chunk_id": str(candidate["chunk_id"]),
                    "source": str(candidate["source"]),
                    "text": str(candidate["text"]),
                    "dense_rank": int(candidate["dense_rank"]),
                    "dense_score": float(candidate["dense_score"]),
                    "gold_source": str(candidate["source"]) in gold_sources,
                    "exact_span": answer_span in _normalize(candidate["text"]),
                }
                for candidate in dense
            ]
            positive_text = _positive_text(query, roots)
            negatives = [
                candidate
                for candidate in candidates
                if answer_span not in _normalize(candidate["text"])
            ][:NEGATIVES_PER_QUERY]
            if len(negatives) != NEGATIVES_PER_QUERY:
                raise RuntimeError(f"row {query['id']} has fewer than four dense negatives")
            if any(_normalize(candidate["text"]) == positive_text for candidate in negatives):
                raise RuntimeError(f"row {query['id']} labels its positive as a negative")
            rows.append(
                {
                    "id": str(query["id"]),
                    "query": str(query["query"]),
                    "split": str(query["split"]),
                    "positive_text": positive_text,
                    "negative_chunk_ids": [str(candidate["chunk_id"]) for candidate in negatives],
                    "candidates": candidates,
                }
            )
            _write(args.output, _collection_payload(args, rows, started=started, status="RUNNING"))
            print(f"selector dense collection {completed}/{len(queries)}", flush=True)
    finally:
        client.close()
    _write(args.output, _collection_payload(args, rows, started=started, status="COMPLETE"))
    print(json.dumps({"output": str(args.output), "rows": len(rows), "status": "COMPLETE"}))


def write_model_inputs(args: argparse.Namespace) -> None:
    collection = json.loads(args.collection.read_text(encoding="utf-8"))
    if collection.get("status") != "COMPLETE" or len(collection.get("rows", [])) != 248:
        raise RuntimeError("collection is incomplete")
    rows = list(collection["rows"])
    train_rows = []
    score_rows: dict[str, list[dict[str, Any]]] = {"validation": [], "internal_test": []}
    for row in rows:
        split = str(row["split"])
        if split == "train":
            by_id = {str(candidate["chunk_id"]): candidate for candidate in row["candidates"]}
            negatives = [by_id[str(chunk_id)] for chunk_id in row["negative_chunk_ids"]]
            train_rows.append(
                {
                    "id": str(row["id"]),
                    "query": str(row["query"]),
                    "positive_text": str(row["positive_text"]),
                    "negative_texts": [str(candidate["text"]) for candidate in negatives],
                }
            )
        else:
            score_rows[split].append(
                {
                    "id": str(row["id"]),
                    "query": str(row["query"]),
                    "candidates": [
                        {
                            "chunk_id": str(candidate["chunk_id"]),
                            "text": str(candidate["text"]),
                            "dense_rank": int(candidate["dense_rank"]),
                        }
                        for candidate in row["candidates"]
                    ],
                }
            )
    if (len(train_rows), len(score_rows["validation"]), len(score_rows["internal_test"])) != (
        186,
        33,
        29,
    ):
        raise RuntimeError("model input split counts changed")
    common = {
        "schema_version": 1,
        "protocol": "2026-09-16-task-specific-dense-hard-negative-selector-model-input",
        "collection_sha256": _sha256(args.collection),
    }
    _write(
        args.train_validation_output,
        {**common, "train_rows": train_rows, "score_rows": score_rows["validation"]},
    )
    _write(
        args.internal_test_output,
        {**common, "score_rows": score_rows["internal_test"]},
    )
    print(
        json.dumps(
            {
                "train_validation": str(args.train_validation_output),
                "internal_test": str(args.internal_test_output),
                "train": len(train_rows),
                "validation": len(score_rows["validation"]),
                "internal_test_rows": len(score_rows["internal_test"]),
            }
        )
    )


def _root(value: str) -> tuple[str, Path]:
    prefix, separator, raw_path = value.partition("=")
    if not separator or not prefix.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("source root must be PREFIX=PATH")
    return prefix.strip().strip("/"), Path(raw_path)


def _add_lineage(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument("--pipeline-fingerprint", required=True)
    parser.add_argument("--corpus-fingerprint", required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--query-pool", type=Path, required=True)
    collect_parser.add_argument("--audit", type=Path, required=True)
    collect_parser.add_argument("--source-root", action="append", type=_root, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--tenant", default="memory")
    collect_parser.add_argument("--embedder", default="voyage-context:voyage-context-4")
    collect_parser.add_argument("--index-root", default="/home/sentiment/recall-repos/memory")
    collect_parser.add_argument("--profile", default="fast")
    collect_parser.add_argument("--timeout", type=float, default=240)
    _add_lineage(collect_parser)
    inputs_parser = sub.add_parser("write-model-inputs")
    inputs_parser.add_argument("--collection", type=Path, required=True)
    inputs_parser.add_argument("--train-validation-output", type=Path, required=True)
    inputs_parser.add_argument("--internal-test-output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args)
    else:
        write_model_inputs(args)


if __name__ == "__main__":
    main()
