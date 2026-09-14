"""Replay guarded source rescue only into unused alpha 0.08 context slots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recall.source_conditioning import (  # noqa: E402
    SourceConditioningArtifact,
    chunk_identifier_hash,
    load_source_conditioning_artifact,
    select_source_conditioned,
    source_features,
    source_support,
)
from scripts.run_live_source_conditioned_admission import _score_selection  # noqa: E402


EXPECTED_TRACE_SHA256 = "facdac77945c820c80af76e8adaa3fd106f34598c88dd56a291d001f3fa2bfd9"
EXPECTED_MODEL_SHA256 = "fb304c68a6ded04e28bfd9f0e9f244e45c133609101f788b5741f1a06e81242f"
EXPECTED_ROWS = 72
ITEM_BUDGET = 5
DUAL_MAX_RANK = 5
DUAL_COSINE_FLOOR = 0.35
LEXICAL_MAX_RANK = 1
LEXICAL_TOP10_MIN_CHUNKS = 2
LEXICAL_COSINE_FLOOR = 0.28
PRECISION_TOLERANCE = 0.05
ALLOWED_VERDICTS = {"ok", "low_confidence"}


def _source_rank(rows: list[dict[str, Any]], source: str) -> int | None:
    ranks = [int(item["rank"]) for item in rows if str(item["source"]) == source]
    return min(ranks) if ranks else None


def _select_spare_slot_rescue(
    model: SourceConditioningArtifact,
    row: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = list(row["alpha008_items"])
    if len(base) >= ITEM_BUDGET:
        return base, []

    trace = row["trace"]
    pool = list(trace["pool"])
    dense = list(trace["dense"])
    sparse = list(trace["sparse"])
    features = source_features(pool, dense, sparse)
    supports = source_support(model, features)
    represented = {str(item["source"]) for item in base}
    rescue_sources: list[
        tuple[tuple[int, int, float, str], str, str, float, int | None, int | None, list[Any]]
    ] = []

    for source, values in features.items():
        if source in represented:
            continue
        dense_rank = _source_rank(dense, source)
        sparse_rank = _source_rank(sparse, source)
        sparse_top10 = sum(
            int(item["rank"]) <= 10 and str(item["source"]) == source for item in sparse
        )
        max_cosine = float(values[0])
        dual = (
            dense_rank is not None
            and dense_rank <= DUAL_MAX_RANK
            and sparse_rank is not None
            and sparse_rank <= DUAL_MAX_RANK
            and max_cosine >= DUAL_COSINE_FLOOR
        )
        lexical = (
            sparse_rank is not None
            and sparse_rank <= LEXICAL_MAX_RANK
            and sparse_top10 >= LEXICAL_TOP10_MIN_CHUNKS
            and max_cosine >= LEXICAL_COSINE_FLOOR
        )
        if not dual and not lexical:
            continue
        lane = "dual_leg" if dual else "lexical_dominant"
        floor = DUAL_COSINE_FLOOR if dual else LEXICAL_COSINE_FLOOR
        items = [
            item
            for item in pool
            if str(item["source"]) == source
            and item.get("verdict") in ALLOWED_VERDICTS
            and float(item["cosine"]) >= floor
        ]
        if not items:
            continue
        items.sort(key=lambda item: (-float(item["cosine"]), int(item["pool_rank"])))
        rank_value = (
            int(dense_rank) + int(sparse_rank)
            if dual and dense_rank is not None and sparse_rank is not None
            else int(sparse_rank or 99)
        )
        priority = (0 if dual else 1, rank_value, -float(supports[source]), source)
        rescue_sources.append(
            (
                priority,
                source,
                lane,
                float(supports[source]),
                dense_rank,
                sparse_rank,
                items,
            )
        )

    rescue_sources.sort(key=lambda value: value[0])
    receipts: list[dict[str, Any]] = []
    while len(base) < ITEM_BUDGET and any(value[6] for value in rescue_sources):
        for _, source, lane, support, dense_rank, sparse_rank, items in rescue_sources:
            if len(base) >= ITEM_BUDGET:
                break
            if not items:
                continue
            item = items.pop(0)
            base.append(item)
            receipts.append(
                {
                    "chunk_hash": chunk_identifier_hash(item["chunk_id"]),
                    "source": source,
                    "ordinal": int(item["ordinal"]),
                    "lane": lane,
                    "cosine": float(item["cosine"]),
                    "source_support": support,
                    "dense_rank": dense_rank,
                    "sparse_rank": sparse_rank,
                }
            )
    return base, receipts


def _arm_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    answerable = [row for row in rows if row["label"] is not None]
    unanswerable = [row for row in rows if row["label"] is None]
    scores = [row["scores"][arm] for row in answerable]
    gold_items = sum(int(value["gold_items"]) for value in scores)
    total_items = sum(int(value["items"]) for value in scores)
    return {
        "complete_queries": sum(bool(value["complete"]) for value in scores),
        "covered_facts": sum(len(value["covered_facts"]) for value in scores),
        "source_hit_queries": sum(bool(value["source_hit"]) for value in scores),
        "gold_context_items": gold_items,
        "total_context_items": total_items,
        "context_precision": round(gold_items / total_items, 4) if total_items else None,
        "unanswerable_answers": sum(
            bool(row["scores"][arm]["answered_unanswerable"]) for row in unanswerable
        ),
    }


def _comparison(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {
        "complete_gain_ids": [],
        "complete_loss_ids": [],
        "fact_gain_ids": [],
        "fact_loss_ids": [],
    }
    for row in rows:
        if row["label"] is None:
            continue
        query_id = str(row["query"]["id"])
        base = row["scores"]["alpha008"]
        candidate = row["scores"]["candidate"]
        if bool(candidate["complete"]) and not bool(base["complete"]):
            result["complete_gain_ids"].append(query_id)
        if bool(base["complete"]) and not bool(candidate["complete"]):
            result["complete_loss_ids"].append(query_id)
        if len(candidate["covered_facts"]) > len(base["covered_facts"]):
            result["fact_gain_ids"].append(query_id)
        if len(candidate["covered_facts"]) < len(base["covered_facts"]):
            result["fact_loss_ids"].append(query_id)
    return result


def _decision(
    arms: dict[str, dict[str, Any]],
    comparison: dict[str, list[str]],
    integrity: dict[str, int],
) -> str:
    if (
        integrity["rows"] != EXPECTED_ROWS
        or integrity["capture_fixed_hash_parity"] != EXPECTED_ROWS
        or integrity["recomputed_fixed_parity"] != EXPECTED_ROWS
        or integrity["base_prefix_preserved"] != EXPECTED_ROWS
    ):
        return "REPAIR"
    base = arms["alpha008"]
    candidate = arms["candidate"]
    precision_safe = float(candidate["context_precision"] or 0.0) >= (
        float(base["context_precision"] or 0.0) - PRECISION_TOLERANCE
    )
    if (
        int(candidate["covered_facts"]) > int(base["covered_facts"])
        and not comparison["complete_loss_ids"]
        and not comparison["fact_loss_ids"]
        and int(candidate["unanswerable_answers"]) <= int(base["unanswerable_answers"])
        and precision_safe
    ):
        return "HEADROOM_FOR_FRESH_VALIDATION"
    return "NO_HEADROOM"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace", default="docs/results/2026-09-13-live-source-admission-trace-capture.json"
    )
    parser.add_argument(
        "--artifact", default="docs/results/2026-09-13-source-conditioning-model.json"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    trace_path = Path(args.trace)
    model_path = Path(args.artifact)
    trace_bytes = trace_path.read_bytes()
    model_bytes = model_path.read_bytes()
    if hashlib.sha256(trace_bytes).hexdigest() != EXPECTED_TRACE_SHA256:
        raise RuntimeError("source admission trace differs from the registered digest")
    if hashlib.sha256(model_bytes).hexdigest() != EXPECTED_MODEL_SHA256:
        raise RuntimeError("source conditioning model differs from the registered digest")
    captured = json.loads(trace_bytes.decode("utf-8"))
    model = load_source_conditioning_artifact(model_path)

    rows: list[dict[str, Any]] = []
    recomputed_parity = 0
    prefix_preserved = 0
    for row in captured["rows"]:
        trace = row["trace"]
        recomputed = select_source_conditioned(
            model,
            trace["pool"],
            trace["dense"],
            trace["sparse"],
            threshold=float(trace["threshold"]),
        )
        if [item["chunk_id"] for item in recomputed] == [
            item["chunk_id"] for item in row["alpha008_items"]
        ]:
            recomputed_parity += 1
        candidate, receipts = _select_spare_slot_rescue(model, row)
        if [item["chunk_id"] for item in candidate[: len(row["alpha008_items"])] ] == [
            item["chunk_id"] for item in row["alpha008_items"]
        ]:
            prefix_preserved += 1
        rows.append(
            {
                "challenge": row["challenge"],
                "query": row["query"],
                "label": row["label"],
                "alpha008_items": row["alpha008_items"],
                "candidate_items": candidate,
                "rescue_receipts": receipts,
                "scores": {
                    "alpha008": _score_selection(row["alpha008_items"], row["label"]),
                    "candidate": _score_selection(candidate, row["label"]),
                },
            }
        )

    arms = {arm: _arm_summary(rows, arm) for arm in ("alpha008", "candidate")}
    holdouts = {
        challenge: {
            arm: _arm_summary(
                [row for row in rows if row["challenge"] == challenge], arm
            )
            for arm in ("alpha008", "candidate")
        }
        for challenge in ("independent", "buried")
    }
    comparison = _comparison(rows)
    integrity = {
        "rows": len(rows),
        "capture_fixed_hash_parity": int(captured["integrity"]["fixed_hash_parity"]),
        "recomputed_fixed_parity": recomputed_parity,
        "base_prefix_preserved": prefix_preserved,
        "changed_queries": sum(bool(row["rescue_receipts"]) for row in rows),
        "added_items": sum(len(row["rescue_receipts"]) for row in rows),
    }
    decision = _decision(arms, comparison, integrity)
    result = {
        "schema_version": 1,
        "protocol": "2026-09-13-source-admission-spare-slot-development-replay",
        "trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "policy": {
            "item_budget": ITEM_BUDGET,
            "dual_max_rank": DUAL_MAX_RANK,
            "dual_cosine_floor": DUAL_COSINE_FLOOR,
            "lexical_max_rank": LEXICAL_MAX_RANK,
            "lexical_top10_min_chunks": LEXICAL_TOP10_MIN_CHUNKS,
            "lexical_cosine_floor": LEXICAL_COSINE_FLOOR,
            "preserve_base_prefix": True,
        },
        "integrity": integrity,
        "decision": decision,
        "arms": arms,
        "holdouts": holdouts,
        "comparison": comparison,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "decision": decision,
                "integrity": integrity,
                "arms": arms,
                "holdouts": holdouts,
                "comparison": comparison,
            }
        )
    )


if __name__ == "__main__":
    main()
