"""Audit live graph tail headroom on the frozen memory query set."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_live_graph_performance_attribution import (  # noqa: E402
    _attribution_row,
    _call_query,
    _initialize,
)
from scripts.run_live_tty_graph_precision import TTYMCP, _command  # noqa: E402


ARM_CONFIGS: tuple[dict[str, str], ...] = (
    {"arm": "linked_tail_true", "relation_control": "none"},
    {"arm": "linked_tail_removed", "relation_control": "removed"},
)


def _arm_order(query_index: int) -> tuple[str, ...]:
    names = tuple(config["arm"] for config in ARM_CONFIGS)
    offset = query_index % len(names)
    return names[offset:] + names[:offset]


def _gold_keys(query: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for value in query.get("relevant_ids", []):
        source, ordinal = str(value).rsplit(":", 1)
        if not source.startswith("recall/"):
            source = f"recall/{source}"
        keys.add(f"{source}:{int(ordinal)}")
    return keys


def _item_key(item: dict[str, Any]) -> str | None:
    source = item.get("source")
    ordinal = item.get("ordinal")
    if not isinstance(source, str) or isinstance(ordinal, bool) or not isinstance(ordinal, int):
        return None
    return f"{source}:{ordinal}"


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("payload")
    if not isinstance(value, str):
        raise ValueError("row payload must be serialized JSON")
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("row payload must decode to an object")
    return payload


def _audit(payload: dict[str, Any]) -> dict[str, Any]:
    value = (
        payload.get("diagnostics", {})
        .get("performance", {})
        .get("values", {})
        .get("graph_benchmark_audit")
    )
    if not isinstance(value, dict):
        raise ValueError("benchmark candidate audit is missing")
    return value


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_query: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        query_index = int(row["query_index"])
        by_query.setdefault(query_index, {})[str(row["arm"])] = row

    direct_incomplete: list[int] = []
    tail_headroom: list[int] = []
    connected_gold: list[int] = []
    gold_promotions: list[int] = []
    non_gold_promotions: list[int] = []
    recall_gains: list[int] = []
    recall_losses: list[int] = []
    complete_rescues: list[int] = []
    complete_regressions: list[int] = []
    connected_relation_types: Counter[str] = Counter()
    selected_relation_types: Counter[str] = Counter()
    answerable_queries = 0

    for query_index, arm_rows in sorted(by_query.items()):
        true_row = arm_rows.get("linked_tail_true")
        removed_row = arm_rows.get("linked_tail_removed")
        if true_row is None or removed_row is None:
            raise ValueError(f"query {query_index} is missing a paired arm")
        query = true_row["query"]
        gold = _gold_keys(query)
        if not bool(query.get("answerable", bool(gold))) or not gold:
            continue
        answerable_queries += 1

        true_payload = _payload(true_row)
        removed_payload = _payload(removed_row)
        audit = _audit(true_payload)
        raw_hits = audit.get("raw_hits", [])
        linked = audit.get("linked_candidates", [])
        direct_keys = {
            key
            for item in raw_hits
            if int(item.get("rank", 0)) <= 10 and (key := _item_key(item)) is not None
        }
        tail_keys = {
            key
            for item in raw_hits
            if 9 <= int(item.get("rank", 0)) <= 20 and (key := _item_key(item)) is not None
        }
        missing = gold.difference(direct_keys)
        if missing:
            direct_incomplete.append(query_index)
        if missing.intersection(tail_keys):
            tail_headroom.append(query_index)

        linked_gold = False
        selected_gold = False
        selected_non_gold = False
        for item in linked:
            key = _item_key(item)
            relation_types = [str(value) for value in item.get("relation_types", [])]
            if key in missing:
                linked_gold = True
                connected_relation_types.update(relation_types)
            is_promotion = bool(item.get("selected_pre_trust")) and int(
                item.get("raw_rank") or 0
            ) > 10
            if is_promotion:
                selected_relation_types.update(relation_types)
                if key in gold:
                    selected_gold = True
                else:
                    selected_non_gold = True
        if linked_gold:
            connected_gold.append(query_index)
        if selected_gold:
            gold_promotions.append(query_index)
        if selected_non_gold:
            non_gold_promotions.append(query_index)

        def evidence_gold(payload: dict[str, Any]) -> set[str]:
            items = payload.get("trusted_evidence", {}).get("items", [])
            return {
                key
                for item in items
                if (key := _item_key(item)) is not None and key in gold
            }

        true_gold = evidence_gold(true_payload)
        removed_gold = evidence_gold(removed_payload)
        if len(true_gold) > len(removed_gold):
            recall_gains.append(query_index)
        elif len(true_gold) < len(removed_gold):
            recall_losses.append(query_index)
        if gold.issubset(true_gold) and not gold.issubset(removed_gold):
            complete_rescues.append(query_index)
        if gold.issubset(removed_gold) and not gold.issubset(true_gold):
            complete_regressions.append(query_index)

    return {
        "queries": len(by_query),
        "answerable_queries": answerable_queries,
        "direct_incomplete_queries": direct_incomplete,
        "tail_headroom_queries": tail_headroom,
        "connected_gold_queries": connected_gold,
        "gold_promotion_queries": gold_promotions,
        "non_gold_promotion_queries": non_gold_promotions,
        "gold_recall_gain_queries": recall_gains,
        "gold_recall_loss_queries": recall_losses,
        "complete_gold_rescue_queries": complete_rescues,
        "complete_gold_regression_queries": complete_regressions,
        "connected_gold_relation_types": dict(sorted(connected_relation_types.items())),
        "selected_promotion_relation_types": dict(sorted(selected_relation_types.items())),
    }


def _client_command(args: argparse.Namespace, config: dict[str, str]) -> list[str]:
    return _command(
        args.tenant,
        args.embedder,
        args.index_root,
        args.profile,
        args.variant,
        config["relation_control"],
        args.control_seed,
        args.hub_threshold,
        args.cosine_margin,
        args.generation_id,
        candidate_mode="linked_tail",
        tail_replacement_margin="0.05",
        benchmark_graph_audit=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-set", default="docs/preregistrations/2026-08-17-memory-queries.json"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--tenant", default="memory")
    parser.add_argument("--embedder", default="voyage:voyage-4")
    parser.add_argument("--index-root", default="/home/sentiment/recall-repos/memory")
    parser.add_argument("--profile", default="fast")
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-graph-nodes", type=int, default=32)
    parser.add_argument("--max-evidence-tokens", type=int, default=2048)
    parser.add_argument("--variant", default="combined")
    parser.add_argument("--control-seed", type=int, default=20260825)
    parser.add_argument("--hub-threshold", type=int, default=32)
    parser.add_argument("--cosine-margin", type=float, default=0.10)
    args = parser.parse_args()

    query_path = Path(args.query_set)
    query_bytes = query_path.read_bytes()
    queries = json.loads(query_bytes.decode("utf-8"))
    if not isinstance(queries, list) or not all(isinstance(item, dict) for item in queries):
        raise ValueError("query set must be a JSON list of objects")

    configs = {config["arm"]: config for config in ARM_CONFIGS}
    clients: dict[str, TTYMCP] = {}
    request_ids: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    try:
        for arm, config in configs.items():
            client = TTYMCP(_client_command(args, config), args.timeout)
            clients[arm] = client
            request_ids[arm] = _initialize(client)
        for query_index, query in enumerate(queries):
            for arm in _arm_order(query_index):
                print(f"recorded {arm} {query_index + 1}/{len(queries)}", flush=True)
                raw_payload, client_ms = _call_query(
                    clients[arm],
                    request_ids[arm],
                    str(query["query"]),
                    max_steps=args.max_steps,
                    max_graph_nodes=args.max_graph_nodes,
                    max_evidence_tokens=args.max_evidence_tokens,
                    graph_expansion="one_hop",
                )
                rows.append(
                    _attribution_row(
                        raw_payload,
                        client_ms,
                        query_index=query_index,
                        query=query,
                        arm=arm,
                        phase="recorded",
                        pass_index=1,
                        generation_id=args.generation_id,
                    )
                )
                request_ids[arm] += 1
    finally:
        for client in clients.values():
            client.close()

    artifact = {
        "schema_version": 1,
        "protocol": "2026-09-13-live-graph-candidate-headroom",
        "measured_at": datetime.now(UTC).isoformat(),
        "source_commit": os.environ.get("RECALL_SOURCE_COMMIT"),
        "query_set": str(query_path),
        "query_set_sha256": hashlib.sha256(query_bytes).hexdigest(),
        "query_count": len(queries),
        "generation_id": args.generation_id,
        "arm_configs": ARM_CONFIGS,
        "summary": _summarize(rows),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "summary": artifact["summary"], "output": str(output)}))


if __name__ == "__main__":
    main()
