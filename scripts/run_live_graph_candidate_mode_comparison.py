"""Run the preregistered graph candidate mode comparison on a pinned VPS2 generation."""

from __future__ import annotations

import argparse
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
    {
        "arm": "off",
        "graph_expansion": "off",
        "candidate_mode": "outside_pool",
        "relation_control": "none",
        "tail_replacement_margin": "off",
    },
    {
        "arm": "outside_pool",
        "graph_expansion": "one_hop",
        "candidate_mode": "outside_pool",
        "relation_control": "none",
        "tail_replacement_margin": "0.05",
    },
    {
        "arm": "linked_tail",
        "graph_expansion": "one_hop",
        "candidate_mode": "linked_tail",
        "relation_control": "none",
        "tail_replacement_margin": "0.05",
    },
    {
        "arm": "hybrid",
        "graph_expansion": "one_hop",
        "candidate_mode": "hybrid",
        "relation_control": "none",
        "tail_replacement_margin": "0.05",
    },
    {
        "arm": "linked_tail_shuffled",
        "graph_expansion": "one_hop",
        "candidate_mode": "linked_tail",
        "relation_control": "shuffled",
        "tail_replacement_margin": "0.05",
    },
    {
        "arm": "linked_tail_removed",
        "graph_expansion": "one_hop",
        "candidate_mode": "linked_tail",
        "relation_control": "removed",
        "tail_replacement_margin": "0.05",
    },
)


def _arm_order(query_index: int) -> tuple[str, ...]:
    """Rotate first mover assignment deterministically by query index."""
    names = tuple(config["arm"] for config in ARM_CONFIGS)
    offset = query_index % len(names)
    return names[offset:] + names[:offset]


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
        candidate_mode=config["candidate_mode"],
        tail_replacement_margin=config["tail_replacement_margin"],
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
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--warmup-passes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-graph-nodes", type=int, default=32)
    parser.add_argument("--max-evidence-tokens", type=int, default=2048)
    parser.add_argument("--variant", default="combined")
    parser.add_argument("--control-seed", type=int, default=20260825)
    parser.add_argument("--hub-threshold", type=int, default=32)
    parser.add_argument("--cosine-margin", type=float, default=0.10)
    args = parser.parse_args()
    if args.passes < 1 or args.warmup_passes < 0:
        raise ValueError("passes must be positive and warmup-passes cannot be negative")

    query_path = Path(args.query_set)
    query_bytes = query_path.read_bytes()
    queries = json.loads(query_bytes.decode("utf-8"))
    if not isinstance(queries, list) or not all(isinstance(item, dict) for item in queries):
        raise ValueError("query set must be a JSON list of objects")

    configs = {config["arm"]: config for config in ARM_CONFIGS}
    clients: dict[str, TTYMCP] = {}
    request_ids: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    warmup_rows: list[dict[str, Any]] = []
    try:
        for arm, config in configs.items():
            client = TTYMCP(_client_command(args, config), args.timeout)
            clients[arm] = client
            request_ids[arm] = _initialize(client)

        phases = (("warmup", args.warmup_passes), ("recorded", args.passes))
        for phase, phase_passes in phases:
            destination = warmup_rows if phase == "warmup" else rows
            for pass_index in range(1, phase_passes + 1):
                for query_index, query in enumerate(queries):
                    for arm in _arm_order(query_index):
                        print(
                            f"{phase} {arm} pass={pass_index} "
                            f"{query_index + 1}/{len(queries)}",
                            flush=True,
                        )
                        config = configs[arm]
                        raw_payload, client_ms = _call_query(
                            clients[arm],
                            request_ids[arm],
                            str(query["query"]),
                            max_steps=args.max_steps,
                            max_graph_nodes=args.max_graph_nodes,
                            max_evidence_tokens=args.max_evidence_tokens,
                            graph_expansion=config["graph_expansion"],
                        )
                        destination.append(
                            _attribution_row(
                                raw_payload,
                                client_ms,
                                query_index=query_index,
                                query=query,
                                arm=arm,
                                phase=phase,
                                pass_index=pass_index,
                                generation_id=args.generation_id,
                            )
                        )
                        request_ids[arm] += 1
    finally:
        for client in clients.values():
            client.close()

    artifact = {
        "schema_version": 1,
        "protocol": "2026-09-12-production-graph-linked-tail-parity",
        "measured_at": datetime.now(UTC).isoformat(),
        "source_commit": os.environ.get("RECALL_SOURCE_COMMIT"),
        "query_set": str(query_path),
        "query_set_sha256": hashlib.sha256(query_bytes).hexdigest(),
        "query_count": len(queries),
        "generation_id": args.generation_id,
        "passes": args.passes,
        "warmup_passes": args.warmup_passes,
        "variant": args.variant,
        "control_seed": args.control_seed,
        "arm_configs": ARM_CONFIGS,
        "rows": rows,
        "warmup_rows": warmup_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(output)}))


if __name__ == "__main__":
    main()
