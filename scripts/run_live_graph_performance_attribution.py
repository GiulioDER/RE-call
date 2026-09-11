"""Run the preregistered graph off versus one hop attribution benchmark on VPS2."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_live_tty_graph_precision import TTYMCP, _command, _extract_payload  # noqa: E402


def _initialize(client: TTYMCP) -> int:
    client.call(
        1,
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "recall-graph-performance", "version": "1.0"},
        },
    )
    client.stdin.write(b'{"jsonrpc":"2.0","method":"notifications/initialized"}\r')
    client.stdin.flush()
    return 2


def _call_query(
    client: TTYMCP,
    request_id: int,
    query: str,
    *,
    max_steps: int,
    max_graph_nodes: int,
    max_evidence_tokens: int,
    graph_expansion: str,
) -> tuple[str, float]:
    started = time.perf_counter()
    response = client.call(
        request_id,
        "tools/call",
        {
            "name": "recall_reasoning_query",
            "arguments": {
                "query": query,
                "k": 5,
                "mode": "evidence_assembly",
                "max_steps": max_steps,
                "max_graph_nodes": max_graph_nodes,
                "max_evidence_tokens": max_evidence_tokens,
                "graph_expansion": graph_expansion,
            },
        },
    )
    return _extract_payload(response), (time.perf_counter() - started) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-set", default="docs/preregistrations/2026-08-17-memory-queries.json")
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
    parser.add_argument("--control", default="none")
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
    rows: list[dict[str, Any]] = []
    request_id = 2
    for arm in ("off", "one_hop"):
        client = TTYMCP(
            _command(
                args.tenant,
                args.embedder,
                args.index_root,
                args.profile,
                args.variant,
                args.control,
                args.control_seed,
                args.hub_threshold,
                args.cosine_margin,
                args.generation_id,
            ),
            args.timeout,
        )
        try:
            request_id = _initialize(client)
            for warmup_index in range(args.warmup_passes):
                for query_index, query in enumerate(queries, start=1):
                    print(
                        f"warmup {arm} {warmup_index + 1}/{args.warmup_passes} "
                        f"{query_index}/{len(queries)}",
                        flush=True,
                    )
                    _call_query(
                        client,
                        request_id,
                        str(query["query"]),
                        max_steps=args.max_steps,
                        max_graph_nodes=args.max_graph_nodes,
                        max_evidence_tokens=args.max_evidence_tokens,
                        graph_expansion=arm,
                    )
                    request_id += 1
            for pass_index in range(1, args.passes + 1):
                for query_index, query in enumerate(queries, start=1):
                    print(f"recorded {arm} pass={pass_index} {query_index}/{len(queries)}", flush=True)
                    raw_payload, client_ms = _call_query(
                        client,
                        request_id,
                        str(query["query"]),
                        max_steps=args.max_steps,
                        max_graph_nodes=args.max_graph_nodes,
                        max_evidence_tokens=args.max_evidence_tokens,
                        graph_expansion=arm,
                    )
                    payload = json.loads(raw_payload)
                    if payload.get("generation_id") != args.generation_id:
                        raise RuntimeError(
                            f"pinned generation mismatch: expected {args.generation_id}, "
                            f"got {payload.get('generation_id')}"
                        )
                    diagnostics = payload.get("diagnostics", {})
                    rows.append(
                        {
                            "query_index": query_index - 1,
                            "query": query,
                            "arm": arm,
                            "pass_index": pass_index,
                            "client_observed_ms": round(client_ms, 3),
                            "server_performance": diagnostics.get("performance", {}),
                            "outcome": payload.get("outcome"),
                            "refusal_reason": payload.get("refusal_reason"),
                            "trust_state": payload.get("trust_state"),
                            "evidence_ids": [
                                item.get("chunk_id")
                                for item in payload.get("trusted_evidence", {}).get("items", [])
                            ],
                            "payload": raw_payload,
                        }
                    )
                    request_id += 1
        finally:
            client.close()

    artifact = {
        "schema_version": 1,
        "measured_at": datetime.now(UTC).isoformat(),
        "query_set": str(query_path),
        "query_set_sha256": hashlib.sha256(query_bytes).hexdigest(),
        "query_count": len(queries),
        "generation_id": args.generation_id,
        "variant": args.variant,
        "relation_control": args.control,
        "passes": args.passes,
        "warmup_passes": args.warmup_passes,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(output)}))


if __name__ == "__main__":
    main()
