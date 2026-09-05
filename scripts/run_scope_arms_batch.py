"""Run the folder/facet scope arms of the preregistered probe against the remote MCP server.

Registered in `docs/preregistrations/2026-08-28-folder-scope-and-prior.md`. Replays the frozen
`agent-ab-skill-001` population (15 misses, 31 controls) through `recall_search` under three arms
and scores whether the gold memo appears in the top k.

Arms, and what each is FOR:

* ``baseline`` -- no scope. Not a control in the statistical sense, it is the apparatus check: it
  must reproduce the population's own labels, or nothing read from the other arms means anything.
* ``filter``   -- oracle folder. A CEILING probe, not a shippable arm: it hands the system the
  answer it would otherwise have to infer. If the ceiling does not clear the bar, nothing that has
  to infer the folder can.
* ``facet``    -- oracle facet, same framing.

The prior arm is absent on purpose and its absence is recorded in the artifact: `ScopePrior` is a
retriever construction argument and the MCP surface exposes no way to set it, so it cannot be
measured over this transport. The registration makes the ceilings decisive anyway.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

from mcp import ClientSession

from scripts.run_query_construction_batch import (
    _server_command,
    _ssh_stdio_client,
    _tool_payload,
)

ARMS = ("baseline", "filter", "facet")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hit(payload: dict, gold_memo: str) -> bool:
    """Did the gold memo appear among the returned hits?

    Matched on the source STEM. A hit's `source` is the memo's file name, and the population
    records the stem, so comparing raw strings would score every row a miss.
    """
    for hit in payload.get("hits") or []:
        source = str(hit.get("source") or "")
        if Path(source).stem == gold_memo:
            return True
    return False


async def _run(args: argparse.Namespace) -> int:
    rows = json.loads(args.population.read_text(encoding="utf-8"))
    input_sha256 = _digest(args.population)
    if args.expect_input_sha256 and input_sha256 != args.expect_input_sha256:
        print(
            f"VOID: population digest {input_sha256} does not match the registered "
            f"{args.expect_input_sha256}",
            file=sys.stderr,
        )
        return 2

    command, argv = _server_command(
        args.tenant, args.embedder, args.index_root, args.profile, args.pinned_generation_id
    )

    records: list[dict] = []
    generations: set[str] = set()
    started = time.time()

    with open(args.errlog, "w", encoding="utf-8") as errlog:
        async with _ssh_stdio_client(command, argv, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                # Generous: this server does ~15s of database work before its first reply,
                # and a probe that times out reports a healthy server as broken.
                await asyncio.wait_for(session.initialize(), timeout=90)
                for index, row in enumerate(rows):
                    for arm in ARMS:
                        request: dict[str, object] = {"query": row["query"], "k": args.k}
                        if arm == "filter":
                            request["folder"] = args.oracle_folder[row["gold_memo"]]
                        elif arm == "facet":
                            request["facet"] = args.oracle_facet[row["gold_memo"]]
                        result = await session.call_tool("recall_search", request)
                        payload = await _tool_payload(result)
                        generations.add(str(payload.get("generation_id")))
                        records.append(
                            {
                                "index": index,
                                "task_id": row["task_id"],
                                "arm": arm,
                                "gold_memo": row["gold_memo"],
                                "gold_class": row["gold_class"],
                                "query": row["query"],
                                "found": _hit(payload, row["gold_memo"]),
                                "abstained": bool(payload.get("abstained")),
                                "trust_state": payload.get("trust_state"),
                                "n_hits": len(payload.get("hits") or []),
                            }
                        )
                    if (index + 1) % 10 == 0:
                        print(f"  {index + 1}/{len(rows)} rows", file=sys.stderr, flush=True)

    artifact = {
        "measured_at": args.measured_at,
        "population": str(args.population),
        "input_sha256": input_sha256,
        "k": args.k,
        "tenant": args.tenant,
        "embedder": args.embedder,
        "profile": args.profile,
        "generation_ids_seen": sorted(generations),
        "prior_arm": "not measured: ScopePrior is a retriever argument with no MCP surface",
        "elapsed_s": round(time.time() - started, 1),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output} ({len(records)} records)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("population", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--oracle", type=Path, required=True, help="JSON: stem -> {folder, facet}")
    parser.add_argument("--measured-at", required=True)
    parser.add_argument("--expect-input-sha256")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--tenant", default="memory")
    parser.add_argument("--embedder", default="voyage:voyage-4")
    parser.add_argument("--index-root", default="/home/sentiment/recall-repos/memory")
    parser.add_argument("--profile", default="fast")
    parser.add_argument("--pinned-generation-id")
    parser.add_argument("--errlog", type=Path, default=Path("scope_arms.err.log"))
    args = parser.parse_args()

    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    args.oracle_folder = {k: v["folder"] for k, v in oracle.items()}
    args.oracle_facet = {k: v["facet"] for k, v in oracle.items()}
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
