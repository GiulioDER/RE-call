"""Production-path check of the micro atomizer on the memory tenant, before rollout.

Unlike ``memory_atomizer_check.py``, which replayed a dense-only proxy, this calls the memory MCP
server's own search function, ``recall_mcp.service.search_memory``, in process on VPS2: the same
settings snapshot, trust policy, runtime environment, reranker, fusion, admission and calibration
autoload the live ``recall_search`` tool uses. Only the stdio transport is absent. Three arms
differ solely in the atomic keys of that environment:

``off``    ``RECALL_ATOMIC_RESCUE_MODE=off``.
``dense``  active, winner at dense rank six before fusion (today's default).
``fused``  active, winner at final rank six after fusion (the final top five cannot change).

The artifact is built by the production builder, ``scripts/build_atomic_micro_artifact.py``, from
the content-addressed view store, into a private registry, so this run also exercises the builder
that production will use. Questions, per-query rows and views stay on VPS2.

Subcommands: ``spans`` (fresh production-disjoint spans), and ``evaluate``. Questions are written
by ``memory_atomizer_check.py write`` on VPS3, where the writer's key is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import memory_atomizer_check as check  # noqa: E402

SEED = 20260925
SOURCES = 240
K = 10
ARMS = ("off", "dense", "fused")
CUTOFFS = (1, 5, 6, 10)


def gold_sources(records: Iterable[Any]) -> set[str]:
    """Sources named as GOLD in prior atomic evaluation records.

    The first memory check excluded every source a record named at all, and the census records
    list nearly every source as a candidate, which left 156 of 1,652 eligible. A source that was
    only a candidate was never asked about; only gold sources were consumed.
    """

    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"gold_sources", "gold_source"}:
                    for text in item if isinstance(item, list) else [item]:
                        if isinstance(text, str) and "/" in text:
                            found.add(text)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for record in records:
        walk(record)
    return found


def arm_environment(base: Mapping[str, str], arm: str, artifact_root: str) -> dict[str, str]:
    """The runtime environment for one arm: the base, with only the atomic keys changed."""

    values = dict(base)
    if arm == "off":
        values["RECALL_ATOMIC_RESCUE_MODE"] = "off"
        values.pop("RECALL_ATOMIC_RESCUE_PLACEMENT", None)
    elif arm in {"dense", "fused"}:
        values["RECALL_ATOMIC_RESCUE_MODE"] = "active"
        values["RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT"] = artifact_root
        values["RECALL_ATOMIC_RESCUE_PLACEMENT"] = arm
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return values


def first_rank(ranked: Sequence[str | None], wanted: Iterable[str]) -> int | None:
    targets = set(wanted)
    for rank, item in enumerate(ranked, start=1):
        if item in targets:
            return rank
    return None


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(fraction * len(ordered)) - 1)], 2)


def summarise(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Arm counts, paired gains and losses against ``off``, trust changes and atomic latency."""

    report: dict[str, Any] = {"queries": len(rows), "arms": {}, "paired": {}, "latency_ms": {}}
    for arm in ARMS:
        report["arms"][arm] = {
            f"{kind}@{cutoff}": sum(
                1 for row in rows if (r := row[arm][kind]) is not None and r <= cutoff
            )
            for kind in ("exact", "source")
            for cutoff in CUTOFFS
        }
        report["arms"][arm]["abstained"] = sum(1 for row in rows if row[arm]["abstained"])
        report["arms"][arm]["errors"] = sum(1 for row in rows if row[arm]["error"])
    for arm in ("dense", "fused"):
        report["paired"][arm] = {
            f"{kind}@{cutoff}": check.paired(
                [row["off"][kind] for row in rows], [row[arm][kind] for row in rows], cutoff
            )
            for kind in ("exact", "source")
            for cutoff in CUTOFFS
        }
        report["paired"][arm]["trust_state_changed"] = sum(
            1 for row in rows if row[arm]["trust_state"] != row["off"]["trust_state"]
        )
        report["paired"][arm]["top5_changed"] = sum(
            1 for row in rows if row[arm]["top5"] != row["off"]["top5"]
        )
        timings = [row[arm]["atomic_ms"] for row in rows if row[arm]["atomic_ms"] is not None]
        report["latency_ms"][arm] = {
            "atomic_stage_runs": len(timings),
            "p50": percentile(timings, 0.50),
            "p95": percentile(timings, 0.95),
            "p99": percentile(timings, 0.99),
            "total_p95": percentile([row[arm]["total_ms"] for row in rows], 0.95),
        }
    return report


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from recall.embeddings import resolve_embedder
    from recall.generation_store import GenerationStore
    from recall_mcp.service import search_memory
    from recall_mcp.settings import Settings

    base = dict(os.environ)
    embedder = resolve_embedder(args.embedder)
    probes = [
        json.loads(line)
        for line in args.probes.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["split"] == args.split
    ]
    if not probes:
        raise RuntimeError(f"no probes for split {args.split!r}")
    rows: list[dict[str, Any]] = []
    with GenerationStore(args.dsn, embedder.dim, tenant=args.tenant) as raw_store:
        store: Any = raw_store
        store.set_fixed_generation(args.generation)
        source_of = {chunk.id: chunk.metadata.get("file") for chunk in store.iter_chunks()}
        policies = {
            arm: Settings.from_env(arm_environment(base, arm, str(args.artifact_root))).trust_policy
            for arm in ARMS
        }
        for probe in probes:
            row: dict[str, Any] = {"probe_id": probe["probe_id"]}
            for arm in ARMS:
                values = arm_environment(base, arm, str(args.artifact_root))
                started = time.perf_counter()
                try:
                    result = search_memory(
                        store, embedder, probe["question"], k=K, policy=policies[arm], env=values
                    )
                except Exception as exc:  # BROAD-CATCH: an error is a measured outcome here
                    row[arm] = {
                        "exact": None, "source": None, "abstained": False, "trust_state": "error",
                        "top5": [], "atomic_ms": None, "total_ms": 0.0,
                        "error": type(exc).__name__,
                    }
                    continue
                ids = [hit.chunk_id for hit in result.hits]
                row[arm] = {
                    "exact": first_rank(ids, probe["gold_chunk_ids"]),
                    "source": first_rank([source_of.get(i or "") for i in ids], [probe["source"]]),
                    "abstained": bool(result.abstained),
                    "trust_state": result.trust_state,
                    "top5": ids[:5],
                    "atomic_ms": result.stage_ms.get("atomic_rescue"),
                    "total_ms": round((time.perf_counter() - started) * 1000.0, 1),
                    "error": None,
                }
            rows.append(row)
        unchanged = store.active_generation_id() == args.generation
    args.rows.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8", newline="\n")
    os.chmod(args.rows, 0o600)
    report = summarise(rows)
    report.update(
        {
            "protocol": "2026-09-23-memory-atomizer-production-path",
            "split": args.split,
            "generation_id": args.generation,
            "active_generation_unchanged": unchanged,
            "median_off_total_ms": statistics.median(row["off"]["total_ms"] for row in rows),
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=os.environ.get("RECALL_DSN"))
    parser.add_argument("--tenant", default="memory")
    parser.add_argument("--generation", required=True)
    parser.add_argument("--embedder", default="voyage-context:voyage-context-4")
    sub = parser.add_subparsers(dest="command", required=True)
    spans = sub.add_parser("spans")
    spans.add_argument("--evals-root", type=Path, required=True)
    spans.add_argument("--used-probes", type=Path, required=True)
    spans.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("evaluate")
    run.add_argument("--probes", type=Path, required=True)
    run.add_argument("--artifact-root", type=Path, required=True)
    run.add_argument("--split", choices=("dev", "confirm"), required=True)
    run.add_argument("--rows", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.dsn:
        raise SystemExit("RECALL_DSN or --dsn is required")

    if args.command == "spans":
        from recall.generation_store import GenerationStore

        records = []
        for path in sorted(args.evals_root.glob("atomic-fact-*/*.json")):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        used = {
            json.loads(line)["source"]
            for line in args.used_probes.read_text(encoding="utf-8").splitlines()
        }
        excluded = gold_sources(records) | used
        with GenerationStore(args.dsn, 1024, tenant=args.tenant) as raw_store:
            store: Any = raw_store
            store.set_fixed_generation(args.generation)
            chunks = check.load_chunks(store)
        rows = check.span_records(chunks, excluded, seed=SEED, sources=SOURCES)
        args.out.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
            newline="\n",
        )
        os.chmod(args.out, 0o600)
        summary: dict[str, Any] = {
            "spans": len(rows),
            "excluded_sources": len(excluded),
            "excluded_gold": len(gold_sources(records)),
            "excluded_used": len(used),
            "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        }
    else:
        summary = evaluate(args)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
