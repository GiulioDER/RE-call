"""Replay the skill run's recorded queries against the control and aliased rebuilds.

    python -u scripts/agent_ab_probe_alias_index.py \
        --archive ~/.claude/archive/agent-ab-skill-001 \
        --control-dsn <dsn>/probe_control --aliased-dsn <dsn>/probe_aliased

Preregistered in `docs/preregistrations/2026-08-23-alias-index-probe.md`. Every admitted on-arm
`memory_only` session of `agent-ab-skill-001` is replayed: its recorded `recall_search` queries
are asked, over the same stdio transport and top-5, against two corpora built by the same
committed builder from the same recovered frozen sources, differing ONLY in the appended alias
sections. Three numbers come out, in this order:

1. **Apparatus:** does the control rebuild reproduce each session's original outcome? A rebuild
   that cannot reproduce the misses and hits it is supposed to perturb is not an instrument, and
   the registered gate voids the probe rather than interpreting it.
2. **Rescue:** of the sessions that missed their governing memo in the run, how many find it in
   the aliased corpus?
3. **Retention:** of the sessions that hit, how many still hit? Aliases add competing text to
   every memo, and a rescue bought by breaking existing hits is not a rescue.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.agent_ab.recall_server import StdioRecallSpec  # noqa: E402
from benchmarks.agent_ab.schema import RECALL_ON, SessionRecord  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from agent_ab_analyze_tasks import retrieved_governing_memo, searched  # noqa: E402
from agent_ab_probe_expansion import retrieve  # noqa: E402

MEMORY_ONLY = "memory_only"
TOP_K = 5


def sessions(archive: Path) -> list[dict]:
    rows = []
    for line in (archive / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = SessionRecord.from_mapping(json.loads(line))
        if record.variant != RECALL_ON or record.metadata.get("locus") != MEMORY_ONLY:
            continue
        if not searched(record):
            continue
        queries = [
            str(call.get("args", {}).get("query", ""))
            for call in record.tool_calls
            if "recall_search" in str(call.get("name", "")) and call.get("args")
        ]
        rows.append(
            {
                "task_id": record.task_id,
                "base": record.task_id.split("#")[0],
                "memo": str(record.metadata["governing_memo"]),
                "queries": [q for q in queries if q],
                "hit_in_run": bool(retrieved_governing_memo(record)),
            }
        )
    return rows


def outcome(row: dict, retrieved: dict[str, list[str]]) -> bool:
    stem = row["memo"]
    return any(stem in " ".join(retrieved.get(q, [])) for q in row["queries"])


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--control-dsn", required=True)
    parser.add_argument("--aliased-dsn", required=True)
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = sessions(Path(args.archive).expanduser())
    misses = [r for r in rows if not r["hit_in_run"]]
    hits = [r for r in rows if r["hit_in_run"]]
    print(f"{len(rows)} on-arm sessions searched: {len(hits)} hit, {len(misses)} missed\n")

    distinct = sorted({q for r in rows for q in r["queries"]})
    print(f"{len(distinct)} distinct queries to replay against each corpus\n")

    answers: dict[str, dict[str, list[str]]] = {}
    for arm, dsn in (("control", args.control_dsn), ("aliased", args.aliased_dsn)):
        spec = StdioRecallSpec(dsn=dsn, cwd=REPO_ROOT, tenant=args.tenant)
        check = await spec.check()
        print(
            f"{arm}: {check.get('tool_count')} tools, trust_state={check.get('trust_state')} "
            f"calibrated={check.get('calibrated')} generation={str(check.get('generation_id'))[:20]}"
        )
        answers[arm] = await retrieve(spec, distinct)

    apparatus_miss = sum(1 for r in misses if not outcome(r, answers["control"]))
    apparatus_hit = sum(1 for r in hits if outcome(r, answers["control"]))
    print("\nAPPARATUS (control rebuild reproduces the run):")
    print(f"  misses reproduced: {apparatus_miss}/{len(misses)}")
    print(f"  hits reproduced:   {apparatus_hit}/{len(hits)}")

    rescued = [r for r in misses if outcome(r, answers["aliased"])]
    retained = [r for r in hits if outcome(r, answers["aliased"])]
    per_task = defaultdict(lambda: [0, 0])
    for r in misses:
        per_task[r["base"]][1] += 1
        per_task[r["base"]][0] += int(outcome(r, answers["aliased"]))
    print(f"\nRESCUE on aliased: {len(rescued)}/{len(misses)}")
    for base, (won, n) in sorted(per_task.items()):
        print(f"  {base:<28} {won}/{n}")
    print(f"RETENTION on aliased: {len(retained)}/{len(hits)}")

    payload = {
        "archive": str(args.archive),
        "distinct_queries": len(distinct),
        "apparatus": {
            "misses_reproduced": apparatus_miss,
            "of_misses": len(misses),
            "hits_reproduced": apparatus_hit,
            "of_hits": len(hits),
        },
        "rescue": {"rescued": len(rescued), "of": len(misses)},
        "retention": {"retained": len(retained), "of": len(hits)},
        "per_task_rescue": {k: {"rescued": v[0], "n": v[1]} for k, v in sorted(per_task.items())},
        "sessions": [
            {
                **{k: r[k] for k in ("task_id", "memo", "hit_in_run")},
                "control": outcome(r, answers["control"]),
                "aliased": outcome(r, answers["aliased"]),
            }
            for r in rows
        ],
        "retrieved": answers,
    }
    out = Path(args.out) if args.out else REPO_ROOT / "benchmarks" / "artifacts" / "agent_ab" / "alias-probe" / "alias-probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
