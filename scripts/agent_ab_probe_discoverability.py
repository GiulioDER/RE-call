"""Replay the skill run's recorded queries against the discoverability rebuilds.

    python -u scripts/agent_ab_probe_discoverability.py \
        --archive ~/.claude/archive/agent-ab-skill-001 \
        --control-dsn <dsn>/probe_control \
        --arm retitle=<dsn>/probe_retitle \
        --arm restructured=<dsn>/probe_restructured \
        --arm pointer=<dsn>/probe_pointer \
        --exclude-base ts-raise-on-missing

Preregistered in `docs/preregistrations/2026-08-27-memo-discoverability-authoring.md`. Every
admitted on-arm `memory_only` session that searched is replayed against the control rebuild and
each treatment arm, over the same stdio transport and top-5. Excluded task families are replayed
and recorded but read from no endpoint: the registration excludes `ts-raise-on-missing` because
its governing memo is the one reconstruction-approximate source, which is exactly where the alias
probe's apparatus gate failed.

Numbers, in the order they matter:

1. **Apparatus:** the control rebuild must reproduce at least 13 of 14 non-excluded misses AND at
   least 24 of 26 non-excluded hits, or the probe is VOID and no verdict is read from any arm.
2. **Rescue per arm:** of the non-excluded sessions that missed, how many find their governing
   memo. A pointer document whose name contains the memo stem counts (the reader is delivered the
   memo's name and summary); `rescued_direct` counts only the memo itself.
3. **Retention per arm:** of the non-excluded sessions that hit, how many still hit.
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

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from agent_ab_probe_alias_index import outcome, sessions  # noqa: E402
from agent_ab_probe_expansion import retrieve  # noqa: E402

from benchmarks.agent_ab.recall_server import StdioRecallSpec  # noqa: E402

GATE_MISSES = 13
GATE_HITS = 24


def outcome_direct(row: dict, retrieved: dict[str, list[str]]) -> bool:
    """The governing memo itself in top-5, a pointer document not counting."""

    wanted = f"{row['memo']}.md"
    return any(wanted in retrieved.get(q, []) for q in row["queries"])


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--control-dsn", required=True)
    parser.add_argument(
        "--arm", action="append", default=[], metavar="NAME=DSN", help="a treatment arm"
    )
    parser.add_argument("--exclude-base", action="append", default=[])
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    arms: dict[str, str] = {}
    for spec in args.arm:
        name, _, dsn = spec.partition("=")
        if not name or not dsn:
            raise SystemExit(f"--arm wants NAME=DSN, got {spec!r}")
        arms[name] = dsn

    rows = sessions(Path(args.archive).expanduser())
    excluded = set(args.exclude_base)
    scored = [r for r in rows if r["base"] not in excluded]
    misses = [r for r in scored if not r["hit_in_run"]]
    hits = [r for r in scored if r["hit_in_run"]]
    print(
        f"{len(rows)} on-arm sessions searched; {len(scored)} scored after excluding "
        f"{sorted(excluded) or 'nothing'}: {len(hits)} hit, {len(misses)} missed\n"
    )

    distinct = sorted({q for r in rows for q in r["queries"]})
    print(f"{len(distinct)} distinct queries to replay against each corpus\n")

    answers: dict[str, dict[str, list[str]]] = {}
    for arm, dsn in {"control": args.control_dsn, **arms}.items():
        spec = StdioRecallSpec(dsn=dsn, cwd=REPO_ROOT, tenant=args.tenant)
        check = await spec.check()
        print(
            f"{arm}: {check.get('tool_count')} tools, trust_state={check.get('trust_state')} "
            f"calibrated={check.get('calibrated')} generation={str(check.get('generation_id'))[:20]}"
        )
        answers[arm] = await retrieve(spec, distinct)

    apparatus_miss = sum(1 for r in misses if not outcome(r, answers["control"]))
    apparatus_hit = sum(1 for r in hits if outcome(r, answers["control"]))
    void = apparatus_miss < GATE_MISSES or apparatus_hit < GATE_HITS
    print("\nAPPARATUS (control rebuild reproduces the run, excluded families not counted):")
    print(f"  misses reproduced: {apparatus_miss}/{len(misses)}  (gate >= {GATE_MISSES})")
    print(f"  hits reproduced:   {apparatus_hit}/{len(hits)}  (gate >= {GATE_HITS})")
    if void:
        print("  ⛔ GATE FAILED: the probe is VOID; treatment numbers below are not a verdict.")

    arm_summary: dict[str, dict] = {}
    for arm in arms:
        rescued = [r for r in misses if outcome(r, answers[arm])]
        rescued_direct = [r for r in misses if outcome_direct(r, answers[arm])]
        retained = [r for r in hits if outcome(r, answers[arm])]
        per_task = defaultdict(lambda: [0, 0])
        for r in misses:
            per_task[r["base"]][1] += 1
            per_task[r["base"]][0] += int(outcome(r, answers[arm]))
        print(f"\n{arm.upper()}:")
        print(f"  rescue:    {len(rescued)}/{len(misses)}  (direct {len(rescued_direct)})")
        for base, (won, n) in sorted(per_task.items()):
            print(f"    {base:<28} {won}/{n}")
        print(f"  retention: {len(retained)}/{len(hits)}")
        arm_summary[arm] = {
            "rescued": len(rescued),
            "rescued_direct": len(rescued_direct),
            "of_misses": len(misses),
            "retained": len(retained),
            "of_hits": len(hits),
            "per_task_rescue": {
                k: {"rescued": v[0], "n": v[1]} for k, v in sorted(per_task.items())
            },
        }

    payload = {
        "archive": str(args.archive),
        "distinct_queries": len(distinct),
        "excluded_bases": sorted(excluded),
        "apparatus": {
            "misses_reproduced": apparatus_miss,
            "of_misses": len(misses),
            "hits_reproduced": apparatus_hit,
            "of_hits": len(hits),
            "gate_misses": GATE_MISSES,
            "gate_hits": GATE_HITS,
            "void": void,
        },
        "arms": arm_summary,
        "sessions": [
            {
                **{k: r[k] for k in ("task_id", "memo", "hit_in_run")},
                "excluded": r["base"] in excluded,
                "control": outcome(r, answers["control"]),
                **{arm: outcome(r, answers[arm]) for arm in arms},
            }
            for r in rows
        ],
        "retrieved": answers,
    }
    out = (
        Path(args.out)
        if args.out
        else REPO_ROOT
        / "benchmarks"
        / "artifacts"
        / "agent_ab"
        / "discoverability-probe"
        / "discoverability-probe.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
