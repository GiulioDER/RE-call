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
   Both halves of each ratio are enforced: the denominators are asserted against the registered
   14 and 26, an exclusion matching no session is refused, and every arm must answer differently
   from control on at least 10% of replayed queries. Those three checks exist because each one
   silently produces `rescue 0/14, retention 26/26` — character for character the genuine null.
   A VOID run exits non-zero and names its artifact `*.VOID.json`.
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
from agent_ab_probe_expansion import TOP_K, retrieve  # noqa: E402

from benchmarks.agent_ab.recall_server import StdioRecallSpec  # noqa: E402

#: k is a frozen term of the registration ("same top-5"), inherited from the shared retriever.
#: Checked in main() rather than asserted at import: `python -O` strips an assert, so the guard
#: would vanish exactly where somebody has tuned the run. The payload records TOP_K so the
#: artifact proves which k was applied.
REGISTERED_TOP_K = 5

GATE_MISSES = 13
GATE_HITS = 24
#: The denominators the gate above was registered against. They are half of the gate, not
#: context: "at least 13 of 14" applied to a population of 15 is a different, weaker instrument.
REGISTERED_MISSES = 14
REGISTERED_HITS = 26
#: An arm must differ from control on at least this share of replayed queries. Control reproduced
#: every miss, so an arm pointed at the control corpus CANNOT rescue and MUST retain: it prints
#: rescue 0/14 and retention 26/26, character for character the genuine null.
MIN_ARM_DIVERGENCE = 0.10
#: Names an arm may not take, because each would overwrite a field of the same name.
RESERVED_ARM_NAMES = frozenset({"control", "task_id", "memo", "hit_in_run", "excluded"})


def outcome_direct(row: dict, retrieved: dict[str, list[str]]) -> bool:
    """The governing memo itself in top-5, a pointer document not counting."""

    wanted = f"{row['memo']}.md"
    return any(wanted in retrieved.get(q, []) for q in row["queries"])


def population_matches_registration(n_misses: int, n_hits: int) -> tuple[bool, str]:
    """Is the gate about to be applied to the population it was registered against?

    The gate is two absolute counts, and the denominators come from the archive and
    `--exclude-base` at runtime. Drop the exclusion and the population becomes 15 misses and 31
    hits, where 13 and 24 still clear: a verdict then gets published over denominators the
    registration never authorised, and nothing says so.
    """

    if (n_misses, n_hits) == (REGISTERED_MISSES, REGISTERED_HITS):
        return True, ""
    return False, (
        f"scored population is {n_misses} misses / {n_hits} hits, but the gate "
        f"({GATE_MISSES} and {GATE_HITS}) was registered against "
        f"{REGISTERED_MISSES} / {REGISTERED_HITS}; it no longer means what it was registered "
        "to mean"
    )


def unmatched_exclusions(rows: list[dict], excluded: set[str]) -> list[str]:
    """Exclusion values that matched no session, which is how a wrong denominator arrives."""

    seen = {row["base"] for row in rows}
    return sorted(name for name in excluded if name not in seen)


def validate_arm_names(names: list[str]) -> str | None:
    """Reject an arm name that would silently overwrite something, or a duplicate."""

    for name in names:
        if name in RESERVED_ARM_NAMES:
            return (
                f"--arm {name}=... is refused: {name!r} would overwrite "
                + (
                    "the control corpus, making the apparatus gate self-referential"
                    if name == "control"
                    else f"the {name!r} field of every session row in the artifact"
                )
            )
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        return f"--arm given more than once for: {', '.join(duplicates)}"
    return None


def arm_is_distinct(
    control: dict[str, list[str]], arm: dict[str, list[str]]
) -> tuple[bool, float]:
    """Does this arm's corpus actually answer differently from control?

    Returns (ok, share of queries whose top-5 differs). A mis-keyed or duplicated DSN produces a
    treatment arm that IS control, whose rescue and retention are then byte-identical to the
    published null. This is the one false-zero route the apparatus gate cannot catch, because the
    gate only ever examines control.
    """

    queries = sorted(set(control) | set(arm))
    if not queries:
        return False, 0.0
    differing = sum(1 for q in queries if control.get(q) != arm.get(q))
    share = differing / len(queries)
    return share >= MIN_ARM_DIVERGENCE, share


def exit_code(*, void: bool) -> int:
    """A void run must not exit 0: void and verdict were indistinguishable to any caller."""

    return 3 if void else 0


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
    names: list[str] = []
    for spec in args.arm:
        name, _, dsn = spec.partition("=")
        if not name or not dsn:
            raise SystemExit(f"--arm wants NAME=DSN, got {spec!r}")
        names.append(name)
        arms[name] = dsn
    invalid = validate_arm_names(names)
    if invalid:
        raise SystemExit(invalid)
    shared = sorted({n for n, d in arms.items() if d == args.control_dsn})
    if shared:
        raise SystemExit(
            f"arm(s) {', '.join(shared)} point at the control DSN. An arm that IS control scores "
            "rescue 0 and retention 100%, which is indistinguishable from a real null."
        )

    if TOP_K != REGISTERED_TOP_K:
        raise SystemExit(
            f"the registration fixes top-{REGISTERED_TOP_K}; the shared retriever now uses "
            f"top-{TOP_K}, so this would not be the registered instrument"
        )

    rows = sessions(Path(args.archive).expanduser())
    excluded = set(args.exclude_base)
    stale = unmatched_exclusions(rows, excluded)
    if stale:
        raise SystemExit(
            f"--exclude-base matched no session: {', '.join(stale)}. A typo'd exclusion silently "
            "scores the family it was meant to remove, which changes the gate's denominators."
        )
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
    provenance: dict[str, dict] = {}
    for arm, dsn in {"control": args.control_dsn, **arms}.items():
        spec = StdioRecallSpec(dsn=dsn, cwd=REPO_ROOT, tenant=args.tenant)
        check = await spec.check()
        # Recorded, not merely printed: without per-arm provenance the artifact cannot show which
        # corpus produced which number, and a duplicated arm is invisible after the fact.
        provenance[arm] = {
            "generation_id": check.get("generation_id"),
            "trust_state": check.get("trust_state"),
            "calibrated": check.get("calibrated"),
            "tool_count": check.get("tool_count"),
            "embedder": spec.embedder,
            "database": dsn.rsplit("/", 1)[-1].split("?", 1)[0],
        }
        print(
            f"{arm}: {check.get('tool_count')} tools, trust_state={check.get('trust_state')} "
            f"calibrated={check.get('calibrated')} generation={str(check.get('generation_id'))[:20]}"
        )
        answers[arm] = await retrieve(spec, distinct)

    apparatus_miss = sum(1 for r in misses if not outcome(r, answers["control"]))
    apparatus_hit = sum(1 for r in hits if outcome(r, answers["control"]))
    registered, population_reason = population_matches_registration(len(misses), len(hits))
    void = apparatus_miss < GATE_MISSES or apparatus_hit < GATE_HITS or not registered
    print("\nAPPARATUS (control rebuild reproduces the run, excluded families not counted):")
    print(f"  misses reproduced: {apparatus_miss}/{len(misses)}  (gate >= {GATE_MISSES})")
    print(f"  hits reproduced:   {apparatus_hit}/{len(hits)}  (gate >= {GATE_HITS})")
    if not registered:
        print(f"  ⛔ POPULATION: {population_reason}")

    # An arm that is secretly control passes every check above, because every check above looks
    # only at control. This is the one remaining route to a manufactured null.
    divergence: dict[str, float] = {}
    for arm in arms:
        distinct_arm, share = arm_is_distinct(answers["control"], answers[arm])
        divergence[arm] = round(share, 4)
        if not distinct_arm:
            void = True
            print(
                f"  ⛔ ARM {arm}: top-5 differs from control on only {share:.1%} of queries "
                f"(needs >= {MIN_ARM_DIVERGENCE:.0%}); this arm is indistinguishable from control."
            )
    if void:
        print("  ⛔ GATE FAILED: the probe is VOID; treatment numbers below are not a verdict.")

    arm_summary: dict[str, dict] = {}
    for arm in arms:
        rescued = [r for r in misses if outcome(r, answers[arm])]
        rescued_direct = [r for r in misses if outcome_direct(r, answers[arm])]
        retained = [r for r in hits if outcome(r, answers[arm])]
        # Retention needs the same direct twin rescue has: in the pointer arm a `<stem>--tasks.md`
        # document satisfies the substring criterion, so a hit whose memo fell out of top-5 could
        # score as retained on its own pointer. It did not happen in the 2026-08-27 run; nothing
        # in the instrument made that a property rather than luck.
        retained_direct = [r for r in hits if outcome_direct(r, answers[arm])]
        per_task = defaultdict(lambda: [0, 0])
        for r in misses:
            per_task[r["base"]][1] += 1
            per_task[r["base"]][0] += int(outcome(r, answers[arm]))
        print(f"\n{arm.upper()}:")
        print(f"  rescue:    {len(rescued)}/{len(misses)}  (direct {len(rescued_direct)})")
        for base, (won, n) in sorted(per_task.items()):
            print(f"    {base:<28} {won}/{n}")
        print(f"  retention: {len(retained)}/{len(hits)}  (direct {len(retained_direct)})")
        arm_summary[arm] = {
            "rescued": len(rescued),
            "rescued_direct": len(rescued_direct),
            "of_misses": len(misses),
            "retained": len(retained),
            "retained_direct": len(retained_direct),
            "of_hits": len(hits),
            "divergence_from_control": divergence[arm],
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
            "registered_misses": REGISTERED_MISSES,
            "registered_hits": REGISTERED_HITS,
            "population_matches_registration": registered,
            "population_reason": population_reason,
            "min_arm_divergence": MIN_ARM_DIVERGENCE,
            "top_k": TOP_K,
            "void": void,
        },
        "provenance": provenance,
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
    if void:
        # A void run's file is named so no fragment of it can be quoted as a result by accident.
        # `with_suffix` rather than a string replace: a --out without a .json suffix would
        # otherwise land a VOID artifact under an ordinary name.
        out = out.with_suffix(".VOID.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {out}")
    if void:
        print("VOID: no verdict may be read from the arm tables above.")
    return exit_code(void=void)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
