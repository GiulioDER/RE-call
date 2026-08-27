"""Trace the recall / false-trigger frontier for draft-time search over a confidence threshold.

    python -u scripts/agent_ab_threshold_frontier.py \
        --precision benchmarks/artifacts/agent_ab/draft-precision.json \
        --screen ~/.claude/archive/direction-screen-2026-08-27/direction-screen.json

Preregistered in `docs/preregistrations/2026-08-27-draft-threshold-frontier.md`. Pure re-analysis
of committed artifacts: no retrieval, no model, no database.

Why a sweep settles the calibration question rather than merely hinting at it: a calibration fits
a score-to-confidence mapping and a threshold on it, and **a monotone remapping cannot change
which hit outranks which**. So for a fixed ranking the achievable (recall, false_trigger) pairs
are exactly the ones this sweep traces, and no recalibration can reach a point off the frontier.
If no acceptable point exists here, refitting cannot produce one, and the remaining levers are the
ones that change the RANKING.

Endpoints, and their deliberate asymmetry:

- recall(t)        — of the 14 registered miss SESSIONS, how many surface their governing memo in
                     some draft query's top-5 at confidence >= t. A rescue need happen once.
- false_trigger(t) — of the 18 negative draft QUERIES, how many return any hit at confidence >= t.
                     The agent pays this on every write.

Both are also reported the other way round, so the framing that favours the direction cannot hide
the one that does not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STEPS = [round(i / 100, 2) for i in range(101)]


def load(path: Path) -> dict:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--screen", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    precision = load(Path(args.precision))
    screen = load(Path(args.screen))
    miss_ids = {s["task_id"] for s in screen["sessions"]}
    if len(miss_ids) != 14:
        raise SystemExit(f"expected the registered 14 miss sessions, found {len(miss_ids)}")

    positives = [p for p in precision["positives"] if p["task_id"] in miss_ids]
    if len(positives) != 14:
        raise SystemExit(f"precision artifact holds {len(positives)} of the 14 miss sessions")
    neg_queries = [
        d for r in precision["negatives"] for d in r["per_draft"] if not d.get("refused")
    ]
    print(f"population: {len(positives)} miss sessions, {len(neg_queries)} negative draft queries")

    # Per session, the best confidence at which its governing memo appears in any draft's top-5.
    best_conf: list[float | None] = []
    for entry in positives:
        wanted = f"{entry['memo']}.md"
        confidences = [
            hit["confidence"]
            for draft in entry["per_draft"]
            for hit in draft["hits"]
            if hit["source"] == wanted and isinstance(hit["confidence"], (int, float))
        ]
        best_conf.append(max(confidences) if confidences else None)
    never = sum(1 for c in best_conf if c is None)
    print(f"governing memo never returned at all in: {never}/{len(positives)} sessions "
          f"(these bound max recall below 14 and no threshold can fix them)")

    # Per negative query, the best confidence it returns at all.
    neg_best = [
        max(
            (h["confidence"] for h in d["hits"] if isinstance(h["confidence"], (int, float))),
            default=None,
        )
        for d in neg_queries
    ]
    # Per positive session, does ANY draft query fire on something (the other-way-round metric)?
    pos_fire = [
        max(
            (
                h["confidence"]
                for draft in entry["per_draft"]
                for h in draft["hits"]
                if isinstance(h["confidence"], (int, float))
            ),
            default=None,
        )
        for entry in positives
    ]

    rows = []
    for t in STEPS:
        recall = sum(1 for c in best_conf if c is not None and c >= t)
        ft_q = sum(1 for c in neg_best if c is not None and c >= t)
        ft_s = sum(
            1
            for r in precision["negatives"]
            if any(
                h["confidence"] >= t
                for d in r["per_draft"]
                if not d.get("refused")
                for h in d["hits"]
                if isinstance(h["confidence"], (int, float))
            )
        )
        rows.append(
            {
                "t": t,
                "recall": recall,
                "of_sessions": len(positives),
                "false_trigger_queries": ft_q,
                "of_queries": len(neg_queries),
                "false_trigger_rate": round(ft_q / len(neg_queries), 4),
                "false_trigger_sessions": ft_s,
                "of_negative_sessions": len(precision["negatives"]),
                "positive_sessions_firing": sum(1 for c in pos_fire if c is not None and c >= t),
            }
        )

    max_recall = max(r["recall"] for r in rows)
    first_9 = next((r for r in rows if r["recall"] >= 9), None)
    at_035 = [r for r in rows if r["false_trigger_rate"] <= 0.35]
    best_at_035 = max(at_035, key=lambda r: r["recall"]) if at_035 else None

    print("\n   t     recall   false_trigger(queries)   rate")
    for r in rows:
        if r["t"] * 100 % 5:
            continue
        print(f"  {r['t']:.2f}   {r['recall']:>2}/14      "
              f"{r['false_trigger_queries']:>2}/{r['of_queries']:<3}            "
              f"{r['false_trigger_rate']:.3f}")

    print(f"\nmax achievable recall at any threshold: {max_recall}/14")
    if first_9:
        print(f"recall first reaches 9 at t={first_9['t']:.2f}, where false_trigger = "
              f"{first_9['false_trigger_queries']}/{first_9['of_queries']} "
              f"({first_9['false_trigger_rate']:.3f})")
    else:
        print("recall never reaches 9 at any threshold")
    if best_at_035:
        print(f"best recall with false_trigger <= 0.35: {best_at_035['recall']}/14 "
              f"at t={best_at_035['t']:.2f} "
              f"(ft {best_at_035['false_trigger_queries']}/{best_at_035['of_queries']})")
    else:
        print("NO threshold brings false_trigger to <= 0.35")

    viable = [r for r in rows if r["recall"] >= 9 and r["false_trigger_rate"] <= 0.35]
    print(f"\nthresholds with recall >= 9 AND false_trigger <= 0.35: {len(viable)}")

    payload = {
        "population": {"miss_sessions": len(positives), "negative_queries": len(neg_queries)},
        "never_returned": never,
        "max_recall": max_recall,
        "first_recall_9": first_9,
        "best_at_ft_035": best_at_035,
        "viable_points": viable,
        "frontier": rows,
    }
    out = Path(args.out) if args.out else Path(
        "benchmarks/artifacts/agent_ab/threshold-frontier.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
