"""Re-derive the trigger screen's populations from JUDGED labels, not from retrieval.

    python -u scripts/agent_ab_trigger_rederive.py \
        --trigger benchmarks/artifacts/agent_ab/trigger-screen.json \
        --actionable benchmarks/artifacts/agent_ab/actionable-recall.json

**A correction, not a new claim**, and licensed by the trigger record's own "what this licenses"
section: *"Re-derive the trigger screen's populations from `actionable-recall.json`'s judged
labels."* No new predictions are made, because the underlying retrieval data is unchanged and only
the LABELS move. The registered decision grid is applied to the corrected populations.

What changes. The trigger screen called a draft hazard-bearing when it RETRIEVED the governing
memo — 46 of 178. A judge control later showed that proxy wrong for 19 of those 46: three
sessions' only "hazard-bearing" draft is `ls -la benchmarks/`, which retrieves the memo on a
shared directory name. Hazard-bearing now means retrieved AND judged to apply: 27 of 178.

Consequences the corrected numbers must carry:

- **Coverage has a ceiling of 10, not 14.** Four sessions have no actionable draft at all, so no
  trigger can reach them. Coverage is reported over 10 (achievable) and over 14 (comparable to the
  original table), and the record says which is which.
- **`N_wide` grows** from 132 to 151, because the 19 falsely-labelled drafts move into it.
- **`N_clean` is untouched**: those 18 were labelled by the benchmark, not by retrieval.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trigger", required=True)
    parser.add_argument("--actionable", required=True)
    parser.add_argument("--dsn", default=None, help="only needed to rebuild T2 vocabularies")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    trigger = json.loads(Path(args.trigger).expanduser().read_text(encoding="utf-8"))
    judged = json.loads(Path(args.actionable).expanduser().read_text(encoding="utf-8"))

    # Join judged verdicts back onto the trigger records: (task_id, draft prefix) is unique here,
    # and the character count is carried as a check rather than trusted as a key.
    verdict: dict[tuple[str, str], bool] = {}
    for row in judged["rows"]:
        for entry in row["verdicts"]:
            verdict[(row["task_id"], entry["draft"][:200])] = entry["actionable"] is True
    print(f"judged verdicts available: {len(verdict)}")

    records = trigger["records"]
    positives = [r for r in records if r["population"] == "positive"]
    n_clean = [r for r in records if r["population"] == "n_clean"]

    matched = 0
    for record in positives:
        key = (record["task_id"], record["draft"][:200])
        if record["governing_rank"] is None:
            record["bearing"] = False
            continue
        if key not in verdict:
            raise SystemExit(
                f"unjoined retrieving draft {key[0]} ({record['chars']} chars); the two artifacts "
                "disagree about which drafts retrieve the memo, so the re-derivation is not run"
            )
        matched += 1
        record["bearing"] = verdict[key]

    retrieved = sum(1 for r in positives if r["governing_rank"] is not None)
    bearing = [r for r in positives if r["bearing"]]
    n_wide = [r for r in positives if not r["bearing"]]
    print(f"joined {matched}/{retrieved} memo-retrieving drafts")
    print(f"hazard-bearing: was {retrieved} (retrieved), now {len(bearing)} (judged to apply)")
    print(f"N_wide:        was {retrieved and len(positives) - retrieved}, now {len(n_wide)}")

    by_session: dict[str, list[dict]] = defaultdict(list)
    for record in bearing:
        by_session[record["task_id"]].append(record)
    reachable = sorted(by_session)
    print(f"sessions with >=1 actionable draft (the coverage ceiling): {len(reachable)}/14\n")

    vocabs: dict[int, set[str]] = {}
    if args.dsn:
        import psycopg
        from collections import Counter

        with psycopg.connect(args.dsn, connect_timeout=20) as conn:
            rows = conn.execute(
                "SELECT source_uri, string_agg(text,' ') FROM recall_chunks_v1 GROUP BY source_uri"
            ).fetchall()
        df: Counter[str] = Counter()
        for _, text in rows:
            for token in {t.lower() for t in TOKEN_RE.findall(text or "")}:
                df[token] += 1
        for df_max in (1, 2, 3, 5, 10):
            vocabs[df_max] = {t for t, c in df.items() if c <= df_max}

    triggers: dict[str, object] = {"T0_always": lambda r: True}
    for delta in (0.0, 0.01, 0.02, 0.05, 0.1):
        triggers[f"T1_margin_{delta}"] = lambda r, d=delta: r["margin"] > d
    for df_max, vocab in vocabs.items():
        triggers[f"T2_vocab_df{df_max}"] = (
            lambda r, v=vocab: any(t.lower() in v for t in TOKEN_RE.findall(r["draft"]))
        )
    triggers["T3_operation"] = lambda r: r["operation"]
    if any("llm_gate" in r for r in records):
        triggers["T4_llm_gate"] = lambda r: r.get("llm_gate") is True

    results = []
    for name, fires in triggers.items():
        covered = sum(
            1 for task_id in reachable
            if any(fires(r) for r in by_session[task_id])
        )
        ft_clean = sum(1 for r in n_clean if fires(r))
        ft_wide = sum(1 for r in n_wide if fires(r))
        fired = sum(1 for r in positives if fires(r))
        results.append({
            "trigger": name,
            "coverage": covered, "of_reachable": len(reachable), "of_sessions": 14,
            "coverage_rate": round(covered / max(1, len(reachable)), 4),
            "ft_clean": ft_clean, "of_clean": len(n_clean),
            "ft_clean_rate": round(ft_clean / max(1, len(n_clean)), 4),
            "ft_wide": ft_wide, "of_wide": len(n_wide),
            "ft_wide_rate": round(ft_wide / max(1, len(n_wide)), 4),
            "suppression": round(1 - fired / max(1, len(positives)), 4),
        })

    t0 = results[0]
    if not (t0["coverage"] == len(reachable) and t0["ft_clean_rate"] == 1.0):
        raise SystemExit(f"SCORING CONTROL FAILED: T0 must cover everything reachable, got {t0}")
    print("scoring control: OK (T0 covers every reachable session and fires on every negative)\n")

    print(f"{'trigger':<20} {'cover/10':>9} {'/14':>5} {'ft_clean':>9} {'ft_wide':>8} {'suppress':>9}")
    for r in results:
        print(f"{r['trigger']:<20} {r['coverage']:>6}/10 {r['coverage']:>4}/14 "
              f"{r['ft_clean_rate']:>9.3f} {r['ft_wide_rate']:>8.3f} {r['suppression']:>9.3f}")

    # Both selections, reported side by side: the trigger record's literal rule (sort on coverage,
    # which selected a dominated candidate) and the Pareto rule registered afterwards.
    literal = max(results[1:], key=lambda r: r["coverage"])
    frontier = [r for r in results[1:] if r["ft_clean_rate"] <= 0.35]
    pareto = max(frontier, key=lambda r: r["coverage"]) if frontier else None
    print(f"\nliteral rule (max coverage):        {literal['trigger']}  "
          f"coverage {literal['coverage']}/10  ft_clean {literal['ft_clean_rate']:.3f}")
    if pareto:
        print(f"Pareto rule (max coverage, ft<=0.35): {pareto['trigger']}  "
              f"coverage {pareto['coverage']}/10  ft_clean {pareto['ft_clean_rate']:.3f}")
    else:
        print("Pareto rule: no candidate has ft_clean <= 0.35")

    out = Path(args.out) if args.out else Path(
        "benchmarks/artifacts/agent_ab/trigger-rederived.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "populations": {
            "positive_drafts": len(positives),
            "hazard_bearing_retrieved": retrieved,
            "hazard_bearing_judged": len(bearing),
            "n_wide": len(n_wide), "n_clean": len(n_clean),
            "reachable_sessions": len(reachable), "reachable": reachable,
        },
        "results": results,
        "selection": {"literal": literal, "pareto": pareto},
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
