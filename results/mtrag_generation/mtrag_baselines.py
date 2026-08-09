"""Recompute the MTRAG baselines from the released per-task scores, the SAME way we compute ours.

WHY RECOMPUTE AT ALL. Prior work ([[project-recall-mtrag-rbalg-probe-2026-08-05]]) established that
recomputing the published table from these files runs **+0.018 to +0.043 high on every model**: the
aggregation formula is right (it reproduces a self-reported triple to four decimals) but the
instance set is not the published one. So a number computed here is comparable as an ANCHORED LIFT
against baselines computed here, and NOT against the published leaderboard directly. Recomputing
both sides identically is what makes the comparison mean anything.

  reference.json -> Task B baselines (generation over GOLD contexts)
  RAG.json       -> Task C baselines (generation over RETRIEVED contexts)

⚠️ `target` is the human reference answer, not a competing system. It scores near 1.0 by
construction and is reported separately as a ceiling, never ranked against.
"""

import json
import statistics as st
import sys

KEYS = ("rl_f", "rb_llm", "rb_agg")


def unwrap(ann, key):
    node = ann.get(key)
    if not isinstance(node, dict):
        return None
    for wrapper in ("composite", "system"):
        inner = node.get(wrapper)
        if isinstance(inner, dict) and isinstance(inner.get("value"), (int, float)):
            return float(inner["value"])
    return None


def harmonic(vals):
    if any(v <= 0 for v in vals):
        return 0.0
    return len(vals) / sum(1.0 / v for v in vals)


def table(path, label):
    data = json.load(open(path, encoding="utf-8"))
    per_model = {}
    for row in data["evaluations"]:
        ann = row.get("annotations") or {}
        vals = per_model.setdefault(row["model_id"], {k: [] for k in KEYS})
        for k in KEYS:
            v = unwrap(ann, k)
            if v is not None:
                vals[k].append(v)

    rows = []
    for model, vals in per_model.items():
        means = [st.mean(vals[k]) for k in KEYS]
        rows.append((model, means, harmonic(means), min(len(vals[k]) for k in KEYS)))
    rows.sort(key=lambda r: r[2], reverse=True)

    print(f"\n=== {label} — recomputed from {path.split('/')[-1]} ===")
    print(f"{'model':<26} {'RL_F':>8} {'RB_llm':>8} {'RB_alg':>8} {'HARM':>8} {'n':>5}")
    for model, means, hm, n in rows:
        tag = "  <- human reference, not a system" if model == "target" else ""
        print(f"{model:<26} {means[0]:>8.4f} {means[1]:>8.4f} {means[2]:>8.4f} {hm:>8.4f} {n:>5}{tag}")
    return rows


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    table(f"{root}/reference.json", "TASK B (gold contexts)")
    table(f"{root}/RAG.json", "TASK C (retrieved contexts)")
