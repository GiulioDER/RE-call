"""MTRAGEval RB_alg probe -- STEP 1, the invariant.

Pre-registration: RE-call/benchmarks/archive/preregistrations/PREREGISTRATION-mtrag-rbalg.md sec.1

Recompute the per-model Task C harmonic mean from IBM's published RAG.json and check it
reproduces the published MTRAG table to within +/-0.01. If it does not, the apparatus is
wrong and no later number from it counts.

Prior work: none found on RB_alg / MTRAGEval generation / length calibration
(docs_search source_type=memory, "MT-RAG generation RB_alg RougeL response length
calibration answer verbosity benchmark", gap_warning false). Two adjacent hits:
[[project-recall-token-f1-harness-offset-2026-07-29]] and
[[project-recall-mtrag-symmetric-baseline-2026-08-04]]. Full search recorded in
benchmarks/archive/preregistrations/PREREGISTRATION-mtrag-rbalg.md.
"""

import json
import sys
from statistics import mean
from typing import Any

PATH = sys.argv[1] if len(sys.argv) > 1 else "RAG.json"

# The published MTRAG Task C (RAG) harmonic means, transcribed from the paper's table.
PUBLISHED = {
    "Target": 0.81,
    "GPT-4o": 0.53,
    "Llama 3.1 405B": 0.53,
    "Qwen 2.5 (72b)": 0.52,
    "Llama 3.1 70B": 0.52,
    "GPT-4o-mini": 0.51,
    "Command-R+ (104b)": 0.51,
    "Qwen 2.5 (7b)": 0.51,
    "Mixtral 8x22B Instruct": 0.48,
    "Llama 3.1 8B": 0.45,
}

TRIPLE = ("rl_f", "rb_llm", "rb_agg")


def hm(vals: list[float]) -> float:
    if any(v is None or v <= 0 for v in vals):
        return 0.0
    return len(vals) / sum(1.0 / v for v in vals)


def composite(ann: dict[str, Any], key: str) -> float | None:
    node = ann.get(key)
    if node is None:
        return None
    for slot in ("composite", "system"):
        if slot in node and node[slot] is not None:
            v = node[slot].get("value")
            if v is not None:
                return float(v)
    return None


def main() -> int:
    d = json.load(open(PATH, encoding="utf-8"))

    # model_id -> display name
    names = {}
    for m in d["models"]:
        if isinstance(m, dict):
            names[m.get("model_id") or m.get("id") or m.get("name")] = (
                m.get("display_name") or m.get("name")
            )
        else:
            names[m] = m

    per_model: dict[str, dict[str, Any]] = {}
    for ev in d["evaluations"]:
        mid = ev["model_id"]
        ann = ev.get("annotations") or {}
        vals = {k: composite(ann, k) for k in TRIPLE}
        if any(v is None for v in vals.values()):
            per_model.setdefault(mid, {"skipped": 0, "rows": []})["skipped"] += 1
            continue
        per_model.setdefault(mid, {"skipped": 0, "rows": []})["rows"].append(vals)

    print(f"models in file: {len(per_model)}   evaluations: {len(d['evaluations'])}\n")
    hdr = f"{'model':26} {'n':>5} {'skip':>5} {'RL_F':>7} {'RB_llm':>7} {'RB_alg':>7} {'HM(means)':>10} {'mean(HM)':>9} {'pub':>6} {'d':>7}"
    print(hdr)
    print("-" * len(hdr))

    ok = True
    for mid, blob in sorted(
        per_model.items(), key=lambda kv: -len(kv[1]["rows"])
    ):
        rows = blob["rows"]
        if not rows:
            continue
        means = {k: mean(r[k] for r in rows) for k in TRIPLE}
        hm_of_means = hm([means[k] for k in TRIPLE])
        mean_of_hms = mean(hm([r[k] for k in TRIPLE]) for r in rows)
        label = str(names.get(str(mid), mid))
        pub = PUBLISHED.get(label)
        delta = (hm_of_means - pub) if pub is not None else None
        flag = ""
        if delta is not None and abs(delta) > 0.01:
            flag = "  <-- MISMATCH"
            ok = False
        print(
            f"{str(label)[:26]:26} {len(rows):>5} {blob['skipped']:>5} "
            f"{means['rl_f']:>7.4f} {means['rb_llm']:>7.4f} {means['rb_agg']:>7.4f} "
            f"{hm_of_means:>10.4f} {mean_of_hms:>9.4f} "
            f"{(f'{pub:.2f}' if pub is not None else '-'):>6} "
            f"{(f'{delta:+.4f}' if delta is not None else '-'):>7}{flag}"
        )

    print()
    # BUG-004 (bug-auditor, 2026-08-05): `matched` was counted from per_model keys, which the
    # setdefault populates even for a model whose every row was skipped, and the exit code
    # ignored it entirely. A rename in a future RAG.json would then match nothing, print
    # VIOLATED, and still exit 0 -- total lookup failure reading exactly like a pass.
    matched = sum(
        1
        for m, blob in per_model.items()
        if bool(blob["rows"]) and names.get(str(m)) in PUBLISHED
    )
    complete = matched == len(PUBLISHED)
    print(f"models matched to published table: {matched}/{len(PUBLISHED)}")
    if not complete:
        print("  LOOKUP INCOMPLETE: the invariant was not actually exercised on every model.")
    print("INVARIANT:", "HOLDS (all within +/-0.01)" if (ok and complete) else "VIOLATED")
    return 0 if (ok and complete) else 1


if __name__ == "__main__":
    sys.exit(main())
