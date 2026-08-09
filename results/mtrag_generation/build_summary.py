"""Regenerate `mtrag_summary.json`, the artifact every published MTRAG number is marked against.

Prior work: the numbers themselves are established in `../../docs/MTRAG_BENCHMARK.md` and
`CORRECTION-idk-conditioning-2026-08-09.md`; this file only RETAINS them so the claim gate
(`benchmarks/claim_gate.py`) can check a document against something other than prose.

Nothing here is hand-typed. Each figure is recomputed from a committed artifact:

  Task A          `../mtrag_taskA_dev/*.metrics.json`      (777 judged dev queries)
  Task B and C    `runs/*.fixed.jsonl.gz`                  (842 tasks, IDK-conditioned)
  baselines       the MTRAG release `evaluations/*.json`   (NOT in this repo, see below)

⚠️ The published baselines are recomputed from the MTRAG release, which this repository does not
vendor. Pass its checkout with `--mtrag <path>` to refresh them; without it the existing baseline
block is carried forward unchanged and the script says so, rather than silently emitting a file
with those keys missing and breaking every marker that cites them.

Usage:
    python results/mtrag_generation/build_summary.py [--mtrag /path/to/mt-rag-benchmark]
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import random
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK_A = HERE.parent / "mtrag_taskA_dev"
RUNS = HERE / "runs"
OUT = HERE / "mtrag_summary.json"

METRICS = ("RL_F", "RB_llm", "RB_agg")
RUN_STEMS = (
    "taskb",
    "taskb_official",
    "taskc_benchmark",
    "taskc_benchmark_official",
    "taskc_recall",
    "taskc_recall_official",
)


def first(v: object) -> object:
    return v[0] if isinstance(v, list) else v


def harmonic(values: list[float]) -> float:
    return len(values) / sum(1.0 / v for v in values)


def task_a() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for path in sorted(glob.glob(str(TASK_A / "*.metrics.json"))):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        scores = payload["scores"]["overall"]
        out[payload["arm"]["name"]] = {
            "ndcg_at_5": scores["nDCG@5"],
            "recall_at_100": scores["Recall@100"],
            "n": payload["predictions"],
        }
    return out


def paired_rerank() -> dict[str, object]:
    """The Voyage-rerank paired test, recomputed from the two arms' committed per-query scores.

    Deterministic on purpose: a fixed-seed bootstrap and an exhaustively-seeded sign-flip
    permutation, so this artifact holds the same interval every time it is regenerated. An
    interval that moves when you rebuild the file cannot back a published claim.
    """
    a = json.loads((TASK_A / "hybrid_splade.metrics.json").read_text(encoding="utf-8"))
    b = json.loads((TASK_A / "hybrid_splade_voyage.metrics.json").read_text(encoding="utf-8"))
    pa, pb = a["scores"]["per_query"], b["scores"]["per_query"]
    keys = sorted(set(pa) & set(pb))
    diffs = [pb[k]["nDCG@5"] - pa[k]["nDCG@5"] for k in keys]
    n = len(diffs)
    obs = statistics.fmean(diffs)

    rng = random.Random(20260809)
    boot = []
    for _ in range(10000):
        boot.append(statistics.fmean([diffs[rng.randrange(n)] for _ in range(n)]))
    boot.sort()
    lo, hi = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot)) - 1]

    rng2 = random.Random(20260809)
    hits = 0
    trials = 10000
    for _ in range(trials):
        flipped = statistics.fmean([d if rng2.random() < 0.5 else -d for d in diffs])
        if abs(flipped) >= abs(obs):
            hits += 1
    return {
        "metric": "nDCG@5",
        "arms": ["hybrid_splade", "hybrid_splade_voyage"],
        "n": n,
        "delta": obs,
        "ci95_low": lo,
        "ci95_high": hi,
        "p_permutation": (hits + 1) / (trials + 1),
        "better": sum(1 for d in diffs if d > 0),
        "worse": sum(1 for d in diffs if d < 0),
        "unchanged": sum(1 for d in diffs if d == 0),
    }


def run_figures() -> dict[str, dict[str, float]]:
    """Per-run conditioned means, the harmonic mean, and the per-metric n.

    The per-metric n is emitted deliberately: `taskb` averages RL_F over 832 rows and the other
    two over 842 because of twelve RAGAS timeouts, and a summary that hid that would reproduce
    the mistake this whole correction exists to document.
    """
    out: dict[str, dict[str, float]] = {}
    for stem in RUN_STEMS:
        path = RUNS / f"{stem}.fixed.jsonl.gz"
        if not path.is_file():
            continue
        rows = [
            json.loads(line)
            for line in gzip.open(path, "rt", encoding="utf-8")
            if line.strip()
        ]
        per: dict[str, list[float]] = {}
        for metric in METRICS:
            vals = [first(r["metrics"].get(f"{metric}_idk_underspecified")) for r in rows]
            per[metric] = [v for v in vals if v is not None]
        means = [statistics.fmean(per[m]) for m in METRICS]
        entry: dict[str, float] = {
            "rl_f": means[0],
            "rb_llm": means[1],
            "rb_alg": means[2],
            "harmonic": harmonic(means),
            "rows": len(rows),
        }
        for metric in METRICS:
            entry[f"n_{metric.lower()}"] = len(per[metric])
        out[stem] = entry
    return out


def abstention() -> dict[str, object]:
    """Correct refusals on UNANSWERABLE, ours, by the official judge.

    A correct refusal is an exact 1.0 on the conditioned metric for an UNANSWERABLE task, which is
    the same rule applied to the baselines below. ⛔ Not a regex over the answer text: a
    string-matched abstention rate misled this analysis three times.
    """
    path = RUNS / "taskc_recall_official.fixed.jsonl.gz"
    src = path if path.is_file() else RUNS / "taskb_official.fixed.jsonl.gz"
    rows = [
        json.loads(line) for line in gzip.open(src, "rt", encoding="utf-8") if line.strip()
    ]
    hit = total = 0
    for r in rows:
        if first(r.get("Answerability")) != "UNANSWERABLE":
            continue
        total += 1
        if first(r["metrics"].get("RB_agg_idk_underspecified")) == 1:
            hit += 1
    return {
        "source": src.name,
        "correct": hit,
        "total": total,
        "rate_pct": 100.0 * hit / total if total else None,
    }


def baselines(mtrag: Path) -> dict[str, dict[str, object]]:
    """Recompute the published baselines from the release, per task and per system."""

    def value(ann: dict, key: str) -> object:
        node = ann.get(key)
        if not node:
            return None
        for sub in ("composite", "system"):
            if sub in node and isinstance(node[sub], dict):
                return node[sub].get("value")
        return None

    out: dict[str, dict[str, object]] = {}
    for task, name in (("task_c", "RAG.json"), ("task_b", "reference.json")):
        path = mtrag / "mtrag-human" / "evaluations" / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        label = {
            t["task_id"]: (
                t["Answerability"][0]
                if isinstance(t.get("Answerability"), list)
                else t.get("Answerability")
            )
            for t in payload["tasks"]
        }
        per: dict[str, dict[str, list[float]]] = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        refuse: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        for e in payload["evaluations"]:
            mid = e["model_id"]
            for metric in ("rl_f", "rb_llm", "rb_agg"):
                v = value(e["annotations"], metric)
                if v is not None:
                    per[mid][metric].append(v)
            if label.get(e["task_id"]) == "UNANSWERABLE":
                rb = value(e["annotations"], "rb_agg")
                if rb is not None:
                    refuse[mid][1] += 1
                    refuse[mid][0] += 1 if rb == 1 else 0
        systems: dict[str, object] = {}
        for mid, mm in per.items():
            means = [statistics.fmean(mm[m]) for m in ("rl_f", "rb_llm", "rb_agg")]
            entry = {
                "rl_f": means[0],
                "rb_llm": means[1],
                "rb_alg": means[2],
                "harmonic": harmonic(means),
                "n": len(mm["rl_f"]),
            }
            if refuse[mid][1]:
                entry["unanswerable_correct"] = refuse[mid][0]
                entry["unanswerable_total"] = refuse[mid][1]
                entry["unanswerable_pct"] = 100.0 * refuse[mid][0] / refuse[mid][1]
            systems[mid.replace(".", "_")] = entry
        out[task] = systems
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtrag", type=Path, default=None)
    args = ap.parse_args()

    previous: dict = {}
    if OUT.is_file():
        previous = json.loads(OUT.read_text(encoding="utf-8"))

    payload: dict[str, object] = {
        "_note": (
            "Every MTRAG figure this repository publishes, recomputed from committed artifacts by "
            "results/mtrag_generation/build_summary.py. Baselines are RECOMPUTED from the release, "
            "not copied from the paper, and run +0.018 to +0.043 high against the published table; "
            "they are anchored lifts and must not be quoted against the public leaderboard."
        ),
        "task_a": task_a(),
        "runs": run_figures(),
        "abstention_ours": abstention(),
        "paired_rerank": paired_rerank(),
    }

    # Derived differences, computed here rather than in prose. A delta quoted in a document is a
    # claim like any other, and subtracting two figures by hand is exactly where a sign or a digit
    # goes wrong unnoticed.
    runs = payload["runs"]
    ta = payload["task_a"]
    deltas: dict[str, float] = {}
    if "taskc_recall_official" in runs and "taskc_benchmark_official" in runs:
        deltas["recall_vs_benchmark_contexts_official"] = (
            runs["taskc_recall_official"]["harmonic"] - runs["taskc_benchmark_official"]["harmonic"]
        )
    if "taskc_recall" in runs and "taskc_benchmark" in runs:
        deltas["recall_vs_benchmark_contexts_abstain"] = (
            runs["taskc_recall"]["harmonic"] - runs["taskc_benchmark"]["harmonic"]
        )
    if "taskb" in runs and "taskb_official" in runs:
        deltas["prompt_effect_taskb"] = (
            runs["taskb_official"]["harmonic"] - runs["taskb"]["harmonic"]
        )
    if {"hybrid_splade", "hybrid_lexical"} <= ta.keys():
        deltas["splade_over_lexical_recall_at_100"] = (
            ta["hybrid_splade"]["recall_at_100"] - ta["hybrid_lexical"]["recall_at_100"]
        )
    if {"hybrid_splade_voyage", "hybrid_splade"} <= ta.keys():
        deltas["voyage_rerank_ndcg_at_5"] = (
            ta["hybrid_splade_voyage"]["ndcg_at_5"] - ta["hybrid_splade"]["ndcg_at_5"]
        )
    payload["deltas"] = deltas

    if args.mtrag:
        payload["baselines"] = baselines(args.mtrag)
    elif "baselines" in previous:
        payload["baselines"] = previous["baselines"]
        print("note: --mtrag not given, carrying the existing baseline block forward unchanged")
    else:
        print("ERROR: no baseline block and no --mtrag; markers citing baselines would break")
        return 1

    # Deltas against the baselines need the baseline block, so they are computed after it is set.
    bl = payload.get("baselines", {})
    if "task_c" in bl and "taskc_recall_official" in runs:
        deltas["recall_vs_published_gpt4o_task_c"] = (
            runs["taskc_recall_official"]["harmonic"] - bl["task_c"]["gpt-4o"]["harmonic"]
        )
    if "task_b" in bl and "taskb_official" in runs:
        deltas["ours_vs_published_gpt4o_task_b"] = (
            runs["taskb_official"]["harmonic"] - bl["task_b"]["gpt-4o"]["harmonic"]
        )

    # Placement among the published systems, computed rather than counted by hand. `target` is the
    # human reference, not a system, so it is excluded from the field.
    ranks: dict[str, object] = {}
    for task, stems in (("task_b", ("taskb", "taskb_official")),
                        ("task_c", ("taskc_recall", "taskc_recall_official",
                                    "taskc_benchmark", "taskc_benchmark_official"))):
        if task not in bl:
            continue
        field = sorted(
            (v["harmonic"] for k, v in bl[task].items() if k != "target"), reverse=True
        )
        for stem in stems:
            if stem not in runs:
                continue
            ours = runs[stem]["harmonic"]
            ranks[stem] = {
                "harmonic": ours,
                "rank": sum(1 for h in field if h > ours) + 1,
                "of": len(field) + 1,
            }
    payload["ranks"] = ranks

    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(Path.cwd()) if OUT.is_relative_to(Path.cwd()) else OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
