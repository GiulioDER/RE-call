"""Score the ATM answer arms against each other, with the columns the pre-registration requires.

⛔ Never prints QS alone. `docs/preregistrations/2026-08-20-atm-evidence-allocation-and-selection.md`
requires the refusal rate and the gold-abstention score beside every total, because the two arms
under test can buy points on one side and pay them on the other with no sign of it in the mean: the
upside measured for a disposition change is 3.46 QS and the downside is 1.68.

The deterministic half of ATM, `number` and `list_recall`, is recomputed here with the official
scorer, so 60% of the comparison costs nothing. `open_end` needs the official judge and is read
from its output when present, and reported as missing when it is not, never silently as zero.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.atm_subset import POPULATION, weights  # noqa: E402


def _load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["id"])] = row
    return rows


def _official_scores(path: Path) -> dict[str, float]:
    """Read the official evaluator's per-question output, whichever container shape it used."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.values() if isinstance(payload, dict) else payload
    return {str(row.get("id")): float(row.get("accuracy", 0.0)) for row in rows}


def score_arm(
    arm_dir: Path,
    gold: dict[str, dict[str, Any]],
    *,
    official: Path | None,
) -> dict[str, Any]:
    from memqa.utils.evaluator.evaluate_qa import _list_jaccard_core, deterministic_accuracy
    from memqa.utils.evaluator.normalizer import is_abstention

    answers = _load_jsonl(arm_dir / "answers.jsonl")
    diagnostics = _load_jsonl(arm_dir / "diagnostics.jsonl")
    judged = _official_scores(official) if official else {}

    per_type: dict[str, list[float]] = {}
    missing_judge = 0
    scored: dict[str, float] = {}
    for qid, row in answers.items():
        question = gold.get(qid)
        if question is None:
            continue
        qtype = str(question.get("qtype"))
        prediction = str(row.get("answer", ""))
        truth = str(question.get("answer", ""))
        if qtype == "number":
            value = float(deterministic_accuracy(truth, prediction, question.get("question")))
        elif qtype == "list_recall":
            value = _list_jaccard_core(truth, prediction)[0]
        elif qid in judged:
            value = judged[qid]
        else:
            missing_judge += 1
            continue
        scored[qid] = value
        per_type.setdefault(qtype, []).append(value)

    counts = Counter(str(gold[qid].get("qtype")) for qid in scored)
    per_question = weights(counts)
    qs = sum(value * per_question[str(gold[qid]["qtype"])] for qid, value in scored.items())

    refused = [row for row in diagnostics.values() if row.get("refused")]
    abstention_ids = [
        qid for qid in scored if is_abstention(str(gold[qid].get("answer", "")))
    ]
    return {
        "arm": arm_dir.name,
        "answers": len(answers),
        "scored": len(scored),
        "missing_judge": missing_judge,
        # Reweighted to the population shares. The raw mean is a different metric and is not shown,
        # so it cannot be mistaken for this one.
        "qs_reweighted": round(qs * 100, 2) if len(scored) == sum(counts.values()) else None,
        "by_qtype": {
            qtype: round(statistics.mean(values) * 100, 2) for qtype, values in sorted(per_type.items())
        },
        "refusal_rate": round(len(refused) / len(diagnostics), 4) if diagnostics else None,
        "gold_abstention_score": (
            round(statistics.mean(scored[qid] for qid in abstention_ids) * 100, 2)
            if abstention_ids
            else None
        ),
        "gold_abstention_n": len(abstention_ids),
        "answer_chars_median": (
            round(statistics.median(row["answer_chars"] for row in diagnostics.values()), 1)
            if diagnostics
            else None
        ),
        "selection_parse_failures": sum(
            1 for row in diagnostics.values() if row.get("parse_failed")
        ),
        "selection_rescued": sum(1 for row in diagnostics.values() if row.get("rescued_answer")),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-file", type=Path, required=True)
    ap.add_argument("--arms-root", type=Path, required=True)
    ap.add_argument("--official-eval", type=Path, default=None,
                    help="directory holding <arm>.atm.json outputs from the official evaluator")
    ap.add_argument("--atm-bench-dir", type=Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(args.atm_bench_dir))
    gold = {str(row["id"]): row for row in json.loads(args.qa_file.read_text(encoding="utf-8"))}

    rows = []
    for arm_dir in sorted(p for p in args.arms_root.iterdir() if p.is_dir()):
        official = (args.official_eval / f"{arm_dir.name}.atm.json") if args.official_eval else None
        rows.append(score_arm(arm_dir, gold, official=official))

    header = f"{'arm':30s} {'QS':>7s} {'number':>7s} {'list':>7s} {'open':>7s} {'refuse':>7s} {'abst':>6s} {'chars':>6s} {'parse!':>6s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        by = row["by_qtype"]
        print(
            f"{row['arm']:30s} "
            f"{row['qs_reweighted'] if row['qs_reweighted'] is not None else 'n/a':>7} "
            f"{by.get('number', 'n/a'):>7} {by.get('list_recall', 'n/a'):>7} {by.get('open_end', 'n/a'):>7} "
            f"{row['refusal_rate'] if row['refusal_rate'] is not None else 'n/a':>7} "
            f"{row['gold_abstention_score'] if row['gold_abstention_score'] is not None else 'n/a':>6} "
            f"{row['answer_chars_median'] if row['answer_chars_median'] is not None else 'n/a':>6} "
            f"{row['selection_parse_failures']:>6}"
        )
        if row["missing_judge"]:
            print(f"{'':30s} {row['missing_judge']} open_end questions have no judge result yet")
    print()
    print(json.dumps(rows, indent=1))
    print(f"\npopulation shares used for reweighting: {POPULATION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
