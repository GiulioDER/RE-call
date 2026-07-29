"""Build a BLIND labelling set: RE-call vs Mem0 on BEAM, judged by a human.

Why a human judge here rather than a cheaper model
--------------------------------------------------
`benchmarks/labelling/RESULTS.md` measured the LLM judges against blind human labels on 195
contested items: `gpt-4o-mini` was right on **0.144** of them, `gpt-4o` on **0.856**. A weak judge
is not merely noisy — mini's failure mode is *under-crediting correct answers*, which attacks
precisely the signal a system comparison is trying to detect. And a strong judge is the expensive
half of a run: on the BEAM arm, output tokens were two thirds of the bill and the `judgment` field
averages four characters, so almost all of it is judge reasoning.

Removing the LLM judge therefore removes most of the cost AND the largest measurement error at the
same time. What it costs is annotator time, which is why this file exists: to keep that time
proportional and the result publishable.

What is hidden, and why
-----------------------
The CSV shows the question, the gold answer and ONE predicted answer. It does not show which
system produced it, nor what any LLM judge said. The annotator is the author of one of those
systems, so unblinded labels would be worthless no matter how carefully they were made — a reader
can dismiss "the author graded his own system" without reading further. Row order is shuffled with
a fixed seed and the mapping is written to a separate key file.

Keeping it proportional
-----------------------
`--disagreements-only` emits just the questions where the two arms' recorded LLM scores differ.
Questions where both arms already agree carry no information about which system is better, so
labelling them spends attention for nothing — the same argument that reduced the judge-arbitration
set to its 199 contested items. `--category abstention` restricts to the axis this project is
actually about.

::

    python -m benchmarks.labelling.build_beam_labelling \\
        --ours   benchmarks/results/beam1M_recall_<ts>.json \\
        --theirs benchmarks/data/mem0_beam_1m_results.json \\
        --disagreements-only --out benchmarks/labelling/beam
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

from benchmarks.beam.dataset import _rubric_of

#: Mem0's published file nests each arm's answer under a retrieval-budget key.
THEIR_CUTOFF = "top_200"


def _their_cell(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Mem0's answer at the headline retrieval budget, or an error.

    No insertion-order fallback. `next(iter(cutoffs.values()))` used to stand in whenever
    `top_200` was missing OR merely falsy — so a structurally degraded upstream cell silently
    became a `top_50` comparison, and this output is what the human blind-labelling CSV is built
    from. `benchmarks.beam.run._load_published` had the same defect and now raises; leaving it
    here would keep the two readers of the same third-party file disagreeing about which cell is
    the comparator, which is the condition that makes the mismatch invisible.
    """
    cutoffs = evaluation.get("cutoff_results") or {}
    if not cutoffs:
        return {}
    cell: dict[str, Any] = cutoffs.get(THEIR_CUTOFF) or {}
    if not cell:
        raise SystemExit(
            f"question {evaluation.get('question_id')!r} has no usable {THEIR_CUTOFF!r} cell "
            f"(available: {sorted(cutoffs)}). Refusing to substitute another retrieval budget: "
            f"the arms would then be compared at different depths with nothing recording it."
        )
    return cell


#: Characters a spreadsheet treats as the start of a FORMULA rather than as text.
_FORMULA_LEAD = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(value: str) -> str:
    """Neutralise a cell a spreadsheet would execute rather than display.

    This file's whole purpose is to be opened in Excel or LibreOffice by a human annotator, and
    `question`, `gold_answer` and `predicted_answer` are never author-written: they come from a
    third-party downloaded corpus (BEAM's rubrics) and from model output. A cell beginning `=`
    is a formula to both applications, so a rubric reading `=cmd|'/c calc'!A1` runs on open.

    Prefixing with an apostrophe is the standard fix and is visible to the annotator as a leading
    quote rather than silently altering what they grade.
    """
    if value and value[0] in _FORMULA_LEAD:
        return "'" + value
    return value


def _gold(rubric: Any, ground_truth: str | None) -> str:
    """The standard the annotator applies: every rubric nugget, one per line.

    BEAM grades against nuggets rather than a single string, and the judge prompt's rule is that a
    prediction covering only some of them is WRONG. Showing the nuggets rather than a prose gold
    answer is showing the annotator the criterion the systems were actually held to.

    Extraction is delegated to `dataset._rubric_of` rather than re-implemented. The local copy
    read each nugget under the key `"nugget"`; BEAM stores them under `"description"`. Our own
    rows are already flattened to `list[str]` by that same function, so the bug was masked until
    the documented fallback fired — `ours.get("rubric") or their.get("rubric")`, i.e. exactly when
    our arm had no rubric for a question — and then the annotator was shown a raw dict repr as
    the gold standard they grade against.
    """
    items = _rubric_of(rubric)
    if items:
        return "\n".join(f"- {i}" for i in items)
    return ground_truth or ""


def build(
    ours_rows: list[dict[str, Any]],
    theirs: list[dict[str, Any]],
    *,
    seed: int,
    category: str | None,
    disagreements_only: bool,
    limit: int | None,
) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
    theirs_by_id = {e["question_id"]: e for e in theirs}

    pairs: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
    for row in ours_rows:
        qid = row.get("question_id")
        if not isinstance(qid, str):
            continue
        their = theirs_by_id.get(qid)
        if their is None:
            continue  # only questions BOTH arms answered can be compared
        if category and row.get("question_type") != category:
            continue
        cell = _their_cell(their)
        if disagreements_only:
            ours_score = row.get("score")
            their_score = cell.get("score")
            if ours_score is None or their_score is None or ours_score == their_score:
                continue
        pairs.append((qid, str(row.get("question_type", "?")), row, their))

    # Capped by QUESTION, before the rows are expanded to one per arm. Slicing `rows` after the
    # shuffle cut arm pairs in half at random: `score_beam_labels` drops half-pairs from the
    # McNemar table but accumulates `per_arm` over EVERY labelled row, so the two per-arm
    # accuracies were computed over different, randomly-selected question sets and printed side
    # by side as a system comparison. Under --disagreements-only, where exactly one arm is correct
    # per question by construction, an unbalanced split biases them in opposite directions.
    if limit:
        pairs = pairs[: max(1, limit // 2)]

    rows: list[dict[str, str]] = []
    key: dict[str, dict[str, Any]] = {}
    for qid, qtype, ours, their in pairs:
        cell = _their_cell(their)
        gold = _gold(ours.get("rubric") or their.get("rubric"), their.get("ground_truth_answer"))
        for arm, answer, llm_score in (
            ("recall", ours.get("generated_answer", ""), ours.get("score")),
            ("mem0", cell.get("generated_answer", ""), cell.get("score")),
        ):
            rows.append({
                "category": qtype,
                "question": ours.get("question", their.get("question", "")),
                "gold_answer": gold,
                "predicted_answer": (answer or "").strip(),
                "_arm": arm,
                "_question_id": qid,
                "_llm_score": "" if llm_score is None else str(llm_score),
            })

    random.Random(seed).shuffle(rows)

    out_rows: list[dict[str, str]] = []
    for i, row in enumerate(rows, 1):
        key[str(i)] = {
            "arm": row.pop("_arm"),
            "question_id": row.pop("_question_id"),
            "llm_score": row.pop("_llm_score"),
        }
        out_rows.append({"item": str(i), **row, "your_verdict_Y_or_N": ""})
    return out_rows, key


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ours", type=Path, required=True, help="our BEAM result artifact")
    p.add_argument("--theirs", type=Path, required=True, help="Mem0's published beam_*_results.json")
    p.add_argument("--out", type=Path, required=True, help="output prefix (writes .csv and _key.json)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--category", default=None, help="restrict to one question_type, e.g. abstention")
    p.add_argument(
        "--disagreements-only", action="store_true",
        help="only questions where the two arms' recorded LLM scores differ — agreement carries no "
             "information about which system is better",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="cap the number of ITEMS; applied in QUESTION units (limit // 2) so both "
             "arms of a question are always emitted together or not at all",
    )
    args = p.parse_args()

    ours = json.loads(args.ours.read_text(encoding="utf-8"))["rows"]
    theirs = json.loads(args.theirs.read_text(encoding="utf-8"))["evaluations"]
    rows, key = build(
        ours, theirs, seed=args.seed, category=args.category,
        disagreements_only=args.disagreements_only, limit=args.limit,
    )
    if not rows:
        raise SystemExit("no items selected — check --category / --disagreements-only")

    csv_path = args.out.with_suffix(".csv")
    key_path = args.out.parent / (args.out.name + "_key.json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["item", "category", "question", "gold_answer",
                        "predicted_answer", "your_verdict_Y_or_N"],
        )
        writer.writeheader()
        writer.writerows([{k: _csv_safe(v) for k, v in row.items()} for row in rows])
    key_path.write_text(json.dumps(key, indent=1), encoding="utf-8")

    arms = [v["arm"] for v in key.values()]
    print(f"{len(rows)} items ({arms.count('recall')} recall / {arms.count('mem0')} mem0) "
          f"over {len(set(v['question_id'] for v in key.values()))} questions")
    print(f"  {csv_path}")
    print(f"  {key_path}   <- do NOT open until labelling is finished")


if __name__ == "__main__":
    main()
