"""Pair the two BEAM arms question-by-question and test the difference.

Both arms are scored by the same judge instance (see `run.py`), so a pair here is: same question
id, same rubric, same judge — differing only in which memory layer retrieved the context and what
the answerer wrote from it. That is what makes McNemar the right test rather than a two-sample
comparison of two independently-collected accuracies.

Three families are tested and the family-wise error is controlled with Holm across them, because
the interesting claim ("RE-call beats Mem0 on BEAM") is a claim about the whole panel, not about
whichever cell came out smallest:

1. **accuracy** — pass (nugget mean >= 0.5) over all questions
2. **abstention** — correct withholding on the `abstention` category
3. **false abstain** — refusal on the other nine categories, where refusing is the failure

The McNemar primitive is imported from `benchmarks.analyze` rather than re-derived, so the BEAM
arm and the LOCOMO arm cannot drift into two different definitions of the same p-value.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.analyze import _mcnemar_from_discordant

#: Score at or above which a question counts as passed. Upstream's threshold; changing it changes
#: every accuracy cell, so it lives here once rather than as a literal in three places.
PASS_THRESHOLD = 0.5


def _rows(path: Path) -> list[dict[str, Any]]:
    """Rows from a run artifact — the `.json` summary file or its `.partial.jsonl` sidecar."""
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = (
        payload["rows"] if isinstance(payload, dict) and "rows" in payload else payload
    )
    return rows


def _aligned(
    a_rows: Sequence[Mapping[str, Any]], b_rows: Sequence[Mapping[str, Any]]
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Pair on question id, or refuse.

    An intersection would produce a number that reads as a paired comparison of the full 700 and
    is not one, so a mismatch raises instead. Rows the judge failed to parse (`score` NaN) are
    dropped from BOTH sides together — dropping them one-sidedly would let a harness error count
    as evidence against whichever arm it happened to hit.
    """
    a_index = {str(r["question_id"]): r for r in a_rows}
    b_index = {str(r["question_id"]): r for r in b_rows}
    if len(a_index) != len(a_rows) or len(b_index) != len(b_rows):
        raise ValueError("duplicate question_id in an arm; pairing would silently drop one")
    if a_index.keys() != b_index.keys():
        only_a = sorted(a_index.keys() - b_index.keys())
        only_b = sorted(b_index.keys() - a_index.keys())
        raise ValueError(
            "the arms did not answer the same questions, so they cannot be paired: "
            f"{len(only_a)} only in A (e.g. {only_a[:3]}), "
            f"{len(only_b)} only in B (e.g. {only_b[:3]})"
        )
    pairs = []
    for qid in sorted(a_index):
        a, b = a_index[qid], b_index[qid]
        if a["score"] != a["score"] or b["score"] != b["score"]:  # NaN on either side
            continue
        if a["question_type"] != b["question_type"]:
            raise ValueError(f"question {qid!r} is typed differently in the two arms")
        pairs.append((a, b))
    return pairs


def _mcnemar(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    select: Callable[[Mapping[str, Any]], bool],
    good: Callable[[Mapping[str, Any]], bool],
) -> dict[str, Any]:
    both = neither = a_only = b_only = 0
    for a, b in pairs:
        if not select(a):
            continue
        a_good, b_good = good(a), good(b)
        if a_good and b_good:
            both += 1
        elif a_good:
            a_only += 1
        elif b_good:
            b_only += 1
        else:
            neither += 1
    statistic, p_value, test = _mcnemar_from_discordant(a_only, b_only)
    return {
        "n_paired": both + neither + a_only + b_only,
        "both": both,
        "neither": neither,
        "a_only": a_only,
        "b_only": b_only,
        "statistic": statistic,
        "p_value": p_value,
        "test": test,
    }


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, order-enforced so they stay monotone.

    Reported as adjusted p-values rather than as a bare accept/reject at some alpha, so a reader
    picks their own threshold and can see how close a cell came to it.
    """
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = round(running, 6)
    return adjusted


def compare(a_rows: Sequence[Mapping[str, Any]], b_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The full paired panel: three McNemar tests, Holm-adjusted, plus the effect sizes."""
    pairs = _aligned(a_rows, b_rows)
    passed = lambda r: r["score"] >= PASS_THRESHOLD  # noqa: E731 - one-line predicate, used twice
    answerable = lambda r: r["question_type"] != "abstention"  # noqa: E731
    abstention = lambda r: r["question_type"] == "abstention"  # noqa: E731

    tests = {
        "accuracy": _mcnemar(pairs, lambda r: True, passed),
        "abstention": _mcnemar(pairs, abstention, lambda r: bool(r["abstained"])),
        # Scored so `a_only` means "A avoided a false abstain that B committed" — i.e. higher is
        # better for A in every one of the three tables, which is what makes the panel readable.
        "false_abstain": _mcnemar(pairs, answerable, lambda r: not r["abstained"]),
    }
    adjusted = holm({name: t["p_value"] for name, t in tests.items()})
    for name, t in tests.items():
        t["p_holm"] = adjusted[name]

    def _mean(rows: Sequence[Mapping[str, Any]], select: Callable[[Mapping[str, Any]], bool]) -> float | None:
        picked = [r["score"] for r in rows if select(r)]
        return round(statistics.mean(picked), 4) if picked else None

    a_side = [a for a, _ in pairs]
    b_side = [b for _, b in pairs]
    return {
        "n_pairs": len(pairs),
        "avg_score": {"a": _mean(a_side, lambda r: True), "b": _mean(b_side, lambda r: True)},
        "accuracy_pct": {
            "a": round(100 * sum(1 for r in a_side if passed(r)) / len(pairs), 2) if pairs else None,
            "b": round(100 * sum(1 for r in b_side if passed(r)) / len(pairs), 2) if pairs else None,
        },
        "abstain_rate_on_abstention": {
            "a": round(sum(1 for r in a_side if abstention(r) and r["abstained"]) / max(1, sum(1 for r in a_side if abstention(r))), 4),
            "b": round(sum(1 for r in b_side if abstention(r) and r["abstained"]) / max(1, sum(1 for r in b_side if abstention(r))), 4),
        },
        "false_abstain_rate": {
            "a": round(sum(1 for r in a_side if answerable(r) and r["abstained"]) / max(1, sum(1 for r in a_side if answerable(r))), 4),
            "b": round(sum(1 for r in b_side if answerable(r) and r["abstained"]) / max(1, sum(1 for r in b_side if answerable(r))), 4),
        },
        "tests": tests,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="Arm A artifact (RE-call)")
    parser.add_argument("--b", type=Path, required=True, help="Arm B artifact (Mem0, re-judged)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = compare(_rows(args.a), _rows(args.b))
    result["a_artifact"] = str(args.a)
    result["b_artifact"] = str(args.b)
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
