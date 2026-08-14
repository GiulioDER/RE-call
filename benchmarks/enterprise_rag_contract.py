"""Validation for the EnterpriseRAG dense-floor artifact, applied at the write site.

The artifact carries a `dense_floor_summary` derived from its own `rows`. That is the same facts
written twice, and the first version of this file got it wrong in a way nothing caught: every
`median` was `statistics.median_high` rather than the median, biased upward by up to 0.045, while
the commit message quoted the TRUE median for one category, a figure that appeared nowhere in the
file. The disagreement even inverted a published ranking.

A summary that disagrees with its body is worse than a wrong number, because nothing in the file
says which of the two is the typo. So the disagreement is refused by `write_dense_floor_artifact`,
which validates before it writes and therefore leaves no file behind on failure.

⚠️ The SECOND version of this module claimed that in five places while the validator had no caller
outside its own tests. Wiring is the difference between a guard and a comment about a guard, and it
is the reason `write_dense_floor_artifact` exists rather than a bare `validate_...` that a writer is
trusted to remember. `benchmarks/labelling/truth_extraction/census.py:147` is the shape being
followed, and it was already wired when this was not.

Pattern follows `benchmarks/labelling/truth_extraction/artifact_contract.py`.
"""
from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: True question counts in EnterpriseRAG-Bench v1.0.0, the weights that turn a stratified sample
#: rate into a population rate. Pinned here rather than recomputed from the sample, because the
#: whole point of the reweighting is that the sample does NOT carry these proportions.
POPULATION: Mapping[str, int] = {
    "basic": 175,
    "semantic": 125,
    "intra_document_reasoning": 40,
    "project_related": 40,
    "constrained": 30,
    "conflicting_info": 20,
    "completeness": 20,
    "miscellaneous": 20,
    "info_not_found": 20,
    "high_level": 10,
}

FLOOR = 0.50
UNANSWERABLE = "info_not_found"

#: The summary stores four decimals, so a writer emitting full precision differs from a writer
#: emitting `_rounded` output by at most half a unit in the last place. Pinned by two tests, one
#: either side, because an untested tolerance is a number nobody chose.
_TOL = 5e-5

#: Keys a human may add that are prose rather than measurement. Everything else in the summary must
#: be derivable from `rows`, so the next hand-added FIGURE cannot slip past unvalidated the way
#: `floor_catches_no_unanswerable` did.
_PROSE_KEYS = frozenset({"_derived_from", "sampling"})


def _rounded(value: float) -> float:
    return round(value, 4)


def _score(row: Mapping[str, Any]) -> float:
    """The one place a row's score is read, so every caller gets the same refusals.

    `benchmarks/enterprise_rag.py` types `best_dense_score` as `float | None` and emits `None` when
    `query_dense(k=1)` comes back empty. That query filters on tenant alone, with no per-question
    predicate, so an empty result is a property of the TABLE and not of the question: a null never
    means "hard question", it means the tenant is empty or misnamed. A partial null cannot happen;
    an all-null run is one whose index was not there, and every figure derived from it is void.

    Refusing is therefore the right outcome, not a harsh one. The runner raises on the FIRST null
    so the failure lands on question 1 rather than here, after a full run's embedding spend; this
    check is the backstop for a payload assembled some other way.

    A NaN is the more dangerous input and the reason this function exists at all: `abs(nan - x) >
    tol` is False, so a NaN made every comparison against that field vacuously true and an
    arbitrarily wrong summary passed. It also sorts as "not below the floor", so it could only ever
    understate demotion, which is the direction this artifact argues for.
    """
    qid = row.get("question_id", "<unknown>")
    if "best_dense_score" not in row:
        raise ValueError(f"row {qid!r} has no best_dense_score")
    value = row["best_dense_score"]
    if value is None:
        raise ValueError(
            f"row {qid!r} has a null best_dense_score, which means the tenant held no rows for "
            f"the dense query, not that the question was hard: the run measured no index and is "
            f"void. Check the table and tenant."
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"row {qid!r} has a non-numeric best_dense_score {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"row {qid!r} has a non-finite best_dense_score {value!r}")
    return float(value)


def _question_type(row: Mapping[str, Any]) -> str:
    if "question_type" not in row:
        raise ValueError(f"row {row.get('question_id', '<unknown>')!r} has no question_type")
    return str(row["question_type"])


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Derive the summary from the rows. The single definition, so the writer cannot drift."""
    if not rows:
        raise ValueError("dense floor artifact requires non-empty rows")

    scores = sorted(_score(r) for r in rows)
    by_type: dict[str, list[float]] = {}
    for row in rows:
        by_type.setdefault(_question_type(row), []).append(_score(row))

    per_cat: dict[str, Any] = {}
    expected = 0.0
    for cat in sorted(by_type):
        vals = sorted(by_type[cat])
        below = sum(1 for x in vals if x < FLOOR)
        rate = below / len(vals)
        if cat not in POPULATION:
            raise ValueError(f"unknown question_type {cat!r}: it carries no population weight")
        expected += rate * POPULATION[cat]
        per_cat[cat] = {
            "sampled": len(vals),
            "population": POPULATION[cat],
            "below_0_50": below,
            "rate_below_0_50": _rounded(rate),
            "min": _rounded(vals[0]),
            "median": _rounded(statistics.median(vals)),
            "max": _rounded(vals[-1]),
        }

    below_all = sum(1 for x in scores if x < FLOOR)
    unanswerable = sorted(by_type.get(UNANSWERABLE, []))
    return {
        "sample_below_0_50": below_all,
        "sample_n": len(scores),
        "sample_rate_below_0_50": _rounded(below_all / len(scores)),
        "population_weighted_lower_bound_below_0_50": _rounded(expected),
        "population_n": sum(POPULATION.values()),
        "population_rate_lower_bound_below_0_50": _rounded(expected / sum(POPULATION.values())),
        "min": _rounded(scores[0]),
        "median": _rounded(statistics.median(scores)),
        "max": _rounded(scores[-1]),
        # Derived, not hand-written. It was hand-written once, and a reader could have inverted it
        # to claim the floor catches EVERY unanswerable question with nothing to contradict them.
        "unanswerable_below_floor": {
            "question_type": UNANSWERABLE,
            "sampled": len(unanswerable),
            "below_0_50": sum(1 for x in unanswerable if x < FLOOR),
        },
        "by_question_type": per_cat,
    }


def _agrees(got: object, want: object, where: str) -> None:
    if isinstance(want, float):
        # `not (<= tol)` rather than `> tol`, so a NaN on either side fails closed instead of
        # comparing False and passing.
        if not (abs(float(got) - want) <= _TOL):  # type: ignore[arg-type]
            raise ValueError(f"{where} is {got!r}, but rows give {want!r}")
    elif got != want:
        raise ValueError(f"{where} is {got!r}, but rows give {want!r}")


def validate_dense_floor_artifact(payload: Mapping[str, object]) -> None:
    """Raise `ValueError` unless the summary is exactly what `rows` produces."""
    rows = payload.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("dense floor artifact requires non-empty rows")

    summary = payload.get("dense_floor_summary")
    if not isinstance(summary, Mapping):
        raise ValueError("dense floor artifact requires a dense_floor_summary block")

    expected = summarize(rows)

    unknown = set(summary) - set(expected) - _PROSE_KEYS
    if unknown:
        raise ValueError(
            f"dense_floor_summary carries unvalidated keys {sorted(unknown)}: derive them in "
            f"summarize() or add them to _PROSE_KEYS, do not hand-write a figure beside the body"
        )

    for key, want in expected.items():
        if key in {"by_question_type", "unanswerable_below_floor"}:
            continue
        if key not in summary:
            raise ValueError(f"dense_floor_summary is missing {key}")
        _agrees(summary[key], want, f"dense_floor_summary {key}")

    for block in ("unanswerable_below_floor", "by_question_type"):
        got_block = summary.get(block)
        if not isinstance(got_block, Mapping):
            raise ValueError(f"dense_floor_summary requires a {block} block")

    got_unans = summary["unanswerable_below_floor"]
    for key, want in expected["unanswerable_below_floor"].items():
        if key not in got_unans:
            raise ValueError(f"unanswerable_below_floor is missing {key}")
        _agrees(got_unans[key], want, f"unanswerable_below_floor.{key}")

    got_cats = summary["by_question_type"]
    if set(got_cats) != set(expected["by_question_type"]):
        raise ValueError(
            f"by_question_type covers {sorted(got_cats)}, rows give "
            f"{sorted(expected['by_question_type'])}"
        )
    for cat, want_cat in expected["by_question_type"].items():
        got_cat = got_cats[cat]
        if not isinstance(got_cat, Mapping):
            raise ValueError(f"by_question_type[{cat}] is not an object")
        for key, want in want_cat.items():
            if key not in got_cat:
                raise ValueError(f"by_question_type[{cat}] is missing {key}")
            _agrees(got_cat[key], want, f"by_question_type[{cat}].{key}")


def write_dense_floor_artifact(path: Path, payload: Mapping[str, object]) -> None:
    """Validate, then write. A payload that fails validation leaves no file behind.

    This is the wiring. `validate_dense_floor_artifact` on its own is a function a writer has to
    remember to call, and the previous version of this module documented it as a write-site guard
    while every caller lived in a test.
    """
    validate_dense_floor_artifact(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


__all__ = [
    "FLOOR",
    "POPULATION",
    "summarize",
    "validate_dense_floor_artifact",
    "write_dense_floor_artifact",
]
