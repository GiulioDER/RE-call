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

#: Tolerance for a ROW-level identity, which is not the same thing as `_TOL`. Rows are stored at
#: full precision, so `max_returned <= corpus_top_1` needs float-noise slack and nothing more.
#: Reusing `_TOL` here let a genuine 4e-5 violation pass the contract while the runner, which
#: compares exactly, would have refused the same run.
_ROW_TOL = 1e-9

#: Keys a human may add that are prose rather than measurement. Everything else in the summary must
#: be derivable from `rows`, so the next hand-added FIGURE cannot slip past unvalidated the way
#: `floor_catches_no_unanswerable` did.
#: `sampling` was removed once the runner began emitting it under `_provenance`: an
#: exemption that buys nothing still permits an unvalidated block of numbers beside the
#: derived ones, which is the defect this set exists to bound.
_PROSE_KEYS = frozenset({"_derived_from"})

#: Keys the RUNNER must declare because they cannot be derived from `rows`: the scored quantity is
#: the max dense cosine over the returned hits, so it depends on how many hits came back and under
#: which arm, and neither fact is recoverable from the scores themselves. They are validated for
#: presence and shape rather than exempted, because the whole point of recording them is that a
#: reader can tell two runs apart, and a key nobody checks is a key that can go missing.
_DECLARED_KEYS = frozenset({"scored_over_top_k", "scored_arm"})


def _rounded(value: float) -> float:
    return round(value, 4)


#: The field the floor is actually decided on: the best dense cosine among the hits the retrieval
#: RETURNED. The floor is applied per returned hit, so a question loses all its evidence exactly
#: when this is below it. `best_dense_score`, the corpus-wide top-1, is kept in the rows so the
#: old instrument can be compared against this one, but it is NOT what the summary reads.
SCORE_FIELD = "max_returned_dense_score"

#: The previous instrument. Retained for comparison only.
CORPUS_TOP1_FIELD = "best_dense_score"


def _numeric(row: Mapping[str, Any], field: str) -> float:
    """Read one score field with every refusal in one place.

    ⚠️ A null is NOT proof of an empty tenant, and an earlier version of this docstring said it
    was. `query_dense` applies the tenant as a POST-filter over an HNSW index built across the
    whole table, so on a multi-tenant table the walk can come back empty for one query and not
    another: this repo measured 10 of 26 queries empty on one tenant while the rest returned hits.
    A null therefore means "no dense hit was found at the probe depth used", which is a statement
    about the probe and the index, not about the question. Either way the run is void, because a
    missing score cannot be scored.

    A NaN is the more dangerous input and the reason this function exists at all: `abs(nan - x) >
    tol` is False, so a NaN made every comparison against that field vacuously true and an
    arbitrarily wrong summary passed. It also sorts as "not below the floor", so it could only ever
    understate demotion, which is the direction this artifact argues for.
    """
    qid = row.get("question_id", "<unknown>")
    if field not in row:
        raise ValueError(f"row {qid!r} has no {field}")
    value = row[field]
    if value is None:
        raise ValueError(
            f"row {qid!r} has a null {field}: no dense hit was found at the probe depth used. "
            f"That is a statement about the probe and the index, not about the question, and "
            f"nothing derived from this run can be scored."
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"row {qid!r} has a non-numeric {field} {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"row {qid!r} has a non-finite {field} {value!r}")
    return float(value)


def _score(row: Mapping[str, Any]) -> float:
    """The score the floor decides on. See `SCORE_FIELD`."""
    value = _numeric(row, SCORE_FIELD)
    # The corpus top-1 must dominate every returned hit; both are dense cosines of one query over
    # one table. A violation means they are not on one basis, and the runner refuses it at source.
    corpus = _numeric(row, CORPUS_TOP1_FIELD)
    if value > corpus + _ROW_TOL:
        raise ValueError(
            f"row {row.get('question_id', '<unknown>')!r}: {SCORE_FIELD} {value!r} exceeds "
            f"{CORPUS_TOP1_FIELD} {corpus!r}. A returned hit cannot outscore the corpus top-1 "
            f"unless the two were measured against different query vectors or at different "
            f"hnsw.ef_search."
        )
    return value


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

    # ⛔ Every stratum, or nothing. The numerator below sums over the strata PRESENT in `rows`
    # while the denominator is always the full population, so a run missing a stratum publishes a
    # silently deflated rate that passes every other check in this module. Dropping two strata
    # from the committed artifact took its headline from 0.056 to 0.022 with 19 of 19 tests still
    # green, and `--limit-questions 10` (which takes the first N of the file) produces exactly
    # that shape: one stratum, a 0.0 headline, over a population of 500.
    missing = sorted(set(POPULATION) - set(by_type))
    if missing:
        raise ValueError(
            f"sample covers {len(by_type)} of {len(POPULATION)} strata; missing {missing}. "
            f"A population-weighted rate whose numerator skips a stratum and whose denominator "
            f"does not is not a rate over anything."
        )

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
        # ⚠️ ESTIMATE, not a bound, and the previous names said bound. The per-question measure
        # is exact, but nine of the ten strata are 10-question samples of populations of 20 to 175,
        # so the population figure carries sampling error in BOTH directions. Only `high_level` is
        # a census (10 of 10). Calling a stratified estimate a lower bound invites the reader to
        # treat 5.6% as a floor when the interval straddles well below it.
        "population_weighted_estimate_below_0_50": _rounded(expected),
        "population_n": sum(POPULATION.values()),
        "population_rate_estimate_below_0_50": _rounded(expected / sum(POPULATION.values())),
        # The one stratum that is a census, so the reader has a figure with no sampling error.
        "census_strata": sorted(c for c in POPULATION if per_cat.get(c, {}).get("sampled") == POPULATION[c]),
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

    unknown = set(summary) - set(expected) - _PROSE_KEYS - _DECLARED_KEYS
    if unknown:
        raise ValueError(
            f"dense_floor_summary carries unvalidated keys {sorted(unknown)}: derive them in "
            f"summarize() or add them to _PROSE_KEYS, do not hand-write a figure beside the body"
        )

    missing_declared = sorted(_DECLARED_KEYS - set(summary))
    if missing_declared:
        raise ValueError(
            f"dense_floor_summary is missing {missing_declared}: the scored quantity depends on "
            f"the returned-hit depth and the retrieval arm, so a summary that does not name them "
            f"cannot be compared with any other run"
        )
    depth = summary["scored_over_top_k"]
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
        raise ValueError(f"scored_over_top_k must be a positive int, got {depth!r}")
    arm = summary["scored_arm"]
    if not isinstance(arm, Mapping) or "sparse_backend" not in arm or "reranker" not in arm:
        raise ValueError(f"scored_arm must name sparse_backend and reranker, got {arm!r}")

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
    require_provenance(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


#: Fields `tests/test_results_artifact_provenance.py` requires of every committed artifact. They
#: are checked HERE, at the write, because the previous version emitted no `_provenance` at all:
#: the committed file's block was assembled by hand, credits a revision that predates this module,
#: and a re-run to the same path therefore produced a file that failed the repo's own suite. An
#: artifact that cannot be regenerated is not evidence, it is a screenshot.
REQUIRED_PROVENANCE = ("generation", "status", "note", "backs", "generated_at", "stack")

#: Mirrors `tests/test_results_artifact_provenance.py` and `..._model_stack.py`. Duplicated rather
#: than imported because `benchmarks` must not import from `tests`, and a value the suite rejects
#: must be refused at the write rather than discovered when CI runs on the commit.
GENERATIONS = frozenset({"pre-#81/#84", "post-#81/#84"})
STATUSES = frozenset({"current", "superseded", "not re-measured"})


def require_provenance(payload: Mapping[str, object]) -> None:
    """Raise unless the payload carries the provenance the repository's artifact suite demands."""
    prov = payload.get("_provenance")
    if not isinstance(prov, Mapping):
        raise ValueError(
            "dense floor artifact requires a _provenance block; without it the file fails "
            "tests/test_results_artifact_provenance.py the moment it is committed"
        )
    missing = [key for key in REQUIRED_PROVENANCE if not prov.get(key)]
    if missing:
        raise ValueError(f"_provenance is missing or empty for {missing}")
    # ⚠️ Truthiness is not enough, and checking only that was this guard's own defect. The repo's
    # artifact suite requires MEMBERSHIP for two of these fields, so `generation="post-81/84"` or
    # `status="live"` passed here and failed CI on commit, which is the failure this exists to
    # prevent rather than relocate. `--calibrate-generation` is a free-form string, so one typo
    # was enough.
    if prov["generation"] not in GENERATIONS:
        raise ValueError(f"_provenance generation {prov['generation']!r} not in {sorted(GENERATIONS)}")
    if prov["status"] not in STATUSES:
        raise ValueError(f"_provenance status {prov['status']!r} not in {sorted(STATUSES)}")
    if not str(prov["note"]).strip():
        raise ValueError("_provenance note is blank")
    if prov.get("status") == "superseded" and not prov.get("superseded_by"):
        raise ValueError("_provenance says superseded but names no successor")
    if prov.get("status") != "superseded" and prov.get("superseded_by"):
        raise ValueError(
            f"_provenance status {prov.get('status')!r} names a successor "
            f"{prov.get('superseded_by')!r}: a live artifact with a successor is a contradiction"
        )


__all__ = [
    "CORPUS_TOP1_FIELD",
    "FLOOR",
    "POPULATION",
    "REQUIRED_PROVENANCE",
    "SCORE_FIELD",
    "require_provenance",
    "summarize",
    "validate_dense_floor_artifact",
    "write_dense_floor_artifact",
]
