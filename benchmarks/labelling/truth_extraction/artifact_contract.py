"""Validation for the truth-extraction artifacts, applied at the write site.

The census is the artifact every later recall number is read against. Its counts and its lists
are the same facts written twice — a summary a reader quotes and a body a reader recomputes from.
If they disagree, the artifact is not merely wrong, it is unfalsifiable: nothing in it says which
of the two is the typo. So the disagreement is refused at write time, when it costs nothing.

The same argument covers an arm result, plus one more that the census does not need: an arm result
must say what produced it. The first M1 run wrote a file reporting one proposal and 37 refusals,
and that file was indistinguishable from a careful arm. Every call had in fact failed on a
markdown fence, and the prompt has been rewritten twice since. A number whose prompt revision is
not recorded is attributable to nothing.

Pattern follows `benchmarks/artifact_contract.py`.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

_REQUIRED_PROVENANCE = ("peps_sha", "clone_date", "recall_commit", "generated_at", "invocation")

#: Every arm, whatever proposed it.
_REQUIRED_ARM_PROVENANCE = ("peps_sha", "recall_commit", "generated_at", "invocation",
                            "pack_digest")

#: A model arm only. Without these the score names no prompt, no model and no endpoint, and
#: rerunning it is guesswork.
_REQUIRED_MODEL_PROVENANCE = ("prompt_revision", "model_id", "model_revision", "engine_id",
                              "status_vocabulary")

#: Above this share of failed calls a result is an apparatus failure rather than a refusal rate.
#: Not zero, because some documents legitimately fail and that is worth reporting; not `all`,
#: which is what it was, and which let 29 of 30 failures publish a tier computed over the one
#: document that answered.
MAX_FAILED_CALL_FRACTION = 0.10

#: The pre-registration's decision rule, as a closed set. The runner computes the verdict rather
#: than a reader inferring one, so the rule is applied to the number instead of being reread
#: after seeing it.
ARM_VERDICTS = (
    "VACUOUS ARM",
    "UNDERPOWERED",
    # The tier is a property of the PAIR of arms: the underpowered clause reads "fewer than 10
    # proposals in EITHER arm". One artifact cannot answer it alone, and guessing from half the
    # inputs is how a rule gets quietly reinterpreted after seeing a number.
    "PENDING SIBLING ARM",
    # Above the tiers, not among them: its action is "run the four upper falsifier checks first",
    # which is a human step, and its conditions are a superset of high confidence's.
    "SUSPICIOUS",
    "LINT POINTER",
    "REVIEWING AID",
    "BATCH REVIEWABLE",
    # Reachable only after a human has run the four falsifier checks, never by the runner.
    "HIGH CONFIDENCE",
)


def validate_census(payload: Mapping[str, object]) -> None:
    """Raise `ValueError` unless `payload` is a self-consistent, attributable census."""
    provenance = payload.get("_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("census payload requires a _provenance block")
    for field in _REQUIRED_PROVENANCE:
        if not provenance.get(field):
            raise ValueError(f"census _provenance requires {field}")

    edges = payload.get("edges")
    if not isinstance(edges, Sequence) or isinstance(edges, (str, bytes)):
        raise ValueError("census edges must be an array")
    if payload.get("n_header_edges") != len(edges):
        raise ValueError(
            f"n_header_edges {payload.get('n_header_edges')!r} disagrees with "
            f"{len(edges)} entries in edges"
        )

    restatements = payload.get("restatements")
    if not isinstance(restatements, Mapping):
        raise ValueError("census restatements must be an object")
    if payload.get("n_restated_in_prose") != len(restatements):
        raise ValueError(
            f"n_restated_in_prose {payload.get('n_restated_in_prose')!r} disagrees with "
            f"{len(restatements)} entries in restatements"
        )

    # The recall ceiling is a proportion of the header edges. A value above 1.0 means the
    # restatement detector matched something that is not in the gold set at all.
    if len(restatements) > len(edges):
        raise ValueError(
            f"n_restated_in_prose ({len(restatements)}) cannot exceed n_header_edges "
            f"({len(edges)}) — the recall ceiling cannot exceed 100%"
        )


def validate_arm_result(payload: Mapping[str, object]) -> None:
    """Raise `ValueError` unless `payload` is a self-consistent, attributable arm result."""
    provenance = payload.get("_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("arm result requires a _provenance block")
    for field in _REQUIRED_ARM_PROVENANCE:
        if not provenance.get(field):
            raise ValueError(f"arm _provenance requires {field}")

    calls = payload.get("model_calls")
    if not isinstance(calls, int) or isinstance(calls, bool):
        raise ValueError("arm result requires an integer model_calls (0 for a rules arm)")
    if calls:
        for field in _REQUIRED_MODEL_PROVENANCE:
            if not provenance.get(field):
                raise ValueError(f"a model arm's _provenance requires {field}")

    items = payload.get("proposed_items")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise ValueError("arm result proposed_items must be an array")
    if payload.get("proposed") != len(items):
        raise ValueError(
            f"proposed {payload.get('proposed')!r} disagrees with {len(items)} proposed_items"
        )

    scored = payload.get("proposed_scored")
    excluded = payload.get("proposed_undecidable_excluded")
    if not isinstance(scored, int) or not isinstance(excluded, int):
        raise ValueError("proposed_scored and proposed_undecidable_excluded must be integers")
    if scored + excluded != len(items):
        raise ValueError(
            f"proposed_scored ({scored}) plus proposed_undecidable_excluded ({excluded}) must "
            f"account for all {len(items)} proposals"
        )

    tp, fp = payload.get("true_positive"), payload.get("false_positive")
    if not isinstance(tp, int) or not isinstance(fp, int):
        raise ValueError("true_positive and false_positive must be integers")
    if tp + fp != scored:
        raise ValueError(
            f"true_positive ({tp}) plus false_positive ({fp}) must equal proposed_scored ({scored})"
        )

    # `null`, not NaN: an arm that proposed nothing DECLINED to answer, and 0.0 would read as
    # "it was wrong". JSON has no NaN that a strict parser will accept, so the absence is encoded
    # as null and pinned in both directions — a null precision with a non-empty denominator is
    # just as wrong as a number with an empty one.
    precision = payload.get("precision")
    if scored == 0:
        if precision is not None:
            raise ValueError("precision must be null when nothing scored, not a number")
    else:
        if not isinstance(precision, (int, float)) or isinstance(precision, bool):
            raise ValueError(f"precision must be a number when {scored} proposals scored")
        if not math.isclose(float(precision), tp / scored, abs_tol=1e-9):
            raise ValueError(
                f"precision {precision!r} disagrees with true_positive/proposed_scored "
                f"({tp}/{scored} = {tp / scored})"
            )

    failures = payload.get("batch_failures")
    if not isinstance(failures, Mapping):
        raise ValueError("arm result batch_failures must be an object")
    if payload.get("documents_refused_whole") != len(failures):
        raise ValueError(
            f"documents_refused_whole {payload.get('documents_refused_whole')!r} disagrees with "
            f"{len(failures)} entries in batch_failures"
        )

    # The defect this whole validator exists for. An arm whose calls failed produces a file that
    # is arithmetically perfect and reads as a selective arm: few proposals, many refusals.
    #
    # ⚠️ `== calls` was one comparison too tight. It caught only the total wipeout, so 29 of 30
    # failures published, and 15 of 30 published a TIER computed over the half that answered.
    # A fraction is what makes it a gate rather than a tripwire.
    # The RULES arm's equivalent, which had no gate at all: `if calls` skips every check above
    # for an arm with `model_calls == 0`, so an R1 run that located none of the 38 sentences in
    # their source bodies published cleanly as a vacuous arm. That failure mode is not
    # hypothetical — `peps_header.paragraphs` exists because it hit 30 of 38.
    unread = payload.get("candidates_not_read")
    if not isinstance(unread, int) or isinstance(unread, bool) or unread < 0:
        raise ValueError("arm result requires a non-negative integer candidates_not_read")
    reasons = payload.get("refusal_reasons")
    if not isinstance(reasons, Mapping):
        raise ValueError("arm result refusal_reasons must be an object")
    scored_universe = unread + len(items) + sum(int(n) for n in reasons.values())
    if scored_universe and unread > MAX_FAILED_CALL_FRACTION * scored_universe:
        raise ValueError(
            f"{unread} of {scored_universe} candidates were never read by the arm, above the "
            f"{MAX_FAILED_CALL_FRACTION:.0%} ceiling: a candidate the arm could not look at is "
            f"an apparatus failure, and counting it as a refusal flatters precision"
        )

    if calls and len(failures) > MAX_FAILED_CALL_FRACTION * calls:
        raise ValueError(
            f"{len(failures)} of {calls} model calls failed at the batch rung, above the "
            f"{MAX_FAILED_CALL_FRACTION:.0%} ceiling: this is an apparatus failure, not a "
            f"refusal rate. Reasons: {sorted(set(map(str, failures.values())))}"
        )

    if payload.get("verdict") not in ARM_VERDICTS:
        raise ValueError(
            f"verdict {payload.get('verdict')!r} is not one of the pre-registered {ARM_VERDICTS}"
        )

    # The bounds the tier is KEYED ON. Unchecked, a payload could carry an inverted interval, or
    # none at all, and still be stamped with a tier: the two facts that decide the verdict were
    # the two nothing validated.
    bounds = (payload.get("precision_wilson_lower"), payload.get("precision_wilson_upper"))
    if scored == 0:
        if any(bound is not None for bound in bounds):
            raise ValueError("precision_wilson bounds must be null when nothing scored")
    else:
        if not all(isinstance(bound, (int, float)) and not isinstance(bound, bool)
                   for bound in bounds):
            raise ValueError("precision_wilson_lower and _upper must be numbers when scored > 0")
        lower, upper = float(bounds[0]), float(bounds[1])  # type: ignore[arg-type]
        if not lower <= float(precision) <= upper:  # type: ignore[arg-type]
            raise ValueError(
                f"precision {precision!r} is outside its own interval [{lower}, {upper}]"
            )


def validate_fixtures_result(payload: Mapping[str, object]) -> None:
    """Raise `ValueError` unless `payload` is a self-consistent P7 result.

    Separate from `validate_arm_result` because it is a different measurement: no adjudicated
    pack, no precision, no decision-rule tier. Sharing one validator would have meant loosening
    both, which is how a validator stops refusing anything.
    """
    provenance = payload.get("_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("fixtures result requires a _provenance block")
    for field in _REQUIRED_ARM_PROVENANCE:
        if not provenance.get(field):
            raise ValueError(f"fixtures _provenance requires {field}")
    # Always a model arm: the fixtures exist to test a model against rules that already failed.
    for field in ("prompt_revision", "model_id", "model_revision", "engine_id"):
        if not provenance.get(field):
            raise ValueError(f"a fixtures run's _provenance requires {field}")

    total = payload.get("fixtures")
    refused, proposed = payload.get("refused"), payload.get("proposed")
    if not isinstance(total, int) or total <= 0:
        raise ValueError("fixtures result must count at least one fixture")
    if not isinstance(refused, int) or not isinstance(proposed, int):
        raise ValueError("refused and proposed must be integers")
    if refused + proposed != total:
        raise ValueError(
            f"refused ({refused}) plus proposed ({proposed}) must equal fixtures ({total})"
        )

    per_fixture = payload.get("per_fixture")
    if not isinstance(per_fixture, Mapping) or len(per_fixture) != total:
        raise ValueError(
            f"per_fixture must carry one entry per fixture ({total}), so a reader can see WHICH "
            f"refusal came from the language and which from a resolution rung"
        )

    if payload.get("p7_holds") != (refused == total):
        raise ValueError(
            f"p7_holds {payload.get('p7_holds')!r} disagrees with {refused} of {total} refused"
        )

    failures = payload.get("batch_failures")
    if not isinstance(failures, Mapping):
        raise ValueError("fixtures result batch_failures must be an object")
    # ANY failure, not all of them. P7 is the load-bearing public prediction and it has four
    # data points: a fixture the model never read cannot refuse, so it reads as a refusal and
    # inflates the score. `== total` let 3 of 4 failures publish `p7_holds: true`.
    if failures:
        raise ValueError(
            f"{len(failures)} of {total} fixtures failed at the batch rung: a fixture that was "
            f"never read cannot refuse, so an apparatus failure reads here as a perfect P7 "
            f"score. That is the one way this check can lie, and P7 decides shipping"
        )


__all__ = [
    "ARM_VERDICTS",
    "validate_arm_result",
    "validate_census",
    "validate_fixtures_result",
]
