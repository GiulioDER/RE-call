"""Carry-forward and import bounds for calibration v2: error rates, corpus deltas, bundle checks.

Split out of `recall.calibration_v2`, which re-exports every name here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict


from recall.calibration import Calibration, from_samples, separability

from recall.calibration_v2_artifact import (
    CalibrationBindingError,
    _REDERIVE_TOLERANCE,
    _RESCORE_TOLERANCE,
    _CUMULATIVE_DELTA_TOLERANCE,
    CalibrationArtifactV2,
)

#: Largest corpus delta a threshold may be carried across without being refitted, as a fraction of
#: the union of parent and child source URIs.
#:
#: 0.25 is a CEILING on the mechanism, not a claim that a threshold survives a 25% delta. Nothing
#: is carried on the strength of this number alone: the stored labelled query set is re-scored
#: against the child generation and must still certify, so the bound only decides how far the
#: mechanism will bother trying before demanding a fresh fit. It exists because re-scoring says
#: nothing about the queries nobody labelled, and a query set fitted on a corpus half of which has
#: since been replaced is measuring a different corpus while wearing the same digest.
#:
#: ⚠️ Deliberately NOT tuned. The only delta this has been measured at is recorded in
#: `docs/preregistrations/2026-08-20-calibration-carry-forward.md`; treat anything above it as
#: untested, and lower the bound per tenant rather than reading this default as evidence.
DEFAULT_MAX_CORPUS_DELTA = 0.25

#: Largest per-class error the INHERITED threshold may make on the child generation's fresh
#: scores, as a fraction of that class.
#:
#: ⛔ This check is not redundant with certification, and the difference is the entire reason
#: carry-forward needs a rule `calibrate` does not. Separability is threshold-free: it asks
#: whether the two classes are still ORDERED. Adding documents that lift every unanswerable score
#: by the same amount leaves the ordering perfect — AUC 1.00, certified — while sliding the whole
#: unanswerable class above a threshold that is no longer allowed to move. A refit cannot be
#: fooled this way because it puts the cut back between the classes; carry-forward holds the cut
#: still, so it has to check the cut.
#:
#: Found by `test_carry_forward_rejects_when_the_threshold_stops_separating`, which reused 20 of
#: 22 sources and moved only the 2 new ones: separability stayed 1.00 and the inherited threshold
#: admitted 100% of the unanswerable queries.
#:
#: 0.10 is a ceiling, not a measured safe distance. The reference point is the 2026-08-17 fit on
#: this corpus, which measured leave-one-out false-confirm 3.6% and false-abstain 4.5%, and the
#: same session measured the answerable and unanswerable distributions OVERLAPPING in 4 of 4
#: corpora, so a bar at zero is not reachable and would only mean nothing ever carries.
DEFAULT_MAX_CARRY_FORWARD_ERROR = 0.10


class ThresholdErrors(TypedDict):
    """`threshold_error_rates`' result: two rates and the two counts behind them."""

    false_abstain_rate: float
    false_confirm_rate: float
    false_abstain_count: int
    false_confirm_count: int


def threshold_error_rates(
    answerable: Sequence[float], unanswerable: Sequence[float], threshold: float
) -> ThresholdErrors:
    """Per-class error of a FIXED threshold, each named by its own denominator.

    Returns each class's rate (`false_abstain_rate`, `false_confirm_rate`, fractions of their
    own class, 0.0 for an empty class) and the integer counts behind them (`*_count`).

    `>=` is a confirm, matching `recall.trust`, which promotes a hit whose cosine reaches the
    threshold. A `>` here would disagree with serving on exactly the boundary cases the threshold
    was placed to decide.
    """
    false_abstain = sum(1 for score in answerable if score < threshold)
    false_confirm = sum(1 for score in unanswerable if score >= threshold)
    return {
        "false_abstain_rate": (false_abstain / len(answerable)) if answerable else 0.0,
        "false_confirm_rate": (false_confirm / len(unanswerable)) if unanswerable else 0.0,
        "false_abstain_count": false_abstain,
        "false_confirm_count": false_confirm,
    }


def within_error_bound(errors: ThresholdErrors, max_error: float) -> bool:
    """True when a fixed threshold decides BOTH classes within `max_error`.

    The one statement of the carry-forward error rule: `judge_inherited_threshold`, the import
    re-derivation and drift monitoring all apply it, and three hand-written copies could drift.
    """
    return errors["false_abstain_rate"] <= max_error and errors["false_confirm_rate"] <= max_error


@dataclass(frozen=True)
class InheritedThresholdVerdict:
    """How an inherited threshold decides fresh scores: see `judge_inherited_threshold`."""

    #: The inherited threshold and scale beside the separability of the FRESH scores.
    runtime: Calibration
    errors: ThresholdErrors
    certified: bool
    reason: str
    #: What a fresh fit on these scores would choose. Diagnostic only: it reaches nothing.
    refit_threshold: float


def judge_inherited_threshold(
    parent: CalibrationArtifactV2,
    answerable: Sequence[float],
    unanswerable: Sequence[float],
    *,
    max_error: float = DEFAULT_MAX_CARRY_FORWARD_ERROR,
) -> InheritedThresholdVerdict:
    """Judge `parent`'s threshold, held FIXED, on scores measured against a changed corpus.

    The one rule every carry-forward applies, whichever store holds the corpus: the inherited
    threshold must clear the same certification bar a new fit would on the fresh scores, AND decide
    each class within `max_error`. The second condition is the one that catches a shifted class;
    see `DEFAULT_MAX_CARRY_FORWARD_ERROR` for why certification alone would pass a threshold that
    has stopped deciding anything.
    """
    runtime = Calibration(
        embedder=parent.runtime.embedder,
        threshold=parent.threshold,
        scale=parent.scale,
        separability=separability(list(answerable), list(unanswerable)),
        n_answerable=len(answerable),
        n_unanswerable=len(unanswerable),
    )
    if runtime.separability is None or runtime.separability_ci is None:
        raise CalibrationBindingError("both labelled classes are required for carry-forward")
    errors = threshold_error_rates(answerable, unanswerable, parent.threshold)
    within_error = within_error_bound(errors, max_error)
    certified = runtime.certified is True and within_error
    reason = runtime.certification_reason
    if runtime.certified is True and not within_error:
        reason = (
            f"the inherited threshold {parent.threshold:.4f} no longer decides this corpus: "
            f"false abstain {errors['false_abstain_rate']:.1%} of {len(answerable)} "
            f"answerable, false confirm {errors['false_confirm_rate']:.1%} of "
            f"{len(unanswerable)} unanswerable, against a bound of {max_error:.1%}. "
            f"Separability is {runtime.separability:.4f}, so the classes are still ordered "
            f"and only the boundary has moved; recalibrate to place it again."
        )
    refit = from_samples(parent.runtime.embedder, list(answerable), list(unanswerable))
    return InheritedThresholdVerdict(runtime, errors, certified, reason, refit.threshold)


def _cumulative_corpus_delta(carry_forward: Mapping[str, Any] | None) -> float:
    """Total drift a calibration lineage has accumulated across carry-forward steps.

    0.0 for a fitted artifact: its threshold was measured on its own generation, so the chain
    starts fresh there. For a carried artifact the recorded `cumulative_corpus_delta` is read;
    an artifact written before that key existed reads as its own recorded STEP delta, which is
    the conservative floor of what its chain actually drifted (any older steps are unknowable
    once generation gc has pruned their manifests, and assuming 0.0 for them would let a long
    pre-existing chain keep drifting unbounded).
    """
    if carry_forward is None:
        return 0.0
    value = carry_forward.get("cumulative_corpus_delta")
    if value is None:
        value = carry_forward.get("corpus_delta", 0.0)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationBindingError(
            f"carry-forward cumulative corpus delta is not a number: {exc}"
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise CalibrationBindingError(
            "carry-forward cumulative corpus delta must be a finite non-negative number"
        )
    return result


def _exceeds_cumulative_bound(cumulative: float, max_corpus_delta: float) -> bool:
    """True when a chain's summed drift is past the bound, not merely a few ulps onto it."""
    return cumulative > max_corpus_delta + _CUMULATIVE_DELTA_TOLERANCE


def _carried_cumulative_delta(
    parent: CalibrationArtifactV2, delta: Mapping[str, Any], max_corpus_delta: float
) -> float:
    """Refuse a step, or a chain, that drifted past `max_corpus_delta`; return the chain total.

    The step is refused before any embedding work. The chain total is the parent's recorded
    cumulative delta plus this step's (see `_cumulative_corpus_delta`).
    """
    if delta["corpus_delta"] > max_corpus_delta:
        raise CalibrationBindingError(
            f"corpus delta {delta['corpus_delta']:.3f} exceeds the carry-forward bound "
            f"{max_corpus_delta:.3f} ({delta['sources_added']} added, "
            f"{delta['sources_removed']} removed, {delta['sources_modified']} modified over "
            f"{delta['sources_union']} sources); recalibrate against a labelled query set"
        )
    cumulative = _cumulative_corpus_delta(parent.carry_forward) + float(delta["corpus_delta"])
    if _exceeds_cumulative_bound(cumulative, max_corpus_delta):
        raise CalibrationBindingError(
            f"cumulative corpus delta {cumulative:.3f} across this calibration's "
            f"carry-forward chain exceeds the bound {max_corpus_delta:.3f}, although this "
            f"step alone ({delta['corpus_delta']:.3f}) is within it; the corpus has drifted "
            f"too far from the one the threshold was fitted on, recalibrate against a "
            f"labelled query set"
        )
    return cumulative


#: The checksummed fields a stored row reads back as `float`, and as `int`. The checksum covers
#: their JSON SPELLING, so a bundle that spells one differently (`1` for 1.0, `20.0` for 20)
#: verifies on import and then fails `verify_checksum` on every later read, once the
#: DOUBLE PRECISION or INTEGER column has handed back the other spelling.
_BUNDLE_FLOAT_FIELDS = ("threshold", "scale", "separability")
_BUNDLE_INT_FIELDS = ("n_answerable", "n_unanswerable")
_BUNDLE_MAPPING_FIELDS = ("embedder_identity", "scores")


def _require_bundle_field_types(artifact_raw: Mapping[str, Any]) -> None:
    """Refuse a bundle field whose type would not survive storage, or cannot be read at all.

    Only fields that are present are checked; a missing field is refused by name later.
    """
    for key in _BUNDLE_FLOAT_FIELDS:
        if key in artifact_raw and type(artifact_raw[key]) is not float:
            raise CalibrationBindingError(
                f"bundle is malformed: {key} must be a JSON number with a fraction part "
                f"(for example 1.0), got {artifact_raw[key]!r}"
            )
    interval = artifact_raw.get("separability_ci")
    if "separability_ci" in artifact_raw and (
        not isinstance(interval, list)
        or len(interval) != 2
        or any(type(bound) is not float for bound in interval)
    ):
        raise CalibrationBindingError(
            "bundle is malformed: separability_ci must be a pair of bounds, each a JSON number "
            "with a fraction part"
        )
    for key in _BUNDLE_INT_FIELDS:
        if key in artifact_raw and type(artifact_raw[key]) is not int:
            raise CalibrationBindingError(
                f"bundle is malformed: {key} must be a JSON integer, got {artifact_raw[key]!r}"
            )
    if "certified" in artifact_raw and type(artifact_raw["certified"]) is not bool:
        raise CalibrationBindingError("bundle is malformed: certified must be true or false")
    for key in _BUNDLE_MAPPING_FIELDS:
        if key in artifact_raw and not isinstance(artifact_raw[key], Mapping):
            raise CalibrationBindingError(f"bundle is malformed: {key} must be an object")
    carry = artifact_raw.get("carry_forward")
    if carry is not None and not isinstance(carry, Mapping):
        raise CalibrationBindingError("bundle is malformed: carry_forward must be an object")


def _bundle_scores(artifact: CalibrationArtifactV2) -> tuple[list[float], list[float]]:
    """An artifact's recorded answerable and unanswerable scores, as floats, or a named refusal."""
    try:
        answerable = [float(v) for v in artifact.scores.get("answerable", ())]
        unanswerable = [float(v) for v in artifact.scores.get("unanswerable", ())]
    except (TypeError, ValueError) as exc:
        raise CalibrationBindingError(f"bundle scores are malformed: {exc}") from exc
    return answerable, unanswerable


def _import_error_bound(carry_forward: Mapping[str, Any]) -> float:
    """The per-class error bound a CARRIED bundle is judged at on import.

    The bundle's recorded `max_carry_forward_error`, never looser than the default: the bound is
    read from the bundle, so a bundle that recorded 1.0 would certify ANY inherited threshold. A
    carry made locally with a looser `max_error` still serves where it was made; it is only
    refused as an import.
    """
    try:
        recorded = float(
            carry_forward.get("max_carry_forward_error", DEFAULT_MAX_CARRY_FORWARD_ERROR)
        )
    except (TypeError, ValueError) as exc:
        raise CalibrationBindingError(
            f"carry-forward max_carry_forward_error is not a number: {exc}"
        ) from exc
    if not math.isfinite(recorded) or not 0.0 <= recorded <= 1.0:
        raise CalibrationBindingError(
            "carry-forward max_carry_forward_error must be a fraction in [0, 1]"
        )
    return min(recorded, DEFAULT_MAX_CARRY_FORWARD_ERROR)


def _imported_verdict(
    artifact: CalibrationArtifactV2, answerable: Sequence[float], unanswerable: Sequence[float]
) -> tuple[bool, str]:
    """Whether the artifact's threshold certifies on these scores, and why not when it does not.

    The rule `calibrate` and `carry_forward` apply: the separability certification bar, and for a
    CARRIED artifact also both classes decided within `_import_error_bound`. Import applies it
    twice, to the recorded scores (is the bundle consistent?) and to the fresh ones (does this
    generation, on the operator's labels, support it?).
    """
    measured = separability(list(answerable), list(unanswerable))
    if measured is None:
        raise CalibrationBindingError(
            "the scores lack a labelled class, so certification cannot be re-derived"
        )
    check = Calibration(
        embedder=artifact.embedder_model,
        threshold=artifact.threshold,
        scale=artifact.scale,
        separability=measured,
        n_answerable=len(answerable),
        n_unanswerable=len(unanswerable),
    )
    certified, reason = check.certified is True, check.certification_reason
    if artifact.carry_forward is not None:
        bound = _import_error_bound(artifact.carry_forward)
        errors = threshold_error_rates(answerable, unanswerable, artifact.threshold)
        if certified and not within_error_bound(errors, bound):
            certified = False
            reason = (
                f"the carried threshold {artifact.threshold:.4f} misclassifies "
                f"{errors['false_abstain_rate']:.1%} of the answerable and "
                f"{errors['false_confirm_rate']:.1%} of the unanswerable queries, beyond the "
                f"{bound:.0%} a carried import is judged at (never looser than the default "
                f"{DEFAULT_MAX_CARRY_FORWARD_ERROR:.0%})"
            )
    return certified, reason


def _require_scores_reproduce(
    artifact: CalibrationArtifactV2, answerable: Sequence[float], unanswerable: Sequence[float]
) -> None:
    """Refuse a bundle whose recorded scores the named generation does not reproduce.

    The scores are measured again, here, on this database's copy of the generation, from the
    operator's own labelled set, and must agree query by query within `_RESCORE_TOLERANCE`. Both
    lists are in the canonical query set's order, which is the order `calibrate` scored and
    stored them in. Agreeing within the tolerance is not agreeing exactly, so the certification
    is then judged on the fresh scores too (`_require_fresh_certification`).
    """
    recorded_answerable, recorded_unanswerable = _bundle_scores(artifact)
    for name, stored, fresh in (
        ("answerable", recorded_answerable, answerable),
        ("unanswerable", recorded_unanswerable, unanswerable),
    ):
        if len(stored) != len(fresh):
            raise CalibrationBindingError(
                f"bundle records {len(stored)} {name} scores, but its query set re-scores to "
                f"{len(fresh)} on this generation"
            )
        # Before the comparison, because NaN fails it open: `max()` either skips a NaN or
        # returns it, and `nan > tolerance` is False, so the query would go unchecked.
        if not all(math.isfinite(value) for value in fresh):
            raise CalibrationBindingError(
                f"this generation re-scored a {name} query to a non-finite cosine, so the "
                f"bundle cannot be checked against it; recalibrate here instead"
            )
        worst = max((abs(a - b) for a, b in zip(stored, fresh, strict=True)), default=0.0)
        if worst > _RESCORE_TOLERANCE:
            raise CalibrationBindingError(
                f"bundle {name} scores are not what this generation measures: the largest "
                f"difference is {worst:.4f}, beyond {_RESCORE_TOLERANCE}. The bundle was not "
                f"fitted on this corpus, or it was edited; recalibrate here instead"
            )


def _require_fresh_certification(
    artifact: CalibrationArtifactV2, answerable: Sequence[float], unanswerable: Sequence[float]
) -> None:
    """Refuse a certified bundle whose certification this generation's own scores do not earn.

    The recorded scores may each sit up to `_RESCORE_TOLERANCE` from the fresh ones, and that is
    enough to move every near-threshold score across the line: overlapping fresh classes whose
    interval misses the bar can be recorded perfectly separated. So the verdict is re-derived on
    the FRESH scores, by `_imported_verdict`, and a bundle that certifies only on its own copy is
    refused. An uncertified bundle is stored as `rejected` whatever it re-scores to, so it is
    not judged again.
    """
    if not artifact.certified:
        return
    certified, reason = _imported_verdict(artifact, answerable, unanswerable)
    if not certified:
        raise CalibrationBindingError(
            f"bundle claims certification its scores on this generation do not earn: {reason}. "
            f"Recalibrate here instead"
        )


def _rederived_matches(stored: float, derived: float) -> bool:
    return math.isclose(
        stored, derived, rel_tol=_REDERIVE_TOLERANCE, abs_tol=_REDERIVE_TOLERANCE
    )


def _require_scores_support_certification(artifact: CalibrationArtifactV2) -> None:
    """Refuse an artifact whose recorded statistics its own scores do not reproduce.

    `import_bundle`'s checksum only proves the bundle is INTERNALLY consistent: it is unkeyed,
    so anyone can write certified=true beside arbitrary scores and recompute the digest over the
    lie. The separability, its interval, the sample counts, and the certification verdict are
    all pure functions of the scores the bundle itself carries, so they are re-derived here with
    the same rules `calibrate` and `carry_forward` apply, and any disagreement beyond float
    tolerance is refused naming the field that disagreed. That catches a bundle whose claims
    contradict its own scores, a hand edit of one field or a stale export.

    `Calibration.certified` never reads the threshold, so for a FITTED artifact the threshold
    and scale are re-derived too, from `from_samples` over the bundle's own scores, which is
    exact for a legitimate round trip because `calibrate` sets them from `from_samples` over
    those same scores. A CARRIED artifact's threshold is inherited rather than fitted, so
    re-deriving it would refuse every legitimate carried bundle; it is checked against its
    recorded error-rate bound over the bundle's scores, capped at the default
    (`_import_error_bound`).

    ⛔ **No bundle is AUTHENTICATED by this check, fitted or carried.** Every input it reads is
    in the bundle, and a forger chooses the scores and the labels as freely as the threshold:
    `from_samples` places the threshold between the classes, so scores clustered either side of
    any value re-derive exactly that value, certified. Consistency is all this proves.
    `import_bundle` authenticates separately: it requires the operator's own labelled set, scores
    it on the named generation, and requires the recorded scores to reproduce
    (`_require_scores_reproduce`) and the certification to hold on the fresh scores
    (`_require_fresh_certification`).
    """
    answerable, unanswerable = _bundle_scores(artifact)
    if len(answerable) != artifact.n_answerable or len(unanswerable) != artifact.n_unanswerable:
        raise CalibrationBindingError(
            f"bundle sample counts disagree with its own scores: recorded "
            f"{artifact.n_answerable}/{artifact.n_unanswerable}, scores hold "
            f"{len(answerable)}/{len(unanswerable)}"
        )
    recomputed = separability(answerable, unanswerable)
    if recomputed is None:
        raise CalibrationBindingError(
            "bundle scores lack a labelled class, so its separability cannot be re-derived"
        )
    if not _rederived_matches(recomputed, artifact.separability):
        raise CalibrationBindingError(
            f"bundle separability {artifact.separability} disagrees with the value re-derived "
            f"from its own scores ({recomputed})"
        )
    check = Calibration(
        embedder=artifact.embedder_model,
        threshold=artifact.threshold,
        scale=artifact.scale,
        separability=recomputed,
        n_answerable=len(answerable),
        n_unanswerable=len(unanswerable),
    )
    interval = check.separability_ci
    if interval is None:
        raise CalibrationBindingError("bundle separability interval cannot be re-derived")
    for name, stored, derived in (
        ("ci_low", artifact.separability_ci[0], interval[0]),
        ("ci_high", artifact.separability_ci[1], interval[1]),
    ):
        if not _rederived_matches(stored, derived):
            raise CalibrationBindingError(
                f"bundle separability {name} {stored} disagrees with the value re-derived "
                f"from its own scores ({derived})"
            )
    if artifact.carry_forward is None:
        # The threshold is the only field of a fitted artifact that reaches serving, and it is
        # the one `certified` never looks at, so bind it to the scores directly. `calibrate`
        # sets threshold and scale from `from_samples` over exactly these scores, and
        # `from_samples` is deterministic, so a legitimate round trip reproduces both exactly.
        expected = from_samples(artifact.embedder_model, answerable, unanswerable)
        for name, stored, derived in (
            ("threshold", artifact.threshold, expected.threshold),
            ("scale", artifact.scale, expected.scale),
        ):
            if not _rederived_matches(stored, derived):
                raise CalibrationBindingError(
                    f"bundle {name} {stored} disagrees with the value re-derived from its own "
                    f"scores ({derived})"
                )
    # A carried artifact is certified under a SECOND condition too, inside `_imported_verdict`:
    # the inherited threshold must decide both classes within the (capped) recorded error bound.
    verdict, reason = _imported_verdict(artifact, answerable, unanswerable)
    if artifact.certified != verdict:
        raise CalibrationBindingError(
            f"bundle claims certified={artifact.certified} but its own scores re-derive "
            f"certified={verdict}: {reason}"
        )


def corpus_delta(
    parent_objects: Sequence[Mapping[str, Any]],
    child_objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Source-level difference between two generation manifests.

    The denominator is the UNION of source URIs, not the child's count, and that choice is load
    bearing: over the child alone, deleting nine tenths of a corpus scores a delta near zero,
    because the survivors all still match. A deletion is a change to what the index can answer and
    must count as one.

    Compares `(uri, sha256)`, so a file edited in place counts as changed even though its URI is
    unmoved. That is the same identity `_reuse_source` uses to decide whether a chunk may be
    copied forward, so the delta reported here and the work the rebuild actually does cannot
    disagree.
    """
    parent = {str(obj["uri"]): str(obj.get("sha256", "")) for obj in parent_objects}
    child = {str(obj["uri"]): str(obj.get("sha256", "")) for obj in child_objects}
    added = sorted(set(child) - set(parent))
    removed = sorted(set(parent) - set(child))
    modified = sorted(uri for uri in set(parent) & set(child) if parent[uri] != child[uri])
    union = len(set(parent) | set(child))
    changed = len(added) + len(removed) + len(modified)
    return {
        "sources_parent": len(parent),
        "sources_child": len(child),
        "sources_added": len(added),
        "sources_removed": len(removed),
        "sources_modified": len(modified),
        "sources_union": union,
        "corpus_delta": (changed / union) if union else 0.0,
    }

