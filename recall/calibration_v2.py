"""Tenant and generation bound calibration artifacts for the v1 index path."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple, TypedDict

import psycopg
from psycopg.types.json import Jsonb

from recall.calibration import Calibration, from_samples, separability
from recall.embedding_core import profile_lineage_mismatch
from recall.embeddings import Embedder
from recall.lineage import PipelineIdentity, canonical_json, canonical_sha256
from recall.errors import RecallError
from recall.observability import get_logger

ARTIFACT_VERSION = 2

_log = get_logger("calibration")


class CalibrationStatus(StrEnum):
    """A calibration's standing, from one of two producers that answer different questions.

    `CalibrationRepository.resolve_within` says how a GENERATION resolves, and returns only
    CERTIFIED, MISSING, STALE, UNCERTIFIED and DRAFT. `CalibrationArtifactV2.status` says what one
    ROW's lifecycle is, and returns only CERTIFIED, DRAFT, REJECTED and SUPERSEDED, from the
    stored state alone. LEGACY_UNBOUND names a stored lifecycle state and neither returns it.
    """

    #: From `resolve_within`: a published, certified artifact whose lineage and query set still
    #: match the generation. From `CalibrationArtifactV2.status`: a published, certified row,
    #: with no lineage or query-set check.
    CERTIFIED = "certified"
    #: No published calibration exists anywhere in the tenant; this tenant has never certified.
    MISSING = "missing"
    #: The tenant HAS a published calibration, but not one that binds to this generation: either
    #: none was published for it, or the published one no longer matches the generation's
    #: fingerprints or its stored query set. Distinct from MISSING because the remedy is a
    #: recalibration or carry-forward of an existing practice, not a first calibration.
    STALE = "stale"
    #: The newest calibration for this generation exists but failed certification (rejected).
    UNCERTIFIED = "uncertified"
    #: A certified calibration exists for this generation but was never published.
    DRAFT = "draft"
    #: The artifact failed certification; `calibrate` writes it in this state rather than raising.
    REJECTED = "rejected"
    #: A formerly published artifact that a later `publish` replaced.
    SUPERSEDED = "superseded"
    #: Imported from a pre-v2 file: a bare threshold with no tenant, generation, pipeline,
    #: corpus, or query binding. Never resolvable and never publishable.
    LEGACY_UNBOUND = "legacy_unbound"


class CalibrationError(RuntimeError, RecallError):
    pass


class CalibrationNotFound(CalibrationError):
    pass


class CalibrationUncertified(CalibrationError):
    pass


class CalibrationBindingError(CalibrationError):
    pass


class LifecycleState(StrEnum):
    """The stored `lifecycle_state` of a calibration row (the column's CHECK, migration 0011)."""

    DRAFT = "draft"
    PUBLISHED = "published"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"
    LEGACY_UNBOUND = "legacy_unbound"


#: The states a v2 artifact can hold, and the subset in which it is certified.
_ARTIFACT_STATES = frozenset(
    {
        LifecycleState.DRAFT,
        LifecycleState.PUBLISHED,
        LifecycleState.SUPERSEDED,
        LifecycleState.REJECTED,
    }
)
_CERTIFIED_STATES = frozenset(
    {LifecycleState.DRAFT, LifecycleState.PUBLISHED, LifecycleState.SUPERSEDED}
)

#: Model name used when an embedder identity carries none. One constant because the import
#: check re-derives a `Calibration` that must name the same embedder serving does.
_FALLBACK_EMBEDDER = "bound-v2"

#: Float tolerance for re-deriving an imported artifact's statistics from its own scores. A
#: legitimate round trip is exact (every column is DOUBLE PRECISION and the functions are the
#: ones `calibrate` used), so this absorbs only representation noise, never a different fit.
_REDERIVE_TOLERANCE = 1e-9

#: Tolerance on the cumulative carry-forward delta. The chain total is a sum of floats, and a
#: chain that is exactly at the bound can sum a few ulps above it (25 steps of 0.01 sum to
#: 0.25000000000000006), which refused a chain the bound is documented to allow.
_CUMULATIVE_DELTA_TOLERANCE = 1e-9


def _frozen_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _frozen_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_frozen_json(item) for item in value)
    return value


def _utc_isoformat(value: Any) -> str:
    """Render a stored timestamp as the UTC ISO string the checksum was computed over.

    `created_at` is part of `immutable_payload()`, so its *string* form is inside the
    artifact checksum, but the column is `timestamptz` and psycopg decodes it in the
    connection's TimeZone. Rendering it as-is makes the recomputed digest depend on the
    session's `TimeZone`, so an artifact written by one session fails `verify_checksum()`
    in another. `calibrate()` always writes `datetime.now(UTC).isoformat()`, so
    normalising back to UTC on read restores the exact bytes that were hashed.
    """
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    return str(value)


def _require_utc_isoformat(value: str, field_name: str) -> None:
    """Refuse a timestamp the `timestamptz` round trip would not return unchanged.

    The column keeps the instant and discards the rendering, while the *string* is what the
    artifact checksum covers. Any other valid ISO-8601 spelling of the same instant (a non-UTC
    offset, or the `Z`-plus-milliseconds form a JavaScript producer emits) would therefore
    store cleanly and then fail `verify_checksum()` on every later read. Enforcing the
    canonical form at construction keeps that failure at the boundary, where it names the
    cause, instead of committing a row nothing can read back.
    """
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise CalibrationBindingError(f"{field_name} must be an ISO-8601 timestamp") from None
    if parsed.tzinfo is None:
        # `replace(tzinfo=UTC)`, not `astimezone()`: astimezone would read a naive value as
        # machine-local, naming a different instant on every host. Not string concatenation
        # either: `fromisoformat` accepts spellings the canonical form does not (a space
        # separator, minute precision, milliseconds, a bare date), and appending an offset to
        # those produced advice this very guard then rejected. `str()` on a naive datetime
        # yields the space-separated form, so that was the likely shape to hit it.
        raise CalibrationBindingError(
            f"{field_name} must carry a UTC offset "
            f"(e.g. {parsed.replace(tzinfo=UTC).isoformat()!r}); re-export the bundle"
        )
    try:
        canonical = parsed.astimezone(UTC).isoformat()
    except (OverflowError, OSError):
        # astimezone() is not total: it overflows near datetime.min/max. Rendering a value
        # already judged unusable would turn this boundary check into an unnamed crash that
        # escapes the CLI's `except CalibrationError`.
        raise CalibrationBindingError(
            f"{field_name} is not a representable UTC instant"
        ) from None
    if canonical != value:
        raise CalibrationBindingError(
            f"{field_name} must be UTC in the form written by calibrate() "
            f"(e.g. {canonical!r}); re-export the bundle"
        )


def _require_digest(value: str, field_name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise CalibrationBindingError(f"{field_name} must be a lowercase SHA-256 digest")


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


def _require_carry_forward(value: Mapping[str, Any], threshold: float, scale: float) -> None:
    """Validate carry-forward provenance, including that it names the numbers it actually carried.

    The last two checks are the ones worth having. Provenance saying "inherited from cal_X" beside
    a threshold that is not cal_X's threshold is not provenance, it is a decoration, and it would
    survive every other check in this class because nothing else compares the two.
    """
    for key in ("parent_calibration_id", "parent_generation_id", "corpus_delta"):
        if key not in value:
            raise CalibrationBindingError(f"carry-forward provenance is missing {key!r}")
    for key in ("parent_calibration_id", "parent_generation_id"):
        if not str(value[key]).strip():
            raise CalibrationBindingError(f"carry-forward {key} must be non-empty")
    delta = value["corpus_delta"]
    if not isinstance(delta, (int, float)) or isinstance(delta, bool) or not math.isfinite(delta):
        raise CalibrationBindingError("carry-forward corpus_delta must be a finite number")
    if not 0.0 <= float(delta) <= 1.0:
        raise CalibrationBindingError("carry-forward corpus_delta must be a fraction in [0, 1]")
    # `cumulative_corpus_delta` is a NEWER key than the step delta above, so artifacts written
    # before it existed legitimately lack it (see `_cumulative_corpus_delta`, which reads such an
    # artifact's step delta in its place). When present it must at least be a sane number; it is
    # not capped at 1.0 because a sum of per-step fractions is not itself a fraction.
    cumulative = value.get("cumulative_corpus_delta")
    if cumulative is not None:
        if (
            not isinstance(cumulative, (int, float))
            or isinstance(cumulative, bool)
            or not math.isfinite(cumulative)
            or float(cumulative) < 0.0
        ):
            raise CalibrationBindingError(
                "carry-forward cumulative_corpus_delta must be a finite non-negative number"
            )
    inherited_threshold = value.get("inherited_threshold")
    inherited_scale = value.get("inherited_scale")
    # The coercions are wrapped because this validates data from the OUTSIDE (an imported bundle
    # or a stored row), and a string or mapping here would otherwise escape as a bare TypeError
    # or ValueError that the CLI's `except CalibrationError` cannot name.
    try:
        threshold_mismatch = (
            inherited_threshold is not None and float(inherited_threshold) != threshold
        )
        scale_mismatch = inherited_scale is not None and float(inherited_scale) != scale
    except (TypeError, ValueError) as exc:
        raise CalibrationBindingError(
            f"carry-forward inherited threshold or scale is not a number: {exc}"
        ) from exc
    if threshold_mismatch:
        raise CalibrationBindingError(
            "carry-forward provenance names an inherited threshold this artifact does not carry"
        )
    if scale_mismatch:
        raise CalibrationBindingError(
            "carry-forward provenance names an inherited scale this artifact does not carry"
        )


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
    re-deriving it would refuse every legitimate carried bundle; it is checked only against the
    recorded error-rate bound over the bundle's scores.

    ⛔ **No bundle is AUTHENTICATED by this check, fitted or carried.** Every input it reads is
    in the bundle, and a forger chooses the scores as freely as the threshold: `from_samples`
    places the threshold between the classes, so scores clustered either side of any value
    re-derive exactly that value, certified. Consistency is all this proves. Authenticating a
    bundle needs something this function cannot do: re-scoring the stored query set against the
    named generation, or a keyed signature. Treat any bundle from an untrusted source as
    unverified, and do not `publish` one you did not export yourself.
    """
    try:
        answerable = [float(v) for v in artifact.scores.get("answerable", ())]
        unanswerable = [float(v) for v in artifact.scores.get("unanswerable", ())]
    except (TypeError, ValueError) as exc:
        raise CalibrationBindingError(
            f"bundle scores are malformed: {exc}"
        ) from exc
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
    verdict = check.certified is True
    if artifact.carry_forward is not None:
        # A carried artifact is certified under a SECOND condition too: the inherited threshold
        # must still decide the fresh scores within the recorded error bound. Re-deriving the
        # verdict without it would refuse every legitimately carried bundle.
        try:
            max_error = float(
                artifact.carry_forward.get(
                    "max_carry_forward_error", DEFAULT_MAX_CARRY_FORWARD_ERROR
                )
            )
        except (TypeError, ValueError) as exc:
            raise CalibrationBindingError(
                f"carry-forward max_carry_forward_error is not a number: {exc}"
            ) from exc
        errors = threshold_error_rates(answerable, unanswerable, artifact.threshold)
        verdict = verdict and within_error_bound(errors, max_error)
    if artifact.certified != verdict:
        raise CalibrationBindingError(
            f"bundle claims certified={artifact.certified} but its own scores re-derive "
            f"certified={verdict}: {check.certification_reason}"
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


@dataclass(frozen=True)
class CalibrationArtifactV2:
    """One immutable, checksummed fit, bound to a tenant, a generation and a query set.

    Construction refuses an inconsistent artifact rather than storing one: the version must be
    `ARTIFACT_VERSION`, the digests lowercase SHA-256, `created_at` canonical UTC, every statistic
    finite and in range, `certified` must agree with `lifecycle_state` (certified exactly in
    draft, published and superseded), and carry-forward provenance must name the threshold and
    scale this artifact carries. `checksum` covers `immutable_payload()`, i.e. every field except
    `lifecycle_state` and `checksum` itself, so a lifecycle move never invalidates it.
    """

    calibration_id: str
    tenant_id: str
    generation_id: str
    embedder_identity: Mapping[str, Any]
    pipeline_fingerprint: str
    corpus_fingerprint: str
    query_set_digest: str
    threshold: float
    scale: float
    separability: float
    separability_ci: tuple[float, float]
    n_answerable: int
    n_unanswerable: int
    certified: bool
    certification_reason: str
    lifecycle_state: str
    created_at: str
    created_by: str
    scores: Mapping[str, Any]
    checksum: str
    artifact_version: int = ARTIFACT_VERSION
    #: Provenance when this threshold was INHERITED from an earlier generation rather than fitted
    #: on this one. None means fitted here, which is the true statement about every artifact
    #: written before `carry_forward` existed — hence None rather than an empty mapping, and hence
    #: no backfill. See `CalibrationRepository.carry_forward`.
    carry_forward: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.artifact_version != ARTIFACT_VERSION:
            raise CalibrationBindingError(
                f"unsupported calibration artifact version {self.artifact_version}"
            )
        for field_name in (
            "calibration_id",
            "tenant_id",
            "generation_id",
            "created_at",
            "created_by",
            "certification_reason",
        ):
            if not str(getattr(self, field_name)).strip():
                raise CalibrationBindingError(f"{field_name} must be non-empty")
        for field_name in (
            "pipeline_fingerprint",
            "corpus_fingerprint",
            "query_set_digest",
            "checksum",
        ):
            _require_digest(str(getattr(self, field_name)), field_name)
        _require_utc_isoformat(str(self.created_at), "created_at")
        numeric = (self.threshold, self.scale, self.separability, *self.separability_ci)
        if not all(math.isfinite(value) for value in numeric):
            raise CalibrationBindingError("calibration statistics must be finite")
        if not -1.0 <= self.threshold <= 1.0 or self.scale <= 0:
            raise CalibrationBindingError("calibration threshold or scale is invalid")
        low, high = self.separability_ci
        if not 0.0 <= self.separability <= 1.0 or not 0.0 <= low <= high <= 1.0:
            raise CalibrationBindingError("calibration separability interval is invalid")
        if self.n_answerable < 0 or self.n_unanswerable < 0:
            raise CalibrationBindingError("calibration sample counts cannot be negative")
        if self.lifecycle_state not in _ARTIFACT_STATES:
            raise CalibrationBindingError(
                f"invalid bound calibration lifecycle state {self.lifecycle_state!r}"
            )
        if self.certified != (self.lifecycle_state in _CERTIFIED_STATES):
            raise CalibrationBindingError(
                "certified flag is inconsistent with calibration lifecycle state"
            )
        canonical_json(self.embedder_identity)
        canonical_json(self.scores)
        object.__setattr__(self, "embedder_identity", _frozen_json(self.embedder_identity))
        object.__setattr__(self, "scores", _frozen_json(self.scores))
        if self.carry_forward is not None:
            _require_carry_forward(self.carry_forward, self.threshold, self.scale)
            object.__setattr__(self, "carry_forward", _frozen_json(self.carry_forward))

    @property
    def status(self) -> CalibrationStatus:
        """This ROW's lifecycle, from `lifecycle_state` and `certified` alone.

        ⚠️ CERTIFIED here means published and certified, nothing more: it does not re-check the
        generation's lineage or the stored query set, so a row can read CERTIFIED while its
        generation resolves STALE. Ask `CalibrationRepository.resolve` whether a generation is
        served by a valid calibration.
        """
        if self.lifecycle_state == LifecycleState.PUBLISHED and self.certified:
            return CalibrationStatus.CERTIFIED
        if self.lifecycle_state == LifecycleState.REJECTED or not self.certified:
            return CalibrationStatus.REJECTED
        if self.lifecycle_state == LifecycleState.SUPERSEDED:
            return CalibrationStatus.SUPERSEDED
        return CalibrationStatus.DRAFT

    @property
    def embedder_model(self) -> str:
        """The embedder model this artifact names, or `_FALLBACK_EMBEDDER` when it names none."""
        return str(self.embedder_identity.get("model", _FALLBACK_EMBEDDER))

    @property
    def runtime(self) -> Calibration:
        return Calibration(
            embedder=self.embedder_model,
            threshold=self.threshold,
            scale=self.scale,
            separability=self.separability,
            n_answerable=self.n_answerable,
            n_unanswerable=self.n_unanswerable,
        )

    def immutable_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "artifact_version": self.artifact_version,
            "calibration_id": self.calibration_id,
            "tenant_id": self.tenant_id,
            "generation_id": self.generation_id,
            "embedder_identity": dict(self.embedder_identity),
            "pipeline_fingerprint": self.pipeline_fingerprint,
            "corpus_fingerprint": self.corpus_fingerprint,
            "query_set_digest": self.query_set_digest,
            "threshold": self.threshold,
            "scale": self.scale,
            "separability": self.separability,
            "separability_ci": list(self.separability_ci),
            "n_answerable": self.n_answerable,
            "n_unanswerable": self.n_unanswerable,
            "certified": self.certified,
            "certification_reason": self.certification_reason,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "scores": dict(self.scores),
        }
        # Added ONLY when present, so every artifact written before carry-forward existed hashes
        # to exactly the bytes it hashed to then and still verifies. An unconditional key (even
        # `None`) would invalidate every stored checksum on upgrade, and a checksum that fails
        # after a version bump teaches the operator to ignore checksum failures.
        if self.carry_forward is not None:
            payload["carry_forward"] = dict(self.carry_forward)
        return payload

    def verify_checksum(self) -> None:
        actual = canonical_sha256(self.immutable_payload())
        if actual != self.checksum:
            raise CalibrationBindingError(
                f"calibration checksum mismatch: expected {self.checksum}, computed {actual}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.immutable_payload(),
            "lifecycle_state": self.lifecycle_state,
            "checksum": self.checksum,
        }

    @property
    def threshold_was_measured_here(self) -> bool:
        """False when this threshold was inherited from an earlier generation.

        Exposed as a property rather than left to callers testing `carry_forward is not None`,
        because that test reads as "has provenance" and the question every caller actually means
        is the opposite one.
        """
        return self.carry_forward is None


@dataclass(frozen=True)
class CalibrationResolution:
    """How a generation resolves. `artifact` is set exactly when `status` is CERTIFIED."""

    status: CalibrationStatus
    artifact: CalibrationArtifactV2 | None = None


def canonical_query_set(
    entries: Sequence[Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], str]:
    """Normalise a labelled query set and return it with its content digest.

    The canonical form keeps only `query`, `answerable`, and (when present) a sorted,
    deduplicated `relevant_ids`, sorted by each entry's canonical JSON, so the same labels in
    any order and with any extra keys canonicalise identically. A duplicate labelled query is
    refused rather than deduplicated, because two copies of one label would silently weight it.
    Only an identical ENTRY is refused here; a fit also applies `require_distinct_scored_queries`,
    which refuses the same query repeated with different `relevant_ids` or labels.
    The returned digest is `canonical_sha256` of the canonical entries and is what is stored as
    an artifact's `query_set_digest`, so `resolve` and `carry_forward` can detect a stored set
    that no longer matches the evidence a threshold was fitted on.
    """
    normalised: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        query = entry.get("query")
        answerable = entry.get("answerable")
        if not isinstance(query, str) or not query.strip() or not isinstance(answerable, bool):
            raise CalibrationBindingError(
                f"query set entry {index} requires a non-empty query and boolean answerable"
            )
        item: dict[str, Any] = {"query": query, "answerable": answerable}
        if "relevant_ids" in entry:
            relevant = entry["relevant_ids"]
            if not isinstance(relevant, list) or not all(isinstance(v, str) for v in relevant):
                raise CalibrationBindingError(f"query set entry {index} has invalid relevant_ids")
            item["relevant_ids"] = sorted(set(relevant))
        normalised.append(item)
    if not normalised:
        raise CalibrationBindingError("query set must not be empty")
    normalised.sort(key=canonical_json)
    keys = [canonical_json(item) for item in normalised]
    if len(keys) != len(set(keys)):
        raise CalibrationBindingError("query set contains a duplicate labelled query")
    value = tuple(normalised)
    return value, canonical_sha256(list(value))


def require_distinct_scored_queries(labels: Sequence[Mapping[str, Any]]) -> None:
    """Refuse a labelled set that repeats a query the scorer would score twice.

    `canonical_query_set` refuses an identical ENTRY, but `relevant_ids` is part of an entry and
    the scorer (`measure_top_cosines`) never reads it: two entries that differ only there, or in
    their answerable label, are the same cosine counted twice. Twenty such copies of one query
    per class would clear the certification sample floor on two queries' evidence, and one query
    labelled both ways is evidence nobody should fit on.

    Applied where a NEW set enters a fit, not inside `canonical_query_set`, which also
    re-canonicalises sets already stored: refusing there would turn a stored set into a
    resolution error on the serving path instead of a refused fit.
    """
    seen: dict[str, bool] = {}
    for item in labels:
        query = str(item["query"])
        if query in seen:
            label = (
                "with contradictory answerable labels"
                if seen[query] != bool(item["answerable"])
                else "with different relevant_ids"
            )
            raise CalibrationBindingError(
                f"query set repeats the query {query!r} {label}; the scorer would count one "
                f"cosine twice, so keep one entry per query"
            )
        seen[query] = bool(item["answerable"])


def load_query_set(path: str | Path) -> tuple[tuple[dict[str, Any], ...], str]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationBindingError(f"cannot read labelled query set {path}: {exc}") from exc
    if not isinstance(raw, list) or not all(isinstance(item, Mapping) for item in raw):
        raise CalibrationBindingError("labelled query set must be a JSON array of objects")
    return canonical_query_set(raw)


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


#: Passed as `checksum` when an artifact is constructed purely to COMPUTE its checksum from
#: `immutable_payload()`. The checksum field is not part of that payload, so the placeholder never
#: reaches a digest; it exists because the dataclass requires a well-formed value at construction.
#: The row is always read back through `_artifact`, which verifies the REAL checksum.
_UNCOMPUTED_CHECKSUM = "0" * 64


def _require_runtime_embedder(pipeline: PipelineIdentity, embedder: Embedder) -> None:
    """Refuse to score a generation's vectors with an embedder that did not write them.

    Model and dimension first, then the recorded profile, which is the only field that separates
    two profiles of one model (Voyage with and without `input_type`). A threshold fitted on those
    cosines would certify numbers that compare two encoders. Shared by `calibrate` and
    `carry_forward`, the two paths that score labels with the runtime embedder.
    """
    if embedder.name != pipeline.embedder.model or embedder.dim != pipeline.embedder.dimension:
        raise CalibrationBindingError(
            "embedder implementation does not match the generation pipeline identity"
        )
    mismatch = profile_lineage_mismatch(
        embedder, pipeline.embedder.profile_id, pipeline.embedder.profile_fingerprint
    )
    if mismatch is not None:
        raise CalibrationBindingError(f"{mismatch} in the generation pipeline identity")


class _GenerationLineage(NamedTuple):
    """What `CalibrationRepository._generation` reads: the identity and the two fingerprints."""

    pipeline_identity: dict[str, Any]
    pipeline_fingerprint: str
    corpus_fingerprint: str


def write_calibration_bundle(payload: dict[str, Any], path: str | Path) -> Path:
    """Seal `payload` with its `bundle_checksum` and write it as canonical JSON.

    The one bundle format: `CalibrationRepository.export_bundle` and the lite store's export use it.
    """
    payload["bundle_checksum"] = canonical_sha256(payload)
    target = Path(path)
    target.write_bytes(canonical_json(payload) + b"\n")
    return target


class CalibrationRepository:
    """Tenant-scoped store of v2 calibration artifacts and their labelled query sets.

    Lifecycle: `calibrate` and `carry_forward` write a `draft` when certification passes and a
    `rejected` row when it does not, deliberately recording the failure rather than raising, so
    an operator can read WHY a fit failed. `publish` promotes a certified draft under a
    per-generation advisory lock, marking any previously published artifact for the same
    generation `superseded`; `superseded` rows are history, never deleted. `legacy_unbound`
    rows come from `import_bundle`'s legacy fallback (a pre-v2 JSON file holding only a
    threshold and scale) and can never be published or resolved, because they bind to no
    tenant, generation, pipeline, corpus, or query set.

    Every artifact's immutable payload is checksummed at write and re-verified on every read,
    and rows are guarded by database triggers that refuse payload updates, so evidence cannot
    be edited between a fit and a later decision that relies on it.
    """

    def __init__(
        self,
        dsn: str,
        tenant_id: str,
        *,
        actor: str = "recall-cli",
        connect_timeout_s: int = 10,
    ) -> None:
        if not tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")
        self.dsn = dsn
        self.tenant_id = tenant_id
        self.actor = actor
        self._connect_timeout_s = connect_timeout_s

    @contextmanager
    def _connect(self) -> Iterator[psycopg.Connection]:
        """A tenant-scoped autocommit connection.

        A missing table or column is re-raised as `SchemaTooOld`. Only the fitting paths check the
        schema up front (through `score_query_set`), so `list`, `show`, `publish`, `export` and
        `import` against an unmigrated database otherwise ended in a raw psycopg traceback naming a
        column, instead of the message that says to run `recall schema apply`.
        """
        from recall.schema import SchemaTooOld

        with psycopg.connect(
            self.dsn, autocommit=True, connect_timeout=self._connect_timeout_s
        ) as conn:
            conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (self.tenant_id,))
            try:
                yield conn
            except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn) as exc:
                raise SchemaTooOld(
                    f"the calibration tables are older than this package expects "
                    f"({exc.diag.message_primary or exc}); run `recall schema apply`"
                ) from exc

    def _audit(
        self,
        conn: psycopg.Connection,
        event_type: str,
        calibration_id: str,
        generation_id: str | None,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        conn.execute(
            "INSERT INTO recall_audit_events "
            "(tenant_id, event_id, event_type, actor, generation_id, payload) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (
                self.tenant_id,
                _id("evt"),
                event_type,
                self.actor,
                generation_id,
                Jsonb({"calibration_id": calibration_id, **dict(payload or {})}),
            ),
        )

    def _generation(
        self, conn: psycopg.Connection, generation_id: str, *, lock: bool = False
    ) -> _GenerationLineage:
        """A ready, active or retired generation's lineage, refusing any other state.

        `lock` reads the row `FOR SHARE`, for a caller that acts on the fingerprints it reads in
        the same transaction (`publish`). `forget()` rewrites `corpus_fingerprint` under `FOR
        UPDATE`, which a plain read does not wait for, so without the lock a concurrent erasure
        could commit between the check and the write.
        """
        row = conn.execute(
            "SELECT state, pipeline_identity, pipeline_fingerprint, corpus_fingerprint "
            "FROM recall_generations WHERE tenant_id = %s AND generation_id = %s"
            + (" FOR SHARE" if lock else ""),
            (self.tenant_id, generation_id),
        ).fetchone()
        if row is None:
            raise CalibrationBindingError(f"generation {generation_id!r} does not exist")
        if str(row[0]) not in {"ready", "active", "retired"}:
            raise CalibrationBindingError(
                f"generation {generation_id!r} is {row[0]!r}, expected ready, active, or retired"
            )
        if not isinstance(row[1], Mapping):
            raise CalibrationBindingError("stored pipeline identity is malformed")
        return _GenerationLineage(dict(row[1]), str(row[2]), str(row[3]))

    def _query_set_rows(self, conn: psycopg.Connection, digest: str) -> list[Any] | None:
        """The stored labelled queries behind `digest`, or None when absent or not a list."""
        row = conn.execute(
            "SELECT queries FROM recall_calibration_query_sets WHERE tenant_id = %s "
            "AND query_set_digest = %s",
            (self.tenant_id, digest),
        ).fetchone()
        if row is None or not isinstance(row[0], list):
            return None
        return row[0]

    #: The one source of truth for the SELECT column list AND for `_artifact`'s row decoding.
    #: Keeping them a single tuple is what makes it impossible for the projection and the
    #: positional decode to drift apart, which they previously could: the decode indexed
    #: `row[0]..row[21]` against a comma-separated string thirty lines away.
    _COLUMN_NAMES = (
        "calibration_id",
        "tenant_id",
        "generation_id",
        "embedder_identity",
        "pipeline_fingerprint",
        "corpus_fingerprint",
        "query_set_digest",
        "threshold",
        "scale",
        "separability",
        "ci_low",
        "ci_high",
        "n_answerable",
        "n_unanswerable",
        "certified",
        "certification_reason",
        "lifecycle_state",
        "scores",
        "created_at",
        "created_by",
        "artifact_checksum",
        "carry_forward",
    )
    _COLUMNS = ", ".join(_COLUMN_NAMES)

    @classmethod
    def _artifact(cls, row: tuple[Any, ...]) -> CalibrationArtifactV2:
        record = dict(zip(cls._COLUMN_NAMES, row, strict=True))
        identity = record["embedder_identity"]
        scores = record["scores"]
        carry = record["carry_forward"]
        artifact = CalibrationArtifactV2(
            calibration_id=str(record["calibration_id"]),
            tenant_id=str(record["tenant_id"]),
            generation_id=str(record["generation_id"]),
            embedder_identity=dict(identity) if isinstance(identity, Mapping) else {},
            pipeline_fingerprint=str(record["pipeline_fingerprint"]),
            corpus_fingerprint=str(record["corpus_fingerprint"]),
            query_set_digest=str(record["query_set_digest"]),
            threshold=float(record["threshold"]),
            scale=float(record["scale"]),
            separability=float(record["separability"]),
            separability_ci=(float(record["ci_low"]), float(record["ci_high"])),
            n_answerable=int(record["n_answerable"]),
            n_unanswerable=int(record["n_unanswerable"]),
            certified=bool(record["certified"]),
            certification_reason=str(record["certification_reason"]),
            lifecycle_state=str(record["lifecycle_state"]),
            scores=dict(scores) if isinstance(scores, Mapping) else {},
            created_at=_utc_isoformat(record["created_at"]),
            created_by=str(record["created_by"]),
            checksum=str(record["artifact_checksum"]),
            carry_forward=dict(carry) if isinstance(carry, Mapping) else None,
        )
        artifact.verify_checksum()
        return artifact

    def score_query_set(
        self, generation_id: str, embedder: Embedder, labels: Sequence[Mapping[str, Any]]
    ) -> tuple[list[float], list[float]]:
        """Best dense cosine per labelled query against ONE pinned generation.

        Extracted because `calibrate`, `carry_forward` and drift monitoring all need exactly this
        and had two byte-identical copies of it between them. The sampling rule is the single one
        in `recall.eval.calibrate.measure_top_cosines`, so a fit, a re-verification and a drift
        probe cannot come to different answers about the same generation for the reason that they
        sampled it differently.

        `pin_generation` is what makes the scores belong to the generation named rather than to
        whatever is active at the moment the query runs.
        """
        from recall.eval.calibrate import measure_top_cosines
        from recall.generation_store import GenerationStore

        store = GenerationStore(self.dsn, embedder.dim, tenant=self.tenant_id)
        try:
            store.check_schema()
            with store.pin_generation(generation_id):
                # Real dicts, not the stored Mappings. `measure_top_cosines` is annotated
                # `list[dict]`, and `canonical_query_set` hands back frozen mappings.
                return measure_top_cosines(store, embedder, [dict(item) for item in labels])
        finally:
            store.close()

    def stored_query_set(
        self, query_set_digest: str
    ) -> tuple[tuple[Mapping[str, Any], ...], str]:
        """The labelled queries behind a digest, re-canonicalised and re-digested.

        Returned with the digest RECOMPUTED rather than echoed back, so a caller comparing it
        against the artifact's `query_set_digest` is checking the stored bytes and not its own
        argument. That is the same check `resolve` makes, and it exists because the evidence could
        otherwise be edited between the fit and any later re-verification.
        """
        with self._connect() as conn:
            rows = self._query_set_rows(conn, query_set_digest)
        if rows is None:
            raise CalibrationNotFound(
                f"labelled query set {query_set_digest} is not stored for tenant "
                f"{self.tenant_id!r}, so nothing can be re-scored against it"
            )
        return canonical_query_set(rows)

    def manifest_objects_for(self, generation_id: str) -> list[Mapping[str, Any]]:
        """The generation's manifest source objects, on a connection of this repository's own."""
        with self._connect() as conn:
            return self._manifest_objects(conn, generation_id)

    def _manifest_objects(
        self, conn: psycopg.Connection, generation_id: str
    ) -> list[Mapping[str, Any]]:
        row = conn.execute(
            "SELECT manifest FROM recall_generations WHERE tenant_id = %s AND generation_id = %s",
            (self.tenant_id, generation_id),
        ).fetchone()
        if row is None or not isinstance(row[0], Mapping):
            raise CalibrationBindingError(f"generation {generation_id!r} has no stored manifest")
        objects = row[0].get("objects")
        if not isinstance(objects, list) or not objects:
            raise CalibrationBindingError(
                f"generation {generation_id!r} manifest lists no source objects"
            )
        return [obj for obj in objects if isinstance(obj, Mapping)]

    def _insert_payload_row(
        self,
        conn: psycopg.Connection,
        payload: Mapping[str, Any],
        lifecycle: str,
        checksum: str,
        *,
        carried: bool,
    ) -> None:
        """Insert one artifact row whose every checksummed value comes from ``payload``.

        ``payload`` is the mapping the checksum was computed over, so the row cannot hold a value
        the checksum does not; that is also why ``certification_reason`` is right by
        construction. ``carried`` names a carried-forward artifact, whose payload must hold
        ``carry_forward`` (read strictly, so a payload without it fails rather than storing
        NULL); a fresh calibration stores NULL there, as the column's default always did.
        `import_bundle` keeps its own insert: it writes the rebound generation, the verified
        query-set digest and the embedder from the bundle identity, not the payload's.
        """
        conn.execute(
            "INSERT INTO recall_calibrations "
            "(tenant_id, calibration_id, generation_id, embedder_identity, "
            "pipeline_fingerprint, corpus_fingerprint, query_set_digest, threshold, scale, "
            "separability, ci_low, ci_high, n_answerable, n_unanswerable, certified, "
            "certification_reason, lifecycle_state, scores, created_at, created_by, "
            "artifact_checksum, carry_forward) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, "
            "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                self.tenant_id,
                payload["calibration_id"],
                payload["generation_id"],
                Jsonb(payload["embedder_identity"]),
                payload["pipeline_fingerprint"],
                payload["corpus_fingerprint"],
                payload["query_set_digest"],
                payload["threshold"],
                payload["scale"],
                payload["separability"],
                payload["separability_ci"][0],
                payload["separability_ci"][1],
                payload["n_answerable"],
                payload["n_unanswerable"],
                payload["certified"],
                payload["certification_reason"],
                lifecycle,
                Jsonb(payload["scores"]),
                payload["created_at"],
                payload["created_by"],
                checksum,
                Jsonb(payload["carry_forward"]) if carried else None,
            ),
        )

    def _new_artifact(
        self,
        generation_id: str,
        pipeline: PipelineIdentity,
        lineage: _GenerationLineage,
        query_digest: str,
        runtime: Calibration,
        answerable: Sequence[float],
        unanswerable: Sequence[float],
        *,
        certified: bool,
        reason: str,
        carry_forward: Mapping[str, Any] | None = None,
    ) -> tuple[CalibrationArtifactV2, dict[str, Any], str]:
        """Build a new artifact and the payload and checksum its row is written from.

        Shared by `calibrate` and `carry_forward`. The artifact is constructed FIRST and the
        checksum computed over ITS payload, so the bytes hashed here and the bytes
        `verify_checksum` recomputes on every later read come from the same method. A hand-built
        dict previously made three independent copies of the payload shape, and a key added to one
        but not the others would fail every subsequent read as corruption.
        """
        if runtime.separability is None or runtime.separability_ci is None:
            raise CalibrationBindingError("both labelled classes are required for calibration")
        lifecycle = LifecycleState.DRAFT if certified else LifecycleState.REJECTED
        artifact = CalibrationArtifactV2(
            calibration_id=_id("cal"),
            tenant_id=self.tenant_id,
            generation_id=generation_id,
            embedder_identity=pipeline.embedder.to_dict(),
            pipeline_fingerprint=lineage.pipeline_fingerprint,
            corpus_fingerprint=lineage.corpus_fingerprint,
            query_set_digest=query_digest,
            threshold=runtime.threshold,
            scale=runtime.scale,
            separability=runtime.separability,
            separability_ci=runtime.separability_ci,
            n_answerable=len(answerable),
            n_unanswerable=len(unanswerable),
            certified=certified,
            certification_reason=reason,
            # `.value`, a plain str: the row is written from this field, and the enum member is
            # only for comparisons, so the driver never has to choose how to adapt it.
            lifecycle_state=lifecycle.value,
            created_at=datetime.now(UTC).isoformat(),
            created_by=self.actor,
            scores={"answerable": list(answerable), "unanswerable": list(unanswerable)},
            checksum=_UNCOMPUTED_CHECKSUM,
            carry_forward=carry_forward,
        )
        payload = artifact.immutable_payload()
        return artifact, payload, canonical_sha256(payload)

    def _audit_if_rejected(self, conn: psycopg.Connection, artifact: CalibrationArtifactV2) -> None:
        if not artifact.certified:
            self._audit(
                conn,
                "calibration_rejected",
                artifact.calibration_id,
                artifact.generation_id,
                {"reason": artifact.certification_reason},
            )

    def calibrate(
        self,
        generation_id: str,
        queries: Sequence[Mapping[str, Any]],
        embedder: Embedder,
    ) -> CalibrationArtifactV2:
        """Fit a threshold for one generation from a labelled query set, and store it.

        The queries are scored on the named generation with `embedder`, which must be the one
        that wrote its vectors. A certified fit is stored as a `draft` (run `publish` to serve
        it); a fit that fails certification is stored and RETURNED as `rejected`, not raised, so
        check `artifact.certified`. Raises `CalibrationBindingError` for a malformed or repeated
        query, a missing labelled class, or an embedder mismatch. After the artifact is saved,
        near-miss coverage is measured and audited, fail-open (`_record_near_miss_coverage`).
        """
        labels, query_digest = canonical_query_set(queries)
        require_distinct_scored_queries(labels)
        with self._connect() as conn:
            lineage = self._generation(conn, generation_id)
        pipeline = PipelineIdentity.from_dict(lineage.pipeline_identity)
        _require_runtime_embedder(pipeline, embedder)

        answerable, unanswerable = self.score_query_set(generation_id, embedder, labels)
        runtime = from_samples(embedder.name, answerable, unanswerable)
        artifact, payload, checksum = self._new_artifact(
            generation_id,
            pipeline,
            lineage,
            query_digest,
            runtime,
            answerable,
            unanswerable,
            certified=runtime.certified is True,
            reason=runtime.certification_reason,
        )
        calibration_id = artifact.calibration_id
        with self._connect() as conn, conn.transaction():
            conn.execute(
                "INSERT INTO recall_calibration_query_sets "
                "(tenant_id, query_set_digest, queries, sample_count, created_by) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (self.tenant_id, query_digest, Jsonb(list(labels)), len(labels), self.actor),
            )
            self._insert_payload_row(
                conn, payload, artifact.lifecycle_state, checksum, carried=False
            )
            self._audit(conn, "calibration_created", calibration_id, generation_id)
            self._audit_if_rejected(conn, artifact)
        stored = self.get(calibration_id)
        self._record_near_miss_coverage(stored, labels, embedder)
        return stored

    def _record_near_miss_coverage(
        self,
        artifact: CalibrationArtifactV2,
        labels: Sequence[Mapping[str, Any]],
        embedder: Embedder,
    ) -> None:
        """Measure near-miss coverage with a fresh fit and append it as an audit event. Fail-open.

        The PostgreSQL side of the lite fit (`recall.lite.calibration.auto_calibrate`): probes are
        built from this generation's own chunks and the fit's answerable questions
        (`recall.near_miss.near_miss_probes`), scored at the threshold just fitted with the
        unknown-term gate (`near_miss_coverage`), and stored by calibration id, counts only. It runs
        after the artifact is saved and never touches it: certification, lifecycle and threshold
        are decided before this line. A carried calibration is not measured, as on lite, so it reads
        as unmeasured and provisional until the next fit.
        """
        from recall.generation_store import GenerationStore
        from recall.near_miss import fit_coverage
        from recall.support import COVERAGE_EVENT

        try:
            with self._connect() as conn:
                texts = [
                    str(row[0])
                    for row in conn.execute(
                        "SELECT text FROM recall_chunks_v1 WHERE tenant_id = %s "
                        "AND generation_id = %s ORDER BY chunk_id",
                        (self.tenant_id, artifact.generation_id),
                    ).fetchall()
                ]
            store = GenerationStore(self.dsn, embedder.dim, tenant=self.tenant_id)
            try:
                with store.pin_generation(artifact.generation_id):
                    report = fit_coverage(
                        store, embedder, float(artifact.threshold), texts, labels
                    )
            finally:
                store.close()
            with self._connect() as conn:
                self._audit(
                    conn, COVERAGE_EVENT, artifact.calibration_id, artifact.generation_id, report
                )
        except Exception as exc:  # BROAD-CATCH: fail-open
            # The fit is already saved. Optional diagnostics cannot withdraw it, and provider
            # errors may carry private input or credentials, so log only the type.
            _log.warning("Optional near-miss coverage unavailable (%s)", type(exc).__name__)

    def near_miss_coverage(self, calibration_id: str) -> dict[str, Any] | None:
        """The near-miss coverage measured with this calibration's fit, validated, or None.

        Raises ValueError for a stored report that disagrees with the producer's count rules, so a
        reader can report it unavailable rather than print it.
        """
        from recall.support import COVERAGE_EVENT, latest_calibration_event, validated_coverage_report

        with self._connect() as conn:
            payload = latest_calibration_event(conn, self.tenant_id, COVERAGE_EVENT, calibration_id)
        return None if payload is None else validated_coverage_report(payload, calibration_id)

    def _carry_parent(
        self, generation_id: str, parent_calibration_id: str | None
    ) -> CalibrationArtifactV2:
        """The calibration `carry_forward` inherits from: named, or the newest published one.

        Refused when it is already on `generation_id`, or when its row is not published and
        certified.
        """
        with self._connect() as conn:
            if parent_calibration_id is None:
                row = conn.execute(
                    "SELECT calibration_id FROM recall_calibrations WHERE tenant_id = %s "
                    "AND lifecycle_state = 'published' AND generation_id IS NOT NULL "
                    "AND generation_id <> %s ORDER BY published_at DESC, created_at DESC LIMIT 1",
                    (self.tenant_id, generation_id),
                ).fetchone()
                if row is None:
                    raise CalibrationNotFound(
                        f"tenant {self.tenant_id!r} has no published calibration on another "
                        f"generation to carry forward; calibrate {generation_id!r} directly"
                    )
                parent_calibration_id = str(row[0])
        parent = self.get(parent_calibration_id)
        if parent.generation_id == generation_id:
            raise CalibrationBindingError(
                f"calibration {parent.calibration_id!r} is already bound to generation "
                f"{generation_id!r}; there is nothing to carry forward"
            )
        if parent.status is not CalibrationStatus.CERTIFIED:
            # A draft or rejected parent has no certified threshold to carry, and inheriting one
            # would launder an uncertified number into a certified-looking artifact.
            raise CalibrationUncertified(
                f"parent calibration {parent.calibration_id!r} is {parent.status.value}, so it "
                f"has no certified threshold to carry forward"
            )
        return parent

    def carry_forward(
        self,
        generation_id: str,
        embedder: Embedder,
        *,
        parent_calibration_id: str | None = None,
        max_corpus_delta: float = DEFAULT_MAX_CORPUS_DELTA,
        max_error: float = DEFAULT_MAX_CARRY_FORWARD_ERROR,
    ) -> CalibrationArtifactV2:
        """Re-verify a published threshold against a NEW generation, without refitting it.

        This exists because a corpus that changes at all currently costs a full recalibration.
        `resolve` binds a calibration to one `generation_id`, so any rebuild — adding a handful of
        files to a thousand — leaves the new generation with no artifact, resolves `STALE`, and
        strict policy refuses every query. The operational effect is that a live index cannot
        absorb an increment without either a manual recalibration or a period of serving
        uncalibrated, and both of those are worse than the problem.

        **This is not a tolerance and it does not loosen the gate.** Nothing is carried on the
        strength of the delta being small. The parent's own stored labelled query set is re-scored
        against the child generation, and the inherited threshold must clear the SAME
        certification bar on those fresh scores that a new fit would have to clear. What is
        inherited is the threshold, not the certification. If the corpus moved enough to break the
        threshold, this produces a `rejected` artifact and the operator still has to recalibrate,
        which is the correct outcome and the one the delta bound cannot deliver on its own.

        Before any of the bounds below, the parent itself is checked: it must be on a DIFFERENT
        generation from `generation_id`, and its row must be published and certified (a draft,
        rejected or superseded parent has no certified threshold to carry). Then four refusals:

        1. **A different pipeline is refused outright**, never bounded. A threshold is a property
           of an embedder's cosine regime, and 2026-08-17 measured `voyage-4` at 0.269 to 0.413 on
           one corpus where `voyage-code-3` returned 0.480 to 0.834. Carrying a number across that
           is not a small error, and no delta is small enough to make it one. An IDENTICAL corpus
           fingerprint is refused at the same point: carry-forward is for a changed corpus, and
           an unchanged one means the wrong generation was named.
        2. **A delta above `max_corpus_delta` is refused before any embedding work**, because
           re-scoring a query set says nothing about the queries nobody labelled, and past some
           point the labelled set is describing a corpus that no longer exists.
        3. **A CHAIN whose cumulative delta exceeds the same bound is refused**, even when every
           individual step passed it. Each step's delta is measured only against the immediate
           parent, so without this a corpus could be replaced 20% at a time indefinitely while
           the threshold still wears the digest of the original fit. The cumulative figure is
           carried in each artifact's provenance rather than recomputed against the origin
           generation, because generation gc can have pruned the origin's manifest. A chain at
           exactly the bound is allowed: the sum is compared within `_CUMULATIVE_DELTA_TOLERANCE`.
        4. **A query set that no longer canonicalises to its stored digest is refused**, the same
           check `resolve` makes, because otherwise the evidence could be edited between the fit
           and the re-verification.

        `refit_threshold` is recorded in the provenance and **changes nothing**. It is what a
        fresh fit on these scores would have chosen, so an operator can see the inherited number
        drifting away from the data before it drifts far enough to fail. A diagnosis that silently
        moved the boundary would be a different feature wearing this one's name.
        """
        if not 0.0 <= max_corpus_delta <= 1.0:
            raise CalibrationBindingError("max_corpus_delta must be a fraction in [0, 1]")
        if not 0.0 <= max_error <= 1.0:
            raise CalibrationBindingError("max_error must be a fraction in [0, 1]")
        parent = self._carry_parent(generation_id, parent_calibration_id)

        with self._connect() as conn, conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            lineage = self._generation(conn, generation_id)
            if lineage.pipeline_fingerprint != parent.pipeline_fingerprint:
                raise CalibrationBindingError(
                    f"generation {generation_id!r} has a different pipeline fingerprint from "
                    f"calibration {parent.calibration_id!r}; a threshold cannot be carried across "
                    f"a pipeline change, recalibrate instead"
                )
            if lineage.corpus_fingerprint == parent.corpus_fingerprint:
                raise CalibrationBindingError(
                    f"generation {generation_id!r} has the same corpus fingerprint as "
                    f"calibration {parent.calibration_id!r}; carry-forward is for a CHANGED "
                    f"corpus, and an identical one means the wrong generation was named"
                )
            delta = corpus_delta(
                self._manifest_objects(conn, parent.generation_id),
                self._manifest_objects(conn, generation_id),
            )
            query_rows = self._query_set_rows(conn, parent.query_set_digest)
        cumulative = _carried_cumulative_delta(parent, delta, max_corpus_delta)
        if query_rows is None:
            raise CalibrationBindingError(
                f"the labelled query set {parent.query_set_digest} behind calibration "
                f"{parent.calibration_id!r} is missing, so its threshold cannot be re-verified"
            )
        labels, query_digest = canonical_query_set(query_rows)
        if query_digest != parent.query_set_digest:
            raise CalibrationBindingError(
                "stored labelled query set no longer matches its digest"
            )

        pipeline = PipelineIdentity.from_dict(lineage.pipeline_identity)
        _require_runtime_embedder(pipeline, embedder)

        answerable, unanswerable = self.score_query_set(generation_id, embedder, labels)

        # The inherited threshold and scale, judged on the CHILD's scores, by the rule the lite
        # store's carry-forward shares (`judge_inherited_threshold`).
        verdict = judge_inherited_threshold(parent, answerable, unanswerable, max_error=max_error)
        runtime, errors, certified, reason = (
            verdict.runtime,
            verdict.errors,
            verdict.certified,
            verdict.reason,
        )
        if runtime.separability is None or runtime.separability_ci is None:  # narrowed for mypy
            raise CalibrationBindingError("both labelled classes are required for carry-forward")
        provenance = {
            "parent_calibration_id": parent.calibration_id,
            "parent_generation_id": parent.generation_id,
            "parent_corpus_fingerprint": parent.corpus_fingerprint,
            "parent_separability": parent.separability,
            "inherited_threshold": parent.threshold,
            "inherited_scale": parent.scale,
            "refit_threshold": verdict.refit_threshold,
            "max_corpus_delta": max_corpus_delta,
            "max_carry_forward_error": max_error,
            **errors,
            **delta,
            # The chain total, not this step's delta: what the NEXT carry-forward reads to keep
            # cumulative drift bounded. New in this key only for NEW artifacts; older provenance
            # without it is read by `_cumulative_corpus_delta` as its recorded step delta.
            "cumulative_corpus_delta": cumulative,
        }
        artifact, payload, checksum = self._new_artifact(
            generation_id,
            pipeline,
            lineage,
            query_digest,
            runtime,
            answerable,
            unanswerable,
            certified=certified,
            reason=reason,
            carry_forward=provenance,
        )
        calibration_id = artifact.calibration_id
        with self._connect() as conn, conn.transaction():
            self._insert_payload_row(
                conn, payload, artifact.lifecycle_state, checksum, carried=True
            )
            self._audit(
                conn,
                "calibration_carried_forward",
                calibration_id,
                generation_id,
                {
                    "parent_calibration_id": parent.calibration_id,
                    "parent_generation_id": parent.generation_id,
                    "corpus_delta": delta["corpus_delta"],
                    "certified": certified,
                },
            )
            self._audit_if_rejected(conn, artifact)
        return self.get(calibration_id)

    def get(self, calibration_id: str) -> CalibrationArtifactV2:
        """One v2 artifact by id, with its checksum verified.

        A `legacy_unbound` row (no generation) is not an artifact and raises
        `CalibrationNotFound` here, although `list_records` and `show_record` return it; use
        `show_record` to inspect one.
        """
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {self._COLUMNS} FROM recall_calibrations "
                "WHERE tenant_id = %s AND calibration_id = %s AND generation_id IS NOT NULL",
                (self.tenant_id, calibration_id),
            ).fetchone()
        if row is None:
            raise CalibrationNotFound(calibration_id)
        return self._artifact(row)

    def publish(self, calibration_id: str) -> CalibrationArtifactV2:
        """Make a certified artifact the one its generation serves, and return it.

        Accepts a `draft`, or a `superseded` artifact, which is how a rollback to an earlier
        threshold is done; any other published artifact for the generation is marked
        `superseded`, under a per-generation advisory lock. Refuses an uncertified artifact, and
        one whose generation's pipeline or corpus fingerprint changed since the fit. The
        generation row is read `FOR SHARE`, so a concurrent `forget()` either commits first (and
        this refuses on the changed fingerprint) or waits, and then supersedes what this published.
        """
        artifact = self.get(calibration_id)
        if not artifact.certified:
            raise CalibrationUncertified(artifact.certification_reason)
        with self._connect() as conn, conn.transaction():
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"calibration\x1f{self.tenant_id}\x1f{artifact.generation_id}",),
            )
            lineage = self._generation(conn, artifact.generation_id, lock=True)
            if (
                lineage.pipeline_fingerprint != artifact.pipeline_fingerprint
                or lineage.corpus_fingerprint != artifact.corpus_fingerprint
            ):
                raise CalibrationBindingError("generation lineage changed after calibration")
            old = conn.execute(
                "SELECT calibration_id FROM recall_calibrations WHERE tenant_id = %s "
                "AND generation_id = %s AND lifecycle_state = 'published' FOR UPDATE",
                (self.tenant_id, artifact.generation_id),
            ).fetchall()
            for row in old:
                old_id = str(row[0])
                if old_id == calibration_id:
                    continue
                conn.execute(
                    "UPDATE recall_calibrations SET lifecycle_state = 'superseded', "
                    "superseded_at = clock_timestamp() WHERE tenant_id = %s "
                    "AND calibration_id = %s",
                    (self.tenant_id, old_id),
                )
                self._audit(
                    conn,
                    "calibration_superseded",
                    old_id,
                    artifact.generation_id,
                    {"superseded_by": calibration_id},
                )
            conn.execute(
                "UPDATE recall_calibrations SET lifecycle_state = 'published', "
                "published_at = clock_timestamp(), superseded_at = NULL WHERE tenant_id = %s "
                "AND calibration_id = %s "
                "AND lifecycle_state IN ('draft', 'published', 'superseded')",
                (self.tenant_id, calibration_id),
            )
            self._audit(conn, "calibration_published", calibration_id, artifact.generation_id)
        return self.get(calibration_id)

    def resolve(self, generation_id: str) -> CalibrationResolution:
        """Resolve on a connection of this repository's own, in a read-only snapshot."""
        with self._connect() as conn, conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            return self.resolve_within(conn, generation_id)

    def resolve_within(
        self, conn: psycopg.Connection, generation_id: str
    ) -> CalibrationResolution:
        """The same resolution, on a connection and transaction the CALLER already holds.

        ⚠️ **This exists so a gate can decide and act in one transaction.** `promote` locks the
        tenant row and the generation row `FOR UPDATE`, then asks whether the generation is
        certified. Asking through a second connection made that two decisions instead of one:

        * **The verdict came from a different snapshot than the write.** Between `resolve`
          returning CERTIFIED and the promotion committing, a concurrent `forget()` can rewrite the
          generation's `corpus_fingerprint` and make the calibration STALE. The gate would have
          approved a state that no longer existed by the time it was acted on. Read on the
          promoting connection, the generation row is the one the caller holds `FOR UPDATE`, so
          the check and the activation are atomic with respect to it.
        * **A connection was acquired while a tenant lock was held.** Measured: +1 backend for the
          width of the lock window, against `max_connections=100`. Under exhaustion the acquisition
          blocks, and it blocks holding the lock, so every other promotion and rollback for that
          tenant queues behind it rather than behind the work.

        Measured on the same probe: this is **not** a deadlock, which is what it looks like and is
        the thing worth stating. A plain `SELECT` does not wait on `FOR UPDATE` under MVCC, so the
        nested read completed and returned. The failure it produces is a wrong verdict under
        concurrency and a stall under load, not a hang.

        The caller owns the transaction, so no isolation level is set here: `promote`'s transaction
        has already written and cannot be made `REPEATABLE READ READ ONLY`. Reading at the caller's
        isolation is the point rather than a compromise, because the question being asked is about
        the caller's own transaction.
        """
        lineage = self._generation(conn, generation_id)
        row = conn.execute(
            f"SELECT {self._COLUMNS} FROM recall_calibrations WHERE tenant_id = %s "
            "AND generation_id = %s AND lifecycle_state = 'published'",
            (self.tenant_id, generation_id),
        ).fetchone()
        if row is None:
            local = conn.execute(
                "SELECT certified, lifecycle_state FROM recall_calibrations "
                "WHERE tenant_id = %s AND generation_id = %s "
                "ORDER BY created_at DESC, calibration_id DESC LIMIT 1",
                (self.tenant_id, generation_id),
            ).fetchone()
            if local is not None and not bool(local[0]):
                return CalibrationResolution(CalibrationStatus.UNCERTIFIED)
            if local is not None and str(local[1]) == LifecycleState.DRAFT:
                return CalibrationResolution(CalibrationStatus.DRAFT)
            if local is not None and str(local[1]) == LifecycleState.SUPERSEDED:
                return CalibrationResolution(CalibrationStatus.STALE)
            any_published = conn.execute(
                "SELECT 1 FROM recall_calibrations WHERE tenant_id = %s "
                "AND lifecycle_state = 'published' LIMIT 1",
                (self.tenant_id,),
            ).fetchone()
            return CalibrationResolution(
                CalibrationStatus.STALE if any_published else CalibrationStatus.MISSING
            )
        artifact = self._artifact(row)
        query_rows = self._query_set_rows(conn, artifact.query_set_digest)
        if (
            artifact.pipeline_fingerprint != lineage.pipeline_fingerprint
            or artifact.corpus_fingerprint != lineage.corpus_fingerprint
        ):
            return CalibrationResolution(CalibrationStatus.STALE)
        if query_rows is None:
            return CalibrationResolution(CalibrationStatus.STALE)
        _queries, digest = canonical_query_set(query_rows)
        if digest != artifact.query_set_digest:
            return CalibrationResolution(CalibrationStatus.STALE)
        if not artifact.certified:
            return CalibrationResolution(CalibrationStatus.UNCERTIFIED)
        return CalibrationResolution(CalibrationStatus.CERTIFIED, artifact)

    def list_records(self, generation_id: str | None = None) -> list[dict[str, Any]]:
        """Every calibration row of the tenant, newest first, or only one generation's.

        Rows are never deleted, so the unfiltered list grows with every fit; a caller that wants
        one generation passes `generation_id` and reads only that generation's rows.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT calibration_id, generation_id, lifecycle_state, certified, "
                "pipeline_fingerprint, corpus_fingerprint, query_set_digest, created_at, "
                "carry_forward FROM recall_calibrations WHERE tenant_id = %s "
                "AND (%s::text IS NULL OR generation_id = %s) "
                "ORDER BY created_at DESC, calibration_id",
                (self.tenant_id, generation_id, generation_id),
            ).fetchall()
        return [
            {
                "calibration_id": str(row[0]),
                "generation_id": row[1],
                "lifecycle_state": str(row[2]),
                "certified": bool(row[3]),
                "pipeline_fingerprint": row[4],
                "corpus_fingerprint": row[5],
                "query_set_digest": row[6],
                "created_at": _utc_isoformat(row[7]),
                # Listed, not left to `show`. After a chain of rebuilds the question an operator
                # has is which of these thresholds anyone actually measured, and a listing that
                # renders an inherited threshold identically to a fitted one cannot answer it.
                "threshold_was_measured_here": row[8] is None,
                "carried_forward_from": (
                    row[8].get("parent_calibration_id") if isinstance(row[8], Mapping) else None
                ),
                "corpus_delta": (
                    row[8].get("corpus_delta") if isinstance(row[8], Mapping) else None
                ),
            }
            for row in rows
        ]

    def show_record(self, calibration_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT to_jsonb(c) FROM recall_calibrations c "
                "WHERE tenant_id = %s AND calibration_id = %s",
                (self.tenant_id, calibration_id),
            ).fetchone()
        if row is None or not isinstance(row[0], Mapping):
            raise CalibrationNotFound(calibration_id)
        value = dict(row[0])
        for key, item in tuple(value.items()):
            if isinstance(item, datetime):
                value[key] = _utc_isoformat(item)
        # `to_jsonb` renders a timestamptz as a STRING already formatted in the session
        # TimeZone, so the loop above never sees it and `show` would otherwise disagree with
        # `list` about the same field. Re-render as the canonical UTC form: `created_at` is the
        # checksummed string, and `published_at` / `superseded_at` are set by `clock_timestamp()`
        # in `publish`, so all three arrive in whatever TimeZone the session happens to run in.
        for key in ("created_at", "published_at", "superseded_at"):
            item = value.get(key)
            if isinstance(item, str):
                value[key] = _utc_isoformat(datetime.fromisoformat(item))
        return value

    def measure_near_miss(
        self,
        calibration_id: str,
        queries: Sequence[Mapping[str, Any]],
        embedder: Embedder,
        *,
        record: bool = True,
    ) -> dict[str, Any]:
        """Score held-out near-miss questions against one calibration; report, never a gate.

        See `recall.near_miss_certificate`. The questions are scored on the calibration's OWN
        generation by the calibration sampling rule and compared with the answerable scores the
        artifact stores, at its threshold. With `record`, the report is appended to the audit
        ledger as one event bound to the calibration id: counts, scores and the set's digest,
        never a question's text. Nothing that decides serving reads that event.
        """
        from recall.near_miss_certificate import EVENT_TYPE, certificate_report

        artifact = self.get(calibration_id)
        labels, digest = canonical_query_set(queries)
        if any(item["answerable"] is not False for item in labels):
            raise CalibrationBindingError(
                "a near-miss set holds unanswerable questions only (answerable: false); an "
                "answerable entry would be scored as a near-miss"
            )
        with self._connect() as conn:
            lineage = self._generation(conn, artifact.generation_id)
        _require_runtime_embedder(PipelineIdentity.from_dict(lineage.pipeline_identity), embedder)
        _answerable, near_miss = self.score_query_set(artifact.generation_id, embedder, labels)
        report = certificate_report(
            artifact.scores["answerable"], near_miss, float(artifact.threshold)
        )
        report.update(generation_id=artifact.generation_id, query_set_digest=digest)
        if record:
            with self._connect() as conn:
                self._audit(conn, EVENT_TYPE, calibration_id, artifact.generation_id, report)
        return {"calibration_id": calibration_id, **report}

    def near_miss_report(self, calibration_id: str) -> dict[str, Any] | None:
        """The newest near-miss certificate recorded for this calibration id, or None.

        Validated against its own stored scores; raises ValueError for a row that disagrees with
        itself, so a reader can report it unavailable rather than print it.
        """
        from recall.near_miss_certificate import EVENT_TYPE, validated_certificate
        from recall.support import latest_calibration_event

        with self._connect() as conn:
            payload = latest_calibration_event(conn, self.tenant_id, EVENT_TYPE, calibration_id)
        return None if payload is None else validated_certificate(payload, calibration_id)

    def served_gap(
        self,
        calibration_id: str,
        embedder: Embedder,
        search: Callable[[str], tuple[float | None, bool]],
        *,
        tolerance: float | None = None,
    ) -> dict[str, Any]:
        """Compare served and exact best cosines over this calibration's labelled queries.

        See `recall.served_gap`. `search(query)` runs the search path being measured and returns
        `(best dense cosine among the hits it served, or None; abstained)`; it must search the
        calibration's own generation. The exact side is the calibration sampling rule. Read only:
        nothing is stored, and nothing that decides serving reads the result.
        """
        from recall.served_gap import DEFAULT_TOLERANCE, ServedGapRow, served_gap_report

        artifact = self.get(calibration_id)
        labels, digest = self.stored_query_set(artifact.query_set_digest)
        if digest != artifact.query_set_digest:
            raise CalibrationBindingError("stored labelled query set does not match its digest")
        with self._connect() as conn:
            lineage = self._generation(conn, artifact.generation_id)
        _require_runtime_embedder(PipelineIdentity.from_dict(lineage.pipeline_identity), embedder)
        answerable, unanswerable = self.score_query_set(artifact.generation_id, embedder, labels)
        # `measure_top_cosines` keeps each class in label order, so walking the labels in the same
        # order pairs every query with its own exact score.
        exact = {True: iter(answerable), False: iter(unanswerable)}
        rows = []
        for item in labels:
            served, abstained = search(str(item["query"]))
            rows.append(ServedGapRow(
                bool(item["answerable"]), next(exact[bool(item["answerable"])]), served, abstained
            ))
        report = served_gap_report(
            rows, float(artifact.threshold),
            DEFAULT_TOLERANCE if tolerance is None else tolerance,
        )
        report.update(calibration_id=calibration_id, generation_id=artifact.generation_id,
                      query_set_digest=digest)
        return report

    def export_bundle(self, calibration_id: str, path: str | Path) -> Path:
        artifact = self.get(calibration_id)
        with self._connect() as conn:
            rows = self._query_set_rows(conn, artifact.query_set_digest)
        if rows is None:
            raise CalibrationBindingError("calibration query set is missing")
        return write_calibration_bundle({"artifact": artifact.to_dict(), "query_set": rows}, path)

    def import_bundle(self, path: str | Path) -> str:
        """Store a calibration from a bundle file, and return its calibration id.

        Two input forms. A v2 bundle (`export_bundle`'s output: an `artifact` and its
        `query_set`, sealed by `bundle_checksum`) is verified and stored. Any other JSON object is
        read as a pre-v2 file holding a bare threshold and scale and stored `legacy_unbound`,
        which can never be published or resolved (`_import_legacy`).

        A v2 bundle is refused unless its bundle checksum and artifact checksum verify, its field
        types are the ones a stored row reads back, its statistics re-derive from its own scores,
        and its tenant, generation lineage, embedder identity and query-set digest match this
        database. Its lifecycle is NOT imported: it arrives as `draft` when certified and
        `rejected` otherwise, so it serves nothing until `publish`.

        ⛔ None of that authenticates the bundle: see `_require_scores_support_certification`.
        Every refusal is a `CalibrationError`, so the CLI names it.
        """
        try:
            raw = Path(path).read_bytes()
        except OSError as exc:
            raise CalibrationBindingError(f"cannot read calibration bundle {path}: {exc}") from exc
        try:
            data = json.loads(raw)
        except ValueError as exc:
            # ValueError, not only JSONDecodeError: bytes that are not UTF-8 raise
            # UnicodeDecodeError, which is a ValueError and not a JSONDecodeError.
            raise CalibrationBindingError(f"invalid calibration JSON: {exc}") from exc
        if not isinstance(data, Mapping) or "artifact" not in data:
            return self._import_legacy(data, raw)
        supplied = data.get("bundle_checksum")
        unsigned = {key: value for key, value in data.items() if key != "bundle_checksum"}
        if supplied != canonical_sha256(unsigned):
            raise CalibrationBindingError("calibration bundle checksum mismatch")
        artifact_raw = data.get("artifact")
        query_raw = data.get("query_set")
        if not isinstance(artifact_raw, Mapping) or not isinstance(query_raw, list):
            raise CalibrationBindingError("calibration bundle is malformed")
        if not all(isinstance(item, Mapping) for item in query_raw):
            raise CalibrationBindingError("bundle is malformed: query_set must hold objects")
        version = artifact_raw.get("artifact_version", 0)
        if type(version) is not int or version != ARTIFACT_VERSION:
            raise CalibrationBindingError("unsupported calibration artifact version")
        _require_bundle_field_types(artifact_raw)
        labels, digest = canonical_query_set(query_raw)
        if digest != artifact_raw.get("query_set_digest"):
            raise CalibrationBindingError("imported query set digest does not match artifact")
        if artifact_raw.get("tenant_id") != self.tenant_id:
            raise CalibrationBindingError("imported calibration belongs to another tenant")
        generation_id = str(artifact_raw.get("generation_id", ""))
        with self._connect() as conn:
            lineage = self._generation(conn, generation_id)
        identity = lineage.pipeline_identity
        if (
            artifact_raw.get("pipeline_fingerprint") != lineage.pipeline_fingerprint
            or artifact_raw.get("corpus_fingerprint") != lineage.corpus_fingerprint
        ):
            raise CalibrationBindingError("imported calibration does not match generation lineage")
        immutable_keys = CalibrationArtifactV2.__dataclass_fields__.keys()
        # Validated by name BEFORE construction: a missing key would otherwise surface as a bare
        # KeyError (from the separability_ci coercion) or TypeError (from the constructor), both
        # of which escape the CLI's `except CalibrationError` as an unnamed crash.
        optional_keys = {"artifact_version", "carry_forward", "lifecycle_state"}
        for key in immutable_keys:
            if key not in optional_keys and key not in artifact_raw:
                raise CalibrationBindingError(f"bundle is malformed: missing {key!r}")
        values = {key: artifact_raw[key] for key in immutable_keys if key in artifact_raw}
        try:
            values["separability_ci"] = tuple(values["separability_ci"])
        except TypeError:
            raise CalibrationBindingError(
                "bundle is malformed: separability_ci must be a pair of bounds"
            ) from None
        values["lifecycle_state"] = (
            LifecycleState.DRAFT.value if values.get("certified") else LifecycleState.REJECTED.value
        )
        try:
            artifact = CalibrationArtifactV2(**values)
        except (TypeError, ValueError) as exc:
            raise CalibrationBindingError(f"bundle is malformed: {exc}") from exc
        artifact.verify_checksum()
        # The checksum above only proves internal consistency; it is unkeyed, so a forger can
        # recompute it over any content. The recorded statistics and the certification verdict
        # must also FOLLOW from the scores the bundle carries.
        _require_scores_support_certification(artifact)
        generation_embedder = identity.get("embedder")
        if not isinstance(generation_embedder, Mapping) or canonical_sha256(
            artifact.embedder_identity
        ) != canonical_sha256(generation_embedder):
            raise CalibrationBindingError("imported calibration embedder identity does not match")
        with self._connect() as conn, conn.transaction():
            conn.execute(
                "INSERT INTO recall_calibration_query_sets "
                "(tenant_id, query_set_digest, queries, sample_count, created_by) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (self.tenant_id, digest, Jsonb(list(labels)), len(labels), self.actor),
            )
            try:
                conn.execute(
                    "INSERT INTO recall_calibrations "
                    "(tenant_id, calibration_id, generation_id, embedder_identity, "
                    "pipeline_fingerprint, corpus_fingerprint, query_set_digest, threshold, "
                    "scale, "
                    "separability, ci_low, ci_high, n_answerable, n_unanswerable, certified, "
                    # `carry_forward` is written here for the same reason it is checksummed: it
                    # is part of the immutable payload, so an import that dropped it would store
                    # a row whose checksum can never verify again, and the artifact would come
                    # back from `get` as a corruption rather than as an import.
                    "certification_reason, lifecycle_state, scores, created_at, created_by, "
                    "artifact_checksum, carry_forward) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (
                        self.tenant_id,
                        artifact.calibration_id,
                        generation_id,
                        Jsonb(dict(identity["embedder"])),
                        artifact.pipeline_fingerprint,
                        artifact.corpus_fingerprint,
                        digest,
                        artifact.threshold,
                        artifact.scale,
                        artifact.separability,
                        artifact.separability_ci[0],
                        artifact.separability_ci[1],
                        artifact.n_answerable,
                        artifact.n_unanswerable,
                        artifact.certified,
                        artifact.certification_reason,
                        artifact.lifecycle_state,
                        Jsonb(dict(artifact.scores)),
                        artifact.created_at,
                        artifact.created_by,
                        artifact.checksum,
                        Jsonb(dict(artifact.carry_forward))
                        if artifact.carry_forward is not None
                        else None,
                    ),
                )
            except psycopg.errors.UniqueViolation as exc:
                # A re-import of an id that is already stored. Named rather than allowed to
                # escape as a raw UniqueViolation, which the CLI cannot catch and which reads
                # as a crash instead of as the benign fact it is.
                raise CalibrationBindingError(
                    f"calibration already imported: {artifact.calibration_id}"
                ) from exc
            self._audit(conn, "calibration_imported", artifact.calibration_id, generation_id)
        return artifact.calibration_id

    def _import_legacy(self, data: Any, raw: bytes) -> str:
        if not isinstance(data, Mapping):
            raise CalibrationBindingError("legacy calibration JSON must be an object")
        try:
            threshold = float(data["threshold"])
            scale = float(data["scale"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationBindingError("legacy calibration is malformed") from exc
        # The range check mirrors the table's CHECK (threshold BETWEEN -1.0 AND 1.0): without it
        # a legacy file with a cosine outside that range reaches the INSERT and surfaces as a raw
        # CheckViolation instead of a named refusal at the boundary.
        if (
            not math.isfinite(threshold)
            or not math.isfinite(scale)
            or scale <= 0
            or not -1.0 <= threshold <= 1.0
        ):
            raise CalibrationBindingError("legacy calibration contains invalid numeric values")
        calibration_id = _id("legacy_cal")
        with self._connect() as conn, conn.transaction():
            conn.execute(
                "INSERT INTO recall_calibrations "
                "(tenant_id, calibration_id, embedder_identity, threshold, scale, certified, "
                "certification_reason, lifecycle_state, scores, created_by, artifact_checksum) "
                "VALUES (%s,%s,%s,%s,%s,false,%s,'legacy_unbound','{}'::jsonb,%s,%s)",
                (
                    self.tenant_id,
                    calibration_id,
                    Jsonb({"model": str(data.get("embedder", "unknown"))}),
                    threshold,
                    scale,
                    "legacy artifact has no tenant, generation, pipeline, corpus, or query binding",
                    self.actor,
                    hashlib.sha256(raw).hexdigest(),
                ),
            )
            self._audit(conn, "calibration_imported_legacy_unbound", calibration_id, None)
        return calibration_id
