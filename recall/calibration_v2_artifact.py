"""The calibration v2 artifact: its record, its states and errors, and the checks it runs on itself.

Split out of `recall.calibration_v2`, which re-exports every name here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any


from recall.calibration import Calibration
from recall.lineage import canonical_json, canonical_sha256
from recall.errors import RecallError

ARTIFACT_VERSION = 2



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

#: How far a re-scored query's best cosine may sit from the score a bundle recorded for it, on
#: import. Not a float-noise bound: embedding the same query again is not deterministic on a hosted
#: model, measured moving a best cosine by up to 0.0051 between calls (`recall.served_gap`), so
#: an honest bundle re-scores within about that. It is also the price of the check: a forger who
#: must stay within it can move each score, and so a fitted threshold, by at most this much.
_RESCORE_TOLERANCE = 0.01

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

