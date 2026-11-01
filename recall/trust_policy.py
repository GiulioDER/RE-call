"""Trust policy: strict by default, degradation only when explicitly asked for.

The behaviour this module exists to prevent
------------------------------------------

Before it, `trusted_search` resolved calibration and then did this::

    cal = calibration or _UNCALIBRATED   # the 0.50 floor

So a generation whose calibration was **missing**, **stale**, or **uncertified** still ran the
retrieval, still returned corpus text, and still stamped every hit with a `verdict` computed from
a threshold nobody ever certified for that corpus. The caller saw a normal-looking answer. Nothing
in the payload said the number behind it was invented.

Strict mode refuses instead, and refuses *before* retrieval runs, so a refusal cannot carry corpus
bytes it never fetched. `DEPENDENCY_UNAVAILABLE` refuses under every policy, because the gate could
not run at all. Development mode still retrieves, but has to say so in band: `trust_state`
is `degraded` and the result is never `calibrated`. With no calibration at all, every hit is
`unverified` and no abstention decision is claimed. With a calibration that is not
certified-bound (one the caller passed in, or a certified artifact while the dependency graph is
not ready), the verdicts and any abstention computed under it are kept.

Why the failure codes are an API
--------------------------------

Callers automate on these. A code is a promise: `CALIBRATION_STALE` will keep meaning "a certified
artifact exists but no longer binds to this generation's lineage" across versions. Renaming one is
a breaking change, which is why `TestFailureCodes` pins the exact spelling of every code rather
than just iterating the enum.

The distinction the codes are really carrying is requirement 13: a working gate that found nothing
is *not* the same event as a gate that could not run. The first is an answer. The second is an
outage wearing an answer's clothes, and it is the one that quietly corrupts downstream decisions.
Every code in here is the second kind, which is why every `advice` string says so.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from recall.calibration_v2 import CalibrationStatus
from recall.errors import RecallError


class TrustMode(StrEnum):
    """How the trust gate behaves when it cannot certify an answer."""

    #: Refuse. The production default, for both the library and the network service.
    STRICT = "strict"
    #: Retrieve, but degrade explicitly and never claim a trustworthy decision.
    DEVELOPMENT = "development"


class TrustFailureCode(StrEnum):
    """Stable, machine-readable reasons a trustworthy answer was not possible."""

    #: No active generation, or its physical index is not usable.
    INDEX_NOT_READY = "INDEX_NOT_READY"
    #: The runtime embedder (model, dimension or profile) is not the one the active generation
    #: was built with. A fingerprint that drifted from the artifact is `CALIBRATION_STALE`.
    LINEAGE_MISMATCH = "LINEAGE_MISMATCH"
    #: No calibration artifact exists for this tenant and generation.
    CALIBRATION_MISSING = "CALIBRATION_MISSING"
    #: An artifact exists but was never certified, so its threshold is not usable.
    CALIBRATION_UNCERTIFIED = "CALIBRATION_UNCERTIFIED"
    #: A certified artifact exists but no longer binds to the current lineage.
    CALIBRATION_STALE = "CALIBRATION_STALE"
    #: A dependency the trust gate needs (database, control plane) could not be reached.
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    #: The opt-in dependency invalidation projection is absent for the active generation.
    DEPENDENCY_GRAPH_NOT_READY = "DEPENDENCY_GRAPH_NOT_READY"


class TrustState(StrEnum):
    """The trust standing of a result, carried in band so it cannot be lost in transit."""

    #: A certified artifact exactly bound to this generation backed the verdicts.
    TRUSTED = "trusted"
    #: Development mode returned corpus text with no certified, generation-bound threshold behind
    #: it. Verdicts are voided when no calibration exists, and kept when one exists but is not
    #: certified-bound (passed in by the caller, or certified while the dependency graph is not
    #: ready).
    DEGRADED = "degraded"
    #: The gate refused: strict mode, or `DEPENDENCY_UNAVAILABLE` under any policy. No corpus text
    #: was returned.
    REFUSED = "refused"


#: Which calibration statuses can never back a trustworthy verdict, and what to call each.
#:
#: `DRAFT` is here deliberately. A draft artifact is certified in the statistical sense but has not
#: been published, so it is not the artifact an operator chose to serve; treating it as usable
#: would let an unpublished measurement silently become the runtime threshold.
_STATUS_CODES: dict[CalibrationStatus, TrustFailureCode] = {
    CalibrationStatus.MISSING: TrustFailureCode.CALIBRATION_MISSING,
    CalibrationStatus.STALE: TrustFailureCode.CALIBRATION_STALE,
    CalibrationStatus.SUPERSEDED: TrustFailureCode.CALIBRATION_STALE,
    CalibrationStatus.UNCERTIFIED: TrustFailureCode.CALIBRATION_UNCERTIFIED,
    CalibrationStatus.REJECTED: TrustFailureCode.CALIBRATION_UNCERTIFIED,
    CalibrationStatus.DRAFT: TrustFailureCode.CALIBRATION_UNCERTIFIED,
    CalibrationStatus.LEGACY_UNBOUND: TrustFailureCode.CALIBRATION_UNCERTIFIED,
}


def code_for_status(status: CalibrationStatus | str) -> TrustFailureCode | None:
    """Map a calibration status to its stable failure code, or None when it is certified.

    Accepts the raw string form too, because `TrustedResult.calibration_status` is a `str` on the
    wire and callers should not have to re-parse it into an enum to ask this question.
    """
    if isinstance(status, str) and not isinstance(status, CalibrationStatus):
        try:
            status = CalibrationStatus(status)
        except ValueError:
            # An unrecognised status is not evidence of certification. Fail closed.
            return TrustFailureCode.CALIBRATION_UNCERTIFIED
    if status is CalibrationStatus.CERTIFIED:
        return None
    return _STATUS_CODES.get(status, TrustFailureCode.CALIBRATION_UNCERTIFIED)


#: Remedy text per code. Every entry states that no trustworthy decision was possible, because
#: that is the fact a caller must not confuse with "the gate ran and found nothing".
_ADVICE: dict[TrustFailureCode, str] = {
    TrustFailureCode.INDEX_NOT_READY: (
        "No trustworthy decision was possible: this tenant has no active, usable generation, so "
        "the trust gate never ran. Build and promote a generation, then retry."
    ),
    TrustFailureCode.LINEAGE_MISMATCH: (
        "No trustworthy decision was possible: the runtime embedder (model, dimension or "
        "profile) is not the one the active generation was built with, so its cosines are not "
        "comparable to any threshold and the trust gate never ran. Serve with the generation's "
        "embedder, or build and promote a generation for this one."
    ),
    TrustFailureCode.CALIBRATION_MISSING: (
        "No trustworthy decision was possible: no calibration artifact is bound to this tenant "
        "and generation, so there is no certified threshold and the trust gate never ran. "
        "Calibrate against a labelled query set and publish the artifact."
    ),
    TrustFailureCode.CALIBRATION_UNCERTIFIED: (
        "No trustworthy decision was possible: the calibration for this generation is not a "
        "certified, published artifact bound to it (uncertified, rejected, a draft, a threshold "
        "passed in unbound, or of unknown status), so its threshold is not usable and the trust "
        "gate never ran. Certify and publish an artifact for this generation."
    ),
    TrustFailureCode.CALIBRATION_STALE: (
        "No trustworthy decision was possible: a certified calibration exists but no longer binds "
        "to this generation's lineage (a rebuild or a privacy erasure changed the corpus "
        "fingerprint), so the trust gate never ran. Recalibrate against the current generation."
    ),
    TrustFailureCode.DEPENDENCY_UNAVAILABLE: (
        "No trustworthy decision was possible: a dependency the trust gate needs could not be "
        "reached, so the gate never ran. This is an outage, not an empty result. Retry once the "
        "dependency is healthy."
    ),
    TrustFailureCode.DEPENDENCY_GRAPH_NOT_READY: (
        "No trustworthy decision was possible: dependency invalidation is enabled but the active "
        "generation has no ready dependency projection. Rebuild the generation graph and retry."
    ),
}


@dataclass(frozen=True)
class TrustPolicy:
    """Whether an uncertifiable answer is refused or degraded.

    Strict is the default on purpose. A policy object that defaults to permissive would make every
    caller that forgets to pass one fail open, which is the exact class of bug this session closes.
    """

    mode: TrustMode = TrustMode.STRICT

    def __post_init__(self) -> None:
        # `strict` tests identity, so a plain string mode, even "strict", would otherwise read as
        # not strict and take the degrading path. Coercing raises on an unknown value instead.
        object.__setattr__(self, "mode", TrustMode(self.mode))

    @property
    def strict(self) -> bool:
        return self.mode is TrustMode.STRICT

    @classmethod
    def strict_policy(cls) -> TrustPolicy:
        return cls(mode=TrustMode.STRICT)

    @classmethod
    def development(cls) -> TrustPolicy:
        """Explicit opt-in for local workflows. Never reachable by omission."""
        return cls(mode=TrustMode.DEVELOPMENT)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None, /) -> TrustPolicy:
        """Resolve from `RECALL_TRUST_MODE`, defaulting to strict.

        The value is matched after `strip().lower()`, so `development`, `Development` and
        `  DEVELOPMENT  ` all relax the gate. Anything else is strict, and that specifically
        includes a **misspelling**: `developmnet` stays strict. A typo must not be the thing that
        opens the gate, while a capital letter is not a typo and refusing it only produces a
        confusing strict server that the operator believes is relaxed.

        This docstring previously said "the exact string", which the code has never done. The
        distinction is worth stating precisely rather than strictly, because someone will read this
        to decide whether their configuration is safe.

        Raises ValueError when development is asked for under `RECALL_ENV=production`: a relaxed
        gate never errors, so in production it would stamp `degraded` on every answer unseen.
        """
        import os

        from recall._env import env_is_production

        source = os.environ if env is None else env
        raw = str(source.get("RECALL_TRUST_MODE", "")).strip().lower()
        if raw == TrustMode.DEVELOPMENT.value and env_is_production(env):
            raise ValueError(
                "RECALL_TRUST_MODE=development is refused when RECALL_ENV=production; unset "
                "RECALL_TRUST_MODE (strict is the default) or RECALL_ENV"
            )
        if raw not in {"", *TrustMode}:
            # Still strict, but said: otherwise every search refuses and nothing tells the
            # operator that the value they set was ignored. The value itself is not logged.
            logging.getLogger("recall.trust_policy").warning(
                "RECALL_TRUST_MODE is not 'strict' or 'development'; the strict trust gate is "
                "in force"
            )
        return cls(mode=TrustMode.DEVELOPMENT if raw == TrustMode.DEVELOPMENT.value else
                   TrustMode.STRICT)


#: The dependency invalidation modes, from an argument, the store, or RECALL_DEPENDENCY_INVALIDATION.
DEPENDENCY_MODES = frozenset({"off", "enforce"})


def dependency_mode_from_env(env: Mapping[str, str]) -> str:
    """`RECALL_DEPENDENCY_INVALIDATION`, normalised; raises ValueError for anything but the modes.

    Normalised like RECALL_ATOMIC_RESCUE_MODE, and named in the error, because an operator's
    `Enforce` would otherwise pass startup and fail every search as a code bug.
    """
    mode = str(env.get("RECALL_DEPENDENCY_INVALIDATION", "off")).strip().lower()
    if mode not in DEPENDENCY_MODES:
        raise ValueError("RECALL_DEPENDENCY_INVALIDATION must be 'off' or 'enforce'")
    return mode


def validate_trust_env(env: Mapping[str, str]) -> None:
    """Refuse a trust configuration at startup that every search would otherwise refuse later."""
    TrustPolicy.from_env(dict(env))
    dependency_mode_from_env(env)


class TrustRefusal(RecallError):
    """The trust gate refused to answer. Carries identity and a code, never corpus bytes.

    Strict mode refuses on every failure code; `DEPENDENCY_UNAVAILABLE` refuses under every
    policy, development included, because the gate could not run at all.

    The payload is built only from fields the *system* controls: the code, the calibration status,
    and the tenant and generation identifiers. Chunk text, source text, previews, scores, and the
    query itself are all absent by construction rather than by filtering, so there is no sanitiser
    to forget to call.

    The query is excluded too. It is caller data rather than corpus data, but echoing it into an
    exception puts it into every log line and traceback that touches the refusal, which is exactly
    what the observability rules say not to do.
    """

    def __init__(
        self,
        *,
        code: TrustFailureCode,
        calibration_status: str,
        tenant_id: str | None,
        generation_id: str | None,
        calibration_id: str | None = None,
        pipeline_fingerprint: str | None = None,
        corpus_fingerprint: str | None = None,
        query_set_digest: str | None = None,
    ) -> None:
        self.code = code
        self.calibration_status = calibration_status
        self.tenant_id = tenant_id
        self.generation_id = generation_id
        self.calibration_id = calibration_id
        self.pipeline_fingerprint = pipeline_fingerprint
        self.corpus_fingerprint = corpus_fingerprint
        self.query_set_digest = query_set_digest
        super().__init__(
            f"{code.value}: refused in strict trust mode "
            f"(tenant={tenant_id!r}, generation={generation_id!r}, "
            f"calibration_status={calibration_status!r})"
        )

    @property
    def trust_state(self) -> TrustState:
        return TrustState.REFUSED

    @property
    def advice(self) -> str:
        return _ADVICE[self.code]

    def to_dict(self) -> dict[str, Any]:
        """Wire form. Deliberately has no hits, chunks, text, or preview key at all."""
        return {
            "code": self.code.value,
            "trust_state": TrustState.REFUSED.value,
            "calibration_status": self.calibration_status,
            "calibrated": False,
            "tenant_id": self.tenant_id,
            "generation_id": self.generation_id,
            "calibration_id": self.calibration_id,
            "pipeline_fingerprint": self.pipeline_fingerprint,
            "corpus_fingerprint": self.corpus_fingerprint,
            "query_set_digest": self.query_set_digest,
            "advice": self.advice,
        }


__all__ = [
    "TrustFailureCode",
    "TrustMode",
    "TrustPolicy",
    "TrustRefusal",
    "TrustState",
    "code_for_status",
]
