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
bytes it never fetched. Development mode still retrieves, but has to say so in band: `trust_state`
is `degraded`, every hit is `unverified`, and no abstention decision is claimed.

Why the failure codes are an API
--------------------------------

Callers automate on these. A code is a promise: `CALIBRATION_STALE` will keep meaning "a certified
artifact exists but no longer binds to this generation's lineage" across versions. Renaming one is
a breaking change, which is why `TestFailureCodes` pins the exact spelling of all six rather than
just iterating the enum.

The distinction the codes are really carrying is requirement 13: a working gate that found nothing
is *not* the same event as a gate that could not run. The first is an answer. The second is an
outage wearing an answer's clothes, and it is the one that quietly corrupts downstream decisions.
Every code in here is the second kind, which is why every `advice` string says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from recall.calibration_v2 import CalibrationStatus


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
    #: The generation's pipeline or corpus fingerprint is not what the artifact was bound to.
    LINEAGE_MISMATCH = "LINEAGE_MISMATCH"
    #: No calibration artifact exists for this tenant and generation.
    CALIBRATION_MISSING = "CALIBRATION_MISSING"
    #: An artifact exists but was never certified, so its threshold is not usable.
    CALIBRATION_UNCERTIFIED = "CALIBRATION_UNCERTIFIED"
    #: A certified artifact exists but no longer binds to the current lineage.
    CALIBRATION_STALE = "CALIBRATION_STALE"
    #: A dependency the trust gate needs (database, control plane) could not be reached.
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"


class TrustState(StrEnum):
    """The trust standing of a result, carried in band so it cannot be lost in transit."""

    #: A certified artifact exactly bound to this generation backed the verdicts.
    TRUSTED = "trusted"
    #: Development mode returned corpus text with no certified threshold behind it.
    DEGRADED = "degraded"
    #: Strict mode refused. No corpus text was returned.
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
        "No trustworthy decision was possible: the active generation's pipeline or corpus "
        "fingerprint does not match what the calibration was bound to, so the trust gate never "
        "ran. Recalibrate against the current generation."
    ),
    TrustFailureCode.CALIBRATION_MISSING: (
        "No trustworthy decision was possible: no calibration artifact is bound to this tenant "
        "and generation, so there is no certified threshold and the trust gate never ran. "
        "Calibrate against a labelled query set and publish the artifact."
    ),
    TrustFailureCode.CALIBRATION_UNCERTIFIED: (
        "No trustworthy decision was possible: the calibration artifact for this generation was "
        "not certified, so its threshold is not usable and the trust gate never ran. The evidence "
        "is retained as a rejected artifact; certify and publish a replacement."
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
}


@dataclass(frozen=True)
class TrustPolicy:
    """Whether an uncertifiable answer is refused or degraded.

    Strict is the default on purpose. A policy object that defaults to permissive would make every
    caller that forgets to pass one fail open, which is the exact class of bug this session closes.
    """

    mode: TrustMode = TrustMode.STRICT

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
        """
        import os

        source = os.environ if env is None else env
        raw = str(source.get("RECALL_TRUST_MODE", "")).strip().lower()
        return cls(mode=TrustMode.DEVELOPMENT if raw == TrustMode.DEVELOPMENT.value else
                   TrustMode.STRICT)


class TrustRefusal(Exception):
    """Strict mode refused to answer. Carries identity and a code, never corpus bytes.

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
