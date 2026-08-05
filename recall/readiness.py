"""Fail-closed enterprise readiness checks.

Readiness is deliberately two questions that used to be one:

- `process_readiness` — can this process serve *anyone*? Depends only on SHARED dependencies
  (database, object store). This is what a Kubernetes readiness probe should call.
- `tenant_readiness` — can this process serve *this tenant*? Depends on that tenant's generation
  and calibration.

Folding the second into the first is a fleet-wide outage waiting to happen: one tenant with a
stale calibration would fail the probe, the pod would leave the Service endpoints, and every
healthy tenant would lose that capacity. A per-tenant fault must cost exactly one tenant, so
`process_readiness` *reports* unready tenants without ever becoming unready because of them.

Neither function returns corpus text. They answer questions about the system, and a status
endpoint that could be induced to echo a chunk would be a retrieval side channel around the
trust gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from recall.calibration import Calibration
from recall.control_plane import SERVABLE_ACTIVE_STATES, ControlPlane
from recall.embeddings import Embedder, embedding_profile, embedding_profile_id
from recall.store import PgVectorStore
from recall.trust_policy import TrustFailureCode, TrustState, code_for_status


@dataclass(frozen=True)
class ReadinessResult:
    ready: bool
    degraded: bool
    failures: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class DependencyStatus:
    """One shared dependency. `name` is an operator-facing label, never a connection string."""

    name: str
    healthy: bool
    detail: str = ""


@dataclass(frozen=True)
class TenantReadiness:
    """Whether one tenant can be served, and the stable code when it cannot."""

    tenant_id: str
    ready: bool
    trust_state: str
    failure_code: str | None
    generation_id: str | None
    calibration_id: str | None
    calibration_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "ready": self.ready,
            "trust_state": self.trust_state,
            "failure_code": self.failure_code,
            "generation_id": self.generation_id,
            "calibration_id": self.calibration_id,
            "calibration_status": self.calibration_status,
        }


@dataclass(frozen=True)
class ProcessReadiness:
    """Whether the PROCESS can serve at all. Never false because of one tenant."""

    ready: bool
    failure_code: str | None
    unavailable: tuple[str, ...]
    tenants_ready: tuple[str, ...]
    tenants_unready: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "failure_code": self.failure_code,
            "unavailable": list(self.unavailable),
            "tenants_ready": list(self.tenants_ready),
            "tenants_unready": list(self.tenants_unready),
        }


def tenant_readiness(
    tenant_id: str,
    *,
    generation_id: str | None,
    calibration_status: str,
    calibration_id: str | None = None,
) -> TenantReadiness:
    """Trust readiness for one tenant, with no corpus access and no corpus text in the answer.

    The two failure conditions are checked in the order an operator would act on them. A tenant
    with no active generation is `INDEX_NOT_READY` regardless of its calibration state, because
    telling someone to recalibrate against a generation that does not exist is useless advice.

    `calibration_id` is required even when the status says `certified`: a status string on its own
    is a claim, not evidence, and the same conjunction guards `TrustedResult.calibrated`.
    """
    if not generation_id:
        return TenantReadiness(
            tenant_id=tenant_id,
            ready=False,
            trust_state=TrustState.REFUSED.value,
            failure_code=TrustFailureCode.INDEX_NOT_READY.value,
            generation_id=None,
            calibration_id=calibration_id,
            calibration_status=calibration_status,
        )
    code = code_for_status(calibration_status)
    if code is None and calibration_id is None:
        code = TrustFailureCode.CALIBRATION_MISSING
    ready = code is None
    return TenantReadiness(
        tenant_id=tenant_id,
        ready=ready,
        trust_state=TrustState.TRUSTED.value if ready else TrustState.REFUSED.value,
        failure_code=None if ready else code.value,  # type: ignore[union-attr]
        generation_id=generation_id,
        calibration_id=calibration_id,
        calibration_status=calibration_status,
    )


def process_readiness(
    dependencies: list[DependencyStatus],
    *,
    tenants: list[TenantReadiness] | None = None,
) -> ProcessReadiness:
    """Whether the process can serve at all.

    `tenants` is reported, never counted. That asymmetry IS requirement 12: a tenant-scoped fault
    is visible to an operator here, but it can never flip `ready` to False and evict a pod that is
    still serving every other tenant correctly.
    """
    unavailable = tuple(dep.name for dep in dependencies if not dep.healthy)
    states = tenants or []
    return ProcessReadiness(
        ready=not unavailable,
        failure_code=TrustFailureCode.DEPENDENCY_UNAVAILABLE.value if unavailable else None,
        unavailable=unavailable,
        tenants_ready=tuple(t.tenant_id for t in states if t.ready),
        tenants_unready=tuple(t.tenant_id for t in states if not t.ready),
    )


def check_enterprise_readiness(
    store: PgVectorStore,
    embedder: Embedder,
    *,
    control_plane: ControlPlane | None = None,
    calibration: Calibration | None = None,
    allow_legacy_profile: bool = False,
) -> ReadinessResult:
    """Check model identity, generation metadata, both schema ledgers, indexes, RLS, calibration.

    Every branch here is a FAILURE except the two calibration warnings, and the split is
    deliberate. A failure means the process must not serve: it cannot prove which model wrote the
    vectors it is about to search, or which schema they live in, or that another tenant cannot
    read them. A warning means retrieval still works and one capability is degraded.

    The calibration branches are warnings for a reason worth stating, because the opposite reading
    is tempting. An uncertified calibration does not make retrieval wrong; it makes ABSTENTION
    unreliable, because the abstention threshold is what a calibration certifies. Refusing to
    serve would trade a degraded capability for no capability. What must never happen is the third
    option: quietly using the uncertified numbers as if they were certified. This function reports
    and returns; it holds no reference through which it could adjust a threshold, and
    `tests/test_enterprise_readiness.py` asserts that the embedder and store are unchanged across
    the call, so "reports degraded" cannot drift into "silently modifies retrieval".
    """
    failures: list[str] = []
    warnings: list[str] = []
    profile = embedding_profile(embedder)
    if profile.dimension != embedder.dim:
        failures.append("embedding profile dimension does not match the runtime embedder")
    if profile.artifact_digest == "legacy-unverified" and not allow_legacy_profile:
        failures.append("model artifact is not pinned by an immutable digest")
    if control_plane is not None:
        try:
            route = control_plane.route(store.tenant)
        except Exception as exc:
            # An unreachable control plane used to propagate out of this function, so the caller
            # got a traceback instead of `ready=False`. Startup treats a readiness FAILURE and an
            # exception differently, and "the database is down" is the readiness question, not an
            # unexpected condition.
            route = None
            failures.append(f"control plane query failed: {type(exc).__name__}")
        else:
            if route is None:
                failures.append("tenant route is missing")
        if route is not None:
            if route.active.generation_id != store.generation_id:
                failures.append("acquired store does not match the active tenant generation")
            if route.active.embedding_profile != embedding_profile_id(embedder):
                failures.append("active generation embedding profile does not match runtime")
            if route.active.dimension != embedder.dim:
                failures.append("active generation vector dimension does not match runtime")
            if route.active.state not in SERVABLE_ACTIVE_STATES:
                failures.append(
                    f"active generation is {route.active.state} and cannot serve requests"
                )
        # The SECOND ledger. `check_schema` below covers `recall_schema_migrations`, which is
        # scoped per chunk table; `recall_schema_versions` is database-global and nothing checked
        # it, so an enterprise process could boot against a control plane that was behind, or
        # whose applied SQL no longer matched the bytes this package ships. The two ledgers stay
        # separate by decision (docs/MIGRATIONS.md); readiness is where that decision is made
        # safe, by verifying both rather than one.
        try:
            ledger = control_plane.ledger_state()
        except Exception as exc:
            # `str(exc)` included: these are catalog/permission errors naming a TABLE, not a
            # DSN, and "InsufficientPrivilege" alone does not tell an operator which GRANT
            # is missing. `redacted_dsn` guards the connection-string case elsewhere.
            failures.append(
                f"control plane ledger query failed: {type(exc).__name__}: {exc}"
            )
        else:
            if not ledger.current:
                failures.append(f"control plane schema is not current: {ledger.describe()}")
    try:
        facts = store.readiness_facts()
    except Exception as exc:
        failures.append(f"database readiness query failed: {type(exc).__name__}")
    else:
        if not facts["rls_enabled"] or not store.check_rls_effective():
            failures.append("row level security is ineffective for the runtime database role")
        if not facts["indexes_valid"]:
            failures.append("required HNSW or full text index is missing or invalid")
        if facts["dimension"] != embedder.dim:
            failures.append("physical table vector dimension does not match runtime")
        if facts["rows"] and facts["rows_without_profile"] and not allow_legacy_profile:
            failures.append("table has rows without an explicit embedding profile")
    try:
        store.check_schema()
    except Exception as exc:
        # Same reasoning: SchemaTooOld names the pending migration versions, and
        # MigrationChecksumMismatch names the file. Dropping that left the operator with
        # a class name and no next step.
        failures.append(f"chunk table schema is not current: {type(exc).__name__}: {exc}")
    if calibration is None:
        warnings.append(
            "no profile matched calibration is loaded, so abstention capability is degraded"
        )
    elif calibration.embedder != embedding_profile_id(embedder):
        failures.append("calibration identity does not match the embedding profile")
    elif calibration.certified is not True:
        warnings.append(
            "calibration is present but not certified, so abstention capability is degraded"
        )
    return ReadinessResult(
        ready=not failures,
        degraded=bool(warnings),
        failures=tuple(failures),
        warnings=tuple(warnings),
    )
