"""Fail-closed enterprise readiness checks."""
from __future__ import annotations

from dataclasses import dataclass

from recall.calibration import Calibration
from recall.control_plane import ControlPlane
from recall.embeddings import Embedder, embedding_profile, embedding_profile_id
from recall.store import PgVectorStore


@dataclass(frozen=True)
class ReadinessResult:
    ready: bool
    degraded: bool
    failures: tuple[str, ...]
    warnings: tuple[str, ...]


def check_enterprise_readiness(
    store: PgVectorStore,
    embedder: Embedder,
    *,
    control_plane: ControlPlane | None = None,
    calibration: Calibration | None = None,
    allow_legacy_profile: bool = False,
) -> ReadinessResult:
    """Check model identity, generation metadata, indexes, RLS, and calibration identity."""
    failures: list[str] = []
    warnings: list[str] = []
    profile = embedding_profile(embedder)
    if profile.dimension != embedder.dim:
        failures.append("embedding profile dimension does not match the runtime embedder")
    if profile.artifact_digest == "legacy-unverified" and not allow_legacy_profile:
        failures.append("model artifact is not pinned by an immutable digest")
    if control_plane is not None:
        route = control_plane.route(store.tenant)
        if route is None:
            failures.append("tenant route is missing")
        else:
            if route.active.generation_id != store.generation_id:
                failures.append("acquired store does not match the active tenant generation")
            if route.active.embedding_profile != embedding_profile_id(embedder):
                failures.append("active generation embedding profile does not match runtime")
            if route.active.dimension != embedder.dim:
                failures.append("active generation vector dimension does not match runtime")
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
    if calibration is None:
        warnings.append("no profile matched calibration is loaded")
    elif calibration.embedder != embedding_profile_id(embedder):
        failures.append("calibration identity does not match the embedding profile")
    elif calibration.certified is not True:
        warnings.append("calibration is present but not certified")
    return ReadinessResult(
        ready=not failures,
        degraded=bool(warnings),
        failures=tuple(failures),
        warnings=tuple(warnings),
    )
