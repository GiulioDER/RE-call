"""Serving adapter from versioned route plans to bounded specialist federation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from threading import Lock
from typing import Any

from recall.embeddings import Embedder, embedding_profile_id
from recall.federation import (
    FederationConfig,
    FederationLeg,
    FederationResult,
    federate,
)
from recall.retrieval_plan import (
    DEFAULT_RETRIEVAL_ROUTE_ID,
    RetrievalPlan,
    RetrievalPlanError,
    TenantIdentity,
)
from recall.trust import decision_state_for
from recall.types import TrustedResult
from recall.security_policy import AccessContext
from recall_mcp.factories import make_profile_embedder
from recall_mcp.retrieval import _Retrieval, _retrieve_trusted
from recall_mcp.stores import StoreRegistry


RetrieveTrusted = Callable[..., _Retrieval]


@dataclass
class FederationExecution:
    """The selected primary execution boundary and optional bounded merge callback."""

    store: Any
    embedder: Embedder
    retrieve_trusted: RetrieveTrusted | None
    diagnostics: dict[str, object]


def _identity_for(tenant: str, identity: TenantIdentity | None) -> TenantIdentity:
    if identity is None:
        raise RetrievalPlanError(
            f"retrieval plan tenant {tenant!r} has no resolved generation identity"
        )
    required = {
        "generation": identity.generation,
        "embedding_profile": identity.embedding_profile,
        "calibration": identity.calibration,
        "calibration_status": identity.calibration_status,
        "trust_state": identity.trust_state,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RetrievalPlanError(
            f"retrieval plan tenant {tenant!r} has unresolved identity fields: "
            + ", ".join(missing)
        )
    if identity.calibration_status != "certified" or identity.trust_state != "trusted":
        raise RetrievalPlanError(
            f"retrieval plan tenant {tenant!r} is not certified for federation"
        )
    return identity


def prepare_federated_execution(
    plan: RetrievalPlan,
    *,
    config: FederationConfig,
    registry: StoreRegistry | None,
    current_store: Any,
    current_embedder: Embedder,
    env: Mapping[str, str],
    multimodal_enabled: bool = False,
    embedder_cache: dict[str, Embedder] | None = None,
    embedder_cache_lock: Lock | None = None,
    make_embedder_fn: Callable[..., Embedder] | None = None,
    retrieve_trusted_fn: RetrieveTrusted = _retrieve_trusted,
) -> FederationExecution:
    """Bind a validated route plan to isolated stores and profile specific embedders.

    The compatibility route remains the existing single tenant path. Configured specialist routes
    require complete generation, profile, calibration, and trust identities before a request can
    execute. This prevents a plan from becoming an implicit cross tenant fallback when one leg is
    not independently servable.
    """

    empty: dict[str, object] = {}
    if config.mode == "off" or plan.route_id == DEFAULT_RETRIEVAL_ROUTE_ID:
        return FederationExecution(current_store, current_embedder, None, empty)
    if registry is None:
        raise RetrievalPlanError("configured federation requires an authenticated store registry")
    if len(plan.selected_legs) > config.max_legs:
        raise RetrievalPlanError(
            f"retrieval plan fanout {len(plan.selected_legs)} exceeds federation max "
            f"{config.max_legs}"
        )

    cache = embedder_cache if embedder_cache is not None else {}
    cache_lock = embedder_cache_lock or Lock()
    stores: dict[str, Any] = {}
    embedders: dict[str, Embedder] = {}
    identities: dict[str, TenantIdentity] = {}
    for leg in plan.selected_legs:
        identity = _identity_for(leg.tenant, leg.identity)
        assert identity.embedding_profile is not None
        if identity.embedding_profile.startswith("voyage-multimodal") and not multimodal_enabled:
            raise RetrievalPlanError("multimodal federation requires explicit feature enablement")
        identities[leg.tenant] = identity
        stores[leg.tenant] = registry._get_federation_store(
            leg.tenant,
            identity.embedding_profile,
        )
        if embedding_profile_id(current_embedder) == identity.embedding_profile:
            embedders[leg.tenant] = current_embedder
            continue
        with cache_lock:
            embedder = cache.get(identity.embedding_profile)
            if embedder is None:
                if make_embedder_fn is None:
                    embedder = make_profile_embedder(identity.embedding_profile, env=dict(env))
                else:
                    embedder = make_embedder_fn(identity.embedding_profile, dict(env))
                cache[identity.embedding_profile] = embedder
        embedders[leg.tenant] = embedder

    primary_tenant = plan.primary_tenant
    primary_store = stores[primary_tenant]
    primary_embedder = embedders[primary_tenant]
    if len(plan.selected_legs) == 1:
        return FederationExecution(primary_store, primary_embedder, None, empty)

    holder: dict[str, object] = {}

    def retrieve_trusted(
        store: Any,
        embedder: Embedder,
        query: str,
        source: str | None = None,
        k: int = 5,
        calibration: object | None = None,
        policy: object | None = None,
        entailment: object | None = None,
        security_policy: object | None = None,
        access_context: object | None = None,
        env: Mapping[str, str] | None = None,
        **kwargs: object,
    ) -> _Retrieval:
        del store, embedder, kwargs
        retrievals: dict[str, _Retrieval] = {}

        def make_leg_retriever(tenant: str) -> Callable[[str, int], TrustedResult]:
            def retrieve(query_text: str, candidate_k: int) -> TrustedResult:
                leg_context = (
                    AccessContext(
                        principal=access_context.principal,
                        tenant=tenant,
                        purpose=access_context.purpose,
                        clearance=access_context.clearance,
                        egress_allowed=access_context.egress_allowed,
                    )
                    if isinstance(access_context, AccessContext)
                    else access_context
                )
                retrieval = retrieve_trusted_fn(
                    stores[tenant],
                    embedders[tenant],
                    query_text,
                    source,
                    candidate_k,
                    calibration,
                    policy,
                    entailment,
                    security_policy,
                    leg_context,
                    env,
                )
                retrievals[tenant] = retrieval
                return retrieval.result

            return retrieve

        federation_legs = [
            FederationLeg(
                tenant_id=leg.tenant,
                generation_id=identities[leg.tenant].generation or "",
                embedding_profile=identities[leg.tenant].embedding_profile or "",
                calibration_id=identities[leg.tenant].calibration,
                calibration_status=identities[leg.tenant].calibration_status or "",
                route=plan.route_id,
                retrieve=make_leg_retriever(leg.tenant),
                candidate_k=leg.limit,
            )
            for leg in plan.selected_legs
        ]
        result = federate(
            query,
            federation_legs,
            config,
            latency_budget_ms=plan.latency_budget_ms,
        )
        primary_retrieval = retrievals.get(primary_tenant)
        if primary_retrieval is None:
            raise RuntimeError("primary federation leg did not produce a trusted retrieval")
        served_hits = (
            [candidate.hit for candidate in result.served]
            if result.active
            else list(primary_retrieval.result.hits)
        )
        merged = replace(
            primary_retrieval.result,
            hits=served_hits[:k],
            abstained=not served_hits,
            reason="" if served_hits else "no trustworthy federated memory",
            decision_state=decision_state_for(
                served_hits[:k],
                gap_warning=primary_retrieval.result.gap_warning,
            ),
        )
        holder["result"] = result
        return replace(primary_retrieval, result=merged)

    return FederationExecution(
        primary_store,
        primary_embedder,
        retrieve_trusted,
        holder,
    )


def federation_diagnostics(execution: FederationExecution) -> dict[str, object] | None:
    """Return JSON safe bounded federation diagnostics after execution."""
    value = execution.diagnostics.get("result")
    if not isinstance(value, FederationResult):
        return None
    return {
        "active": value.active,
        "mode": value.diagnostics.mode,
        "fanout_count": value.diagnostics.fanout_count,
        "executed_leg_count": value.diagnostics.executed_leg_count,
        "candidate_count": value.diagnostics.candidate_count,
        "fused_candidate_count": value.diagnostics.fused_candidate_count,
        "primary_prefix_count": value.diagnostics.primary_prefix_count,
        "rescue_count": value.diagnostics.rescue_count,
        "rejected_leg_count": value.diagnostics.rejected_leg_count,
        "rejected_candidate_count": value.diagnostics.rejected_candidate_count,
        "latency_ms": value.diagnostics.latency_ms,
        "max_concurrency": value.diagnostics.max_concurrency,
        "legs": [
            {
                "tenant": leg.tenant_id,
                "route": leg.route,
                "embedding_profile": leg.embedding_profile,
                "generation": leg.generation_id,
                "candidate_count": leg.candidate_count,
                "trusted_candidate_count": leg.trusted_candidate_count,
                "rejected_candidate_count": leg.rejected_candidate_count,
                "latency_ms": leg.latency_ms,
                "accepted": leg.accepted,
                "rejection_reason": leg.rejection_reason,
            }
            for leg in value.diagnostics.legs
        ],
    }


__all__ = ["FederationExecution", "federation_diagnostics", "prepare_federated_execution"]
