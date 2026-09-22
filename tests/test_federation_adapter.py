from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal, cast

import pytest

from recall.embeddings import HashingEmbedder, embedding_profile_id
from recall.federation import FederationConfig
from recall.retrieval_plan import RetrievalLeg, RetrievalPlan, TenantIdentity
from recall.retrieval_plan import RetrievalPlanError
from recall.types import (
    Chunk,
    Provenance,
    RetrievalDiagnostics,
    StalenessReport,
    TrustedHit,
    TrustedResult,
    Validity,
)
from recall.timing import TimedEmbedder
from recall_mcp.federation_adapter import (
    federation_diagnostics,
    prepare_federated_execution,
)
from recall_mcp.retrieval import _Retrieval
from recall_mcp.stores import StoreRegistry
from recall.profiles import resolve_retrieval_profile


NOW = datetime.now(timezone.utc)


class _Registry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def get(self, tenant: str, *, embedding_profile: str | None = None) -> object:
        self.calls.append((tenant, embedding_profile))
        return f"store:{tenant}"

    def _get_federation_store(self, tenant: str, expected_profile: str) -> object:
        return self.get(tenant, embedding_profile=expected_profile)


def _result(tenant: str, chunk_id: str, *, profile_id: str = "hashing-64") -> TrustedResult:
    return TrustedResult(
        query="q",
        hits=[
            TrustedHit(
                chunk=Chunk(chunk_id, f"{tenant}.md", tenant),
                cosine=0.2,
                confidence=0.9,
                verdict="ok",
                provenance=Provenance(f"{tenant}.md", f"{tenant}.md", 1, NOW),
                validity=Validity(None, None, None),
            )
        ],
        abstained=False,
        reason="",
        gap_warning=False,
        staleness=StalenessReport(False, NOW, timedelta(days=1), timedelta(days=2)),
        diagnostics=RetrievalDiagnostics(embedding_profile=profile_id),
        calibration_id=f"{tenant}-calibration",
        calibration_status="certified",
        tenant_id=tenant,
        generation_id=f"{tenant}-generation",
        trust_state="trusted",
    )


def _plan() -> RetrievalPlan:
    def identity(tenant: str) -> TenantIdentity:
        return TenantIdentity(
            tenant=tenant,
            generation=f"{tenant}-generation",
            embedding_profile="hashing-64",
            calibration=f"{tenant}-calibration",
            calibration_status="certified",
            trust_state="trusted",
            provenance_identity=f"{tenant}:generation",
        )

    return RetrievalPlan(
        route_id="memory-specialists",
        primary_tenant="memory",
        allowed_tenants=frozenset({"memory", "code"}),
        rescue_tenant="code",
        primary_limit=3,
        rescue_limit=1,
        max_fanout=2,
        selected_legs=(
            RetrievalLeg("primary", "memory", 3, identity("memory")),
            RetrievalLeg("rescue", "code", 1, identity("code")),
        ),
    )


@pytest.mark.parametrize(
    ("mode", "expected_chunks", "expected_active"),
    [
        ("active", ("memory-chunk", "code-chunk"), True),
        ("shadow", ("memory-chunk",), False),
    ],
)
def test_plan_adapter_binds_each_leg_and_returns_federation_diagnostics(
    mode: Literal["active", "shadow"],
    expected_chunks: tuple[str, ...],
    expected_active: bool,
) -> None:
    """Mutation proof: skipping the second selected leg must remove its retrieval and diagnostics.

    The targeted production symbol is ``prepare_federated_execution``. The assertion observes the
    actual per tenant callback calls and the bounded serving diagnostics, not only a plan object.
    """
    registry = _Registry()
    embedder = HashingEmbedder(dim=64)
    profile = resolve_retrieval_profile({})

    def retrieve(store: object, _embedder: object, query: str, *_args: object) -> _Retrieval:
        tenant = str(store).split(":", 1)[1]
        return _Retrieval(
            result=_result(tenant, f"{tenant}-chunk"),
            timed=cast(TimedEmbedder, object()),
            profile=profile,
            request_started=0.0,
            admission_wait_ms=0.0,
            effective_k=5,
        )

    execution = prepare_federated_execution(
        _plan(),
        config=FederationConfig(
            mode=mode, max_legs=2, result_k=2, primary_prefix=1, rescue_slots=1
        ),
        registry=registry,  # type: ignore[arg-type]
        current_store="store:memory",
        current_embedder=embedder,
        env={},
        retrieve_trusted_fn=retrieve,
    )
    assert execution.retrieve_trusted is not None

    merged = execution.retrieve_trusted(
        execution.store,
        execution.embedder,
        "q",
        None,
        5,
        None,
        None,
        None,
        None,
        None,
        {},
    )

    assert tuple(hit.chunk.id for hit in merged.result.hits) == expected_chunks
    assert registry.calls == [("memory", "hashing-64"), ("code", "hashing-64")]
    diagnostics = federation_diagnostics(execution)
    assert diagnostics is not None
    assert diagnostics["active"] is expected_active
    assert diagnostics["fanout_count"] == 2
    assert diagnostics["rescue_count"] == 1


def test_adapter_keeps_tenant_generation_and_profile_bound_per_leg() -> None:
    registry = _Registry()
    primary_embedder = HashingEmbedder(dim=64)
    profile = resolve_retrieval_profile({})

    def identity(tenant: str, profile_id: str) -> TenantIdentity:
        return TenantIdentity(
            tenant=tenant,
            generation=f"{tenant}-generation",
            embedding_profile=profile_id,
            calibration=f"{tenant}-calibration",
            calibration_status="certified",
            trust_state="trusted",
            provenance_identity=f"{tenant}:generation",
        )

    plan = RetrievalPlan(
        route_id="profile-specialists",
        primary_tenant="memory",
        allowed_tenants=frozenset({"memory", "code"}),
        rescue_tenant="code",
        primary_limit=2,
        rescue_limit=1,
        max_fanout=2,
        selected_legs=(
            RetrievalLeg("primary", "memory", 2, identity("memory", "hashing-64")),
            RetrievalLeg("rescue", "code", 1, identity("code", "hashing-32")),
        ),
    )
    seen: list[tuple[str, str, int]] = []

    def make_embedder(profile_id: str, _env: dict[str, str]) -> HashingEmbedder:
        return HashingEmbedder(dim=int(profile_id.rsplit("-", 1)[1]))

    def retrieve(store: object, embedder: object, query: str, *_args: object) -> _Retrieval:
        tenant = str(store).split(":", 1)[1]
        profile_id = embedding_profile_id(cast(HashingEmbedder, embedder))
        seen.append((tenant, profile_id, cast(int, _args[1])))
        return _Retrieval(
            result=_result(tenant, f"{tenant}-chunk", profile_id=profile_id),
            timed=cast(TimedEmbedder, object()),
            profile=profile,
            request_started=0.0,
            admission_wait_ms=0.0,
            effective_k=5,
        )

    execution = prepare_federated_execution(
        plan,
        config=FederationConfig(
            mode="active",
            max_legs=2,
            candidate_k=3,
            result_k=2,
            primary_prefix=1,
            rescue_slots=1,
        ),
        registry=registry,  # type: ignore[arg-type]
        current_store="store:memory",
        current_embedder=primary_embedder,
        env={},
        make_embedder_fn=make_embedder,
        retrieve_trusted_fn=retrieve,
    )
    assert execution.retrieve_trusted is not None

    merged = execution.retrieve_trusted(
        execution.store,
        execution.embedder,
        "q",
        None,
        5,
        None,
        None,
        None,
        None,
        None,
        {},
    )

    assert seen == [("memory", "hashing-64", 2), ("code", "hashing-32", 1)]
    assert registry.calls == [("memory", "hashing-64"), ("code", "hashing-32")]
    assert [hit.chunk.id for hit in merged.result.hits] == ["memory-chunk", "code-chunk"]
    diagnostics = federation_diagnostics(execution)
    assert diagnostics is not None
    legs = cast(list[dict[str, object]], diagnostics["legs"])
    assert legs == [
        {
            "tenant": "memory",
            "route": "profile-specialists",
            "embedding_profile": "hashing-64",
            "generation": "memory-generation",
            "candidate_count": 1,
            "trusted_candidate_count": 1,
            "rejected_candidate_count": 0,
            "latency_ms": legs[0]["latency_ms"],
            "accepted": True,
            "rejection_reason": None,
        },
        {
            "tenant": "code",
            "route": "profile-specialists",
            "embedding_profile": "hashing-32",
            "generation": "code-generation",
            "candidate_count": 1,
            "trusted_candidate_count": 1,
            "rejected_candidate_count": 0,
            "latency_ms": legs[1]["latency_ms"],
            "accepted": True,
            "rejection_reason": None,
        },
    ]


def test_adapter_rejects_multimodal_leg_when_feature_is_disabled() -> None:
    identity = TenantIdentity(
        tenant="re-call-multimodal",
        generation="multimodal-generation",
        embedding_profile="voyage-multimodal-3.5-v1",
        calibration="multimodal-calibration",
        calibration_status="certified",
        trust_state="trusted",
        provenance_identity="multimodal:generation",
    )
    plan = RetrievalPlan(
        route_id="multimodal-specialist",
        primary_tenant="re-call-multimodal",
        allowed_tenants=frozenset({"re-call-multimodal"}),
        primary_limit=1,
        selected_legs=(RetrievalLeg("primary", "re-call-multimodal", 1, identity),),
    )

    with pytest.raises(RetrievalPlanError, match="feature enablement"):
        prepare_federated_execution(
            plan,
            config=FederationConfig(mode="active", max_legs=1, max_concurrency=1),
            registry=cast(StoreRegistry, _Registry()),
            current_store="store:re-call-multimodal",
            current_embedder=HashingEmbedder(dim=64),
            env={},
            multimodal_enabled=False,
        )
