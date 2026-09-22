"""Focused proofs for bounded, rank only federation across isolated tenants."""

from __future__ import annotations

from datetime import datetime, timezone
import threading
import time

import pytest

from recall.federation import (
    FederationConfig,
    FederationConfigurationError,
    FederationLeg,
    federate,
)
from recall.observability import METRICS
from recall.types import (
    Chunk,
    Provenance,
    RetrievalDiagnostics,
    StalenessReport,
    TrustedHit,
    TrustedResult,
    Validity,
)


NOW = datetime.now(timezone.utc)


def _result(
    tenant: str,
    generation: str,
    profile: str,
    calibration: str,
    hits: list[TrustedHit],
    *,
    trust_state: str = "trusted",
    calibration_status: str = "certified",
    gap_warning: bool = False,
) -> TrustedResult:
    return TrustedResult(
        query="query",
        hits=hits,
        abstained=not hits,
        reason="" if hits else "no memory retrieved at all",
        gap_warning=gap_warning,
        staleness=StalenessReport(False, NOW, None, __import__("datetime").timedelta(days=2)),
        diagnostics=RetrievalDiagnostics(embedding_profile=profile),
        calibration_id=calibration,
        calibration_status=calibration_status,
        tenant_id=tenant,
        generation_id=generation,
        trust_state=trust_state,
    )


def _hit(chunk_id: str, source: str, text: str, cosine: float, *, verdict: str = "ok") -> TrustedHit:
    return TrustedHit(
        chunk=Chunk(chunk_id, source, text),
        cosine=cosine,
        confidence=0.9,
        verdict=verdict,  # type: ignore[arg-type]
        provenance=Provenance(source, source, 1, NOW),
        validity=Validity(None, None, None),
    )


def _leg(
    tenant: str,
    profile: str,
    result: TrustedResult,
    *,
    route: str,
) -> FederationLeg:
    return FederationLeg(
        tenant_id=tenant,
        generation_id=f"{tenant}-generation",
        embedding_profile=profile,
        calibration_id=f"{tenant}-calibration",
        calibration_status="certified",
        route=route,
        retrieve=lambda _query, _k: result,
        pipeline_fingerprint=f"{tenant}-pipeline",
    )


def test_off_mode_preserves_single_tenant_control_and_never_fans_out() -> None:
    """The default control is one leg, even when a plan contains specialists.

    Red proof mutation: changing ``effective_legs`` from ``selected[:1]`` to ``selected`` makes
    the assertion on the uncalled secondary leg fail. The targeted production symbol is
    ``recall.federation.federate`` and the failure is a behavioral fanout assertion, not setup.
    """
    calls: list[str] = []
    primary = _result(
        "memory", "memory-generation", "memory-profile", "memory-calibration", [_hit("p", "p.md", "primary", 0.1)]
    )
    secondary = _result(
        "re-call-code-gen", "code-generation", "code-profile", "code-calibration", [_hit("s", "s.md", "secondary", 0.99)]
    )
    first = _leg("memory", "memory-profile", primary, route="primary")
    second = _leg("re-call-code-gen", "code-profile", secondary, route="code")
    first = FederationLeg(**{**first.__dict__, "retrieve": lambda _q, _k: (calls.append("primary") or primary)})
    second = FederationLeg(**{**second.__dict__, "retrieve": lambda _q, _k: (calls.append("secondary") or secondary)})

    result = federate("query", [first, second])

    assert calls == ["primary"]
    assert result.active is False
    assert result.diagnostics.fanout_count == 1
    assert [candidate.hit.chunk.id for candidate in result.served] == ["p"]


def test_single_leg_latency_budget_is_enforced() -> None:
    primary = _result(
        "memory", "memory-generation", "memory-profile", "memory-calibration", [_hit("p", "p.md", "primary", 0.1)]
    )

    def slow(_query: str, _k: int) -> TrustedResult:
        time.sleep(0.2)
        return primary

    leg = FederationLeg(
        tenant_id="memory",
        generation_id="memory-generation",
        embedding_profile="memory-profile",
        calibration_id="memory-calibration",
        calibration_status="certified",
        route="primary",
        retrieve=slow,
        pipeline_fingerprint="memory-pipeline",
    )
    result = federate(
        "query", [leg], FederationConfig(mode="active"), latency_budget_ms=10
    )

    assert result.served == ()
    assert result.diagnostics.legs[0].rejection_reason == "latency_budget_exceeded"


def test_from_env_default_matches_the_bounded_route_contract() -> None:
    config = FederationConfig.from_env({})

    assert config.mode == "off"
    assert config.max_legs == 2
    assert config.max_legs == FederationConfig().max_legs


def test_active_merge_uses_ranks_protects_prefix_and_keeps_lineage() -> None:
    """Different tenant cosine magnitudes cannot reorder the protected primary prefix.

    Red proof mutation: replacing the secondary selection with a cosine sort lets ``s1`` displace
    the primary prefix. This targets ``recall.federation.federate``'s rank merge and fails in the
    assertion on the served order.
    """
    primary_hits = [
        _hit("p1", "primary.md", "one", 0.10),
        _hit("p2", "primary.md", "two", 0.09),
        _hit("p3", "primary.md", "three", 0.08),
        _hit("p4", "primary.md", "four", 0.07),
    ]
    secondary_hits = [
        _hit("s1", "specialist.md", "specialist one", 0.99),
        _hit("s2", "specialist.md", "specialist two", 0.98),
    ]
    primary = _leg(
        "memory",
        "memory-profile",
        _result("memory", "memory-generation", "memory-profile", "memory-calibration", primary_hits),
        route="primary",
    )
    secondary = _leg(
        "re-call-code-gen",
        "code-profile",
        _result("re-call-code-gen", "re-call-code-gen-generation", "code-profile", "re-call-code-gen-calibration", secondary_hits),
        route="code",
    )
    METRICS.reset()

    result = federate(
        "query",
        [primary, secondary],
        FederationConfig(mode="active", result_k=5, primary_prefix=3, rescue_slots=1),
    )

    assert [candidate.hit.chunk.id for candidate in result.served] == ["p1", "p2", "p3", "p4", "s1"]
    rescued = result.served[-1]
    assert rescued.tenant_id == "re-call-code-gen"
    assert rescued.generation_id == "re-call-code-gen-generation"
    assert rescued.embedding_profile == "code-profile"
    assert rescued.calibration_id == "re-call-code-gen-calibration"
    assert rescued.route == "code"
    assert rescued.leg_rank == 1
    assert rescued.fused_rank >= 1
    assert result.diagnostics.rescue_count == 1
    assert result.diagnostics.fanout_count == 2
    assert "recall_federation_total_ms{mode=active}" in METRICS.snapshot()["histograms"]


def test_untrusted_secondary_is_rejected_before_rescue() -> None:
    """A low confidence specialist candidate cannot enter the evidence tail.

    Red proof mutation: replacing the ``is_trusted`` filter with ``list(result.hits)`` admits
    ``s1`` and fails the exact tail assertion. The test therefore observes the candidate trust
    gate rather than merely checking that a result object exists.
    """
    primary = _leg(
        "memory",
        "memory-profile",
        _result("memory", "memory-generation", "memory-profile", "memory-calibration", [_hit("p", "p.md", "primary", 0.5)]),
        route="primary",
    )
    secondary_result = _result(
        "re-call-code-gen",
        "re-call-code-gen-generation",
        "code-profile",
        "re-call-code-gen-calibration",
        [_hit("s1", "s.md", "untrusted", 0.99, verdict="low_confidence")],
    )
    secondary = _leg("re-call-code-gen", "code-profile", secondary_result, route="code")

    result = federate(
        "query",
        [primary, secondary],
        FederationConfig(mode="active", result_k=2, primary_prefix=1, rescue_slots=1),
    )

    assert [candidate.hit.chunk.id for candidate in result.served] == ["p"]
    assert result.diagnostics.rejected_candidate_count == 1
    assert result.diagnostics.legs[1].rejected_candidate_count == 1


def test_secondary_duplicate_is_not_novel_and_fanout_is_bounded() -> None:
    """Copied primary content cannot consume a rescue slot, and leg count is bounded."""
    primary = _leg(
        "memory",
        "memory-profile",
        _result("memory", "memory-generation", "memory-profile", "memory-calibration", [_hit("same", "same.md", "same text", 0.5)]),
        route="primary",
    )
    duplicate = _leg(
        "re-call-docs",
        "docs-profile",
        _result("re-call-docs", "re-call-docs-generation", "docs-profile", "re-call-docs-calibration", [_hit("same", "same.md", "same text", 0.99)]),
        route="docs",
    )
    result = federate(
        "query",
        [primary, duplicate],
        FederationConfig(mode="active", result_k=2, primary_prefix=1, rescue_slots=1),
    )
    assert [candidate.hit.chunk.id for candidate in result.served] == ["same"]
    assert result.diagnostics.rescue_count == 0
    with pytest.raises(FederationConfigurationError, match="over max_legs"):
        federate("query", [primary, duplicate, duplicate, duplicate], FederationConfig(max_legs=3))


def test_secondary_same_chunk_id_is_scoped_to_its_tenant() -> None:
    """Red proof: bare cross tenant IDs must not collapse distinct specialist candidates."""
    primary = _leg(
        "memory",
        "primary-generation",
        _result(
            "memory",
            "memory-generation",
            "primary-generation",
            "memory-calibration",
            [_hit("same", "memory.md", "primary text", 0.9)],
        ),
        route="primary",
    )
    secondary = _leg(
        "re-call-code-gen",
        "code-generation",
        _result(
            "re-call-code-gen",
            "re-call-code-gen-generation",
            "code-generation",
            "re-call-code-gen-calibration",
            [_hit("same", "code.md", "secondary text", 0.8)],
        ),
        route="code",
    )

    result = federate(
        "query",
        [primary, secondary],
        FederationConfig(mode="active", result_k=2, primary_prefix=1, rescue_slots=1),
    )

    assert [(candidate.tenant_id, candidate.hit.chunk.id) for candidate in result.served] == [
        ("memory", "same"),
        ("re-call-code-gen", "same"),
    ]
    assert result.diagnostics.fused_candidate_count == 2

def test_concurrency_is_bounded_by_configuration() -> None:
    """Federation never starts more worker legs than its configured bound."""
    active = 0
    maximum = 0
    lock = threading.Lock()

    def retrieve(_query: str, _k: int) -> TrustedResult:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        tenant = "memory"
        return _result(tenant, "memory-generation", "memory-profile", "memory-calibration", [_hit("p", "p.md", "p", 0.5)])

    leg = FederationLeg(
        tenant_id="memory",
        generation_id="memory-generation",
        embedding_profile="memory-profile",
        calibration_id="memory-calibration",
        calibration_status="certified",
        route="primary",
        retrieve=retrieve,
    )
    result = federate(
        "query",
        [leg, leg],
        FederationConfig(mode="shadow", max_legs=2, max_concurrency=1),
    )
    assert maximum == 1
    assert result.diagnostics.max_concurrency == 1
