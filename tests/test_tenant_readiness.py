"""Readiness is two different questions, and conflating them takes a fleet down.

`process_readiness` answers "can this pod serve anyone at all", and depends only on SHARED
dependencies. `tenant_readiness` answers "can this pod serve tenant X", and depends on that
tenant's generation and calibration.

The bug this guards against: folding per-tenant calibration state into the process-level answer.
One tenant with a stale calibration then makes the Kubernetes readiness probe fail, the pod leaves
the Service endpoints, and every *healthy* tenant loses its capacity. A per-tenant fault must cost
exactly that tenant.
"""

from __future__ import annotations

from recall.readiness import (
    DependencyStatus,
    TenantReadiness,
    process_readiness,
    tenant_readiness,
)
from recall.trust_policy import TrustFailureCode, TrustState


class TestProcessReadinessDependsOnSharedDependenciesOnly:
    def test_ready_when_shared_dependencies_are_healthy(self) -> None:
        result = process_readiness([DependencyStatus("database", True), DependencyStatus("s3", True)])
        assert result.ready is True
        assert result.failure_code is None

    def test_unready_when_a_shared_dependency_is_down(self) -> None:
        result = process_readiness(
            [DependencyStatus("database", False, "connection refused"), DependencyStatus("s3", True)]
        )
        assert result.ready is False
        assert result.failure_code == TrustFailureCode.DEPENDENCY_UNAVAILABLE.value
        assert "database" in result.unavailable

    def test_one_unready_tenant_does_not_make_the_process_unready(self) -> None:
        """The load-bearing assertion of requirement 12."""
        tenants = [
            tenant_readiness("healthy", generation_id="gen_1", calibration_status="certified",
                             calibration_id="cal_1"),
            tenant_readiness("broken", generation_id="gen_2", calibration_status="stale",
                             calibration_id="cal_2"),
        ]
        result = process_readiness([DependencyStatus("database", True)], tenants=tenants)
        assert result.ready is True, "a per-tenant fault must not evict the pod for everyone"
        assert result.tenants_unready == ("broken",)

    def test_all_tenants_unready_still_does_not_make_the_process_unready(self) -> None:
        tenants = [
            tenant_readiness("a", generation_id=None, calibration_status="missing"),
            tenant_readiness("b", generation_id=None, calibration_status="missing"),
        ]
        result = process_readiness([DependencyStatus("database", True)], tenants=tenants)
        assert result.ready is True
        assert set(result.tenants_unready) == {"a", "b"}


class TestTenantReadiness:
    def test_certified_tenant_is_ready_and_trusted(self) -> None:
        state = tenant_readiness(
            "acme", generation_id="gen_1", calibration_status="certified", calibration_id="cal_1"
        )
        assert state.ready is True
        assert state.trust_state == TrustState.TRUSTED.value
        assert state.failure_code is None

    def test_missing_generation_is_index_not_ready(self) -> None:
        state = tenant_readiness("acme", generation_id=None, calibration_status="certified",
                                 calibration_id="cal_1")
        assert state.ready is False
        assert state.failure_code == TrustFailureCode.INDEX_NOT_READY.value

    def test_certified_status_without_an_artifact_id_is_not_ready(self) -> None:
        """A status string alone is not evidence; the artifact id must be there too."""
        state = tenant_readiness("acme", generation_id="gen_1", calibration_status="certified",
                                 calibration_id=None)
        assert state.ready is False
        assert state.failure_code == TrustFailureCode.CALIBRATION_MISSING.value

    def test_each_bad_status_maps_to_its_stable_code(self) -> None:
        for status, code in [
            ("missing", TrustFailureCode.CALIBRATION_MISSING),
            ("stale", TrustFailureCode.CALIBRATION_STALE),
            ("uncertified", TrustFailureCode.CALIBRATION_UNCERTIFIED),
            ("rejected", TrustFailureCode.CALIBRATION_UNCERTIFIED),
            ("legacy_unbound", TrustFailureCode.CALIBRATION_UNCERTIFIED),
        ]:
            state = tenant_readiness("acme", generation_id="gen_1", calibration_status=status,
                                     calibration_id="cal_1")
            assert state.ready is False
            assert state.failure_code == code.value, status


class TestReadinessReturnsNoCorpusText:
    """Requirement 11: status reporting must not become a retrieval side channel."""

    def test_tenant_readiness_payload_has_no_text_channel(self) -> None:
        state = tenant_readiness("acme", generation_id="gen_1", calibration_status="stale",
                                 calibration_id="cal_1")
        payload = state.to_dict()
        for forbidden in ("text", "chunk", "chunks", "hits", "preview", "snippet", "content"):
            assert forbidden not in payload, forbidden
        assert set(payload) <= {
            "tenant_id", "ready", "trust_state", "failure_code", "generation_id",
            "calibration_id", "calibration_status",
        }

    def test_process_readiness_payload_has_no_text_channel(self) -> None:
        payload = process_readiness(
            [DependencyStatus("database", True)],
            tenants=[tenant_readiness("acme", generation_id="gen_1",
                                      calibration_status="certified", calibration_id="cal_1")],
        ).to_dict()
        for forbidden in ("text", "chunk", "chunks", "hits", "preview", "snippet", "content"):
            assert forbidden not in payload, forbidden

    def test_tenant_states_are_frozen(self) -> None:
        assert isinstance(
            tenant_readiness("a", generation_id=None, calibration_status="missing"),
            TenantReadiness,
        )
