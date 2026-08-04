"""Regression tests for the strict trust gate on the search path.

Every test here fails against the predecessor behaviour, where `trusted_search` resolved an
absent, stale, or uncertified calibration and then ran the retrieval anyway with
`cal = calibration or _UNCALIBRATED` — returning corpus text scored against the 0.50 floor.

The strongest assertion in this file is not about message content. It is `_RefusingStore`, whose
`query_dense` and `query_sparse` raise `AssertionError` if they are ever called. A strict refusal
that touched the corpus cannot pass, regardless of what the refusal says. That is the difference
between "the payload happened to have no text in it" and "no corpus byte was ever fetched".
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from recall.calibration_v2 import CalibrationResolution, CalibrationStatus
from recall.trust import trusted_search
from recall.trust_policy import TrustFailureCode, TrustPolicy, TrustRefusal, TrustState
from recall.types import Chunk, ScoredChunk

_CORPUS_SENTINEL = "SENTINEL_CORPUS_TEXT_9f3a17c4"


class _Embedder:
    model = "strict-trust-test"
    name = "strict-trust-test"
    dim = 8

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * (self.dim - 1) for _ in texts]


def _hit(chunk_id: str = "c1", score: float = 0.99) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            id=chunk_id,
            text=_CORPUS_SENTINEL,
            source="sec",
            metadata={"file": "secret.md", "ord": 0},
        ),
        score=score,
        indexed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


class _RefusingStore:
    """A store that proves the corpus was never touched: retrieval raises if reached."""

    tenant = "acme"
    generation_id = "gen_1"

    def __init__(self, status: CalibrationStatus) -> None:
        self._status = status
        self.searched = False

    def resolve_calibration(self) -> CalibrationResolution:
        return CalibrationResolution(self._status)

    def generation_binding(self) -> dict[str, str]:
        return {
            "tenant_id": self.tenant,
            "generation_id": self.generation_id,
            "pipeline_fingerprint": "a" * 64,
            "corpus_fingerprint": "b" * 64,
        }

    def query_dense(self, *args: object, **kwargs: object) -> list[ScoredChunk]:
        self.searched = True
        raise AssertionError("strict refusal must happen BEFORE any corpus read")

    def query_sparse(self, *args: object, **kwargs: object) -> list[ScoredChunk]:
        self.searched = True
        raise AssertionError("strict refusal must happen BEFORE any corpus read")

    def newest_indexed_at(self) -> datetime | None:
        return datetime(2026, 1, 1, tzinfo=UTC)

    def supersession(self) -> tuple[dict[str, str], frozenset[str]]:
        return {}, frozenset()


class _ServingStore(_RefusingStore):
    """Same identity surface, but retrieval actually returns a corpus hit."""

    def query_dense(self, *args: object, **kwargs: object) -> list[ScoredChunk]:
        self.searched = True
        return [_hit()]

    def query_sparse(self, *args: object, **kwargs: object) -> list[ScoredChunk]:
        self.searched = True
        return []


_UNTRUSTWORTHY = [
    (CalibrationStatus.MISSING, TrustFailureCode.CALIBRATION_MISSING),
    (CalibrationStatus.STALE, TrustFailureCode.CALIBRATION_STALE),
    (CalibrationStatus.SUPERSEDED, TrustFailureCode.CALIBRATION_STALE),
    (CalibrationStatus.UNCERTIFIED, TrustFailureCode.CALIBRATION_UNCERTIFIED),
    (CalibrationStatus.REJECTED, TrustFailureCode.CALIBRATION_UNCERTIFIED),
    (CalibrationStatus.DRAFT, TrustFailureCode.CALIBRATION_UNCERTIFIED),
    (CalibrationStatus.LEGACY_UNBOUND, TrustFailureCode.CALIBRATION_UNCERTIFIED),
]


class TestStrictRefusesBeforeAnyCorpusRead:
    @pytest.mark.parametrize(("status", "code"), _UNTRUSTWORTHY)
    def test_each_status_refuses_with_its_stable_code(
        self, status: CalibrationStatus, code: TrustFailureCode
    ) -> None:
        store = _RefusingStore(status)
        with pytest.raises(TrustRefusal) as excinfo:
            trusted_search(store, _Embedder(), "what is the policy", k=3)
        assert excinfo.value.code is code
        assert store.searched is False, "retrieval ran before the trust gate refused"

    @pytest.mark.parametrize(("status", "code"), _UNTRUSTWORTHY)
    def test_refusal_carries_no_corpus_bytes(
        self, status: CalibrationStatus, code: TrustFailureCode
    ) -> None:
        store = _ServingStore(status)  # would happily serve text if the gate let it
        with pytest.raises(TrustRefusal) as excinfo:
            trusted_search(store, _Embedder(), "what is the policy", k=3)
        refusal = excinfo.value
        rendered = f"{refusal}|{refusal.to_dict()!r}|{refusal.advice}|{refusal.args!r}"
        assert _CORPUS_SENTINEL not in rendered
        assert "secret.md" not in rendered
        assert store.searched is False

    def test_strict_is_the_default_policy(self) -> None:
        """Omitting the policy must not open the gate."""
        store = _RefusingStore(CalibrationStatus.MISSING)
        with pytest.raises(TrustRefusal):
            trusted_search(store, _Embedder(), "q", k=1)

    def test_certified_still_serves(self) -> None:
        """The gate must not be a blanket denial: certified calibration still answers."""
        from recall.calibration import Calibration

        class _Certified(_ServingStore):
            def resolve_calibration(self) -> CalibrationResolution:
                from recall.calibration_v2 import CalibrationArtifactV2

                class _Artifact:
                    calibration_id = "cal_ok"
                    query_set_digest = "c" * 64
                    runtime = Calibration(
                        embedder="strict-trust-test", threshold=0.2, scale=0.1
                    )

                assert CalibrationArtifactV2 is not None
                return CalibrationResolution(CalibrationStatus.CERTIFIED, _Artifact())  # type: ignore[arg-type]

        result = trusted_search(_Certified(CalibrationStatus.CERTIFIED), _Embedder(), "q", k=1)
        assert result.calibrated is True
        assert result.trust_state == TrustState.TRUSTED
        assert result.calibration_status == "certified"


class TestUncertifiedCanNeverReportCalibrated:
    """Requirement 1: reproduce certified=False presenting as calibrated=True, prove impossible."""

    @pytest.mark.parametrize(("status", "_code"), _UNTRUSTWORTHY)
    def test_development_mode_never_reports_calibrated(
        self, status: CalibrationStatus, _code: TrustFailureCode
    ) -> None:
        result = trusted_search(
            _ServingStore(status),
            _Embedder(),
            "q",
            k=1,
            policy=TrustPolicy.development(),
        )
        assert result.calibrated is False
        assert result.calibration_status != "certified"

    def test_calibrated_is_false_without_an_artifact_id(self) -> None:
        from recall.types import RetrievalDiagnostics, StalenessReport, TrustedResult

        forged = TrustedResult(
            query="q",
            hits=[],
            abstained=False,
            reason="",
            gap_warning=False,
            staleness=StalenessReport(False, None, None, timedelta(days=2)),
            diagnostics=RetrievalDiagnostics(),
            calibration_id=None,
            calibration_status="certified",  # claims certified with no artifact behind it
        )
        assert forged.calibrated is False


class TestDevelopmentDegradation:
    """Requirements 7 and 8."""

    @pytest.mark.parametrize(("status", "_code"), _UNTRUSTWORTHY)
    def test_degraded_state_and_unverified_hits(
        self, status: CalibrationStatus, _code: TrustFailureCode
    ) -> None:
        result = trusted_search(
            _ServingStore(status), _Embedder(), "q", k=1, policy=TrustPolicy.development()
        )
        assert result.trust_state == TrustState.DEGRADED
        assert result.calibration_status == status.value
        assert result.hits, "development mode still retrieves"
        assert all(h.verdict == "unverified" for h in result.hits)

    @pytest.mark.parametrize(("status", "_code"), _UNTRUSTWORTHY)
    def test_never_verdict_ok_and_never_abstained(
        self, status: CalibrationStatus, _code: TrustFailureCode
    ) -> None:
        result = trusted_search(
            _ServingStore(status), _Embedder(), "q", k=1, policy=TrustPolicy.development()
        )
        assert not any(h.verdict == "ok" for h in result.hits)
        assert result.abstained is False, "an abstention would be a trustworthy decision"

    def test_degraded_reason_names_the_code(self) -> None:
        result = trusted_search(
            _ServingStore(CalibrationStatus.MISSING),
            _Embedder(),
            "q",
            k=1,
            policy=TrustPolicy.development(),
        )
        assert result.failure_code == TrustFailureCode.CALIBRATION_MISSING.value


class TestDependencyFailureIsDistinct:
    """Requirement 7 of the regression list: dependency failure is not calibration failure."""

    def test_resolver_raising_maps_to_dependency_unavailable(self) -> None:
        class _BrokenDependency(_ServingStore):
            def resolve_calibration(self) -> CalibrationResolution:
                raise OSError("control plane unreachable")

        store = _BrokenDependency(CalibrationStatus.CERTIFIED)
        with pytest.raises(TrustRefusal) as excinfo:
            trusted_search(store, _Embedder(), "q", k=1)
        assert excinfo.value.code is TrustFailureCode.DEPENDENCY_UNAVAILABLE
        assert store.searched is False

    def test_dependency_failure_is_not_reported_as_missing_calibration(self) -> None:
        class _BrokenDependency(_ServingStore):
            def resolve_calibration(self) -> CalibrationResolution:
                raise OSError("control plane unreachable")

        with pytest.raises(TrustRefusal) as excinfo:
            trusted_search(_BrokenDependency(CalibrationStatus.MISSING), _Embedder(), "q", k=1)
        assert excinfo.value.code is not TrustFailureCode.CALIBRATION_MISSING

    def test_dependency_advice_differs_from_a_working_gate_finding_nothing(self) -> None:
        refusal = TrustRefusal(
            code=TrustFailureCode.DEPENDENCY_UNAVAILABLE,
            calibration_status="missing",
            tenant_id="acme",
            generation_id=None,
        )
        assert "outage" in refusal.advice.lower()


class TestTenantIsolation:
    """Requirement 6: one tenant's failure cannot affect another tenant."""

    def test_one_tenant_refusing_does_not_affect_another(self) -> None:
        broken = _RefusingStore(CalibrationStatus.MISSING)
        broken.tenant = "tenant_broken"

        from recall.calibration import Calibration

        class _HealthyArtifact:
            calibration_id = "cal_ok"
            query_set_digest = "c" * 64
            runtime = Calibration(embedder="strict-trust-test", threshold=0.2, scale=0.1)

        class _Healthy(_ServingStore):
            tenant = "tenant_healthy"

            def resolve_calibration(self) -> CalibrationResolution:
                return CalibrationResolution(
                    CalibrationStatus.CERTIFIED, _HealthyArtifact()  # type: ignore[arg-type]
                )

        with pytest.raises(TrustRefusal):
            trusted_search(broken, _Embedder(), "q", k=1)

        healthy = trusted_search(
            _Healthy(CalibrationStatus.CERTIFIED), _Embedder(), "q", k=1
        )
        assert healthy.calibrated is True
        assert healthy.trust_state == TrustState.TRUSTED
