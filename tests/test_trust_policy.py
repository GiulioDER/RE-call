"""Strict trust policy: an untrustworthy calibration can never present as trustworthy.

These are regression tests for the fail-open path that existed before this module: when
calibration resolution returned MISSING, STALE, or UNCERTIFIED, `trusted_search` fell back to
`_UNCALIBRATED` (the 0.50 floor) and returned corpus text anyway, with verdicts computed from a
threshold nobody certified. Every test here fails against that predecessor behaviour.
"""

from __future__ import annotations

import pytest

from recall.calibration_v2 import CalibrationStatus
from recall.trust_policy import (
    TrustFailureCode,
    TrustMode,
    TrustPolicy,
    TrustRefusal,
    TrustState,
    code_for_status,
)


class TestFailureCodes:
    """The seven codes are a stable machine-readable contract. Their spelling is the API."""

    def test_all_seven_codes_exist_with_exact_spelling(self) -> None:
        assert TrustFailureCode.INDEX_NOT_READY.value == "INDEX_NOT_READY"
        assert TrustFailureCode.LINEAGE_MISMATCH.value == "LINEAGE_MISMATCH"
        assert TrustFailureCode.CALIBRATION_MISSING.value == "CALIBRATION_MISSING"
        assert TrustFailureCode.CALIBRATION_UNCERTIFIED.value == "CALIBRATION_UNCERTIFIED"
        assert TrustFailureCode.CALIBRATION_STALE.value == "CALIBRATION_STALE"
        assert TrustFailureCode.DEPENDENCY_UNAVAILABLE.value == "DEPENDENCY_UNAVAILABLE"
        assert TrustFailureCode.DEPENDENCY_GRAPH_NOT_READY.value == "DEPENDENCY_GRAPH_NOT_READY"

    def test_no_unexpected_codes(self) -> None:
        assert {code.value for code in TrustFailureCode} == {
            "INDEX_NOT_READY",
            "LINEAGE_MISMATCH",
            "CALIBRATION_MISSING",
            "CALIBRATION_UNCERTIFIED",
            "CALIBRATION_STALE",
            "DEPENDENCY_UNAVAILABLE",
            "DEPENDENCY_GRAPH_NOT_READY",
        }


class TestStatusMapping:
    """Each calibration status maps to exactly one stable code, tested independently."""

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (CalibrationStatus.MISSING, TrustFailureCode.CALIBRATION_MISSING),
            (CalibrationStatus.STALE, TrustFailureCode.CALIBRATION_STALE),
            (CalibrationStatus.SUPERSEDED, TrustFailureCode.CALIBRATION_STALE),
            (CalibrationStatus.UNCERTIFIED, TrustFailureCode.CALIBRATION_UNCERTIFIED),
            (CalibrationStatus.REJECTED, TrustFailureCode.CALIBRATION_UNCERTIFIED),
            (CalibrationStatus.DRAFT, TrustFailureCode.CALIBRATION_UNCERTIFIED),
            (CalibrationStatus.LEGACY_UNBOUND, TrustFailureCode.CALIBRATION_UNCERTIFIED),
        ],
    )
    def test_status_maps_to_stable_code(
        self, status: CalibrationStatus, expected: TrustFailureCode
    ) -> None:
        assert code_for_status(status) is expected

    def test_certified_has_no_failure_code(self) -> None:
        assert code_for_status(CalibrationStatus.CERTIFIED) is None


class TestPolicyDefaults:
    """Strict is the production default. Development must be asked for by name."""

    def test_default_policy_is_strict(self) -> None:
        assert TrustPolicy().mode is TrustMode.STRICT
        assert TrustPolicy().strict is True

    def test_development_requires_explicit_construction(self) -> None:
        policy = TrustPolicy.development()
        assert policy.mode is TrustMode.DEVELOPMENT
        assert policy.strict is False

    def test_strict_constructor(self) -> None:
        assert TrustPolicy.strict_policy().mode is TrustMode.STRICT

    def test_policy_is_frozen(self) -> None:
        policy = TrustPolicy()
        with pytest.raises(AttributeError):
            policy.mode = TrustMode.DEVELOPMENT  # type: ignore[misc]


class TestRefusalCarriesNoCorpusBytes:
    """A strict refusal is an answer about the SYSTEM, never about the corpus."""

    def test_refusal_message_excludes_supplied_corpus_text(self) -> None:
        secret = "SENTINEL_CORPUS_TEXT_bd41f2"
        refusal = TrustRefusal(
            code=TrustFailureCode.CALIBRATION_MISSING,
            calibration_status="missing",
            tenant_id="acme",
            generation_id="gen_1",
        )
        rendered = f"{refusal} {refusal.to_dict()!r} {refusal.advice}"
        assert secret not in rendered

    def test_refusal_exposes_code_and_identity_only(self) -> None:
        refusal = TrustRefusal(
            code=TrustFailureCode.CALIBRATION_STALE,
            calibration_status="stale",
            tenant_id="acme",
            generation_id="gen_9",
        )
        payload = refusal.to_dict()
        assert payload["code"] == "CALIBRATION_STALE"
        assert payload["trust_state"] == "refused"
        assert payload["tenant_id"] == "acme"
        assert payload["generation_id"] == "gen_9"
        assert payload["calibration_status"] == "stale"
        # The refusal must not carry a hits/chunks/text channel at all.
        assert "hits" not in payload
        assert "chunks" not in payload
        assert "text" not in payload
        assert "preview" not in payload

    def test_query_text_is_not_echoed(self) -> None:
        """Query text is caller data and is deliberately not echoed into refusals or logs."""
        refusal = TrustRefusal(
            code=TrustFailureCode.INDEX_NOT_READY,
            calibration_status="missing",
            tenant_id="acme",
            generation_id=None,
        )
        assert "query" not in refusal.to_dict()


class TestAdviceDistinguishesGateFromOutage:
    """Requirement 13: 'nothing found' and 'the gate was down' are different answers."""

    def test_dependency_unavailable_advice_says_no_decision_was_possible(self) -> None:
        refusal = TrustRefusal(
            code=TrustFailureCode.DEPENDENCY_UNAVAILABLE,
            calibration_status="missing",
            tenant_id="acme",
            generation_id=None,
        )
        advice = refusal.advice.lower()
        assert "no trustworthy decision" in advice
        # It must NOT be confusable with a working gate that found nothing.
        assert "no answer was found" not in advice

    def test_calibration_missing_advice_names_the_remedy(self) -> None:
        refusal = TrustRefusal(
            code=TrustFailureCode.CALIBRATION_MISSING,
            calibration_status="missing",
            tenant_id="acme",
            generation_id="gen_1",
        )
        assert "no trustworthy decision" in refusal.advice.lower()

    def test_every_code_has_advice(self) -> None:
        for code in TrustFailureCode:
            refusal = TrustRefusal(
                code=code,
                calibration_status="missing",
                tenant_id="t",
                generation_id=None,
            )
            assert refusal.advice.strip(), f"{code} has no advice"


class TestTrustState:
    def test_states(self) -> None:
        assert TrustState.TRUSTED.value == "trusted"
        assert TrustState.DEGRADED.value == "degraded"
        assert TrustState.REFUSED.value == "refused"
