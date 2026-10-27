"""CCA regressions for diagnostic transport and stored report consistency.

Red proof baseline: 86ce3921c046d4deea285730e9692de3dd9013b9, with only this test file added. Command:
python -m pytest tests/test_query_support_reasoning.py -q.
Before production repairs, 18 intended assertion failures and 7 passing controls, zero errors:
test_reasoning_wire_roundtrip_preserves_coverage[measured,unavailable] lost diagnostics;
test_reasoning_wire_rejects_invalid_diagnostics (all seven cases) failed to reject bad fields;
test_reasoning_expansion_preserves_compatible_coverage[same,different_report] lost coverage;
test_proof_repair_preserves_compatible_coverage[same,different_report] lost coverage;
test_inconsistent_stored_coverage_is_unavailable (all five cases) accepted damaged reports.
All node IDs use this module's tests/test_query_support_reasoning.py:: prefix.
The legacy, mixed-fit and valid-report controls already passed the baseline.
After the repairs, all 25 cases passed, alongside all 23 original contract cases.
"""
from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from recall.evidence import EvidenceValidationError
from recall.lite.calibration import auto_calibrate
from recall.proof_obligations import run_proof_obligations
from recall.reasoning import ReasoningPolicy, reason, reasoning_response_from_dict
from recall.reasoning_planner import ReasoningBudget
from recall.support import CalibrationCoverage
from tests.test_proof_obligations import _assessment, _bundle, _item
from tests.test_query_support_contract import _report
from tests.test_reasoning_api import _chunk, _hit, _request, _result
from tests.test_lite_calibration import ContentWordEmbedder, _store, _strict


@pytest.fixture
def served(tmp_path):
    store = _store(tmp_path, 48)
    outcome = auto_calibrate(store, ContentWordEmbedder())
    assert outcome.artifact is not None and outcome.artifact.certified
    yield store
    store.close()


def _coverage(status="measured", calibration_id="cal-1"):
    return CalibrationCoverage(
        calibration_id=calibration_id, status=status,
        method="synthetic_corpus_probes" if status == "measured" else None,
        report=_report(calibration_id) if status == "measured" else None,
    )


def _response():
    retrieval = replace(_result(_hit(_chunk("c1", "owner.md", "Ada owns rollout."))),
                        calibration_coverage=_coverage())
    return reason(_request(retrieval))


@pytest.mark.parametrize("status", ["measured", "unavailable"])
def test_reasoning_wire_roundtrip_preserves_coverage(status):
    response = _response()
    response = replace(response, trusted_evidence=replace(
        response.trusted_evidence, calibration_coverage=_coverage(status)))
    decoded = reasoning_response_from_dict(json.loads(json.dumps(response.to_dict())))
    assert decoded.trusted_evidence == response.trusted_evidence, "reasoning decoder lost diagnostics"


def test_reasoning_wire_legacy_defaults():
    payload = _response().to_dict()
    payload["trusted_evidence"].pop("query_support")
    payload["trusted_evidence"].pop("calibration_coverage")
    decoded = reasoning_response_from_dict(payload)
    assert decoded.trusted_evidence.query_support.status == "not_checked"
    assert decoded.trusted_evidence.calibration_coverage == CalibrationCoverage()


@pytest.mark.parametrize("field,key,value", [
    ("query_support", "status", "supported"),
    ("query_support", "scope", "whole_corpus"),
    ("query_support", "method", "semantic_verifier"),
    ("calibration_coverage", "status", "certified"),
    ("calibration_coverage", "method", "answerability"),
    ("calibration_coverage", "report", None),
    ("calibration_coverage", "calibration_id", "different-fit"),
])
def test_reasoning_wire_rejects_invalid_diagnostics(field, key, value):
    payload = _response().to_dict()
    payload["trusted_evidence"][field][key] = value
    with pytest.raises(EvidenceValidationError):
        reasoning_response_from_dict(payload)


@pytest.mark.parametrize("kind", ["same", "different_fit", "different_report"])
def test_reasoning_expansion_preserves_compatible_coverage(kind):
    initial = replace(_result(_hit(_chunk("first", "first.md", "Project identity.")), gap_warning=True),
                      calibration_coverage=_coverage())
    coverage = (_coverage(calibration_id="cal-2") if kind == "different_fit" else
                _coverage("unavailable") if kind == "different_report" else _coverage())
    expanded = replace(_result(_hit(_chunk("second", "second.md", "Ada owns rollout."))),
                       calibration_id=coverage.calibration_id, calibration_coverage=coverage)
    response = reason(_request(
        initial, policy=ReasoningPolicy(allow_retrieval_expansion=True),
        budget=ReasoningBudget(max_model_calls=1), expansion_retriever=lambda *_: expanded,
        answer=lambda *_: {"answer": "Ada owns rollout.", "citations": ["second"],
                           "insufficient_evidence": False},
    ))
    assert response.outcome == "answered"
    assert response.diagnostics.retrieval_expansion.accepted_chunk_ids == ("second",)
    expected = _coverage() if kind == "same" else CalibrationCoverage(
        calibration_id=None if kind == "different_fit" else "cal-1")
    assert response.trusted_evidence.calibration_coverage == expected, "expansion misrepresented coverage"
    assert response.trusted_evidence.query_support.status == "not_checked"


@pytest.mark.parametrize("kind", ["same", "different_fit", "different_report"])
def test_proof_repair_preserves_compatible_coverage(kind):
    initial = replace(_bundle(_item("one", "Project alpha ships.")), calibration_coverage=_coverage())
    coverage = (_coverage(calibration_id="cal-2") if kind == "different_fit" else
                _coverage("unavailable") if kind == "different_report" else _coverage())
    repair = replace(_bundle(_item("two", "The release is 2026-10-15.")), calibration_coverage=coverage)
    outputs = iter([
        _assessment(date_supported=False, repair={"query": "release date", "missing_slot_ids": ["release_date"]}),
        _assessment(date_supported=True, repair=None),
    ])
    result = run_proof_obligations(initial, lambda *_: next(outputs), lambda *_: repair)
    assert result.decision == "sufficient"
    expected = _coverage() if kind == "same" else CalibrationCoverage(
        calibration_id=None if kind == "different_fit" else "cal-1")
    assert result.evidence.calibration_coverage == expected, "proof repair misrepresented coverage"
    assert result.evidence.query_support.status == "not_checked"


@pytest.mark.parametrize("kind", ["zero_measured", "few_measured", "wrong_rate", "wrong_state", "warning"])
def test_inconsistent_stored_coverage_is_unavailable(served, kind):
    before = _strict(served)
    report = _report(before.calibration_id)
    if kind in {"zero_measured", "few_measured"}:
        n = 0 if kind == "zero_measured" else 9
        report["in_vocabulary"] = {"n": n, "rejected": n, "rate": 1.0 if n else None, "state": "measured"}
    elif kind == "wrong_rate":
        report["in_vocabulary"]["rate"] = 1.0
    elif kind == "wrong_state":
        report["in_vocabulary"]["state"] = "unmeasured"
    else:
        report["provisional_in_vocabulary"] = False
    served.save_near_miss_coverage(before.calibration_id, report)
    after = _strict(served)
    assert after.calibration_coverage.status == "unavailable", "inconsistent report survived serving"
    assert after.hits == before.hits and after.abstained == before.abstained
    assert after.trust_state == before.trust_state


@pytest.mark.parametrize("n,rejected", [(0, 0), (9, 9), (209, 188), (10, 9)])
def test_valid_stored_coverage_preserves_exact_warning(served, n, rejected):
    before = _strict(served)
    report = _report(before.calibration_id)
    report["in_vocabulary"] = {"n": n, "rejected": rejected,
                               "rate": round(rejected / n, 3) if n else None,
                               "state": "measured" if n >= 10 else "unmeasured"}
    report["provisional_in_vocabulary"] = n < 10 or rejected / n < 0.9
    served.save_near_miss_coverage(before.calibration_id, report)
    coverage = asdict(_strict(served).calibration_coverage)
    assert coverage["status"] == ("measured" if n >= 10 else "unmeasured")
    assert coverage["report"] == report
