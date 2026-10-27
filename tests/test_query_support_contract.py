"""Additive answerability and calibration coverage contract.

Red proof recorded 2026-10-18 against baseline b7f5d9c032791477f046ebc374eb77a2233b052f:
only this test file was added to the baseline. Command: python -m pytest
 tests/test_query_support_contract.py -q. All 23 cases failed in their intended assertions,
with zero setup/collection errors, then the same 23 passed with production restored.
No verifier or model inference is exercised.

Node IDs below are prefixed by tests/test_query_support_contract.py:::
- test_certified_retrieval_does_not_claim_semantic_support: TrustedResult lacked query_support.
- test_coverage_is_bound_to_the_served_calibration[measured,partial,missing,mismatch,error,malformed]:
  _trusted_search did not read or carry the selected calibration's coverage.
- test_every_bundle_branch_preserves_the_diagnostics[answer,abstain,budget,slots]:
  build_evidence_bundle omitted the diagnostics on each response branch.
- test_mcp_and_compact_surfaces_preserve_coverage[answer-search,answer-evidence,abstain-search,
  abstain-evidence]: search_memory/evidence_memory omitted the new response fields.
- test_cli_reports_unchecked_support_and_provisional_coverage: _print_result omitted the limits.
- test_model_instructions_distinguish_citation_eligibility[search,evidence]: the SDK's canonical
  description copies omitted semantic-check and coverage qualifications.
- test_carried_fit_does_not_inherit_parent_near_miss_measurements: carried results lacked the
  explicit unmeasured state for the selected calibration.
- test_coverage_is_read_inside_the_serving_snapshot: _trusted_search never read coverage.
- test_framework_documents_keep_the_same_limits: result_trust_metadata omitted the fields.
- test_backend_without_coverage_reader_is_unmeasured: unavailable backend capability was implicit.
- test_unbound_development_calibration_does_not_borrow_stored_coverage: unbound results omitted
  the explicit unmeasured state.

Additional production mutations, each restored byte for byte before the final green run:
- build_evidence_bundle: remove its three calibration_coverage projections; all four bundle
  cases fail "bundle dropped calibration coverage".
- AssistantMemos: omit both fields from SEARCH_FIELDS/EVIDENCE_FIELDS; all four MCP/compact cases
  fail "compact response lost the evidence limits". This also challenges the updated assistant
  surface snapshot's projection contract without weakening its existing assertions.
- result_trust_metadata: omit recall_calibration_coverage; the framework case fails
  "framework metadata dropped calibration coverage".
- search_memory/evidence_memory: omit ELIGIBILITY_NOTE; all four MCP/compact cases fail
  "response omitted citation eligibility limits".
- calibration_coverage: return unmeasured instead of unavailable on reader errors; error and
  malformed cases fail "coverage status was lost or overstated".
"""
from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from recall.cli_commands._shared import _print_result
from recall.evidence import AnswerSlot, EvidencePolicy, build_evidence_bundle
from recall.lite.calibration import auto_calibrate
from recall_mcp import retrieval
from recall_mcp.assistant import AssistantMemos
from recall_mcp.models import EvidenceResult, SearchResult
from recall_agent._descriptions import RECALL_EVIDENCE_DESCRIPTION, RECALL_SEARCH_DESCRIPTION
from tests.test_lite_calibration import ContentWordEmbedder, _store, _strict

EXPECTED = {"status": "not_checked", "scope": "returned_evidence", "method": None}


@pytest.fixture
def served(tmp_path):
    store = _store(tmp_path, 48)
    outcome = auto_calibrate(store, ContentWordEmbedder())
    assert outcome.artifact is not None and outcome.artifact.certified
    yield store
    store.close()


def _report(calibration_id):
    return {
        "calibration_id": calibration_id,
        "unknown_entity": {"n": 12, "rejected": 12, "rate": 1.0, "state": "measured"},
        "in_vocabulary": {"n": 12, "rejected": 3, "rate": 0.25, "state": "measured"},
        "unknown_term_gate": "checked",
        "provisional_in_vocabulary": True,
    }


def test_certified_retrieval_does_not_claim_semantic_support(served):
    result = _strict(served)
    assert result.calibrated and not result.abstained
    assert result.hits[0].verdict == "ok"
    assert asdict(result).get("query_support") == EXPECTED, "certified retrieval hid unchecked answerability"


@pytest.mark.parametrize("kind", ["measured", "partial", "missing", "mismatch", "error", "malformed"])
def test_coverage_is_bound_to_the_served_calibration(served, monkeypatch, kind):
    seen = []
    def reader(calibration_id):
        seen.append(calibration_id)
        if kind == "error":
            raise RuntimeError("optional report unavailable")
        if kind == "malformed":
            return {"calibration_id": calibration_id}
        if kind == "missing":
            return None
        report = _report(calibration_id if kind != "mismatch" else "another-fit")
        if kind == "partial":
            report["in_vocabulary"] = {"n": 0, "rejected": 0, "rate": None, "state": "unmeasured"}
        return report
    monkeypatch.setattr(served, "near_miss_coverage", reader)
    result = _strict(served)
    assert result.calibrated and not result.abstained, "optional coverage changed retrieval"
    coverage = asdict(result).get("calibration_coverage")
    status = "unavailable" if kind in {"error", "malformed"} else "measured" if kind == "measured" else "unmeasured"
    assert coverage is not None and coverage["status"] == status, "coverage status was lost or overstated"
    assert seen == [result.calibration_id], "coverage was read for a different calibration"
    assert coverage["calibration_id"] == result.calibration_id
    if kind in {"measured", "partial"}:
        assert coverage["method"] == "synthetic_corpus_probes"
        assert coverage["report"]["provisional_in_vocabulary"] is True
    else:
        assert coverage["report"] is None


@pytest.mark.parametrize("path", ["answer", "abstain", "budget", "slots"])
def test_every_bundle_branch_preserves_the_diagnostics(served, path):
    result = _strict(served)
    policy = EvidencePolicy()
    if path == "abstain":
        result = replace(result, hits=[], abstained=True, reason="empty")
    elif path == "budget":
        class Tokenizer:
            def count_tokens(self, text):
                return 1000
        policy = EvidencePolicy(max_tokens=1, tokenizer=Tokenizer())
    elif path == "slots":
        policy = EvidencePolicy(answer_slots=(AnswerSlot("missing", ("nonexistent-requirement",)),))
    bundle = asdict(build_evidence_bundle(result, policy))
    assert bundle.get("query_support") == EXPECTED, "bundle dropped the unchecked support state"
    assert bundle.get("calibration_coverage") == asdict(result).get("calibration_coverage"), "bundle dropped calibration coverage"
    assert bundle["decision"] == ("answer" if path == "answer" else "abstain")


@pytest.mark.parametrize("surface", ["search", "evidence"])
@pytest.mark.parametrize("answerable", [True, False], ids=["answer", "abstain"])
def test_mcp_and_compact_surfaces_preserve_coverage(served, tmp_path, surface, answerable):
    kwargs = dict(k=5, env={})
    query = _strict(served).query if answerable else "remote absent"
    if surface == "search":
        result = retrieval.search_memory(served, ContentWordEmbedder(), query, **kwargs)
    else:
        result = retrieval.evidence_memory(served, ContentWordEmbedder(), query, **kwargs,
                                          register_evidence_cards_fn=lambda *a, **k: None)
    payload = result.model_dump()
    assert payload.get("query_support") == EXPECTED, "MCP omitted unchecked answerability"
    assert payload.get("calibration_coverage", {}).get("calibration_id") == _strict(served).calibration_id
    assert "does not establish" in payload["advice"], "response omitted citation eligibility limits"
    assert (not payload["abstained"] if surface == "search" else payload["decision"] == "answer") == answerable
    memos = AssistantMemos(tmp_path / "assistant", index_root=tmp_path)
    compact = memos.render_search(payload) if surface == "search" else memos.render_evidence(payload)
    for key in ("query_support", "calibration_coverage"):
        assert compact.get(key) == payload[key], "compact response lost the evidence limits"
    legacy = {k: v for k, v in payload.items() if k not in {"query_support", "calibration_coverage"}}
    decoded = (SearchResult if surface == "search" else EvidenceResult).model_validate(legacy).model_dump()
    assert decoded.get("query_support") == EXPECTED
    assert decoded.get("calibration_coverage", {}).get("status") == "unmeasured"


def test_cli_reports_unchecked_support_and_provisional_coverage(served, capsys):
    _print_result(_strict(served))
    output = capsys.readouterr().out
    assert "query support: not_checked" in output, "CLI implied answerability from retrieval"
    assert "calibration coverage:" in output
    assert "provisional" in output


@pytest.mark.parametrize("description", [RECALL_SEARCH_DESCRIPTION, RECALL_EVIDENCE_DESCRIPTION], ids=["search", "evidence"])
def test_model_instructions_distinguish_citation_eligibility(description):
    assert "does not establish" in description and "query_support" in description, "tool instructions imply semantic support"
    assert "calibration_coverage" in description and "unmeasured" in description


def test_carried_fit_does_not_inherit_parent_near_miss_measurements(served, tmp_path):
    from tests.test_lite_carry_forward import _add

    parent = _strict(served).calibration_id
    _add(served, tmp_path, 1)
    result = _strict(served)
    assert result.calibrated and result.calibration_id != parent
    coverage = asdict(result).get("calibration_coverage")
    assert coverage is not None and coverage["status"] == "unmeasured", "carried fit inherited unmeasured coverage"
    assert coverage["calibration_id"] == result.calibration_id and coverage["report"] is None


def test_coverage_is_read_inside_the_serving_snapshot(served, monkeypatch):
    original = served.near_miss_coverage
    snapshots = []
    def reader(calibration_id):
        snapshots.append(served._conn.in_transaction)
        return original(calibration_id)
    monkeypatch.setattr(served, "near_miss_coverage", reader)
    _strict(served)
    assert snapshots == [True], "coverage was read outside the serving snapshot"


def test_framework_documents_keep_the_same_limits(served):
    from recall.integrations import result_trust_metadata

    result = _strict(served)
    metadata = result_trust_metadata(result)
    assert metadata.get("recall_query_support") == EXPECTED, "framework metadata lost unchecked support"
    assert metadata.get("recall_calibration_coverage") == asdict(result).get("calibration_coverage"), "framework metadata dropped calibration coverage"


def test_backend_without_coverage_reader_is_unmeasured(served, monkeypatch):
    monkeypatch.setattr(served, "near_miss_coverage", None)
    result = _strict(served)
    coverage = asdict(result).get("calibration_coverage")
    assert coverage is not None and coverage["status"] == "unmeasured", "missing backend support became a pass"
    assert coverage["calibration_id"] == result.calibration_id and coverage["report"] is None


def test_unbound_development_calibration_does_not_borrow_stored_coverage(served, monkeypatch):
    from recall.calibration import Calibration
    from recall.trust import trusted_search
    from recall.trust_policy import TrustPolicy

    seen = []
    monkeypatch.setattr(served, "near_miss_coverage", lambda cid: seen.append(cid))
    result = trusted_search(served, ContentWordEmbedder(), "who services flange",
                            calibration=Calibration(embedder="fixture", threshold=0.1),
                            policy=TrustPolicy.development(), env={})
    assert result.trust_state == "degraded" and result.calibration_id is None
    coverage = asdict(result).get("calibration_coverage")
    assert coverage is not None and coverage["status"] == "unmeasured", "unbound retrieval borrowed a certificate"
    assert coverage["calibration_id"] is None and seen == []
    assert asdict(result).get("query_support") == EXPECTED


class _CoverageStore:
    """A store whose only capability is the coverage reader, returning one fixed report."""

    def __init__(self, report):
        self._report = report

    def near_miss_coverage(self, _calibration_id):
        return self._report


def _three_class_report(calibration_id):
    report = _report(calibration_id)
    report["paraphrase"] = {"n": 3, "rejected": 1, "rate": 0.333, "state": "unmeasured"}
    return report


def test_cca945_every_probe_class_decides_measured(monkeypatch):
    """Node tests/test_query_support_contract.py::test_cca945_every_probe_class_decides_measured.

    CCA DEEP run 20261027T143257Z, CODE-001 (with ENV-002). The top-level status must follow every
    class in `recall.support.CLASSES`, which the validator already requires. Red against bde16292:
    both the reader and the decoder judge only the two classes they spell out, so a report whose
    added class is unmeasured reads `measured` and `coverage overstated an unmeasured class` fails.
    Target: `recall.support.coverage_status`.
    """
    from recall import support
    from recall.reasoning_response import EvidenceValidationError, _coverage_from_dict

    monkeypatch.setattr(support, "CLASSES", (*support.CLASSES, "paraphrase"))
    coverage = support.calibration_coverage(_CoverageStore(_three_class_report("fit-1")), "fit-1")
    assert coverage.status == "unmeasured", f"coverage overstated an unmeasured class: {coverage.status}"
    payload = {"calibration_id": "fit-1", "status": "measured", "method": "synthetic_corpus_probes",
               "report": _three_class_report("fit-1")}
    with pytest.raises(EvidenceValidationError, match="disagrees"):
        _coverage_from_dict(payload)
