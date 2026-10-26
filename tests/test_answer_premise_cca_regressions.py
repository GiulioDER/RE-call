"""PR989 profile, successor identity and strict diagnostic decoding regressions."""

from __future__ import annotations

from dataclasses import replace

import pytest

from recall.evidence import AnswerEnvelope, EvidenceValidationError
from recall.reasoning import ReasoningProviderPorts, reason, reasoning_response_from_dict
from tests.test_answer_premise_profile import CORRECTION, _hit, _request, _result


@pytest.mark.parametrize("profile", ["plain", "dated"])
def test_typed_provider_cannot_enable_premise_correction_outside_profile(profile: str) -> None:
    """BUG-001: reason must reject a typed correction without the opted-in profile.

    Baseline 9bd54aaa: _answer_from_evidence emits an uncited corrected response instead of raising.
    """
    hits = [_hit("current", "new.md", "ok")]
    request = _request(profile, {}, hits)
    providers = ReasoningProviderPorts(
        retriever=lambda _request: _result(hits),
        answer_provider=lambda _system, _user: AnswerEnvelope(None, (), True, CORRECTION),
    )
    with pytest.raises(EvidenceValidationError):
        reason(replace(request, providers=providers))


@pytest.mark.parametrize("successor", ["/one/new.md", "C:\\one\\new.md"])
def test_absolute_successor_cannot_cite_another_roots_same_named_memo(successor: str) -> None:
    """BUG-002: a lossy relative label must not impersonate the declared absolute source.

    Baseline 9bd54aaa: stale_premise_notice auto-cites wrong-root instead of refusing the correction.
    """
    hits = [
        _hit("wrong-root", "new.md", "ok", source="/two/new.md"),
        _hit("old", "old.md", "superseded", superseded_by=successor),
    ]
    output = {"answer": None, "citations": [], "insufficient_evidence": True, "premise_correction": CORRECTION}
    with pytest.raises(EvidenceValidationError, match="requires at least one citation"):
        reason(_request("dated-premise", output, hits))


def test_relative_successor_cannot_cite_a_nested_same_named_memo() -> None:
    """SEC-001: nested/new.md and new.md remain different root-relative identities.

    Baseline 9bd54aaa: suffix matching auto-cites the nested memo instead of refusing.
    """
    hits = [
        _hit("nested", "nested/new.md", "ok"),
        _hit("old", "old.md", "superseded", superseded_by="new.md"),
    ]
    output = {"answer": None, "citations": [], "insufficient_evidence": True, "premise_correction": CORRECTION}
    with pytest.raises(EvidenceValidationError, match="requires at least one citation"):
        reason(_request("dated-premise", output, hits))


@pytest.mark.parametrize("invalid", ["false", "true", "", 0, 1, None, [], {}, ["x"]])
def test_premise_grounding_diagnostic_rejects_non_boolean_values(invalid: object) -> None:
    """DAT-001: the strict wire decoder must reject coercion into citation provenance.

    Baseline 9bd54aaa: reasoning_response_from_dict accepts all these values via bool().
    """
    output = {"answer": "a group", "citations": ["current"], "insufficient_evidence": False}
    payload = reason(_request("dated", output, [_hit("current", "new.md", "ok")])).to_dict()
    payload["diagnostics"]["premise_citations_from_notice"] = invalid  # type: ignore[index]
    with pytest.raises(EvidenceValidationError, match="expected boolean"):
        reasoning_response_from_dict(payload)
