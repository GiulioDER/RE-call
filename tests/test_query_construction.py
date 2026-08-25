from __future__ import annotations

import json

import pytest

from recall.query_construction import (
    QueryConstructionRequest,
    QueryProposal,
    RetrievalSignal,
    build_original_model_challenge,
    parse_query_frame,
    should_request_original_model_refinement,
    validate_query_proposals,
)


def _request() -> QueryConstructionRequest:
    return QueryConstructionRequest(
        original_prompt="Bump the package version with a rerunnable release script.",
        original_query="version bump release script",
        trusted_evidence=({"chunk_id": "c1", "source": "project.md", "text": "release", "verdict": "ok"},),
        graph_anchors=("project.md",),
        gap_reason="governing memory not found",
    )


def test_original_model_challenge_is_data_delimited_and_bounded() -> None:
    challenge = build_original_model_challenge(_request())
    assert "Return JSON only" in challenge.prompt
    assert "<retrieval_data>" in challenge.prompt
    payload = json.loads(challenge.prompt.split("<retrieval_data>", 1)[1].split("</retrieval_data>", 1)[0])
    assert payload["evidence"][0]["chunk_id"] == "c1"
    assert challenge.round_index == 0


def test_query_frame_parser_keeps_model_frame_separate_from_evidence() -> None:
    frame = parse_query_frame(
        {
            "task_object": "package release",
            "intended_action": "bump version safely",
            "failure_or_risk": "release drift",
            "memory_need": "the governing release procedure",
            "artifacts": ["version.py", "release script"],
            "query": "release version procedure version.py",
        }
    )
    assert frame.artifacts == ("version.py", "release script")
    assert frame.query.endswith("version.py")


def test_query_controls_reject_untrusted_parents_duplicates_and_frame_rewrites_without_novelty() -> None:
    result = validate_query_proposals(
        _request(),
        [
            QueryProposal("release version procedure", "intent", parent_chunk_ids=("c1",)),
            QueryProposal("release version procedure", "anchor", parent_chunk_ids=("c1",)),
            QueryProposal("new risk", "intent", parent_chunk_ids=("missing",)),
            QueryProposal("version bump release script", "intent"),
        ],
    )
    assert [item.query for item in result.accepted] == ["release version procedure"]
    assert {reason for _, reason in result.rejected} == {
        "duplicate_query",
        "untrusted_parent",
        "no_query_novelty",
    }


def test_refinement_is_bounded_to_one_followup_round() -> None:
    signal = RetrievalSignal(trusted_items=2, new_trusted_items=0, gap_warning=False, agent_says_need_more=False)
    assert should_request_original_model_refinement(signal, round_index=0)
    assert not should_request_original_model_refinement(signal, round_index=1)


def test_query_request_rejects_invalid_round() -> None:
    with pytest.raises(ValueError, match="round_index"):
        QueryConstructionRequest(original_prompt="task", original_query="query", round_index=2)
