"""C8 graph relations over Code4 content-only word windows."""

from __future__ import annotations

from recall.types import ScoredChunk
from recall_aml.graph import attach_grounded_relations, promote_grounded_raw
from recall_aml.models import AddRequest, CodingMemoryRecord, EvidenceSpan
from recall_aml.service import build_chunks


def test_c8_word_window_carries_message_membership_for_grounded_graph_edges() -> None:
    """A C8 raw window can receive a relation from an exact span in its second message.

    Red proof: without ``message_ordinals`` the compiled record has no relation because a Code4
    word window lacks the character offsets used by the legacy raw chunk format.
    """
    marker = "C8_WINDOW_GRAPH_MARKER"
    request = AddRequest.model_validate(
        {
            "request_id": "c8-window-graph",
            "user_id": "c8-window-user",
            "session_id": "c8-window-session",
            "messages": [
                {"role": "user", "content": "diagnose graph relation admission"},
                {"role": "assistant", "content": f"{marker} fixed in src/graph_target.py"},
            ],
        }
    )
    second = request.messages[1].content
    assert isinstance(second, str)
    record = CodingMemoryRecord(
        kind="successful repair",
        action=marker,
        evidence_spans=[
            EvidenceSpan(
                message_ordinal=1,
                start=0,
                end=len(marker),
                quote=marker,
            )
        ],
        source_session_id=request.session_id,
    )
    chunks = build_chunks(
        request,
        [record],
        word_window_size=160,
        word_window_stride=120,
        content_only_windows=True,
        stable_window_identity=True,
    )
    linked = attach_grounded_relations(request, chunks)
    raw = [chunk for chunk in linked if chunk.metadata["record_type"] == "raw"]
    compiled = next(chunk for chunk in linked if chunk.metadata["record_type"] == "compiled")

    assert raw[0].metadata["message_ordinals"] == [0, 1]
    assert len(compiled.metadata["recall_graph"]["relations"]) == 1
    promoted = promote_grounded_raw(
        [ScoredChunk(raw[0], 0.9)], [ScoredChunk(compiled, 0.9)]
    )
    assert promoted.relation_hits == 1
