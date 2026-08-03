import pytest

from recall.context import ContextPolicy, contextual_passages
from recall.profiles import resolve_retrieval_profile


def test_context_embedding_preserves_public_chunk_text() -> None:
    body = "# Title\n\n## Section\n\nImportant fact.\n\nNext fact."
    chunks = ["# Title\n\n## Section", "Important fact.", "Next fact."]
    structured, passages = contextual_passages(
        body, body, chunks, "team/notes.md", ContextPolicy(mode="neighbor")
    )
    assert [chunk.text for chunk in structured] == chunks
    assert "content: Important fact." in passages[1]
    assert "source: team/notes.md" in passages[1]
    assert "previous:" in passages[1] and "following:" in passages[1]


def test_retrieval_profiles_refuse_conflicting_legacy_switch() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        resolve_retrieval_profile(
            {"RECALL_RETRIEVAL_PROFILE": "fast", "RECALL_RERANK": "true"}
        )
    assert resolve_retrieval_profile({"RECALL_RETRIEVAL_PROFILE": "quality"}).reranker
    assert not resolve_retrieval_profile({"RECALL_RETRIEVAL_PROFILE": "fast"}).reranker
