from __future__ import annotations

from recall.query_anchor_admission import (
    query_anchor_candidate_eligible,
    query_anchor_features,
)


def _item(source: str, text: str) -> dict[str, object]:
    return {"chunk_id": source, "source": source, "text": text}


def test_anchor_features_prioritize_the_rarest_query_token() -> None:
    """The absent anchor must outrank generic terms in the query.

    Red proof targets ``query_anchor_features``. The deliberate mutation sorted document frequency
    descending, so the common token appeared first and the intended rare-anchor assertion failed.
    """
    proposal = _item("gold.md", "A durable result appears in this source.")
    other = _item("other.md", "A different result appears in another source.")

    features = query_anchor_features(
        "What result was recorded for velnora durable?", proposal, [proposal, other]
    )

    assert features["anchor_document_frequencies"][0] == 0
    assert features["zero_document_frequency_anchors"] == 1
    assert features["uncovered_zero_document_frequency_anchors"] == 1
    assert features["covered_by_proposal_source"] == 2


def test_anchor_features_join_all_pool_chunks_from_the_proposed_source() -> None:
    proposal = _item("gold.md", "Primary passage without the durable anchor.")
    sibling = _item("gold.md", "The durable evidence appears in a sibling chunk.")

    features = query_anchor_features(
        "What durable evidence was recorded?", proposal, [proposal, sibling]
    )

    assert features["covered_by_proposal_chunk"] == 1
    assert features["covered_by_proposal_source"] == 2
    assert features["source_coverage_fraction"] == 1.0


def test_anchor_candidate_only_fills_an_empty_base() -> None:
    """A nonempty base must not receive an anchor proposal.

    Red proof targets ``query_anchor_candidate_eligible``. The deliberate mutation accepted a
    one-item base, so this behavioral assertion failed.
    """
    features = {
        "anchor_count": 3,
        "zero_document_frequency_anchors": 0,
        "chunk_coverage_fraction": 2.0 / 3.0,
    }

    assert query_anchor_candidate_eligible(0, features)
    assert not query_anchor_candidate_eligible(1, features)


def test_anchor_candidate_requires_three_supported_anchors() -> None:
    features = {
        "anchor_count": 3,
        "zero_document_frequency_anchors": 0,
        "chunk_coverage_fraction": 2.0 / 3.0,
    }

    assert query_anchor_candidate_eligible(0, features)
    assert not query_anchor_candidate_eligible(0, {**features, "anchor_count": 2})
    assert not query_anchor_candidate_eligible(
        0, {**features, "zero_document_frequency_anchors": 1}
    )
    assert not query_anchor_candidate_eligible(
        0, {**features, "chunk_coverage_fraction": 0.5}
    )
