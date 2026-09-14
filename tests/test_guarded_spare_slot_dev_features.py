from __future__ import annotations

from pathlib import Path

from recall.source_conditioning import chunk_identifier_hash, load_source_conditioning_artifact
from scripts.capture_guarded_spare_slot_dev_features import analyze_query


def _item(chunk_id: str, source: str, rank: int = 1) -> dict:
    return {
        "chunk_id": chunk_id,
        "source": source,
        "ordinal": 0,
        "pool_rank": rank,
        "text": "private evidence",
        "cosine": 0.4,
        "confidence": 0.7,
        "verdict": "ok",
    }


def test_dev_features_label_additions_by_frozen_source() -> None:
    """Red proof for ``analyze_query``: chunk-hash comparison mislabeled the gold source."""
    artifact = load_source_conditioning_artifact(
        Path("docs/results/2026-09-13-source-conditioning-model.json")
    )
    item = _item("gold:0", "gold.md")
    item_hash = chunk_identifier_hash(item["chunk_id"])
    query = {
        "id": "q1",
        "expected_answerability": "answerable",
        "gold_sources": ["gold.md"],
    }
    capture = {
        "base_hashes": [],
        "candidate_hashes": [item_hash],
    }
    pool_audit = {"items": [item]}
    legs = {
        "dense": [{"chunk_id": "gold:0", "source": "gold.md", "rank": 1, "cosine": 0.4}],
        "sparse": [{"chunk_id": "gold:0", "source": "gold.md", "rank": 1, "cosine": 0.4}],
    }

    row = analyze_query(query, capture, pool_audit, legs, artifact)

    assert row["base_gold_source_hit"] is False
    assert len(row["additions"]) == 1
    assert row["additions"][0]["is_gold_source"] is True
    assert row["additions"][0]["dense_rank"] == 1
    assert row["additions"][0]["sparse_rank"] == 1
