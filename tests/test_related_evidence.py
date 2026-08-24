from datetime import UTC, datetime

import pytest

from recall.calibration import Calibration
from recall.related import trusted_related
from recall.trust_policy import TrustPolicy
from recall.types import Chunk


class Store:
    tenant = "tenant-a"
    generation_id = "gen-1"

    def __init__(self):
        self.chunks = [
            Chunk("seed", "notes.md", "seed", {"file": "notes.md", "ord": 1}),
            Chunk("neighbor", "notes.md", "neighbor", {"file": "notes.md", "ord": 2}),
            Chunk("other", "other.md", "other", {"file": "other.md", "ord": 1}),
        ]

    def iter_chunks(self, batch_size=1000):
        del batch_size
        return iter(self.chunks)

    def supersession_all(self):
        return {}, frozenset(), {}


def test_related_evidence_filters_by_relation_and_trusts_each_candidate() -> None:
    result = trusted_related(
        Store(),
        "seed",
        relation="source",
        calibration=Calibration(embedder="fixture", threshold=0.5),
        now=datetime(2026, 8, 24, tzinfo=UTC),
        explain=True,
    )
    assert [item.chunk.id for item in result.items] == ["neighbor"]
    assert result.rejected_count == 0
    assert result.explanation["reason_code"] == "structural_relatedness"


def test_related_evidence_uses_store_bounded_path() -> None:
    store = Store()
    calls = []

    def related_chunks(seed_chunk_id, relation, max_items):
        calls.append((seed_chunk_id, relation, max_items))
        return store.chunks[0], [store.chunks[1]]

    store.related_chunks = related_chunks
    store.iter_chunks = lambda batch_size=1000: (_ for _ in ()).throw(
        AssertionError("bounded relation should not scan the corpus")
    )
    result = trusted_related(
        store,
        "seed",
        relation="source",
        max_items=1,
        calibration=Calibration(embedder="fixture", threshold=0.5),
        policy=TrustPolicy.development(),
        now=datetime(2026, 8, 24, tzinfo=UTC),
    )
    assert [item.chunk.id for item in result.items] == ["neighbor"]
    assert calls == [("seed", "source", 1)]


def test_related_evidence_rejects_unbounded_item_limit() -> None:
    with pytest.raises(ValueError, match="<= 50"):
        trusted_related(Store(), "seed", max_items=51)
