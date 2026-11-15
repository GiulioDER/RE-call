from datetime import UTC, datetime

import pytest

from recall.calibration import Calibration
from recall.related import trusted_related
from recall.security_policy import AccessContext, SourceRule, SourceSecurityPolicy
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


def test_related_evidence_does_not_return_stale_redaction_rows() -> None:
    """At 65b7a791c the old neighbor and the current neighbor were both returned."""
    policy = SourceSecurityPolicy((SourceRule("notes.md", redactions=("email",)),))
    store = Store()
    store.chunks[0].metadata["security_policy_digest"] = policy.digest
    store.chunks.append(Chunk(
        "current", "notes.md", "current redacted text",
        {"file": "notes.md", "ord": 3, "security_policy_digest": policy.digest},
    ))
    result = trusted_related(
        store, "seed", relation="source", calibration=Calibration(embedder="fixture", threshold=0.5),
        security_policy=policy, access_context=AccessContext("reader", store.tenant),
    )
    assert [item.chunk.id for item in result.items] == ["current"]


def test_related_evidence_rejects_unbounded_item_limit() -> None:
    with pytest.raises(ValueError, match="<= 50"):
        trusted_related(Store(), "seed", max_items=51)


class _BoundedSourceStore(Store):
    """`related_chunks` as the PostgreSQL stores answer `source`: resolver order, then LIMIT."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.windows = []

    def iter_chunks(self, batch_size=1000):
        raise AssertionError("a bounded relation must not scan the corpus")

    def related_chunks(self, seed_chunk_id, relation, max_items):
        self.windows.append(max_items)
        seed = next(c for c in self.chunks if c.id == seed_chunk_id)
        same = [c for c in self.chunks if c.source == seed.source and c.id != seed.id]
        same.sort(key=lambda c: (abs(c.metadata["ord"] - seed.metadata["ord"]), c.id))
        return seed, same[:max_items]


def _stale_neighbours(policy, denied_ords):
    """One source, ords 0..10, seed 5; the ords listed carry an older policy digest."""
    return [
        Chunk(
            f"c{i}", "notes.md", f"part {i}",
            {"file": "notes.md", "ord": i} | ({} if i in denied_ords else {"security_policy_digest": policy.digest}),
        )
        for i in range(11)
    ]


def test_related_evidence_widens_past_chunks_the_policy_denies() -> None:
    """The bounded window must not be spent on rows the policy then drops.

    Invariant: with a security policy, `trusted_related` returns the max_items nearest ALLOWED
    chunks of the relation, as the unbounded scan would. Failure mode: the store's LIMIT was
    filled by the stale rows next to the seed (c4, c6), the policy dropped them, and the current
    chunks one step further out were never fetched, so nothing came back.
    """
    policy = SourceSecurityPolicy((SourceRule("notes.md"),))
    store = _BoundedSourceStore(_stale_neighbours(policy, {4, 6}))
    result = trusted_related(
        store, "c5", relation="source", max_items=2,
        calibration=Calibration(embedder="fixture", threshold=0.5),
        security_policy=policy, access_context=AccessContext("reader", store.tenant),
    )
    assert [item.chunk.id for item in result.items] == ["c3", "c7"]


def test_related_evidence_widening_stops_when_the_relation_is_exhausted() -> None:
    """Every neighbour denied: nothing comes back, and the window grows only until it is short."""
    policy = SourceSecurityPolicy((SourceRule("notes.md"),))
    store = _BoundedSourceStore(_stale_neighbours(policy, set(range(11)) - {5}))
    result = trusted_related(
        store, "c5", relation="source", max_items=2,
        calibration=Calibration(embedder="fixture", threshold=0.5),
        security_policy=policy, access_context=AccessContext("reader", store.tenant),
    )
    assert result.items == ()
    assert store.windows[-1] > len(store.chunks) - 1
    assert len(store.windows) <= 5
