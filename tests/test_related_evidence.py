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


@pytest.mark.parametrize("relation", ["source", "ordinal"])
@pytest.mark.parametrize("deny_window", [False, True])
def test_related_expansion_reads_source_metadata_once(tmp_path, monkeypatch, relation, deny_window):
    """PERF-001. Denied neighbors must not cause repeated ranked full-text queries.

    Red proof against 8b3bb14d3: windows contains repeated, growing LIMITs. The same
    public request must keep nearest allowed rows, ordinal membership and stored text.
    """
    from recall.lite import LiteStore

    policy = SourceSecurityPolicy((SourceRule("notes.md"),))
    denied = {3, 4, 6, 7} if deny_window else {4, 6}
    chunks = _stale_neighbours(policy, denied)
    chunks.append(Chunk("unrelated", "other.md", "other body", {"ord": 5}))
    expected = [] if relation == "ordinal" and deny_window else (["c2", "c8"] if deny_window else ["c3", "c7"])
    with LiteStore(tmp_path / "related.db", dim=1) as store:
        store.upsert(chunks, [[1.0]] * len(chunks))
        windows, streams, hydrated = [], [], []
        original_query, original_stream, original_by_id = store.related_chunks, store.iter_scoped_chunks, store.chunk_by_id

        def query(seed, requested_relation, max_items):
            windows.append(max_items)
            return original_query(seed, requested_relation, max_items)

        def stream(scope, **kwargs):
            streams.append((kwargs.get("sources"), kwargs.get("include_text")))
            yield from original_stream(scope, **kwargs)

        def by_id(chunk_id):
            hydrated.append(chunk_id)
            return original_by_id(chunk_id)

        monkeypatch.setattr(store, "related_chunks", query)
        monkeypatch.setattr(store, "iter_scoped_chunks", stream)
        monkeypatch.setattr(store, "chunk_by_id", by_id)
        result = trusted_related(
            store, "c5", relation=relation, max_items=2,
            calibration=Calibration(embedder="fixture", threshold=0.5),
            security_policy=policy, access_context=AccessContext("reader", store.tenant),
        )
    assert [hit.chunk.id for hit in result.items] == expected
    assert [hit.chunk.text for hit in result.items] == [f"part {chunk_id[1:]}" for chunk_id in expected]
    assert windows == [2], f"repeated ranked text windows: {windows}"
    assert streams == [(["notes.md"], False)]
    assert hydrated == ["c5", *expected]


def test_related_metadata_expansion_supports_batch_hydration():
    """PERF-001. A native batch reader hydrates only the selected permitted IDs.

    Red proof with related.py restored to 8b3bb14d3: repeated text windows are observed.
    """
    policy = SourceSecurityPolicy((SourceRule("notes.md"),))
    store = _BoundedSourceStore(_stale_neighbours(policy, {4, 6}))
    hydrated = []

    def stream(scope, *, sources, include_text):
        assert sources == ["notes.md"] and include_text is False
        return (Chunk(chunk.id, chunk.source, "", chunk.metadata) for chunk in store.chunks)

    def by_ids(ids):
        hydrated.append(ids)
        return {chunk.id: chunk for chunk in store.chunks if chunk.id in ids}

    store.iter_scoped_chunks = stream
    store.chunks_by_ids = by_ids
    result = trusted_related(
        store, "c5", relation="source", max_items=2,
        calibration=Calibration(embedder="fixture", threshold=0.5),
        security_policy=policy, access_context=AccessContext("reader", store.tenant),
    )
    assert [hit.chunk.id for hit in result.items] == ["c3", "c7"]
    assert store.windows == [2], f"repeated ranked text windows: {store.windows}"
    assert hydrated == [["c3", "c7"]]
