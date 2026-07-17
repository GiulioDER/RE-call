from recall.types import Chunk
from recall_mcp.service import search_memory

from tests.conftest import requires_db


class DictEmbedder:
    dim = 3
    name = "dict"

    def __init__(self, mapping, default):
        self._mapping, self._default = mapping, default

    def embed(self, texts):
        return [self._mapping.get(t, self._default) for t in texts]


@requires_db
def test_search_memory_answerable(make_store):
    store = make_store(3)
    store.upsert([Chunk("a", "notes.md", "cats")], [[1.0, 0.0, 0.0]])
    emb = DictEmbedder({"cats": [1.0, 0.0, 0.0]}, default=[0.0, 0.0, 1.0])
    result = search_memory(store, emb, "cats")
    assert result.gap_warning is False
    assert result.hits and result.hits[0].source == "notes.md"
    assert "relevant" in result.advice.lower()


@requires_db
def test_search_memory_gap(make_store):
    store = make_store(3)
    store.upsert([Chunk("a", "notes.md", "cats")], [[1.0, 0.0, 0.0]])
    emb = DictEmbedder({}, default=[0.0, 0.0, 1.0])  # query orthogonal -> gap
    result = search_memory(store, emb, "unicorns")
    assert result.gap_warning is True
    assert "gap" in result.advice.lower() or "unreliable" in result.advice.lower()


@requires_db
def test_search_memory_superseded_redirects_and_reports_fields(make_store):
    store = make_store(3)
    store.upsert(
        [
            Chunk("old", "v1.md", "cats", metadata={"file": "v1.md", "ord": 0}),
            Chunk(
                "new",
                "v2.md",
                "cats update",
                metadata={"file": "v2.md", "ord": 0, "supersedes": "v1.md"},
            ),
        ],
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]],
    )
    emb = DictEmbedder({"cats": [1.0, 0.0, 0.0]}, default=[0.0, 0.0, 1.0])
    result = search_memory(store, emb, "cats")
    assert result.abstained is False
    assert result.hits[0].verdict == "ok"
    assert result.hits[0].source == "v2.md"
    stale = next(h for h in result.hits if h.source == "v1.md")
    assert stale.verdict == "superseded"
    assert stale.superseded_by == "v2.md"
    assert stale.indexed_at is not None
    assert 0.0 <= stale.confidence <= 1.0
    assert "superseded" in result.advice.lower()
    assert result.calibrated is False  # no calibration passed
    assert "uncalibrated" in result.advice.lower()


@requires_db
def test_search_memory_abstains_when_only_superseded(make_store):
    store = make_store(3)
    store.upsert(
        [
            Chunk("old", "v1.md", "cats", metadata={"file": "v1.md", "ord": 0}),
            Chunk(
                "new",
                "v2.md",
                "dogs entirely",
                metadata={"file": "v2.md", "ord": 0, "supersedes": "v1.md"},
            ),
        ],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    # k=1 keeps only the top fused hit: the superseded v1 (the orthogonal successor ranks
    # below it), so no verdict-ok hit remains and the service must abstain
    emb = DictEmbedder({"cats": [1.0, 0.0, 0.0]}, default=[0.0, 0.0, 1.0])
    result = search_memory(store, emb, "cats", k=1)
    assert result.abstained is True
    assert "do not answer" in result.advice.lower()
    assert result.reason != ""
