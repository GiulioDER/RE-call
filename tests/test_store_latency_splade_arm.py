"""The splade arm of the latency split, end to end on a tiny corpus.

No checkpoint: a deterministic keyword encoder drives the same production retrieval path, so
this asserts the ARM is selectable and its guards hold, at a size a test suite can afford.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.store_latency_share import measure  # noqa: E402
from recall.sparse import SparseProfile, backfill_learned_sparse  # noqa: E402
from recall.types import Chunk  # noqa: E402
from tests.conftest import requires_db  # noqa: E402

PROFILE_ID = "kw-arm-test"
VOCAB = {"aardvark": 7, "beta": 9, "gamma": 11, "delta": 13}


class KeywordSparseEncoder:
    def __init__(self) -> None:
        self.profile = SparseProfile(
            profile_id=PROFILE_ID, model_name="test/keyword",
            artifact_digest="sha256:test", dimension=30522, top_k=1000,
        )

    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        return [
            {VOCAB[w]: 1.0 for w in text.lower().split() if w in VOCAB} for text in texts
        ]


class StubEmbedder:
    dim = 64
    name = "stub"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 64 for _ in texts]


@requires_db
def test_the_splade_arm_reports_a_learned_fire_rate_and_no_lexical_leg(make_store) -> None:
    """`splade` REPLACES ts_rank rather than adding to it.

    So under it the LEXICAL leg is the one asserted idle, and its fire rate must read null rather
    than 0.0. Zero is the issue #81 alarm value (a leg present but matching nothing), and a
    healthy splade run must not publish the alarm.
    """
    store = make_store(64)
    chunks = [
        Chunk(id=f"c{i}", source=f"/c/{i}.md", text=text, metadata={"file": f"{i}.md"})
        for i, text in enumerate(["aardvark facts", "beta prose", "gamma notes", "delta text"])
    ]
    store.upsert(chunks, [[0.1] * 64 for _ in chunks])
    encoder = KeywordSparseEncoder()
    backfill_learned_sparse(store, encoder)

    split = measure(
        store, StubEmbedder(),
        [{"query": "aardvark"}, {"query": "beta"}],
        candidate_k=10, reranker=None, n_chunks=store.count(), repeats=1,
        sparse_backend="splade", sparse_encoder=encoder,
    )

    assert split.sparse_backend == "splade"
    assert split.learned_sparse_fire_rate is not None
    assert split.learned_sparse_fire_rate > 0.0
    assert split.sparse_fire_rate is None
    assert split.max_nesting_violation_ms <= 0.0


@requires_db
def test_the_lexical_arm_still_reports_a_null_learned_fire_rate(make_store) -> None:
    """The mirror image, so neither column can be hard-coded to a constant."""
    store = make_store(64)
    chunks = [
        Chunk(id=f"c{i}", source=f"/c/{i}.md", text=text, metadata={"file": f"{i}.md"})
        for i, text in enumerate(["aardvark facts", "beta prose", "gamma notes", "delta text"])
    ]
    store.upsert(chunks, [[0.1] * 64 for _ in chunks])

    split = measure(
        store, StubEmbedder(),
        [{"query": "aardvark"}, {"query": "beta"}],
        candidate_k=10, reranker=None, n_chunks=store.count(), repeats=1,
        sparse_backend="lexical",
    )

    assert split.sparse_backend == "lexical"
    assert split.learned_sparse_fire_rate is None
    assert split.sparse_fire_rate is not None
