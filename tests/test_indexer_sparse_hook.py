"""`Indexer(sparse_encoder=...)`: does the learned sparse sidecar actually get written?

Both assertions below are ROW COUNTS, not call counts. `Indexer`'s other optional secondary
write, `shadow`, passed every structural review and wrote nothing (b0e74e5, PR #218): every
active fingerprint matched, every file was skipped, and the shadow write lived past the
`continue`. The run reported success. A call count would not have caught that; a row count does.
"""

from __future__ import annotations

from pathlib import Path

from recall.index import Indexer
from recall.sparse import SparseProfile, assert_sparse_coverage
from tests.conftest import requires_db

PROFILE_ID = "kw-hook-test"


class KeywordSparseEncoder:
    def __init__(self, vocabulary: dict[str, int]) -> None:
        self._vocabulary = vocabulary
        self.profile = SparseProfile(
            profile_id=PROFILE_ID, model_name="test/keyword",
            artifact_digest="sha256:test", dimension=30522, top_k=1000,
        )

    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        return [
            {self._vocabulary[w]: 1.0 for w in text.lower().split() if w in self._vocabulary}
            for text in texts
        ]


class StubEmbedder:
    dim = 64
    name = "stub"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 64 for _ in texts]


def _corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "one.md").write_text("aardvark facts here\n", encoding="utf-8")
    (root / "two.md").write_text("beta prose here\n", encoding="utf-8")
    return root


VOCAB = {"aardvark": 7, "beta": 9, "facts": 11, "prose": 13, "here": 15}


@requires_db
def test_indexing_with_a_sparse_encoder_fills_the_sidecar(make_store, tmp_path) -> None:
    store = make_store(64)
    encoder = KeywordSparseEncoder(VOCAB)

    Indexer(store, StubEmbedder(), sparse_encoder=encoder).index_path(_corpus(tmp_path))

    assert store.count() > 0
    assert store.sparse_row_count(PROFILE_ID) == store.count()
    assert_sparse_coverage(store, PROFILE_ID)


@requires_db
def test_attaching_a_sparse_encoder_to_an_indexed_corpus_fills_the_sidecar(
    make_store, tmp_path
) -> None:
    """The exact sequence the shadow dual-write got wrong, pinned in both directions.

    Index the corpus with no encoder, then attach one and re-index. Every dense fingerprint still
    matches, so every file is a candidate for `continue`, and a sparse write placed past that
    `continue` would leave the sidecar empty while the run reported success with a skipped count.

    A third `index_path` call, once the corpus is fully covered, pins the other direction: a
    covered corpus must actually be SKIPPED. Without this, a silent regression in the sparse
    membership check (`str(f) in known_sparse` no longer matching) would show up only as every
    run re-embedding the whole corpus through the active embedder, a cost with no failing test
    to catch it.
    """
    store = make_store(64)
    root = _corpus(tmp_path)
    Indexer(store, StubEmbedder()).index_path(root)
    assert store.sparse_row_count(PROFILE_ID) == 0

    encoder = KeywordSparseEncoder(VOCAB)
    Indexer(store, StubEmbedder(), sparse_encoder=encoder).index_path(root)
    assert store.sparse_row_count(PROFILE_ID) == store.count()

    stats = Indexer(store, StubEmbedder(), sparse_encoder=encoder).index_path(root)
    assert stats.skipped == 2
    assert stats.chunks == 0
