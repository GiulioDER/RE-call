"""Corpus to learned sparse sidecar, as a library operation rather than two offline scripts."""

from __future__ import annotations

import pytest

from recall.sparse import SparseIndexResult, SparseProfile, store_sparse_vectors
from recall.types import Chunk
from tests.conftest import requires_db

PROFILE_ID = "kw-index-test"


class KeywordSparseEncoder:
    """A real, deterministic encoder: one term id per known word.

    Not a mock of the system under test. It implements the same encoder protocol the production
    path depends on, and is chosen over a SPLADE checkpoint so this file needs no download and
    no network. The store path it drives is the production one.
    """

    def __init__(self, vocabulary: dict[str, int]) -> None:
        self._vocabulary = vocabulary
        self.batches: list[int] = []
        self.profile = SparseProfile(
            profile_id=PROFILE_ID, model_name="test/keyword",
            artifact_digest="sha256:test", dimension=30522, top_k=1000,
        )

    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        self.batches.append(len(texts))
        return [
            {self._vocabulary[w]: 1.0 for w in text.lower().split() if w in self._vocabulary}
            for text in texts
        ]


class StubEmbedder:
    dim = 64
    name = "stub"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 64 for _ in texts]


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(id=cid, source=f"/c/{cid}.md", text=text, metadata={"file": f"{cid}.md"})


@requires_db
def test_store_sparse_vectors_writes_under_the_encoders_own_profile(make_store) -> None:
    """The profile id is READ OFF the encoder, never passed alongside it.

    Filing vectors under a name a different model produced is the failure the profile column
    exists to prevent, and it produces plausible scores rather than an error.
    """
    store = make_store(64)
    encoder = KeywordSparseEncoder({"aardvark": 7, "beta": 9})

    result = store_sparse_vectors(
        store, encoder, [("a", "aardvark"), ("b", "beta")], batch_size=1
    )

    assert isinstance(result, SparseIndexResult)
    assert result.written == 2
    assert result.empty_ids == []
    assert store.sparse_row_count(PROFILE_ID) == 2
    # batch_size is honoured, not merely accepted: two calls of one text each.
    assert encoder.batches == [1, 1]


@requires_db
def test_a_term_free_chunk_is_skipped_and_named_rather_than_fatal(make_store) -> None:
    """One punctuation-only chunk must not kill a 20,000 chunk index.

    `upsert_sparse` refuses an empty weights mapping outright, and it is right to: the table's
    CHECK requires nnz > 0 and an all-empty run means a broken encoder. So the decision splits by
    level. Here, at the ROW level, the empty vector is skipped and its id recorded. The refusal
    lives at the CORPUS level, in `assert_sparse_coverage`, where an operator can act on it.
    """
    store = make_store(64)
    encoder = KeywordSparseEncoder({"aardvark": 7})

    result = store_sparse_vectors(store, encoder, [("a", "aardvark"), ("b", "!!! ???")])

    assert result.written == 1
    assert result.empty_ids == ["b"]
    assert store.sparse_row_count(PROFILE_ID) == 1


@requires_db
def test_progress_reports_the_running_written_count(make_store) -> None:
    """A silent twenty minute CPU encode is indistinguishable from a hang."""
    store = make_store(64)
    encoder = KeywordSparseEncoder({"aardvark": 7, "beta": 9, "gamma": 11})
    seen: list[int] = []

    store_sparse_vectors(
        store, encoder,
        [("a", "aardvark"), ("b", "beta"), ("c", "gamma")],
        batch_size=2, progress=seen.append,
    )

    assert seen == [2, 3]
