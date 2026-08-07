"""Corpus to learned sparse sidecar, as a library operation rather than two offline scripts."""

from __future__ import annotations

import pytest

from recall.sparse import (
    SparseCoverageError,
    SparseIndexResult,
    SparseProfile,
    assert_sparse_coverage,
    backfill_learned_sparse,
    store_sparse_vectors,
)
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


@requires_db
def test_coverage_passes_when_every_chunk_is_encoded(make_store) -> None:
    store = make_store(64)
    store.upsert(
        [_chunk("a", "aardvark"), _chunk("b", "beta")], [[0.1] * 64, [0.1] * 64]
    )
    encoder = KeywordSparseEncoder({"aardvark": 7, "beta": 9})
    store_sparse_vectors(store, encoder, [("a", "aardvark"), ("b", "beta")])

    assert_sparse_coverage(store, PROFILE_ID)  # must not raise


@requires_db
def test_coverage_refuses_a_half_encoded_corpus(make_store) -> None:
    """"The corpus is half encoded" is a state that exists, and a query over it returns
    plausible thin results rather than an error. So it is refused here, loudly, with both counts
    in the message."""
    store = make_store(64)
    store.upsert(
        [_chunk("a", "aardvark"), _chunk("b", "beta")], [[0.1] * 64, [0.1] * 64]
    )
    encoder = KeywordSparseEncoder({"aardvark": 7})
    store_sparse_vectors(store, encoder, [("a", "aardvark")])

    with pytest.raises(SparseCoverageError, match="1 of 2"):
        assert_sparse_coverage(store, PROFILE_ID)


@requires_db
def test_coverage_refuses_an_overcounted_sidecar_and_names_the_cause(make_store) -> None:
    """More sidecar rows than chunks is an orphan, not a shortfall, and needs its own message.

    The sidecar keys its parent as a column VALUE, not a relation, so nothing cascades: deleting
    a chunk row leaves its sidecar row behind. The overcount branch must still raise, but the
    message has to name the mechanism, not print a ratio that reads backwards ("holds 3 of 2").
    """
    store = make_store(64)
    store.upsert(
        [_chunk("a", "aardvark"), _chunk("b", "beta"), _chunk("c", "gamma")],
        [[0.1] * 64, [0.1] * 64, [0.1] * 64],
    )
    encoder = KeywordSparseEncoder({"aardvark": 7, "beta": 9, "gamma": 11})
    store_sparse_vectors(
        store, encoder, [("a", "aardvark"), ("b", "beta"), ("c", "gamma")]
    )
    # Orphan the sidecar: remove one chunk row with no sidecar cleanup, exactly as
    # `delete_sources` does today (a later task fixes that; this test only needs the state).
    store.delete_sources(["/c/c.md"])

    assert store.sparse_row_count(PROFILE_ID) == 3
    assert store.count() == 2

    with pytest.raises(SparseCoverageError) as excinfo:
        assert_sparse_coverage(store, PROFILE_ID)

    message = str(excinfo.value)
    assert "3 of 2" not in message
    assert "sidecar" in message
    assert "nothing cascades" in message
    assert "delete_sources" in message
    assert "drop_table" in message


@requires_db
def test_coverage_names_the_empty_chunks_as_the_explanation(make_store) -> None:
    """A shortfall fully explained by term-free passages still refuses, but says WHY.

    An operator who knows the two missing chunks are punctuation can proceed. One who is told
    only "1 of 2" cannot tell that from a broken encoder.
    """
    store = make_store(64)
    store.upsert(
        [_chunk("a", "aardvark"), _chunk("b", "!!!")], [[0.1] * 64, [0.1] * 64]
    )
    encoder = KeywordSparseEncoder({"aardvark": 7})
    result = store_sparse_vectors(store, encoder, [("a", "aardvark"), ("b", "!!!")])

    with pytest.raises(SparseCoverageError, match="encoded to an empty vector: b"):
        assert_sparse_coverage(store, PROFILE_ID, empty_ids=result.empty_ids)


@requires_db
def test_backfill_encodes_a_corpus_that_was_already_indexed(make_store) -> None:
    """The case that actually matters: every corpus in existence today is already indexed.

    `Indexer` has never written the sidecar, so there is no corpus anywhere whose learned sparse
    vectors were written at index time. The backfill is the only path that can reach them, and it
    is what `benchmarks/store_latency_share.py` calls after `_throwaway_store` has already run.
    """
    store = make_store(64)
    store.upsert(
        [_chunk("a", "aardvark"), _chunk("b", "beta")], [[0.1] * 64, [0.1] * 64]
    )
    encoder = KeywordSparseEncoder({"aardvark": 7, "beta": 9})

    result = backfill_learned_sparse(store, encoder)

    assert result.written == 2
    assert_sparse_coverage(store, PROFILE_ID, empty_ids=result.empty_ids)


@requires_db
def test_backfill_is_idempotent(make_store) -> None:
    """`upsert_sparse` is ON CONFLICT DO UPDATE, so a re-run re-encodes rather than duplicating.

    Deliberately NOT resumable: skipping ids already present would need a new
    `store.sparse_ids(profile_id)`, and at the corpus sizes this serves that buys nothing.
    """
    store = make_store(64)
    store.upsert([_chunk("a", "aardvark")], [[0.1] * 64])
    encoder = KeywordSparseEncoder({"aardvark": 7})

    backfill_learned_sparse(store, encoder)
    backfill_learned_sparse(store, encoder)

    assert store.sparse_row_count(PROFILE_ID) == 1
