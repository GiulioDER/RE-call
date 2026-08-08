"""Corpus to learned sparse sidecar, as a library operation rather than two offline scripts."""

from __future__ import annotations

from contextlib import closing
import gc

import pytest

from recall.sparse import (
    SparseCoverageError,
    SparseIndexResult,
    SparseProfile,
    assert_sparse_coverage,
    backfill_learned_sparse,
    store_sparse_vectors,
)
from recall.store import PgVectorStore
from recall.types import Chunk
from tests.conftest import TEST_DSN, requires_db

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


class ExplodingSparseEncoder(KeywordSparseEncoder):
    """Encodes normally until `fail_on_batch`, then raises.

    Stands in for the whole family of mid-corpus failures the streaming path has to survive: a
    transient DB error, an encoder OOM on a long passage, `upsert_sparse`'s nnz > 0 CHECK firing
    on unexpected input. They differ only in the exception type; what matters here is that one
    arrives while the source iterator is suspended mid-scan.
    """

    def __init__(self, vocabulary: dict[str, int], *, fail_on_batch: int) -> None:
        super().__init__(vocabulary)
        self._fail_on_batch = fail_on_batch

    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        if len(self.batches) + 1 == self._fail_on_batch:
            raise RuntimeError("encoder failed partway through the corpus")
        return super().encode(texts)


def _checked_out(store: PgVectorStore) -> int:
    """Connections the pool has handed out and not got back.

    `pool_size` is how many the pool has opened, `pool_available` how many are sitting idle in
    it; the difference is what someone is still holding.
    """
    stats = store._pool.get_stats()
    return stats["pool_size"] - stats["pool_available"]


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
    assert "replace_sources" in message


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

    Pins both halves of the idempotence contract. First, no duplication: the row count stays 1
    across two runs. Second, and the half a row count alone cannot tell apart from a regression,
    always RE-ENCODES rather than skipping ids already present: each call's own
    `SparseIndexResult.written` must report 1, because `upsert_sparse` always runs its
    INSERT ... ON CONFLICT DO UPDATE and returns `len(rows)` whether the row was inserted or
    updated. A hypothetical future version that silently skipped already-encoded ids would still
    leave the row count at 1, but would report `written == 0` on the second call, which is
    what asserting on both `written` values catches and a row-count-only assertion would not.

    Deliberately NOT resumable: skipping ids already present would need a new
    `store.sparse_ids(profile_id)`, and at the corpus sizes this serves that buys nothing.
    """
    store = make_store(64)
    store.upsert([_chunk("a", "aardvark")], [[0.1] * 64])
    encoder = KeywordSparseEncoder({"aardvark": 7})

    first = backfill_learned_sparse(store, encoder)
    second = backfill_learned_sparse(store, encoder)

    assert first.written == 1
    assert second.written == 1
    assert store.sparse_row_count(PROFILE_ID) == 1


def _id_text_stream(store: PgVectorStore):
    """A caller's own streaming source: the pooled cursor, adapted to `(id, text)` pairs.

    A generator FUNCTION rather than the generator EXPRESSION `backfill_learned_sparse` builds,
    and the difference is load-bearing. Closing a genexp only drops the genexp's reference to the
    iterator it wraps; the wrapped generator is closed only if that was the last reference, which
    puts it back on refcounting. A generator function has its own frame, so `close()` runs this
    `with` block's exit and the release is the language's, not the collector's.
    """
    with closing(store.iter_chunks(batch_size=1)) as chunks:
        for chunk in chunks:
            yield chunk.id, chunk.text


@requires_db
def test_a_failed_encode_closes_the_iterator_it_was_handed(make_store) -> None:
    """A mid-corpus failure must not strand the source iterator's pooled connection.

    `iter_chunks` is a server-side cursor, and `store.py::_borrowed` holds ONE pooled connection
    open inside an explicit transaction for the whole of its scan. When `encode` raises partway
    through, `store_sparse_vectors`' driving loop is abandoned with the source suspended
    mid-scan, and nothing in the language hands the connection back: cleanup falls to CPython
    refcounting finalising the orphan. That is usually prompt and is NOT guaranteed — anything
    retaining the traceback (a retry loop storing the exception, a caller logging
    `sys.exc_info()`, a test framework's `excinfo`) keeps this function's frame alive, and its
    `items` local with it, so the connection stays checked out as long as that reference lives.

    The test holds `items` itself rather than leaning on GC to drop it. The reference is alive at
    the assertion by construction, so refcounting CANNOT be what brings the count to zero: only
    an explicit `close()` inside `store_sparse_vectors` can. Closing it again in `finally` is
    what stops a RED run leaving an open transaction on the test table and deadlocking the
    fixture's `DROP TABLE` behind it.
    """
    corpus = make_store(64)
    ids = ["a", "b", "c", "d"]
    corpus.upsert([_chunk(cid, f"{cid} aardvark") for cid in ids], [[0.1] * 64] * len(ids))

    # A POOLED store over the same table: the default connection mode holds one long-lived
    # connection and borrows nothing, so it has no checkout to strand and cannot show this bug.
    # max_size=2 is required, not incidental — the cursor holds one for the whole scan while
    # `upsert_sparse` borrows a second, and a max_size of 1 would deadlock instead of leaking.
    pooled = PgVectorStore(TEST_DSN, dim=64, table=corpus.table, pool_size=2)
    items = _id_text_stream(pooled)
    try:
        encoder = ExplodingSparseEncoder({"aardvark": 7}, fail_on_batch=3)

        with pytest.raises(RuntimeError, match="failed partway through"):
            store_sparse_vectors(pooled, encoder, items, batch_size=1)

        # Two batches were written before the third raised, so the scan really was suspended
        # mid-corpus rather than exhausted — an exhausted iterator releases on its own and would
        # make the assertion below pass for the wrong reason.
        assert encoder.batches == [1, 1]
        assert pooled.sparse_row_count(PROFILE_ID) == 2
        assert _checked_out(pooled) == 0
    finally:
        items.close()
        pooled.close()


@requires_db
def test_a_failed_backfill_releases_the_streaming_cursors_connection(make_store) -> None:
    """End-to-end: the real backfill path strands nothing when a caller keeps the exception.

    ⚠️ What this pins, exactly, measured by mutation rather than assumed. It fails only when BOTH
    releases are removed; either one alone satisfies it:

    | code state                          | this test |
    |-------------------------------------|-----------|
    | both releases present               | pass      |
    | no `store_sparse_vectors` close     | pass      |
    | no `backfill` `closing()`           | pass      |
    | neither (the original code)         | FAIL      |

    So it is a regression guard on the OUTCOME — "the backfill leaks no connection" — and NOT a
    guard on the backfill's own `closing()`. Nothing here can distinguish that call, because
    `store_sparse_vectors` closing the generator expression drops the expression's only reference
    to the cursor and CPython finalises it there and then. The redundancy is deliberate
    defence-in-depth, not a second thing under test, and saying otherwise would make this a guard
    that reads as protection and cannot fire.
    `test_a_failed_encode_closes_the_iterator_it_was_handed` is what pins the consumer's close.

    The exception is retained in `held`, which is what makes this a test rather than a
    measurement of the collector. That reference keeps the whole traceback — and every frame's
    locals along it — alive across the assertion, which is precisely the production shape (a
    retry loop that stores the failure before deciding whether to retry) that turns "the GC will
    get to it" into a connection checked out indefinitely.

    `held` is dropped in `finally` before pytest captures a failure, so a RED run releases the
    cursor and the fixture's `DROP TABLE` is not left blocking behind an idle transaction.
    """
    corpus = make_store(64)
    ids = ["a", "b", "c", "d"]
    corpus.upsert([_chunk(cid, f"{cid} aardvark") for cid in ids], [[0.1] * 64] * len(ids))

    pooled = PgVectorStore(TEST_DSN, dim=64, table=corpus.table, pool_size=2)
    encoder = ExplodingSparseEncoder({"aardvark": 7}, fail_on_batch=3)
    held: BaseException | None = None
    try:
        try:
            backfill_learned_sparse(pooled, encoder, batch_size=1)
        except RuntimeError as exc:
            held = exc

        assert held is not None, "the encoder was supposed to fail on the third batch"
        assert encoder.batches == [1, 1]
        assert _checked_out(pooled) == 0
    finally:
        held = None
        gc.collect()
        pooled.close()


def test_both_entry_points_refuse_a_non_integer_batch_size_by_name() -> None:
    """One validation, reached the same way from both doors.

    `None` is the plausible caller mistake — an unset CLI option threaded through — and it has to
    arrive as the descriptive `ValueError` a `0` gets. Before this, neither door managed that:
    `store_sparse_vectors` compared `None < 1` and raised `TypeError: '<' not supported between
    instances of 'NoneType' and 'int'`, and `backfill_learned_sparse` never got that far because
    `max(None, 1)` raised its own `TypeError` one frame earlier, naming neither the argument nor
    the constraint.

    That `max()` was also the reason the two doors could disagree about what "batch size" meant:
    it clamped the value handed to `iter_chunks` while passing the raw one to
    `store_sparse_vectors`, so the FETCH size and the encode size were separately derived from
    one argument. They now share a single validated value.

    Neither `store` nor `encoder` is touched, and that is the second half of the contract rather
    than a shortcut: the refusal has to land before a cursor is opened, so passing `None` for
    both is itself the assertion. No database is needed to prove it.
    """
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        store_sparse_vectors(None, None, [], batch_size=None)

    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        backfill_learned_sparse(None, None, batch_size=None)
