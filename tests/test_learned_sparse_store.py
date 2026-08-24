"""Learned sparse sidecar: write path, read path, and the two refusals.

These hit a real PostgreSQL. `RECALL_TEST_DSN` should point at a throwaway database — the
conftest refuses a remote or shared-with-RECALL_DSN target, and `make_store` drops what it creates.
"""

from __future__ import annotations

import psycopg
import pytest

from recall.sparse import SparseProfile
from recall.store import SPARSE_TABLE
from recall.types import Chunk
from tests.conftest import TEST_DSN, requires_db

PROFILE = SparseProfile(
    profile_id="test-splade",
    model_name="test/splade",
    artifact_digest="sha256:test",
    dimension=30522,
    top_k=1000,
)


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(id=cid, source=f"/corpus/{cid}.md", text=text, metadata={"file": f"{cid}.md"})


@requires_db
def test_querying_an_unencoded_corpus_raises_instead_of_returning_nothing(make_store) -> None:
    """Fail CLOSED. An empty sidecar means "this corpus was never encoded", not "no matches".

    Returning `[]` here is the failure this project has already shipped once: the conjunctive
    tsquery matched 0 of 150 real questions, `_rrf` was handed a single non-empty list, and the
    hybrid retriever silently degraded to dense-only. Nothing failed, no test noticed, and the
    sparse leg was inert for as long as it took someone to measure it. An unencoded corpus is
    indistinguishable from an unlucky one unless the store says so.
    """
    store = make_store(64)

    with pytest.raises(LookupError, match="not indexed"):
        store.query_learned_sparse({5: 1.0}, k=5, profile_id=PROFILE.profile_id)


@requires_db
def test_stored_vectors_are_retrievable_by_inner_product(make_store) -> None:
    """The round trip: write sparse vectors, then rank by dot product against a query vector.

    `alpha` shares term 7 with the query and `beta` does not, so `alpha` must come first. This is
    the capability claim of the whole feature, executed rather than asserted.
    """
    store = make_store(64)
    store.upsert([_chunk("alpha", "alpha text"), _chunk("beta", "beta text")], [[0.1] * 64] * 2)
    store.upsert_sparse(
        PROFILE.profile_id,
        {"alpha": {7: 2.0, 11: 0.5}, "beta": {99: 3.0}},
    )

    hits = store.query_learned_sparse({7: 1.0}, k=5, profile_id=PROFILE.profile_id)

    assert [hit.chunk.id for hit in hits] == ["alpha"]


@requires_db
def test_a_second_profile_does_not_answer_for_the_first(make_store) -> None:
    """Two encoders' vectors share one table, so the profile filter is what keeps them apart.

    Without it, a corpus encoded by model A would be served to a query encoded by model B: the
    dimensions match, the dot products are finite, and the results are plausible garbage. That is
    the whole reason the profile is in the primary key rather than in a comment.
    """
    store = make_store(64)
    store.upsert([_chunk("alpha", "alpha text")], [[0.1] * 64])
    store.upsert_sparse("profile-a", {"alpha": {7: 2.0}})

    with pytest.raises(LookupError, match="not indexed"):
        store.query_learned_sparse({7: 1.0}, k=5, profile_id="profile-b")


@requires_db
def test_an_over_budget_vector_is_refused_by_the_writer(make_store) -> None:
    """1001 non-zeros is not storable, and the writer says so before Postgres does.

    pgvector raises on INSERT past the HNSW ceiling, so relying on that means a 366k-passage load
    dies partway through with an arbitrary number of rows already committed. Refusing here makes
    it a caller error at the first bad vector.
    """
    store = make_store(64)
    store.upsert([_chunk("alpha", "alpha text")], [[0.1] * 64])

    with pytest.raises(ValueError, match="1000"):
        store.upsert_sparse(PROFILE.profile_id, {"alpha": {i: 1.0 for i in range(1, 1002)}})


@requires_db
def test_a_query_never_returns_rows_encoded_under_a_different_profile(make_store) -> None:
    """Pins the profile filter IN THE QUERY, which the count check above does not reach.

    Both profiles are populated on the same table here, so `sparse_row_count` is non-zero for the
    one being queried and the fail-closed refusal never fires. Only the WHERE clause can keep the
    two apart — and without this test, removing that clause leaves the suite green, because the
    other profile test is satisfied by the count check alone.

    `beta` is encoded ONLY under profile-b and is the sole holder of term 7 there, so if it
    appears in a profile-a answer the filter is gone.
    """
    store = make_store(64)
    store.upsert([_chunk("alpha", "alpha text"), _chunk("beta", "beta text")], [[0.1] * 64] * 2)
    store.upsert_sparse("profile-a", {"alpha": {11: 1.0}})
    store.upsert_sparse("profile-b", {"beta": {7: 5.0}})

    hits = store.query_learned_sparse({7: 1.0}, k=5, profile_id="profile-a")

    assert [hit.chunk.id for hit in hits] == []


@requires_db
def test_a_source_is_covered_only_once_every_one_of_its_chunks_is_encoded(make_store) -> None:
    """Partial coverage reads as NOT covered. `index_path`'s skip predicate depends on this.

    One source, two chunks. Encoding only the first must not mark the source covered: a caller
    that treated it as covered would skip the second chunk on the next run and leave a permanent
    hole in the sidecar. Only once both chunks are encoded does the source appear.
    """
    store = make_store(64)
    one = Chunk(id="one", source="/corpus/two-chunks.md", text="first half", metadata={})
    two = Chunk(id="two", source="/corpus/two-chunks.md", text="second half", metadata={})
    store.upsert([one, two], [[0.1] * 64, [0.1] * 64])

    store.upsert_sparse(PROFILE.profile_id, {"one": {7: 1.0}})
    assert store.sparse_covered_sources(PROFILE.profile_id) == set()

    store.upsert_sparse(PROFILE.profile_id, {"two": {11: 1.0}})
    assert store.sparse_covered_sources(PROFILE.profile_id) == {"/corpus/two-chunks.md"}


@requires_db
def test_dropping_the_table_removes_its_sidecar_rows(make_store) -> None:
    """The sidecar has no FOREIGN KEY, so nothing cascades on its behalf.

    `chunk_table` is a column VALUE, not a relation, so a dropped table leaves its sparse rows
    addressable by a name that no longer resolves. Every throwaway eval store would orphan a
    uuid-named row set, permanently, and nothing would ever look for them again.
    """
    store = make_store(64)
    store.upsert([Chunk(id="alpha", source="/c/a.md", text="a", metadata={})], [[0.1] * 64])
    store.upsert_sparse("drop-probe", {"alpha": {7: 1.0}})
    table = store.table
    assert store.sparse_row_count("drop-probe") == 1

    store.drop_table()

    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        remaining = conn.execute(
            f"SELECT count(*) FROM {SPARSE_TABLE} WHERE chunk_table = %s", (table,)
        ).fetchone()
    assert remaining is not None and remaining[0] == 0


def _sidecar_ids(table: str) -> set[str]:
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        rows = conn.execute(
            f"SELECT id FROM {SPARSE_TABLE} WHERE chunk_table = %s", (table,)
        ).fetchall()
    return {row[0] for row in rows}


@requires_db
def test_delete_sources_erases_the_sidecar_rows_of_the_deleted_chunks(make_store) -> None:
    """Right-to-erasure reaches the sidecar: a forgotten chunk's term weights die with it,
    under EVERY profile, while the surviving chunk's rows stay."""
    store = make_store(64)
    store.upsert([_chunk("alpha", "alpha text"), _chunk("beta", "beta text")], [[0.1] * 64] * 2)
    store.upsert_sparse("profile-a", {"alpha": {7: 1.0}, "beta": {9: 1.0}})
    store.upsert_sparse("profile-b", {"alpha": {11: 1.0}})

    removed = store.delete_sources(["/corpus/alpha.md"])

    assert removed == 1
    assert _sidecar_ids(store.table) == {"beta"}, (
        "alpha's sidecar rows must be gone under both profiles; beta's must survive"
    )


@requires_db
def test_replace_sources_leaves_no_orphaned_tail_rows(make_store) -> None:
    """The exact scenario SparseCoverageError names: re-chunking a source into fewer chunks
    used to leave the tail's old sidecar rows behind as permanent orphans."""
    store = make_store(64)
    one = Chunk(id="one", source="/corpus/doc.md", text="first half", metadata={})
    two = Chunk(id="two", source="/corpus/doc.md", text="second half", metadata={})
    store.upsert([one, two], [[0.1] * 64] * 2)
    store.upsert_sparse(PROFILE.profile_id, {"one": {7: 1.0}, "two": {9: 1.0}})

    merged = Chunk(id="merged", source="/corpus/doc.md", text="both halves", metadata={})
    store.replace_sources(["/corpus/doc.md"], [merged], [[0.2] * 64])

    assert _sidecar_ids(store.table) == set(), (
        "the replaced chunks' sidecar rows must not survive as orphans"
    )


@requires_db
def test_delete_sources_across_scrubs_both_tables_sidecars(make_store) -> None:
    """Each generation table scrubs its own sidecar rows, keyed under that table's name."""
    store_a = make_store(64)
    store_b = make_store(64)
    for store in (store_a, store_b):
        store.upsert([_chunk("alpha", "alpha text")], [[0.1] * 64])
        store.upsert_sparse(PROFILE.profile_id, {"alpha": {7: 1.0}})

    removed = store_a.delete_sources_across(
        [store_a.table, store_b.table], ["/corpus/alpha.md"]
    )

    assert removed == 2
    assert _sidecar_ids(store_a.table) == set()
    assert _sidecar_ids(store_b.table) == set()
