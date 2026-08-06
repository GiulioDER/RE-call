"""Learned sparse sidecar: write path, read path, and the two refusals.

These hit a real PostgreSQL. `RECALL_TEST_DSN` should point at a throwaway database — the
conftest refuses a remote or shared-with-RECALL_DSN target, and `make_store` drops what it creates.
"""

from __future__ import annotations

import pytest

from recall.sparse import SparseProfile
from recall.types import Chunk
from tests.conftest import requires_db

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

