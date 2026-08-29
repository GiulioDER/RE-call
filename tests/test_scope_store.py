"""The scope predicate against a real Postgres, on all three retrieval legs.

`test_scope.py` proves the predicate is well formed. This file proves it selects the rows it
claims to, which is the half that a string assertion cannot reach: an `ESCAPE` clause Postgres
parses differently, a `lower()` on one side only, or a `LIKE` that quietly matches a sibling
folder all produce valid SQL and a confident wrong answer.
"""

from __future__ import annotations

import pytest

from recall.scope import Scope
from recall.types import Chunk
from tests.conftest import requires_db

# A tiny corpus with two folders, a root file, a wildcard-trap folder name, and two facets.
# Vectors are axis-aligned so "nearest" is decidable by eye in every assertion below.
CORPUS: list[tuple[str, str, str | None, list[float]]] = [
    # (chunk id, file, facet, embedding)
    ("py1", "python/asyncio.md", "reference", [1.0, 0.0, 0.0]),
    ("py2", "python/typing.md", "feedback", [0.9, 0.1, 0.0]),
    ("inf1", "infra/postgres.md", "reference", [0.0, 1.0, 0.0]),
    ("inf2", "infra/docker.md", None, [0.0, 0.9, 0.1]),
    ("root", "readme.md", "project", [0.0, 0.0, 1.0]),
    # A literal underscore in the folder name: `draft_1` must not be matched by a scope naming
    # `draftX1`, and must be matched by one naming itself.
    ("d1", "draft_1/plan.md", "project", [0.5, 0.5, 0.0]),
    ("dx", "draftX1/plan.md", "project", [0.5, 0.5, 0.0]),
    # A nested folder, to pin down that a scope on the parent reaches it.
    ("deep", "python/lib/inner.md", "reference", [0.8, 0.2, 0.0]),
]


@pytest.fixture
def scoped_store(make_store):
    store = make_store(3)
    chunks, vectors = [], []
    for cid, file, facet, vector in CORPUS:
        metadata: dict[str, object] = {"file": file, "ord": 0}
        if facet is not None:
            metadata["type"] = facet
        chunks.append(Chunk(cid, f"/abs/{file}", f"text of {cid}", metadata=metadata))
        vectors.append(vector)
    store.upsert(chunks, vectors)
    return store


def _ids(hits) -> set[str]:
    return {h.chunk.id for h in hits}


@requires_db
def test_a_folder_scope_returns_that_folder_and_everything_beneath_it(scoped_store):
    hits = scoped_store.query_dense([1.0, 0.0, 0.0], k=10, scope=Scope(folder="python"))
    assert _ids(hits) == {"py1", "py2", "deep"}


@requires_db
def test_a_folder_scope_reaches_a_nested_child_by_its_parent(scoped_store):
    hits = scoped_store.query_dense([1.0, 0.0, 0.0], k=10, scope=Scope(folder="python/lib"))
    assert _ids(hits) == {"deep"}


@requires_db
def test_an_underscore_in_a_folder_name_is_a_literal_and_not_a_wildcard(scoped_store):
    """The silent-widening case: without ESCAPE, `draft_1` also matches `draftX1`."""
    hits = scoped_store.query_dense([0.5, 0.5, 0.0], k=10, scope=Scope(folder="draft_1"))
    assert _ids(hits) == {"d1"}


@requires_db
def test_the_root_scope_is_the_root_and_not_the_whole_corpus(scoped_store):
    hits = scoped_store.query_dense([0.0, 0.0, 1.0], k=10, scope=Scope(folder="/"))
    assert _ids(hits) == {"root"}


@requires_db
def test_a_facet_scope_crosses_folders_and_ignores_case(scoped_store):
    hits = scoped_store.query_dense([1.0, 0.0, 0.0], k=10, scope=Scope(facet="ReFeReNcE"))
    assert _ids(hits) == {"py1", "inf1", "deep"}


@requires_db
def test_folder_and_facet_together_are_an_intersection(scoped_store):
    hits = scoped_store.query_dense(
        [1.0, 0.0, 0.0], k=10, scope=Scope(folder="python", facet="reference")
    )
    assert _ids(hits) == {"py1", "deep"}


@requires_db
def test_a_chunk_with_no_facet_is_matched_by_no_facet_scope(scoped_store):
    """`inf2` declares none, so it must not fall into some default bucket."""
    for facet in ("reference", "feedback", "project"):
        assert "inf2" not in _ids(
            scoped_store.query_dense([0.0, 1.0, 0.0], k=10, scope=Scope(facet=facet))
        )
    assert "inf2" in _ids(scoped_store.query_dense([0.0, 1.0, 0.0], k=10))


@requires_db
def test_an_empty_scope_still_returns_the_whole_corpus(scoped_store):
    assert len(scoped_store.query_dense([1.0, 0.0, 0.0], k=10, scope=Scope())) == len(CORPUS)
    assert len(scoped_store.query_dense([1.0, 0.0, 0.0], k=10)) == len(CORPUS)


@requires_db
def test_the_legacy_source_filter_still_works_and_still_means_one_file(scoped_store):
    assert _ids(scoped_store.query_dense([1.0, 0.0, 0.0], k=10, source="python/asyncio.md")) == {
        "py1"
    }
    assert _ids(
        scoped_store.query_dense([1.0, 0.0, 0.0], k=10, scope=Scope(source="python/asyncio.md"))
    ) == {"py1"}


@requires_db
def test_passing_both_forms_of_the_same_filter_is_refused(scoped_store):
    with pytest.raises(ValueError, match="not both"):
        scoped_store.query_dense([1.0, 0.0, 0.0], k=10, source="a.md", scope=Scope(folder="python"))


@requires_db
def test_the_sparse_leg_applies_the_same_scope(scoped_store):
    """A scope honoured by one leg and not another leaks rows through fusion."""
    unscoped = _ids(scoped_store.query_sparse("text", k=10))
    assert len(unscoped) == len(CORPUS)
    scoped = _ids(scoped_store.query_sparse("text", k=10, scope=Scope(folder="python")))
    assert scoped == {"py1", "py2", "deep"}


@requires_db
def test_scope_centroids_group_by_folder_and_report_size(scoped_store):
    rows = scoped_store.scope_centroids(dimension="folder")
    sizes = {value: n for value, n, _vec in rows}
    assert sizes == {"python": 2, "python/lib": 1, "infra": 2, "draft_1": 1, "draftX1": 1, "": 1}
    # The centroid of a folder holding one chunk IS that chunk's vector, which is the cheapest
    # available check that the aggregate is a mean of the right rows rather than of the table.
    deep = next(vec for value, _n, vec in rows if value == "python/lib")
    assert deep == pytest.approx([0.8, 0.2, 0.0], abs=1e-6)
    python = next(vec for value, _n, vec in rows if value == "python")
    assert python == pytest.approx([0.95, 0.05, 0.0], abs=1e-6)


@requires_db
def test_scope_centroids_group_by_facet_and_exclude_the_undeclared(scoped_store):
    rows = scoped_store.scope_centroids(dimension="facet")
    sizes = {value: n for value, n, _vec in rows}
    # `inf2` declares no facet and must not aggregate into a bucket of leftovers.
    assert sizes == {"reference": 3, "feedback": 1, "project": 3}


@requires_db
def test_min_chunks_drops_the_folders_too_small_to_have_a_meaningful_centroid(scoped_store):
    rows = scoped_store.scope_centroids(dimension="folder", min_chunks=2)
    assert {value for value, _n, _vec in rows} == {"python", "infra"}


@requires_db
def test_scope_centroids_refuses_a_dimension_it_does_not_define(scoped_store):
    with pytest.raises(ValueError, match="unknown scope dimension"):
        scoped_store.scope_centroids(dimension="metadata->>'anything'")


@requires_db
def test_scope_inventory_lists_what_a_filter_can_be_given(scoped_store):
    rows = scoped_store.scope_inventory("folder")
    assert dict((value, chunks) for value, chunks, _docs in rows) == {
        "python": 2,
        "infra": 2,
        "python/lib": 1,
        "draft_1": 1,
        "draftX1": 1,
        "": 1,
    }
    # Documents, not chunks: the two `python` chunks are two separate files here.
    assert dict((value, docs) for value, _chunks, docs in rows)["python"] == 2


@requires_db
def test_scope_inventory_counts_the_undeclared_separately_from_the_values(scoped_store):
    """`inf2` has no facet. It must not appear as a value, and must not vanish either."""
    values = {value for value, _c, _d in scoped_store.scope_inventory("facet")}
    assert values == {"reference", "feedback", "project"}
    assert scoped_store.scope_undeclared_count("facet") == 1
    # Every chunk has a folder, even the one at the root, so nothing is undeclared there.
    assert scoped_store.scope_undeclared_count("folder") == 0


@requires_db
def test_the_inventory_and_a_scoped_search_agree_on_what_exists(scoped_store):
    """The listing is only useful if every value it prints is one a filter accepts."""
    for value, chunks, _docs in scoped_store.scope_inventory("folder"):
        scope = Scope(folder=value if value else "/")
        hits = scoped_store.query_dense([1.0, 0.0, 0.0], k=20, scope=scope)
        # A folder scope reaches beneath itself, so its own count is a lower bound.
        assert len(hits) >= chunks, value
