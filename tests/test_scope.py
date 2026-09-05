"""Scope predicates and the folder-affinity prior.

These need no database: `Scope.predicate` is a pure function from a scope to (sql, params), and
the prior is a pure reordering. The DB-backed half — that the predicate actually selects the rows
it claims to — lives in `test_scope_store.py`, because a predicate that is well-formed and wrong
is exactly the failure this layer can produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from recall.frontmatter import FACET_MAX_LENGTH, parse_frontmatter
from recall.scope import Scope, coerce_scope, folder_of, group_expression
from recall.scope_prior import (
    MAX_WEIGHT,
    ScopePrior,
    affinities,
    apply_scope_prior,
    cosine,
    scope_value_of,
)
from recall.types import Chunk, ScoredChunk


# ── the predicate ────────────────────────────────────────────────────────────────────────────


def test_an_empty_scope_contributes_no_sql_at_all():
    """A caller splices the fragment unconditionally, so empty must be the empty string."""
    assert Scope().predicate("c") == ("", {})
    assert Scope().is_empty


def test_a_folder_matches_at_or_beneath_itself_and_binds_every_value():
    sql, params = Scope(folder="python").predicate("c")
    assert "LIKE %(scope_folder_prefix)s" in sql
    assert params["scope_folder_prefix"] == "python/%"
    # No caller value appears in the SQL text itself. This is the property that matters more than
    # the exact shape of the clause, so it is asserted on the value rather than on the string.
    assert "python" not in sql


def test_a_folder_containing_a_like_wildcard_cannot_widen_the_scope():
    """`draft_1` must not match `draftX1`.

    This is the failure the whole module exists to prevent, and it is silent: a widened scope
    returns a confident answer from a region the caller did not name, rather than an error.
    """
    _sql, params = Scope(folder="draft_1").predicate("c")
    assert params["scope_folder_prefix"] == "draft\\_1/%"

    _sql, params = Scope(folder="100%").predicate("c")
    assert params["scope_folder_prefix"] == "100\\%/%"


def test_the_root_folder_is_a_selectable_value_and_is_not_everything():
    """`folder='/'` means files with no separator, not the whole corpus."""
    sql, params = Scope(folder="/").predicate("c")
    assert "position('/' in c.metadata->>'file') = 0" in sql
    assert "LIKE" not in sql
    assert params == {}


def test_a_trailing_slash_and_a_backslash_name_the_same_folder():
    assert Scope(folder="python/").normalized_folder == "python"
    assert Scope(folder="a\\b").normalized_folder == "a/b"


def test_a_facet_is_compared_case_insensitively_on_both_sides():
    sql, params = Scope(facet="Feedback").predicate("c")
    assert params["scope_facet"] == "feedback"
    assert "lower(c.metadata->>'type')" in sql


def test_scope_fields_are_conjunctive():
    sql, _params = Scope(folder="python", facet="reference").predicate("c")
    assert sql.count("AND") >= 2


def test_an_empty_string_is_refused_rather_than_read_as_no_filter():
    """The dangerous reading of `folder=''` is 'everything'. It is refused instead."""
    for kwargs in ({"folder": ""}, {"facet": "   "}, {"source": ""}):
        with pytest.raises(ValueError, match="empty or whitespace"):
            Scope(**kwargs)


def test_a_non_string_scope_value_is_refused_at_construction():
    with pytest.raises(TypeError, match="must be a str"):
        Scope(folder=["python"])  # type: ignore[arg-type]


def test_the_alias_must_be_an_identifier():
    for bad in ("c; DROP TABLE x", "c c", ""):
        with pytest.raises(ValueError, match="bare identifier"):
            Scope(folder="python").predicate(bad)
        with pytest.raises(ValueError, match="bare identifier"):
            group_expression("folder", bad)


def test_an_unknown_dimension_raises_here_rather_than_reaching_postgres():
    with pytest.raises(ValueError, match="unknown scope dimension"):
        group_expression("metadata->>'evil'", "c")


def test_coerce_scope_refuses_two_different_filters_instead_of_picking_one():
    with pytest.raises(ValueError, match="not both"):
        coerce_scope(Scope(source="a"), "b")
    # Each form alone is fine, and the legacy form still produces a source scope.
    assert coerce_scope(None, "a") == Scope(source="a")
    assert coerce_scope(None, None).is_empty
    assert coerce_scope(Scope(folder="p"), None) == Scope(folder="p")


@pytest.mark.parametrize(
    "file,expected",
    [("python/notes.md", "python"), ("a/b/c.md", "a/b"), ("c.md", ""), ("a\\b.md", "a")],
)
def test_folder_of_matches_what_the_sql_group_expression_computes(file, expected):
    assert folder_of(file) == expected


# ── the facet, at index time ─────────────────────────────────────────────────────────────────


def test_the_agent_memory_facet_is_recognized_at_any_indent():
    """The memory format nests `type:` under a `metadata:` key; a flat reader must still see it."""
    nested = "---\nname: x\nmetadata: \n  node_type: memory\n  type: feedback\n---\nbody"
    assert parse_frontmatter(nested)[0]["type"] == "feedback"
    assert parse_frontmatter("---\ntype: project\n---\nbody")[0]["type"] == "project"


def test_a_quoted_facet_matches_the_unquoted_one():
    assert parse_frontmatter('---\ntype: "reference"\n---\nbody')[0]["type"] == "reference"


def test_an_absent_empty_or_overlong_facet_leaves_the_document_facet_less():
    """Inventing a category is worse than having none, so all three yield no facet key."""
    assert "type" not in parse_frontmatter("---\nname: x\n---\nbody")[0]
    assert "type" not in parse_frontmatter("---\ntype:   \n---\nbody")[0]
    long = "x" * (FACET_MAX_LENGTH + 1)
    assert "type" not in parse_frontmatter(f"---\ntype: {long}\n---\nbody")[0]
    edge = "y" * FACET_MAX_LENGTH
    assert parse_frontmatter(f"---\ntype: {edge}\n---\nbody")[0]["type"] == edge


def test_recognizing_the_facet_did_not_disturb_the_validity_keys():
    """The quote-stripping was factored out; these are the cases it existed for."""
    meta, _body = parse_frontmatter('---\nsupersedes: "v1.md"\nvalid_from: 2026-01-01\n---\nb')
    assert meta["supersedes"] == "v1.md"
    assert meta["valid_from"] == "2026-01-01"


# ── the prior ────────────────────────────────────────────────────────────────────────────────


@dataclass
class _Hit:
    """A ScoredChunk is frozen and carries a Chunk; this builds one with the metadata under test."""

    metadata: dict[str, Any] = field(default_factory=dict)

    def scored(self, score: float = 0.5) -> ScoredChunk:
        return ScoredChunk(
            chunk=Chunk(id="i", source="s", text="t", metadata=self.metadata), score=score
        )


def _hits(values: list[str | None], key: str = "file") -> list[ScoredChunk]:
    out = []
    for i, value in enumerate(values):
        metadata: dict[str, Any] = {} if value is None else {key: value}
        out.append(
            ScoredChunk(
                chunk=Chunk(id=str(i), source="s", text="t", metadata=metadata), score=0.5
            )
        )
    return out


def test_a_disabled_prior_is_an_exact_identity():
    """Off must mean bit-identical, not merely 'contributes zero'."""
    hits = _hits(["a/1.md", "b/2.md", "a/3.md"])
    out = apply_scope_prior(hits, {"a": 1.0, "b": 0.0}, ScopePrior(weight=0.0))
    assert out == hits
    assert [id(h) for h in out] == [id(h) for h in hits]


def test_the_prior_promotes_a_matching_candidate_without_dropping_any():
    """`w * (n - 1)` positions is the bound, and it is REACHED rather than missed by one.

    Three hits at w=0.5 buys exactly one position, which lands level with the hit above; the
    affinity tie-break is what turns that level score into an actual promotion. Assert the exact
    position, not merely 'it moved', because the off-by-one this guards is invisible otherwise.
    """
    hits = _hits(["b/1.md", "b/2.md", "a/3.md"])
    out = apply_scope_prior(hits, {"a": 1.0, "b": 0.0}, ScopePrior(weight=MAX_WEIGHT))
    assert [h.chunk.metadata["file"] for h in out] == ["b/1.md", "a/3.md", "b/2.md"]
    assert {h.chunk.id for h in out} == {h.chunk.id for h in hits}


def test_the_prior_never_rewrites_a_score_because_trust_reads_it():
    """A blended score reaching the trust gate would be a silently miscalibrated one."""
    hits = _hits(["b/1.md", "a/2.md"])
    out = apply_scope_prior(hits, {"a": 1.0, "b": 0.0}, ScopePrior(weight=0.4))
    assert [h.score for h in out] == [0.5, 0.5]
    assert all(h in hits for h in out)


def test_a_weight_too_small_to_cross_a_rank_changes_nothing():
    """The bound is real: with w, a hit climbs at most w*(n-1) places."""
    hits = _hits(["b/1.md"] * 10 + ["a/11.md"])
    out = apply_scope_prior(hits, {"a": 1.0, "b": 0.0}, ScopePrior(weight=0.05))
    assert out[-1].chunk.metadata["file"] == "a/11.md"


def test_a_hit_with_no_value_on_the_dimension_gets_no_bonus():
    """A legacy row with no `file` metadata is not assigned a default bucket.

    It keeps its own rank while the hits that DO carry a value collect their bonus, so it ends up
    below one that started beneath it. That is the discriminating observation: an implementation
    that defaulted the missing value to the mean, or to the best folder, would keep it at index 1.
    """
    hits = _hits(["b/1.md", None, "a/3.md"])
    out = apply_scope_prior(hits, {"a": 1.0, "b": 1.0}, ScopePrior(weight=MAX_WEIGHT))
    assert [h.chunk.metadata.get("file") for h in out] == ["b/1.md", "a/3.md", None]


def test_ties_keep_the_incoming_order():
    hits = _hits(["a/1.md", "a/2.md", "a/3.md"])
    out = apply_scope_prior(hits, {"a": 1.0}, ScopePrior(weight=0.4))
    assert [h.chunk.id for h in out] == [h.chunk.id for h in hits]


def test_scope_value_of_reads_the_folder_and_the_facet_from_the_right_key():
    folder_hit = _hits(["python/x.md"])[0]
    assert scope_value_of(folder_hit, "folder") == "python"
    facet_hit = _hits(["Feedback"], key="type")[0]
    assert scope_value_of(facet_hit, "facet") == "feedback"
    assert scope_value_of(facet_hit, "folder") is None


def test_affinity_is_inert_when_there_is_nothing_to_choose_between():
    """A flat corpus has one folder, so enabling the prior on it must change nothing."""
    assert affinities([1.0, 0.0], [("only", 10, [1.0, 0.0])]) == {"only": 0.0}
    assert affinities([1.0, 0.0], []) == {}
    # Two folders whose centroids are identical are equally close: no basis to prefer either.
    same = affinities([1.0, 0.0], [("a", 5, [0.0, 1.0]), ("b", 5, [0.0, 1.0])])
    assert same == {"a": 0.0, "b": 0.0}


def test_affinity_is_min_maxed_across_the_folders_present():
    out = affinities([1.0, 0.0], [("near", 9, [1.0, 0.0]), ("far", 9, [-1.0, 0.0])])
    assert out == {"near": 1.0, "far": 0.0}


def test_cosine_gives_a_zero_vector_no_direction_rather_than_raising():
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    with pytest.raises(ValueError, match="differ in length"):
        cosine([1.0], [1.0, 0.0])


def test_a_prior_strong_enough_to_be_a_router_is_refused():
    with pytest.raises(ValueError, match="MAX_WEIGHT"):
        ScopePrior(weight=MAX_WEIGHT + 0.01)


def test_a_negative_weight_is_a_sign_error_and_is_refused():
    with pytest.raises(ValueError, match="weight must be >= 0"):
        ScopePrior(weight=-0.1)


def test_a_nan_weight_is_refused_rather_than_disabling_the_prior_by_accident():
    """`nan > 0.0` is False, so an unchecked nan would silently read as 'off'."""
    with pytest.raises(ValueError, match="finite"):
        ScopePrior(weight=float("nan"))


def test_an_unknown_prior_dimension_is_refused_at_construction():
    with pytest.raises(ValueError, match="must be 'folder' or 'facet'"):
        ScopePrior(dimension="colour", weight=0.1)


# ── the retriever, without a database ────────────────────────────────────────────────────────


class _RecordingStore:
    """Records how each leg was called, and can serve centroids for the prior."""

    def __init__(self, hits: list[ScoredChunk], centroids=None) -> None:
        self._hits = hits
        self._centroids = centroids or []
        self.dense_calls: list[dict[str, Any]] = []
        self.centroid_calls: list[dict[str, Any]] = []

    def query_dense(self, vector, k, **kwargs):
        self.dense_calls.append(kwargs)
        return self._hits[:k]

    def query_sparse(self, query, k, **kwargs):
        return []

    def scope_centroids(self, dimension="folder", min_chunks=1):
        self.centroid_calls.append({"dimension": dimension, "min_chunks": min_chunks})
        return self._centroids

    def newest_indexed_at(self):
        from datetime import datetime, timezone

        return datetime.now(timezone.utc)


class _ConstantEmbedder:
    name = "constant"
    dim = 2

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    def embed(self, texts):
        return [list(self._vector) for _ in texts]


def _retriever(store, **kwargs):
    from recall.retriever import HybridRetriever

    return HybridRetriever(
        store, _ConstantEmbedder([1.0, 0.0]), candidate_k=10, use_sparse=False, **kwargs
    )


def test_an_unscoped_search_still_calls_the_store_the_legacy_way():
    """A store that predates scoping must keep working, so `scope=` is not sent when unneeded.

    This is the compatibility contract the retriever documents: eleven test doubles in this suite
    alone take `(vector, k, source=None)`, and so may a downstream store, this package being on
    PyPI. Sending the new keyword unconditionally would break all of them on queries that use no
    scoping at all.
    """
    store = _RecordingStore(_hits(["a/1.md"]))
    _retriever(store).search("q", k=1)
    assert store.dense_calls == [{"source": None}]

    store = _RecordingStore(_hits(["a/1.md"]))
    _retriever(store).search("q", k=1, source="a/1.md")
    assert store.dense_calls == [{"source": "a/1.md"}]


def test_a_folder_or_facet_search_sends_the_scope_through():
    store = _RecordingStore(_hits(["a/1.md"]))
    _retriever(store).search("q", k=1, scope=Scope(folder="a"))
    assert store.dense_calls == [{"scope": Scope(folder="a")}]


def test_the_retriever_refuses_two_conflicting_filters_before_touching_the_store():
    store = _RecordingStore(_hits(["a/1.md"]))
    with pytest.raises(ValueError, match="not both"):
        _retriever(store).search("q", k=1, source="x.md", scope=Scope(folder="a"))
    assert store.dense_calls == []


def test_the_prior_is_off_by_default_and_costs_no_centroid_query():
    """An unmeasured feature must not be paid for by anybody who did not ask for it."""
    store = _RecordingStore(_hits(["b/1.md", "a/2.md"]))
    result = _retriever(store).search("q", k=2)
    assert store.centroid_calls == []
    assert [h.chunk.metadata["file"] for h in result.hits] == ["b/1.md", "a/2.md"]


def test_an_enabled_prior_reorders_the_result_and_leaves_the_scores_alone():
    """Three candidates, because `w * (n - 1)` is the climb and two would buy half a position.

    The pool is what the prior acts on, so the assertion is on `candidate_k` hits reordered and
    THEN cut to k, not on the k the caller asked for.
    """
    store = _RecordingStore(
        _hits(["b/1.md", "b/2.md", "a/3.md"]),
        centroids=[("a", 9, [1.0, 0.0]), ("b", 9, [-1.0, 0.0])],
    )
    prior = ScopePrior(dimension="folder", weight=MAX_WEIGHT, min_chunks=5)
    result = _retriever(store, scope_prior=prior).search("q", k=3)
    assert [h.chunk.metadata["file"] for h in result.hits] == ["b/1.md", "a/3.md", "b/2.md"]
    # The dense cosine each hit carries is what the trust layer compares against the certified
    # threshold. The prior must not have touched it.
    assert [h.score for h in result.hits] == [0.5, 0.5, 0.5]
    assert store.centroid_calls == [{"dimension": "folder", "min_chunks": 5}]


def test_the_prior_can_promote_a_candidate_from_outside_the_requested_k():
    """The reason it runs before the cut: rescuing a hit the pool ranked below the line."""
    store = _RecordingStore(
        _hits(["b/1.md", "b/2.md", "a/3.md"]),
        centroids=[("a", 9, [1.0, 0.0]), ("b", 9, [-1.0, 0.0])],
    )
    prior = ScopePrior(dimension="folder", weight=MAX_WEIGHT, min_chunks=5)
    result = _retriever(store, scope_prior=prior).search("q", k=2)
    assert [h.chunk.metadata["file"] for h in result.hits] == ["b/1.md", "a/3.md"]


def test_centroids_are_fetched_once_per_retriever_not_once_per_query():
    store = _RecordingStore(
        _hits(["b/1.md", "a/2.md"]),
        centroids=[("a", 9, [1.0, 0.0]), ("b", 9, [-1.0, 0.0])],
    )
    retriever = _retriever(store, scope_prior=ScopePrior(weight=0.2))
    for _ in range(3):
        retriever.search("q", k=2)
    assert len(store.centroid_calls) == 1


def test_a_store_that_cannot_serve_centroids_degrades_to_no_prior_rather_than_failing():
    """The prior is an optional tilt; it must never become a new way for search to break."""

    class _OldStore(_RecordingStore):
        def scope_centroids(self, dimension="folder", min_chunks=1):
            raise AttributeError("this store predates scope centroids")

    store = _OldStore(_hits(["b/1.md", "a/2.md"]))
    result = _retriever(store, scope_prior=ScopePrior(weight=0.4)).search("q", k=2)
    assert [h.chunk.metadata["file"] for h in result.hits] == ["b/1.md", "a/2.md"]


def test_a_failed_centroid_fetch_is_not_retried_on_every_query():
    calls: list[int] = []

    class _OldStore(_RecordingStore):
        def scope_centroids(self, dimension="folder", min_chunks=1):
            calls.append(1)
            raise RuntimeError("nope")

    retriever = _retriever(_OldStore(_hits(["a/1.md"])), scope_prior=ScopePrior(weight=0.4))
    for _ in range(3):
        retriever.search("q", k=1)
    assert len(calls) == 1
