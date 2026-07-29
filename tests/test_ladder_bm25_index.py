"""The DB-free BM25 core the ladder builder ranks with.

The builder must run with no Postgres, and the ring order it produces is frozen into a released
manifest — so ties cannot break by dict insertion order, or two builds of the same corpus produce
two different benchmarks. These tests pin the ranking property, the IDF property, and determinism.
"""
from __future__ import annotations

from recall.eval.bm25 import BM25Index

DOCS = [
    ("d1", "the cache was replaced with a read-through cache"),
    ("d2", "retry policy uses exponential backoff"),
    ("d3", "the cache warms on deploy"),
]


def test_ranks_the_document_containing_the_query_terms_first():
    index = BM25Index(DOCS)
    assert index.rank("backoff")[0][0] == "d2"


def test_a_term_in_every_document_scores_far_below_a_rare_one():
    """IDF drives a term appearing in every document to near-zero weight.

    The discriminating comparison is a universal term against a RARE one in the same corpus.
    Three identical documents scoring identically proves nothing — that holds by symmetry even
    with IDF removed, which is what the assertion this replaces actually tested.
    """
    index = BM25Index(
        [
            ("a", "shared shared shared rare"),
            ("b", "shared shared shared"),
            ("c", "shared shared shared"),
        ]
    )
    universal = max(index.score("shared"))
    rare = max(index.score("rare"))
    assert rare > universal
    assert universal < 0.35 * rare


def test_ties_break_by_doc_id_ascending_not_insertion_order():
    forward = BM25Index([("z", "same text here"), ("a", "same text here")])
    reverse = BM25Index([("a", "same text here"), ("z", "same text here")])
    assert [d for d, _ in forward.rank("same")] == ["a", "z"]
    assert [d for d, _ in reverse.rank("same")] == ["a", "z"]


def test_unknown_term_scores_everything_zero():
    index = BM25Index(DOCS)
    assert index.score("xyzzy") == [0.0, 0.0, 0.0]


def test_empty_corpus_scores_nothing_rather_than_dividing_by_zero():
    index = BM25Index([])
    assert len(index) == 0
    assert index.score("anything") == []


def test_doc_ids_are_corpus_order():
    assert BM25Index(DOCS).doc_ids == ["d1", "d2", "d3"]
