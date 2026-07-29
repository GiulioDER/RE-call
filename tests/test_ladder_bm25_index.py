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


def test_a_term_in_every_document_cannot_decide_the_ranking():
    """IDF drives a term appearing in every document to near-zero weight."""
    index = BM25Index([("a", "shared term"), ("b", "shared term"), ("c", "shared term")])
    scores = index.score("shared")
    assert len(set(scores)) == 1


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
