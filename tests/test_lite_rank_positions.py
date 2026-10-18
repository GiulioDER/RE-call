"""Keyword occurrence saturation follows the independently evaluated PostgreSQL score.

Baseline: PR897 original head 01149368ed0e43618ae3990930bcd4ed20170475.
test_keyword_scores_match_retained_positions targets query_sparse's count/cap and
fails "keyword rank differs from retained positions" at all six boundary inputs.
test_saturated_ranks_keep_id_ties fails "saturated ranks lost their id tie".
The two Hypothesis nodes preserve the declared retained-count and terminal-position
plateau properties from the CCA numeric reproductions. Their public query_sparse
wrappers target the same count/cap and fail the named plateau assertions.
"""
from __future__ import annotations

import os
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import psycopg
import pytest
from hypothesis import example, given, settings, strategies as st

from recall.lite import LiteStore
from recall.types import Chunk

_BOUNDARIES = [(256, 0), (512, 0), (2, 16383), (8, 16400), (3, 16381), (4, 16380)]


def _text(prefix: int, repetitions: int) -> str:
    return "the " * prefix + " ".join(["valve"] * repetitions)


def _keyword_score(text: str) -> float:
    with TemporaryDirectory() as directory:
        with closing(LiteStore(Path(directory) / "memory.db", dim=2)) as store:
            store.upsert([Chunk(id="a", source="/m/a.md", text=text)], [[1.0, 0.0]])
            return store.query_sparse("valve", k=1)[0].score


@pytest.mark.parametrize("repetitions,prefix", _BOUNDARIES)
def test_keyword_scores_match_retained_positions(repetitions: int, prefix: int) -> None:
    dsn = os.environ.get("RECALL_TEST_DSN")
    if not dsn:
        pytest.skip("requires the checkout's own PostgreSQL reference database")
    text = _text(prefix, repetitions)
    with psycopg.connect(dsn) as conn:
        reference = float(conn.execute(
            "SELECT ts_rank(to_tsvector('english', %s), to_tsquery('english', 'valve'))", (text,),
        ).fetchone()[0])
    assert abs(_keyword_score(text) - reference) <= 1e-7, "keyword rank differs from retained positions"


def test_saturated_ranks_keep_id_ties(tmp_path: Path) -> None:
    with closing(LiteStore(tmp_path / "memory.db", dim=2)) as store:
        store.upsert([
            Chunk(id="a-255", source="/m/a.md", text=_text(0, 255)),
            Chunk(id="b-256", source="/m/b.md", text=_text(0, 256)),
        ], [[1.0, 0.0], [1.0, 0.0]])
        assert [hit.chunk.id for hit in store.query_sparse("valve", k=2)] == ["a-255", "b-256"], (
            "saturated ranks lost their id tie"
        )


@settings(max_examples=25, deadline=None, derandomize=True)
@example(occurrences=255)
@given(occurrences=st.integers(255, 512))
def test_retained_count_plateau(occurrences: int) -> None:
    next_count = occurrences + 1 if occurrences < 512 else occurrences - 1
    assert _keyword_score(_text(0, occurrences)) == _keyword_score(_text(0, next_count)), (
        "retained count plateau was not respected"
    )


@settings(max_examples=25, deadline=None, derandomize=True)
@example(prefix_words=16383, repetitions=1)
@given(prefix_words=st.integers(16383, 16400), repetitions=st.integers(1, 8))
def test_terminal_position_plateau(prefix_words: int, repetitions: int) -> None:
    next_count = repetitions + 1 if repetitions < 8 else repetitions - 1
    assert _keyword_score(_text(prefix_words, repetitions)) == _keyword_score(_text(prefix_words, next_count)), (
        "terminal position plateau was not respected"
    )
