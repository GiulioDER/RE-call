from __future__ import annotations

from scripts.score_span_grounded_reader_vps2 import _best_span


def test_best_span_uses_context_offsets_and_subtracts_null() -> None:
    """Red proof omits the null subtraction and returns 12 rather than the registered margin 2."""
    start = [5.0, 0.0, 6.0, 1.0]
    end = [5.0, 0.0, 6.0, 1.0]
    offsets = [[0, 0], [0, 0], [4, 8], [9, 12]]
    sequence_ids = [None, 0, 1, 1]

    margin, char_start, char_end = _best_span(start, end, offsets, sequence_ids, 0)

    assert margin == 2.0
    assert (char_start, char_end) == (4, 8)


def test_best_span_rejects_question_tokens() -> None:
    """Red proof treats question tokens as context and selects their larger logits."""
    start = [0.0, 20.0, 3.0]
    end = [0.0, 20.0, 3.0]
    offsets = [[0, 0], [0, 4], [5, 9]]
    sequence_ids = [None, 0, 1]

    margin, char_start, char_end = _best_span(start, end, offsets, sequence_ids, 0)

    assert margin == 6.0
    assert (char_start, char_end) == (5, 9)
