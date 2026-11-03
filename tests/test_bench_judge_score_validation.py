"""Malformed numeric judge replies are harness errors, never evidence about an arm."""
import math

import pytest

from benchmarks.beam.dataset import Question
from benchmarks.beam.run import judge_answer


@pytest.mark.parametrize("raw", ["{}", '{"score":"NaN"}', '{"score":"Infinity"}', '{"score":"-Infinity"}', '{"score":true}', '{"score":false}'])
def test_invalid_judge_score_is_counted_as_an_error(raw):
    """BUG-003: df339c87a fabricates a finite score; fail at the error-count assertion."""
    question = Question("q", "1M", 0, "fact", "question", ["rubric"])
    score, nuggets, errors = judge_answer(question, "answer", lambda *args: raw)
    assert errors == 1
    assert math.isnan(score)
    assert nuggets[0]["score"] is None
    good, _, good_errors = judge_answer(question, "answer", lambda *args: '{"score":"0.5"}')
    assert good == 0.5
    assert good_errors == 0
