"""The CamelCase code-token pattern must stay linear, and must match exactly what it used to.

`recall_aml.retrieval._CAMEL_CASE` runs inside `extract_code_tokens`, which the hosted service
applies to every query and to the text of every retrieved chunk. Its second alternative used to
read `[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+`: a nested quantifier that backtracks exponentially when
a run of capitals is followed by a character that is not a word boundary (CodeQL `py/redos`,
found 2026-09-23). Measured on `"Ab" + "A" * n + "_"`: n=16 0.007 s, n=18 0.027 s, n=20 0.113 s,
n=22 0.457 s, about four times slower per two characters.

`(?:[A-Z][A-Za-z0-9]*)+` accepts exactly the strings `[A-Z][A-Za-z0-9]*` accepts, so the fix is
that substitution and nothing else; the second test holds it to that.

Red proof (2026-09-23):
- `test_camel_case_pattern_is_linear_on_a_capital_run_before_an_underscore` was run against the
  unfixed pattern at `674d9a8b` and failed in its elapsed-time assertion, "code-token extraction
  took 3.681s on a 27 character token", then passed against the fix.
- `test_camel_case_pattern_matches_exactly_what_the_nested_form_matched` was run against a
  deliberate mutation of the fixed pattern, `[A-Z][A-Za-z0-9]*` changed to `[A-Z][A-Za-z]*`
  (digits dropped from the tail), and failed in its equality assertion with a falsifying example,
  then passed against the fix.
"""

from __future__ import annotations

import re
import time

from hypothesis import given, settings
from hypothesis import strategies as st

from recall_aml.retrieval import _CAMEL_CASE, extract_code_tokens

# The pattern as it stood before the fix, verbatim, kept only as the reference for equivalence.
_NESTED_CAMEL_CASE = re.compile(
    r"\b(?:[a-z]+[A-Z][A-Za-z0-9]*|[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+)\b"
)


def test_camel_case_pattern_is_linear_on_a_capital_run_before_an_underscore():
    hostile = "Ab" + "A" * 24 + "_"

    started = time.perf_counter()
    _CAMEL_CASE.findall(hostile)
    extract_code_tokens(hostile)
    elapsed = time.perf_counter() - started

    # The nested form needs seconds here and quadruples with every two more capitals; the linear
    # form needs microseconds. The bound is loose on purpose, so a slow runner cannot flake it.
    assert elapsed < 0.25, f"code-token extraction took {elapsed:.3f}s on a 27 character token"


@settings(max_examples=400, deadline=None)
@given(st.text(alphabet="aAbZz09_ .(-", max_size=14))
def test_camel_case_pattern_matches_exactly_what_the_nested_form_matched(value: str):
    assert _CAMEL_CASE.findall(value) == _NESTED_CAMEL_CASE.findall(value)
