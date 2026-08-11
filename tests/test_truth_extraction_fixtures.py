"""The four transplanted negatives reproduce fix.py's measured false positives.

Each is a labelled NEGATIVE: a sentence a naive extractor reads as declaring a supersession
edge, which on the private corpus was wrong on human review. They are the publishable half of a
private finding — the error MIX transfers even though the corpus cannot.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from recall.fix import extract_edges
from recall.frontmatter import parse_frontmatter
from recall.lint import CLOSURE_MARKERS

FIXTURES = (
    Path(__file__).resolve().parents[1]
    / "benchmarks" / "labelling" / "truth_extraction" / "fixtures"
)

CASES = [
    ("reported_speech.md", "the marker's subject is another document"),
    ("hedged.md", "the author wrote Supersedes/augments and meant augments"),
    ("partial_scope_claim.md", "supersedes a claim inside the target, not the target"),
    ("partial_scope_scope.md", "supersedes the scope in the target, not the target"),
]


def test_all_four_fixtures_exist():
    assert sorted(p.name for p in FIXTURES.glob("*.md")) == sorted(n for n, _ in CASES)


@pytest.mark.parametrize(("name", "why"), CASES)
def test_fixture_is_a_negative(name: str, why: str):
    _, body = parse_frontmatter((FIXTURES / name).read_text(encoding="utf-8"))
    active, passive = extract_edges(body)
    assert not active and not passive, f"{name} must be refused: {why}"


@pytest.mark.parametrize(("name", "_why"), CASES)
def test_fixture_would_tempt_a_naive_extractor(name: str, _why: str):
    # Guards the guard: a fixture that carried no marker at all would pass the test above for
    # the wrong reason, and the set would silently stop covering its error mode.
    _, body = parse_frontmatter((FIXTURES / name).read_text(encoding="utf-8"))
    assert CLOSURE_MARKERS.search(body), f"{name} carries no closure marker — it tests nothing"
