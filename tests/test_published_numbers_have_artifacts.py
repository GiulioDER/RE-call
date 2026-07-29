"""The claim gate: a number in a published document must resolve to a committed artifact.

`results/ARTIFACTS.md` and `test_results_artifact_provenance.py` already enforce the other
direction — an artifact must declare what it is. Nothing stopped a number appearing in a document
that no artifact contains, which is how three defects reached publication on 2026-07-29: a loss
published as a tie, a figure derivable from nothing, and a count that contradicted its own summary.
"""
from __future__ import annotations

from benchmarks.claim_gate import scan_text


def test_a_bare_decimal_is_an_unmarked_claim() -> None:
    claims = scan_text("the shipped reranker reaches 0.777 overall.", doc="x.md")
    assert [(c.text, c.marker) for c in claims] == [("0.777", None)]


def test_a_bare_integer_is_an_unmarked_claim() -> None:
    """The `usable: 1` beside a published `n=17` defect was an integer."""
    claims = scan_text("The clean subset holds 17 records.", doc="x.md")
    assert [c.text for c in claims] == ["17"]


def test_an_artifact_marker_binds_to_the_number_before_it() -> None:
    claims = scan_text(
        "reaches **0.777** <!--@ locomo_rerank/rerank_shipped.json # depth_curve.5.overall.hit -->",
        doc="x.md",
    )
    assert len(claims) == 1
    marker = claims[0].marker
    assert marker is not None
    assert marker.kind == "artifact"
    assert marker.artifact == "locomo_rerank/rerank_shipped.json"
    assert marker.key == "depth_curve.5.overall.hit"


def test_citation_pending_derived_and_withdrawn_markers_parse() -> None:
    text = (
        "a 0.467 <!--@ citation-pending: no artifact retains this -->\n"
        "b 0.106 <!--@ derived: 0.777 - 0.671 -->\n"
        "c 0.945 <!--@ withdrawn: README withdrawn list -->\n"
    )
    kinds = [c.marker.kind for c in scan_text(text, doc="x.md") if c.marker]
    assert kinds == ["citation-pending", "derived", "withdrawn"]


def test_excluded_spans_hide_their_digits() -> None:
    text = (
        "code `k = 5` and ```\nblock 3.14\n``` and https://example.com/9.9 and "
        "v0.7.0 and 2026-07-29 and #1987 and hit@5 and bge-large-en-v1.5 and 2026"
    )
    assert scan_text(text, doc="x.md") == []


def test_line_numbers_are_one_based() -> None:
    claims = scan_text("nothing here\nbut 0.33 here\n", doc="x.md")
    assert [c.line for c in claims] == [2]


def test_a_marker_binds_only_to_the_nearest_preceding_number() -> None:
    """Two numbers before one marker must not both read as backed by it.

    A single marker covering both would let one of them drift unchecked — if a document reports
    `hit@5 improves 0.671 -> 0.777 <!--@ f.json # k -->`, only 0.777 (the nearest preceding
    number) may resolve to the artifact key; 0.671 must come back as an unmarked claim, not a
    second claim silently backed by the same evidence.
    """
    claims = scan_text("a 1 and 2 <!--@ f.json # k -->", doc="x.md")
    assert len(claims) == 2
    first, second = claims
    assert first.text == "1"
    assert first.marker is None
    assert second.text == "2"
    assert second.marker is not None
    assert second.marker.kind == "artifact"
    assert second.marker.artifact == "f.json"
    assert second.marker.key == "k"
