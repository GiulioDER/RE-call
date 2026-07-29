"""The claim gate: a number in a published document must resolve to a committed artifact.

`results/ARTIFACTS.md` and `test_results_artifact_provenance.py` already enforce the other
direction — an artifact must declare what it is. Nothing stopped a number appearing in a document
that no artifact contains, which is how three defects reached publication on 2026-07-29: a loss
published as a tie, a figure derivable from nothing, and a count that contradicted its own summary.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.claim_gate import (
    RESULTS_ROOT,
    Claim,
    ClaimError,
    Marker,
    check_withdrawn,
    load_withdrawn,
    matches,
    resolve,
    scan_text,
)


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


def test_match_rule_rounds_to_the_published_precision() -> None:
    assert matches("0.777", 0.77714)
    assert matches("0.78", 0.7771)
    assert matches("17", 17)


def test_match_rule_rejects_the_suite_design_defect() -> None:
    """SUITE-DESIGN published 0.533 where the cell is 0.536 — a loss printed as a tie."""
    assert not matches("0.533", 0.536)


def test_match_rule_rejects_a_non_number() -> None:
    assert not matches("17", "17")
    assert not matches("1", True)


def _write_artifact(root: Path, payload: dict) -> None:
    (root / "sub").mkdir(parents=True, exist_ok=True)
    (root / "sub" / "a.json").write_text(json.dumps(payload), encoding="utf-8")


def test_resolve_accepts_a_matching_artifact(tmp_path: Path) -> None:
    _write_artifact(tmp_path, {"depth": {"5": {"hit": 0.7771}}})
    claim = Claim("x.md", 1, "0.777", Marker("artifact", artifact="sub/a.json", key="depth.5.hit"))
    resolve(claim, tmp_path)  # does not raise


def test_resolve_rejects_a_mismatching_artifact(tmp_path: Path) -> None:
    _write_artifact(tmp_path, {"depth": {"5": {"hit": 0.536}}})
    claim = Claim("x.md", 1, "0.533", Marker("artifact", artifact="sub/a.json", key="depth.5.hit"))
    with pytest.raises(ClaimError, match="0.536"):
        resolve(claim, tmp_path)


def test_resolve_rejects_a_missing_artifact(tmp_path: Path) -> None:
    claim = Claim("x.md", 1, "0.777", Marker("artifact", artifact="sub/missing.json", key="a"))
    with pytest.raises(ClaimError, match="no such artifact"):
        resolve(claim, tmp_path)


def test_resolve_rejects_a_missing_key(tmp_path: Path) -> None:
    _write_artifact(tmp_path, {"depth": {}})
    claim = Claim("x.md", 1, "0.777", Marker("artifact", artifact="sub/a.json", key="depth.5.hit"))
    with pytest.raises(ClaimError, match="no key"):
        resolve(claim, tmp_path)


def test_citation_pending_needs_a_reason(tmp_path: Path) -> None:
    resolve(Claim("x.md", 1, "0.467", Marker("citation-pending", note="no artifact")), tmp_path)
    with pytest.raises(ClaimError, match="reason"):
        resolve(Claim("x.md", 1, "0.467", Marker("citation-pending", note="")), tmp_path)


def test_withdrawn_needs_a_retraction_reference(tmp_path: Path) -> None:
    resolve(Claim("x.md", 1, "0.945", Marker("withdrawn", note="README list")), tmp_path)
    with pytest.raises(ClaimError, match="retraction"):
        resolve(Claim("x.md", 1, "0.945", Marker("withdrawn", note="")), tmp_path)


def test_derived_checks_the_arithmetic(tmp_path: Path) -> None:
    resolve(Claim("x.md", 1, "0.106", Marker("derived", note="0.777 - 0.671")), tmp_path)
    with pytest.raises(ClaimError, match="derived"):
        resolve(Claim("x.md", 1, "0.200", Marker("derived", note="0.777 - 0.671")), tmp_path)


def test_derived_refuses_anything_that_is_not_literal_arithmetic(tmp_path: Path) -> None:
    """The evaluator walks the AST and applies `operator` functions. It must not reach names,
    calls, attributes or subscripts — a documentation gate is not a place to execute code."""
    for hostile in ("__import__('os').getcwd()", "open('x')", "a + 1", "[1][0]"):
        with pytest.raises(ClaimError, match="literal arithmetic|does not parse"):
            resolve(Claim("x.md", 1, "1.0", Marker("derived", note=hostile)), tmp_path)


def test_an_unmarked_claim_does_not_resolve(tmp_path: Path) -> None:
    with pytest.raises(ClaimError, match="unmarked"):
        resolve(Claim("x.md", 1, "0.777", None), tmp_path)


def test_a_withdrawn_value_may_not_appear_bare() -> None:
    withdrawn = {"0.945": {"figure": "real-corpus recall@5", "retraction_ref": "README"}}
    errors = check_withdrawn([Claim("x.md", 3, "0.945", None)], withdrawn)
    assert len(errors) == 1
    assert "withdrawn" in str(errors[0])


def test_a_withdrawn_value_passes_with_a_withdrawn_marker() -> None:
    withdrawn = {"0.945": {"figure": "real-corpus recall@5", "retraction_ref": "README"}}
    claim = Claim("x.md", 3, "0.945", Marker("withdrawn", note="README withdrawn list"))
    assert check_withdrawn([claim], withdrawn) == []


def test_a_withdrawn_value_passes_when_legitimately_re_measured() -> None:
    """Same digits, arrived at from a committed artifact — a different figure that reads the same."""
    withdrawn = {"0.945": {"figure": "real-corpus recall@5", "retraction_ref": "README"}}
    claim = Claim("x.md", 3, "0.945", Marker("artifact", artifact="a.json", key="hit"))
    assert check_withdrawn([claim], withdrawn) == []


def test_a_citation_pending_marker_does_not_excuse_a_withdrawn_figure() -> None:
    """"We have not sourced it yet" is not the same statement as "this was retracted"."""
    withdrawn = {"0.945": {"figure": "real-corpus recall@5", "retraction_ref": "README"}}
    claim = Claim("x.md", 3, "0.945", Marker("citation-pending", note="later"))
    assert len(check_withdrawn([claim], withdrawn)) == 1


def test_the_registry_on_disk_is_well_formed() -> None:
    withdrawn = load_withdrawn(RESULTS_ROOT)
    assert withdrawn, "an empty registry would make the withdrawn rule vacuous"
    for value, entry in withdrawn.items():
        assert value == value.strip()
        assert entry["figure"].strip()
        assert entry["retraction_ref"].strip()


def test_the_registry_holds_literal_digit_strings_not_floats() -> None:
    """0.615 and its artifact's 0.6152 are different strings; float comparison would conflate a
    retracted figure with a live one at a different precision."""
    for value in load_withdrawn(RESULTS_ROOT):
        assert isinstance(value, str)
