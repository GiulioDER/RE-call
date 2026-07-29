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
    GATED_DOCS,
    RESULTS_ROOT,
    Claim,
    ClaimError,
    Marker,
    build_baseline,
    check_withdrawn,
    load_baseline,
    load_withdrawn,
    matches,
    resolve,
    scan_document,
    scan_text,
    unmarked_counts,
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


def test_k_equals_configuration_remains_excluded() -> None:
    """`k=5` is retrieval-budget configuration, not a claim about the data."""
    assert scan_text("retrieval budget k=5", doc="x.md") == []


def test_n_equals_is_no_longer_masked_a_sample_size_is_gated() -> None:
    """The design's stated defect: `results/gap/summary.json` read `usable: 1` beside a published
    `n=17` — and masking `n=` together with `k=` made this exact claim invisible to the gate.
    `n=` is a claim about the data (a sample size); only `k=` is configuration."""
    claims = scan_text("The clean subset holds n=17 records.", doc="x.md")
    assert [c.text for c in claims] == ["17"]


def test_comma_grouped_sample_size_is_one_claim() -> None:
    """`n=1,536` must scan as the single token `1,536`, not shred into `1` and `536` — a document
    edit from `n=1,536` to `n=2,536` must be visible to the gate as a changed claim."""
    claims = scan_text("n=1,536 answerable", doc="x.md")
    assert [c.text for c in claims] == ["1,536"]


def test_comma_grouped_number_with_decimals() -> None:
    claims = scan_text("total 1,536.25 units", doc="x.md")
    assert [c.text for c in claims] == ["1,536.25"]


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


def test_match_rule_strips_commas_from_the_published_string() -> None:
    """`n=1,536` in the document must match an artifact holding the plain int `1536`."""
    assert matches("1,536", 1536)
    assert not matches("1,536", 1537)


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


#: The ratchet. This number may only ever go DOWN. Lower it when you mark a number; a change that
#: raises it is a change that added an uncited number to a published document.
MAX_BASELINE_ENTRIES = 2431  # generated 2026-07-29: 2431 unmarked occurrences across 4 documents
#: (2438 before marking 6 bare withdrawn-figure occurrences — see WITHDRAWN.json — with
#: `<!--@ withdrawn: ... -->` in FINDINGS.md and README.md; 2432 before marking `0.467`
#: citation-pending in benchmarks/SUITE-DESIGN.md — see Task 5 — with
#: `<!--@ citation-pending: ... -->`.)


def test_unmarked_counts_ignores_marked_numbers() -> None:
    claims = [
        Claim("x.md", 1, "0.33", None),
        Claim("x.md", 2, "0.33", None),
        Claim("x.md", 3, "0.77", Marker("citation-pending", note="why")),
    ]
    assert unmarked_counts(claims) == {"0.33": 2}


def test_every_baseline_entry_is_still_present_and_still_unmarked() -> None:
    """Dead-entry test. Marking a number forces deleting its baseline row, so the file shrinks."""
    assert load_baseline(RESULTS_ROOT) == build_baseline(), (
        "CLAIMS_BASELINE.json no longer matches the documents. If you MARKED a number, remove its "
        "row here and lower MAX_BASELINE_ENTRIES. If you ADDED an unmarked number, mark it instead."
    )


def test_the_baseline_never_grows() -> None:
    total = sum(sum(counts.values()) for counts in load_baseline(RESULTS_ROOT).values())
    assert total <= MAX_BASELINE_ENTRIES


def test_no_withdrawn_value_hides_in_the_baseline() -> None:
    """The known-bad figures may not sit in the ratchet — that is where they would be invisible."""
    withdrawn = set(load_withdrawn(RESULTS_ROOT))
    for doc, counts in load_baseline(RESULTS_ROOT).items():
        assert not (withdrawn & set(counts)), f"{doc} baselines a withdrawn figure"


def test_the_committed_baseline_has_no_crlf() -> None:
    """`scripts/generate_claims_baseline.py` must write LF only, so the file it produces is
    byte-identical whether it was generated on Windows or Linux. Without an explicit `newline`
    argument, `Path.write_text` applies universal-newline translation and emits `os.linesep` —
    on Windows that turns every `\n` `json.dumps(indent=2)` embeds into `\r\n`. A generated
    artifact that comes out different on the OS that produced it than on the OS that consumes it
    is exactly how a ratchet like this one silently diverges. Read as bytes: reading with
    universal newlines would translate away the very bytes this test exists to check, and the
    assertion would pass vacuously."""
    raw = (RESULTS_ROOT / "CLAIMS_BASELINE.json").read_bytes()
    assert b"\r\n" not in raw


# --- The gate, armed over the four published documents -----------------------------------------


@pytest.mark.parametrize("doc", GATED_DOCS)
def test_every_marked_claim_resolves(doc: str) -> None:
    """A marker that does not resolve is worse than no marker: it reads as verified."""
    failures: list[str] = []
    for claim in scan_document(Path(doc)):
        if claim.marker is None:
            continue
        try:
            resolve(claim, RESULTS_ROOT)
        except ClaimError as exc:
            failures.append(str(exc))
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("doc", GATED_DOCS)
def test_no_new_unmarked_numbers(doc: str) -> None:
    baseline = load_baseline(RESULTS_ROOT).get(doc, {})
    current = unmarked_counts(scan_document(Path(doc)))
    changed = {
        value: count for value, count in current.items() if count != baseline.get(value, 0)
    }
    assert not changed, (
        f"{doc}: these numbers are new or changed since the baseline: {changed}. Add a marker — "
        f"`<!--@ <artifact>.json # <key> -->`, or `<!--@ citation-pending: <reason> -->` if no "
        f"artifact retains it — and lower MAX_BASELINE_ENTRIES."
    )


@pytest.mark.parametrize("doc", GATED_DOCS)
def test_no_bare_withdrawn_figures(doc: str) -> None:
    errors = check_withdrawn(scan_document(Path(doc)), load_withdrawn(RESULTS_ROOT))
    assert not errors, "\n".join(str(e) for e in errors)


def test_composition_check_withdrawn_and_resolve_must_both_run_over_the_same_claims(
    tmp_path: Path,
) -> None:
    """Pins the two-test composition that makes the withdrawn rule actually safe.

    `check_withdrawn` passes any claim whose marker KIND is `artifact` without opening the
    artifact file — it only checks that *some* citation exists, not that the citation is real. On
    its own that would let a document exempt a retracted figure with a fabricated `artifact:`
    marker pointing at a file that does not exist. The only thing that catches the fabrication is
    `resolve()`, run over the SAME claim, because `resolve()` is the one that actually opens the
    artifact. `test_no_bare_withdrawn_figures` and `test_every_marked_claim_resolves` must both
    stay in this suite, over the same `GATED_DOCS`, for a withdrawn figure to be genuinely closed
    off — either one alone looks sufficient and is not. If a future edit drops one of the pair,
    this test fails and says why.
    """
    withdrawn = {"0.945": {"figure": "real-corpus recall@5", "retraction_ref": "README"}}
    claim = Claim(
        "x.md", 3, "0.945", Marker("artifact", artifact="sub/does-not-exist.json", key="hit")
    )

    # check_withdrawn alone: a fabricated artifact marker is enough to pass — it never opens the
    # file. This is the gap; without the second check below, this line would be "the" answer.
    assert check_withdrawn([claim], withdrawn) == []

    # resolve alone closes it: it actually opens the artifact, and there is nothing at that path.
    with pytest.raises(ClaimError, match="no such artifact"):
        resolve(claim, tmp_path)
