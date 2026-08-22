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


def test_citation_pending_and_withdrawn_markers_parse() -> None:
    text = (
        "a 0.467 <!--@ citation-pending: no artifact retains this -->\n"
        "c 0.945 <!--@ withdrawn: README withdrawn list -->\n"
    )
    kinds = [c.marker.kind for c in scan_text(text, doc="x.md") if c.marker]
    assert kinds == ["citation-pending", "withdrawn"]


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


# --- P1-A: sign-aware NUMBER_RE ------------------------------------------------------------


def test_a_correctly_cited_negative_claim_resolves(tmp_path: Path) -> None:
    """A correctly-signed claim, backed by a matching artifact, must not be rejected.

    Rewritten against an `artifact:` marker (not `derived:`, removed 2026-07-29) — this pins P1-A
    sign coverage, which does not depend on which marker kind carries it.
    """
    _write_artifact(tmp_path, {"delta": -0.065})
    claims = scan_text("Delta was -0.065 <!--@ sub/a.json # delta --> after the fix.", doc="x.md")
    assert len(claims) == 1
    assert claims[0].text == "-0.065"
    resolve(claims[0], tmp_path)  # does not raise


def test_a_sign_flipped_claim_is_rejected(tmp_path: Path) -> None:
    """Reproduction: the document prints -0.065 but the artifact holds +0.065 — a wrong sign must
    not read as verified. Before the P1-A fix, `Claim.text` dropped the minus entirely and this
    ACCEPTED."""
    _write_artifact(tmp_path, {"delta": 0.065})
    claims = scan_text("Delta was -0.065 <!--@ sub/a.json # delta --> after the fix.", doc="x.md")
    assert len(claims) == 1
    assert claims[0].text == "-0.065"
    with pytest.raises(ClaimError, match="0.065"):
        resolve(claims[0], tmp_path)


def test_a_hyphenated_range_is_not_read_as_negative() -> None:
    """The hyphen in `0.36-0.43` is preceded by a digit, so it is a range separator, not a sign."""
    claims = scan_text("the estimate spans 0.36-0.43 across runs.", doc="x.md")
    assert [c.text for c in claims] == ["0.36", "0.43"]


def test_a_leading_hyphen_after_a_word_boundary_is_a_sign() -> None:
    """Adjacent to the digits and preceded by non-alnum (a space here) -> genuinely a sign."""
    claims = scan_text("score change: -5 points", doc="x.md")
    assert [c.text for c in claims] == ["-5"]


def test_a_bullet_hyphen_with_a_space_is_not_a_sign() -> None:
    """`- 5` (list marker, space before the digit) must not be read as `-5`: the sign must be
    IMMEDIATELY adjacent to the digits, with nothing in between."""
    claims = scan_text("- 5 items were dropped\n", doc="x.md")
    assert [c.text for c in claims] == ["5"]


def test_a_hyphen_after_an_identifier_is_not_a_sign() -> None:
    """`a-1` and `5-3`: a hyphen preceded by an alphanumeric character is never a sign."""
    claims = scan_text("run a-1 scored 5-3 on the rubric.", doc="x.md")
    assert [c.text for c in claims] == ["1", "5", "3"]


def test_unicode_minus_is_captured_as_a_sign() -> None:
    """`results/FINDINGS.md` prints negative deltas with U+2212 (−), not ASCII `-`."""
    claims = scan_text("oov_rate correlates −0.512 with corpus size", doc="x.md")
    assert claims[0].text == "−0.512"


def test_match_rule_normalises_unicode_minus() -> None:
    assert matches("−0.512", -0.512)
    assert not matches("−0.512", 0.512)


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


# --- P1-B: markers inside code spans must not bind ----------------------------------------


def test_a_marker_inside_backticks_does_not_bind_to_a_preceding_bare_number() -> None:
    """Reproduction: documenting marker syntax in backticks must not launder a real number.

    Before the fix, `scan_text` extracted NUMBERS from the masked text but MARKERS from the raw
    text, so a marker written inside inline code still bound to whatever real number preceded it
    in prose on the same line — and the number never landed in `unmarked_counts`, so the ratchet
    could not see it either.
    """
    claims = scan_text(
        "Our score improved to 0.884 this week, e.g. write markers like "
        "`<!--@ citation-pending: example -->`.",
        doc="x.md",
    )
    assert len(claims) == 1
    assert claims[0].text == "0.884"
    assert claims[0].marker is None  # unmarked: the marker was inside code, so it doesn't count
    with pytest.raises(ClaimError, match="unmarked"):
        resolve(claims[0], RESULTS_ROOT)


def test_a_marker_inside_a_fenced_block_does_not_bind() -> None:
    """Same line, so a different-line skip in `_marker_for` cannot be what blocks the bind — only
    the fenced-code mask can be."""
    text = "The headline is 0.777 today. ```<!--@ citation-pending: example -->``` more text"
    claims = scan_text(text, doc="x.md")
    assert len(claims) == 1
    assert claims[0].text == "0.777"
    assert claims[0].marker is None


def test_a_marker_outside_code_still_binds_normally() -> None:
    """The fix must not blind the scanner to REAL markers — only ones sitting inside code."""
    claims = scan_text("reaches 0.777 <!--@ f.json # k -->", doc="x.md")
    assert len(claims) == 1
    assert claims[0].marker is not None
    assert claims[0].marker.kind == "artifact"


def test_match_rule_rounds_to_the_published_precision() -> None:
    assert matches("0.777", 0.77714)
    assert matches("0.78", 0.7771)
    assert matches("17", 17)


def test_match_rule_rejects_the_suite_design_defect() -> None:
    """SUITE-DESIGN published 0.533 where the cell is 0.536 — a loss printed as a tie."""
    assert not matches("0.533", 0.536)


def test_match_rule_rounds_half_to_even_not_half_away_from_zero() -> None:
    """F-07: pin the documented rounding convention. `f"{0.625:.2f}"` is `"0.62"` (2 is already
    even), not the `"0.63"` hand-rounding would produce — behaviour, not a bug; this test exists
    so a future "fix" to the more intuitive convention shows up as a failing test, not a silent
    change to which boundary values this gate accepts."""
    assert matches("0.62", 0.625)
    assert not matches("0.63", 0.625)


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


def test_resolve_rejects_an_artifact_path_that_escapes_results_root(tmp_path: Path) -> None:
    """`MARKER_RE`'s artifact path is `[\\w./-]+\\.json`, which permits `../`. A marker can still
    only CITE something — it cannot fabricate a claim, because the value must match too — but it
    could cite a file outside `results/` that was never committed as a result. Write a real,
    matching file just outside `results_root` to prove the containment check (not the missing-file
    branch above) is what rejects the traversal."""
    results_root = tmp_path / "results"
    results_root.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "escape.json").write_text(json.dumps({"a": 0.777}), encoding="utf-8")
    claim = Claim(
        "x.md", 1, "0.777", Marker("artifact", artifact="../outside/escape.json", key="a")
    )
    with pytest.raises(ClaimError, match="outside|escapes|results_root|containment"):
        resolve(claim, results_root)


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


def test_the_registry_values_parse_as_floats_matching_their_own_precision() -> None:
    """Pins the contract `load_withdrawn` exists for: each key is a digit string that `matches()`
    accepts against its own parsed float value, at its own published precision.

    Not `isinstance(value, str)` (the old version of this test): `json.loads` always produces
    `str` keys, so that assertion could never fail regardless of what the registry held — it
    pinned nothing. Parsing each key as a float AND round-tripping it through `matches()` actually
    exercises why the registry stores literal digit strings rather than floats in the first place
    (0.615 and its artifact's 0.6152 are different strings; float comparison would conflate a
    retracted figure with a live one at a different precision)."""
    for value in load_withdrawn(RESULTS_ROOT):
        parsed = float(value.replace(",", "").replace("−", "-"))
        assert matches(value, parsed)


#: The historical ratchet log. `MAX_BASELINE_ENTRIES` — a hand-maintained second copy of the
#: baseline's total size — was removed 2026-07-29 (deferred CCA second-pass audit):
#: `test_every_baseline_entry_is_still_present_and_still_unmarked` below already asserts full
#: dict equality between `load_baseline` and `build_baseline`, which already makes growth
#: structurally impossible — any new or changed unmarked number fails that test regardless of
#: what this constant said. The constant added no coverage the equality test lacked, and being a
#: hand-edited duplicate figure, it went stale on its own: raised three times in one week and the
#: direct cause of finding F-05 (a stale number in this very ratchet). Kept below as a comment,
#: not a constant, because the growth history itself is worth keeping:
#:
#: 2481 unmarked occurrences across 4 documents as of 2026-07-29 (2438 before marking 6 bare
#: withdrawn-figure occurrences — see WITHDRAWN.json — with `<!--@ withdrawn: ... -->` in
#: FINDINGS.md and README.md; 2432 before marking `0.467` citation-pending in
#: benchmarks/SUITE-DESIGN.md — see Task 5 — with `<!--@ citation-pending: ... -->`; 2431 -> 2444
#: (+13) in the final-review pass: `EXCLUSIONS` stopped masking `n=` alongside `k=` (a sample size
#: is a claim, not configuration — the exact defect class the design spec cites, `usable: 1`
#: beside a published `n=17`) and `NUMBER_RE` stopped shredding comma-grouped integers like
#: `1,536` into `1` + `536`. The `n=` change alone made roughly 60 previously-invisible integers
#: visible; the comma-grouping change partially offsets it by merging pairs of digit fragments
#: back into one token. Net +13 unmarked numbers is the correct, larger, more honest baseline —
#: not a regression to chase back down.
#:
#: 2444 -> 2481 (+37) on merging origin/master: PR #154 rewrote SUITE-DESIGN.md's Track C passage
#: — the false-abstain correction — introducing 21 distinct uncited numbers (9.3, 4.1, 3.3, 0.536,
#: 0.594, 0.650, ...). The gate caught them on the merge, which is the guard working; they are
#: baselined rather than cited because they landed on master BEFORE this gate existed, the same
#: reasoning that froze the original 2431. Numbers written into these documents from here on must
#: carry a marker.
#:
#: 2481 -> 2481 (net 0) after P1-A: `NUMBER_RE`/`Claim.text` now capture a leading sign (ASCII `-`
#: or Unicode minus `−`), so `matches()` can catch a sign-flipped claim instead of silently
#: accepting one — see the note on `NUMBER_RE` in `benchmarks/claim_gate.py`. This RELABELS
#: entries, it does not add or remove any: every occurrence that used to sit under its unsigned
#: digit string (`"0.065"`) now sits under its signed one (`"-0.065"`) if the document actually
#: printed the sign — the per-document and grand totals are identical before and after (RESULTS
#: 826, FINDINGS 1274, README 303, SUITE-DESIGN 78; 2481 throughout). Two effects are visible in
#: the regenerated file: (1) FINDINGS.md and RESULTS.md's negative deltas (`−0.009`, `−0.512`,
#: ...) move from their unsigned to their signed key; (2) README.md's Quickstart anchor slug
#: `#quickstart--2-minutes...` — a `-` preceded by another `-`, which the stated sign rule counts
#: as real — relabels 2 occurrences of `"2"` to `"-2"`. Neither is a real claim in either form;
#: see the stated-limit paragraph on `NUMBER_RE`.)
#:
#: 2481, unchanged, when `derived:` was deleted 2026-07-29 (deferred second pass): `derived:` had
#: zero occurrences in any gated document, so removing it could not move a single baseline row.
#:
#: ⚠️ THE PROSE TOTALS ABOVE ARE NOT DERIVED FROM GIT, and at least one of them never existed. An
#: entry added on 2026-08-07 claimed a shrink "2481 -> 2463"; git holds no commit of this file
#: totalling 2481. Reconstructed by summing the committed file at every commit that has ever touched
#: it, which is the only source that cannot drift:
#:
#:     3d3c905  2026-08-02  2484   (file added)
#:     0341c15  2026-08-04  2480
#:     274bf73  2026-08-06  2463
#:     3509256  2026-08-07  2556   arming the gate over docs/ENTERPRISE_RETRIEVAL.md
#:     2c892f1  2026-08-07  2557   +1, see below
#:
#: ⚠️ That table is current only as far as the commit that last edited this comment, and it will be
#: stale the moment the baseline moves again — including on the commit that added it. Do not patch it
#: by hand; every hand-maintained total in this log that has been checked against git has been wrong.
#: Regenerate it, and treat the output as the record:
#:
#:     for r in $(git log --format=%H --all -- results/CLAIMS_BASELINE.json); do \
#:       python -c "import json,subprocess,sys;b=json.loads(subprocess.run(
#:       ['git','show','$r:results/CLAIMS_BASELINE.json'],capture_output=True,text=True).stdout);
#:       print(sum(sum(v.values()) for k,v in b.items() if k!='_note'))"; done
#:
#: 2463 -> 2556 (+93) on 2026-08-07, arming the gate over `docs/ENTERPRISE_RETRIEVAL.md`. Largest
#: single growth event in the ratchet's history, and the ONLY one that is not "numbers that predate
#: the gate": the document is new and the same session that wrote it armed the gate over it.
#:
#: 2556 -> 2557 (+1) immediately after, and the +1 is the lesson. CI's `pull_request` event scans
#: the MERGE of the branch into master, not the branch tip, and a concurrent commit had added one
#: `~0` to the runbook. The baseline was regenerated against the tip and went red in CI. See the
#: note on `GATED_DOCS` in `benchmarks/claim_gate.py`.
#:
#: SIX OCCURRENCES were MARKED instead of baselined, because freezing them would have been the gate
#: certifying its own author's unbacked numbers: the five Qwen3 latency measurements the document
#: uses to justify a rejection verdict (taken on VPS2 on 2026-08-03, before this repository's
#: artifact convention, with no committed `results/*.json` retaining them) and the `2.2x`
#: disk-headroom figure the document's own prose calls a policy rule of thumb.
#:
#: ⚠️ Occurrences, NOT figures. Three further occurrences of those same unbacked numbers ARE frozen
#: unmarked: `2.2` where the preconditions restate it, and `5.8` and `41` where the prose restates
#: the Qwen3 latencies. So the claim "those numbers are not baselined" is false at the figure level
#: and true only at the six marked sites. Known gap, recorded rather than quietly narrowed.
#:
#: The rest are frozen deliberately and are overwhelmingly structural. The `EXCLUSIONS` soft-spot
#: note applies with unusual force to this document: roughly three fifths of its numeric tokens are
#: masked and the masked set is the command arguments. Read the coverage note on `GATED_DOCS` in
#: `benchmarks/claim_gate.py` before treating a green run here as coverage of the runbook.
#:
#: 2557 -> 2558 (+1) on 2026-08-08, shortening the README into `docs/EVIDENCE.md`,
#: `docs/PRODUCTION.md`, `docs/PRIOR_ART.md` and `docs/ENGINEERING.md`, and arming the gate over all
#: four in the same commit. Almost entirely a MOVE, and the move is the thing to check rather than
#: the total: `README.md` went 299 -> 21 while the four destinations went 0 -> 279, and the multiset
#: over those five documents lost nothing at all. The whole +1 is one occurrence of `0`, from the
#: product name `Mem0` in the README's new "Read next" row.
#:
#: ⚠️ State that one in a checkable form, because every hand-maintained total in this log that has
#: since been checked against git was wrong. The `0` row over those five documents went 17 -> 18;
#: the NAME `Mem0` accounts for 64 -> 65 unmasked occurrences across the whole gated set. The row
#: count and the name count are different quantities and the first is the one the +1 comes from.
#: Either way this is the existing treatment of a name the number regex cannot tell from a figure,
#: not a new unbacked claim.
#:
#: 2544 -> 2555 (+11) on 2026-08-22, arming the gate over `docs/ATM_BENCH.md` in the same change
#: that wrote it. Verified against a regeneration diff rather than by hand: exactly one row was
#: ADDED, no existing row changed by a single occurrence, and the branch was rebased onto
#: `origin/master` first so the regeneration ran against the merge result (master's one intervening
#: commit touched `site/*.html` only, so it could not have moved a row either way).
#:
#: 🔑 The +11 is the SMALLEST arming event in this log, and deliberately so. The document scans to
#: 132 numeric claims and 121 of them carry a marker, so the frozen 11 are structural without
#: exception: seven section headings (`## 1.` to `## 7.`), two occurrences of `20260821` from the
#: artifact filename inside a relative link path, the arXiv identifier `2603.01990`, and one `0`
#: from the harness name `Mem0` in the comparison table -- the same name-versus-figure case the
#: 2026-08-08 entry above describes, arriving for the same reason in a different document.
#:
#: Compare the 2026-08-07 arming of `docs/ENTERPRISE_RETRIEVAL.md`: +93 frozen, six marked. The
#: ratio is inverted here because that document's numbers predated the artifact convention and no
#: `results/*.json` retained them, whereas every RE-call figure in this one resolves to
#: `results/atm/atm_bench_full_20260821.json`. Where a figure genuinely cannot resolve it is
#: escalated with `citation-pending` and a stated reason, not frozen: the maintainers' own
#: leaderboard rows, the judge-transport route comparison, and the zero-cost replay behind section 5.
#:
#: What arming it bought, stated as the defect it would have caught: the document shipped a
#: `list_recall` retrieval-to-QS gap of `30.1010` for a value of `30.10106...`, truncated rather
#: than rounded on its way out of a submission report. A human caught it. `retrieval_to_qs_gap` was
#: added to the artifact in this commit, so that cell and the three beside it now resolve, and the
#: next truncation of one fails `resolve()` instead of needing to be noticed.


def test_unmarked_counts_ignores_marked_numbers() -> None:
    claims = [
        Claim("x.md", 1, "0.33", None),
        Claim("x.md", 2, "0.33", None),
        Claim("x.md", 3, "0.77", Marker("citation-pending", note="why")),
    ]
    assert unmarked_counts(claims) == {"0.33": 2}


def test_every_baseline_entry_is_still_present_and_still_unmarked() -> None:
    """Dead-entry test, and the sole ratchet (see the history comment above): full dict equality
    against `build_baseline()` already makes growth structurally impossible, since any new or
    changed unmarked number fails this assertion regardless of a separately-maintained total."""
    assert load_baseline(RESULTS_ROOT) == build_baseline(), (
        "CLAIMS_BASELINE.json no longer matches the documents. If you MARKED a number, remove its "
        "row here. If you ADDED an unmarked number, mark it instead. Regenerate with "
        "scripts/generate_claims_baseline.py either way."
    )


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


# --- The gate, armed over every document in GATED_DOCS -----------------------------------------


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
    """The advice in the failure message must match the direction of the change: a number that
    got MORE unmarked occurrences needs a marker; a number that got FEWER (partially marked, but
    not down to zero — a full-zero drop is instead caught by the dead-entry equality test) needs
    the baseline row shrunk to match, not another marker on top of the ones already added."""
    baseline = load_baseline(RESULTS_ROOT).get(doc, {})
    current = unmarked_counts(scan_document(Path(doc)))
    changed = {
        value: count for value, count in current.items() if count != baseline.get(value, 0)
    }
    if not changed:
        return
    grew = {v: c for v, c in changed.items() if c > baseline.get(v, 0)}
    shrank = {v: c for v, c in changed.items() if c < baseline.get(v, 0)}
    messages = []
    if grew:
        messages.append(
            f"new or more frequent: {grew}. Add a marker — `<!--@ <artifact>.json # <key> -->`, "
            f"or `<!--@ citation-pending: <reason> -->` if no artifact retains it."
        )
    if shrank:
        messages.append(
            f"less frequent than the baseline: {shrank}. This means a number was marked — shrink "
            f"its row in results/CLAIMS_BASELINE.json to match (regenerate with "
            f"scripts/generate_claims_baseline.py), do not add another marker."
        )
    pytest.fail(f"{doc}: " + " ".join(messages))


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


def test_the_year_row_does_not_eat_a_four_digit_quantity_with_decimals() -> None:
    r"""`results/RESULTS.md:72` publishes `1922.1` ms of rerank latency.

    `\b` is satisfied by the `.`, so the old `\b(?:19|20)\d{2}\b` matched INSIDE `1922.1`,
    masking the leading digits and leaving `.1`. The orphan was not harmless: NUMBER_RE read the
    trailing `1` as a claim, so the document published `1922.1` while the gate recorded `1` — an
    edit to `2922.1` would have changed nothing the gate could see.
    """
    assert [c.text for c in scan_text("rerank adds 1922.1 ms", doc="x.md")] == ["1922.1"]


def test_the_year_row_still_masks_genuine_years() -> None:
    """The fix must not reopen the timestamps this row exists to exclude."""
    for text in (
        "presented at ICLR 2026",
        "in May 2026 we shipped",
        "measured in 2014",
        "Hanley & McNeil (1982)",
        "dated 2026-07-29 exactly",
    ):
        assert scan_text(text, doc="x.md") == [], text


def test_a_bare_four_digit_quantity_in_the_year_range_is_a_KNOWN_hole() -> None:
    """Pins the residual rather than leaving it implied.

    `2048` and `2026` are the same four characters; no local rule separates a chunk count from a
    year. `1024` and `3072` fall outside the range and are gated normally. If a gated document ever
    publishes a 1900-2099 quantity, it must be marked explicitly — this row will not catch it.
    """
    assert scan_text("corpus of 2048 chunks", doc="x.md") == []
    assert [c.text for c in scan_text("dim 1024 vs 3072", doc="x.md")] == ["1024", "3072"]
