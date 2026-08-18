"""The citation checker, and the mutations that prove it can fail.

A gate is not shown to work by passing. `scripts/verify_citations.py` reports 0 STALE against this
repository, and the only reason that number is worth anything is the set of tests below that
deliberately break a citation and watch the exit code go to 1. Its predecessor reported "41
citations, 0 broken" while three of them pointed at a `try:`, a docstring's closing quotes and the
wrong error branch, so a green from this tool means nothing without the red beside it.

Every mutation here hits the decision the checker actually makes -- the line number in the
citation, or the identifier in the prose -- and not an ornament next to it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    """Import `scripts/verify_citations.py`, which is not on any import path.

    It lives in `scripts/` rather than in `recall/` because it is repository maintenance, not
    shipped behaviour: putting it in the package would publish it to PyPI users and add it to the
    coverage denominator of code that has to work on somebody else's machine.
    """
    path = ROOT / "scripts" / "verify_citations.py"
    spec = importlib.util.spec_from_file_location("verify_citations", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_citations"] = module
    spec.loader.exec_module(module)
    return module


vc = _load_module()


# --------------------------------------------------------------------------------------------
# A miniature repository, so the parsing and verdict tests do not depend on this repo's contents.
# --------------------------------------------------------------------------------------------

SOURCE = '''\
"""Module docstring."""

from __future__ import annotations

# Padding, so the module-level constant below sits outside the fallback window of the import
# above it. Without this the two share a window, the test that distinguishes them cannot fail,
# and it becomes a guard that cannot fail.
# Padding.
WIDGET_LIMIT = 32


def compute_widget(value: int) -> int:
    """Docstring that mentions nothing in particular.

    Padding, so the enclosing scope is wider than the fallback window and the test below really
    exercises the scope rule rather than the window.
    Padding.
    """
    scaled = value * WIDGET_LIMIT
    return scaled


def unrelated_helper() -> None:
    pass
'''

#: Line map for SOURCE, asserted below. A hand-maintained line map in a file about stale line
#: numbers is a joke that writes itself, so it is checked rather than trusted.
LINE_IMPORT = 3
LINE_WIDGET_LIMIT = 9
LINE_DEF_COMPUTE = 12
LINE_SCALED = 19
LINE_RETURN = 20


def test_the_fixtures_own_line_map_is_right() -> None:
    lines = SOURCE.splitlines()
    assert lines[LINE_IMPORT - 1].startswith("from __future__")
    assert lines[LINE_WIDGET_LIMIT - 1].startswith("WIDGET_LIMIT")
    assert lines[LINE_DEF_COMPUTE - 1].startswith("def compute_widget")
    assert lines[LINE_SCALED - 1].strip().startswith("scaled =")
    assert lines[LINE_RETURN - 1].strip() == "return scaled"
    # Both gaps must exceed the fallback window, or the two tests that turn on the scope rule
    # would pass under a plain window and prove nothing about it.
    assert LINE_SCALED - LINE_DEF_COMPUTE > vc.ANCHOR_WINDOW
    assert LINE_WIDGET_LIMIT - LINE_IMPORT > vc.ANCHOR_WINDOW


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A `tmp_path` repository with one source file and a docs tree.

    `tmp_path` and not `/tmp`: Python resolves `/tmp` to a directory that does not exist on
    Windows, which turns a harness into a false green.
    """
    (tmp_path / "recall").mkdir()
    (tmp_path / "recall" / "widget.py").write_text(SOURCE, encoding="utf-8", newline="\n")
    (tmp_path / "docs").mkdir()
    return tmp_path


def write_doc(repo: Path, body: str, name: str = "DESIGN.md") -> Path:
    path = repo / "docs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="\n")
    return path


def verdicts(repo: Path) -> list[tuple[str, str]]:
    return [(str(result.citation), result.status) for result in vc.collect(repo)]


# --------------------------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------------------------


def test_it_reads_backticked_bare_range_and_continuation_forms() -> None:
    lines = [
        "A backticked `recall/a.py:10` and a bare recall/b.py:20 in prose.",
        "A range `recall/c.py:30-40`.",
        "Two on one line: `recall/d.py:50` and `:60`.",
    ]
    found = vc.extract_citations("docs/x.md", lines)
    assert [(c.path, c.start, c.end) for c in found] == [
        ("recall/a.py", 10, 10),
        ("recall/b.py", 20, 20),
        ("recall/c.py", 30, 40),
        ("recall/d.py", 50, 50),
        # The continuation inherits the path from the full citation earlier on ITS line, which is
        # the shorthand these documents actually use.
        ("recall/d.py", 60, 60),
    ]


def test_a_continuation_with_no_preceding_path_is_dropped_rather_than_guessed() -> None:
    assert vc.extract_citations("docs/x.md", ["Nothing before it: `:60`."]) == []


def test_citations_inside_fenced_blocks_are_not_claims() -> None:
    body = "Prose cites `recall/widget.py:5`.\n\n```\nrecall/widget.py:999\n```\n"
    lines = vc.strip_fenced_blocks(body.splitlines())
    assert [c.start for c in vc.extract_citations("docs/x.md", lines)] == [5]


def test_stripping_a_fence_preserves_line_numbering() -> None:
    """A reported document line has to match what an editor shows, or the fix costs archaeology."""
    lines = vc.strip_fenced_blocks(["a", "```", "x", "```", "cite recall/widget.py:5"])
    assert len(lines) == 5
    assert vc.extract_citations("docs/x.md", lines)[0].doc_line == 5


# --------------------------------------------------------------------------------------------
# The verdicts
# --------------------------------------------------------------------------------------------


def test_an_anchor_at_the_cited_line_passes(repo: Path) -> None:
    write_doc(repo, f"The cap is `WIDGET_LIMIT` (`recall/widget.py:{LINE_WIDGET_LIMIT}`).\n")
    assert verdicts(repo) == [(f"recall/widget.py:{LINE_WIDGET_LIMIT}", vc.OK)]


def test_an_anchor_at_the_enclosing_def_passes_for_a_line_in_its_body(repo: Path) -> None:
    """A citation into a function body is a claim about that function.

    `compute_widget` is seven lines above the cited line, outside the fallback window. This is the
    case a fixed window got wrong on the real corpus (`recall/index.py:447`, whose anchor is 27
    lines up at the `def`), and it is why the unit is the enclosing scope.
    """
    write_doc(repo, f"`compute_widget` scales it (`recall/widget.py:{LINE_SCALED}`).\n")
    assert verdicts(repo) == [(f"recall/widget.py:{LINE_SCALED}", vc.OK)]


def test_a_citation_to_a_different_top_level_statement_is_stale(repo: Path) -> None:
    """The mirror image, and the case a WIDER window gets wrong.

    The cited line is an `import` and the constant it is cited for is six lines below. Any window
    generous enough to admit the previous test would pass this one too, which is exactly why the
    scope and not the distance is what decides. The real instance was `recall/frontmatter.py:12`.
    """
    write_doc(repo, f"The cap is `WIDGET_LIMIT` (`recall/widget.py:{LINE_IMPORT}`).\n")
    (citation, status), = verdicts(repo)
    assert status == vc.STALE
    assert citation == f"recall/widget.py:{LINE_IMPORT}"


def test_a_stale_report_names_the_line_the_anchor_is_on_now(repo: Path) -> None:
    """The report has to make the fix mechanical, or nobody makes it."""
    write_doc(repo, f"The cap is `WIDGET_LIMIT` (`recall/widget.py:{LINE_IMPORT}`).\n")
    result, = vc.collect(repo)
    assert f"`WIDGET_LIMIT` on line {LINE_WIDGET_LIMIT}" in result.detail
    assert f"Nearest is line {LINE_WIDGET_LIMIT}" in result.detail


def test_prose_with_no_identifier_is_unverifiable_and_not_ok(repo: Path) -> None:
    """The verdict that stops this tool from lying.

    There is no anchor to check, so the honest answer is "cannot decide". Reporting OK here is
    what the existence-only predecessor did for all 41 of its citations.
    """
    write_doc(repo, f"Something happens over there (`recall/widget.py:{LINE_SCALED}`).\n")
    assert verdicts(repo) == [(f"recall/widget.py:{LINE_SCALED}", vc.UNVERIFIABLE)]


def test_an_identifier_absent_from_the_cited_file_is_unverifiable_not_stale(repo: Path) -> None:
    """A sentence routinely cites two files. A token missing from THIS one is evidence about the
    other, so it must not be read as a broken anchor here."""
    write_doc(repo, f"See `TotallyElsewhere` (`recall/widget.py:{LINE_SCALED}`).\n")
    assert verdicts(repo) == [(f"recall/widget.py:{LINE_SCALED}", vc.UNVERIFIABLE)]


def test_a_token_too_common_to_discriminate_cannot_certify_a_citation(repo: Path) -> None:
    """Without the occurrence cap, a token that is everywhere would pass every citation in the
    file, which is how a check manufactures a green."""
    noisy = "\n".join(f"widget_{n} = widget  # widget" for n in range(40))
    (repo / "recall" / "noisy.py").write_text(noisy, encoding="utf-8", newline="\n")
    assert len(vc.anchor_lines(noisy.splitlines(), "widget")) > vc.MAX_ANCHOR_LINES
    write_doc(repo, "It is a `widget` thing (`recall/noisy.py:20`).\n")
    (_citation, status), = verdicts(repo)
    assert status == vc.UNVERIFIABLE


def test_a_common_token_at_the_cited_line_withholds_an_accusation(repo: Path) -> None:
    """Being unable to certify is not the same as having evidence against.

    `recall/control_plane.py:802` is literally `def cutover(...)` in a sentence about `cutover()`,
    but the name occurs too often in that file to certify. Reporting STALE there rested the
    verdict on an unrelated token from the same sentence and accused a correct citation. The
    positive evidence has to beat the negative.
    """
    noisy = "\n".join(f"widget_{n} = widget  # widget" for n in range(40))
    (repo / "recall" / "noisy.py").write_text(noisy, encoding="utf-8", newline="\n")
    write_doc(repo, "About `widget`, and also `WIDGET_LIMIT` (`recall/noisy.py:20`).\n")
    result, = vc.collect(repo)
    assert result.status == vc.UNVERIFIABLE
    assert "occurs too often" in result.detail


def test_a_missing_file_and_an_out_of_range_line_are_distinct_verdicts(repo: Path) -> None:
    write_doc(repo, "Gone `WIDGET_LIMIT` (`recall/absent.py:5`) and `recall/widget.py:9999`.\n")
    assert verdicts(repo) == [
        ("recall/absent.py:5", vc.MISSING_FILE),
        ("recall/widget.py:9999", vc.OUT_OF_RANGE),
    ]


# --------------------------------------------------------------------------------------------
# Anchor matching
# --------------------------------------------------------------------------------------------


def test_an_anchor_matches_through_a_leading_underscore(repo: Path) -> None:
    """A parameter and the attribute it is stored on are the same thing under two names.

    Requiring an exact match reported `recall_mcp/stores.py:154` stale over
    `self._generation_mode`, which is a correct citation.
    """
    (repo / "recall" / "priv.py").write_text(
        "class C:\n"
        "    def __init__(self, widget_mode: bool) -> None:\n"
        "        self._widget_mode = widget_mode\n",
        encoding="utf-8",
        newline="\n",
    )
    write_doc(repo, "The `widget_mode` flag (`recall/priv.py:3`).\n")
    assert verdicts(repo) == [("recall/priv.py:3", vc.OK)]


def test_an_anchor_does_not_match_a_longer_identifier_that_contains_it() -> None:
    """`promote` is not evidence for a line that says `promote_generation`."""
    assert vc.anchor_lines(["promote_generation()"], "promote") == []
    assert vc.anchor_lines(["promote()"], "promote") == [1]


def test_a_hyphenated_span_is_matched_whole(repo: Path) -> None:
    """`bge-small-symmetric-v1` tokenises to `bge`, `small`, `symmetric`, `v1`, none of which
    identifies the profile the sentence is about. The whole span is the anchor."""
    (repo / "recall" / "profiles.py").write_text(
        'A = 1\nB = 2\nPROFILE = "bge-small-symmetric-v1"\n', encoding="utf-8", newline="\n"
    )
    write_doc(repo, "The `bge-small-symmetric-v1` profile (`recall/profiles.py:3`).\n")
    assert verdicts(repo) == [("recall/profiles.py:3", vc.OK)]


def test_an_explicit_anchor_comment_rescues_a_line_with_no_quotable_identifier(
    repo: Path,
) -> None:
    write_doc(
        repo,
        f"Structural line, nothing to quote (`recall/widget.py:{LINE_RETURN}`).\n"
        "<!-- cite-anchor: scaled -->\n",
    )
    assert verdicts(repo) == [(f"recall/widget.py:{LINE_RETURN}", vc.OK)]


def test_an_explicit_anchor_still_has_to_be_there(repo: Path) -> None:
    """The escape hatch is an anchor, not an exemption. A wrong one fails like any other."""
    write_doc(
        repo,
        f"Claim (`recall/widget.py:{LINE_IMPORT}`).\n<!-- cite-anchor: scaled -->\n",
    )
    (_citation, status), = verdicts(repo)
    assert status == vc.STALE


# --------------------------------------------------------------------------------------------
# Document-level dispositions
# --------------------------------------------------------------------------------------------


def test_an_external_document_is_not_verified_against_our_tree(repo: Path) -> None:
    """`docs/their-harness-parity.md` cites mem0's `benchmarks/beam/run.py:690`, and a file of
    that name exists HERE too. Verifying it would yield a confident verdict about the wrong
    repository, which is the failure this whole tool exists to prevent."""
    write_doc(
        repo,
        f"{vc.EXTERNAL_MARKER}\nTheir `WIDGET_LIMIT` (`recall/widget.py:{LINE_IMPORT}`).\n",
    )
    assert verdicts(repo) == [(f"recall/widget.py:{LINE_IMPORT}", vc.EXTERNAL)]


def test_merely_describing_the_external_marker_does_not_take_the_opt_out(repo: Path) -> None:
    """A document that documents the opt-out must not thereby take it.

    Not hypothetical: the pre-registration for this tool described the marker in a sentence, and
    the first version matched it anywhere in the file, so all 8 of that document's citations were
    silently exempted -- including the deliberately stale one it was recording. A silent
    exemption is the same defect class as a stale citation, arrived at from the other side.
    """
    write_doc(
        repo,
        f"Writing `{vc.EXTERNAL_MARKER}` in prose opts a document out.\n"
        f"The cap is `WIDGET_LIMIT` (`recall/widget.py:{LINE_IMPORT}`).\n",
    )
    (_citation, status), = verdicts(repo)
    assert status == vc.STALE


def test_the_external_marker_inside_a_code_fence_does_not_take_the_opt_out(repo: Path) -> None:
    """The same hole from the other common direction: a tutorial showing the marker."""
    write_doc(
        repo,
        "How to opt out:\n\n```markdown\n"
        f"{vc.EXTERNAL_MARKER}\n```\n\n"
        f"The cap is `WIDGET_LIMIT` (`recall/widget.py:{LINE_IMPORT}`).\n",
    )
    (_citation, status), = verdicts(repo)
    assert status == vc.STALE


def test_the_external_marker_on_its_own_line_does_take_the_opt_out(repo: Path) -> None:
    """The allow path. A gate that blocks the correct usage gets deleted, taking its coverage."""
    write_doc(
        repo,
        f"{vc.EXTERNAL_MARKER}\n\nThe cap is `WIDGET_LIMIT` (`recall/widget.py:{LINE_IMPORT}`).\n",
    )
    (_citation, status), = verdicts(repo)
    assert status == vc.EXTERNAL


def test_a_frozen_document_reports_its_stale_citations_without_failing(repo: Path) -> None:
    write_doc(
        repo,
        f"The cap is `WIDGET_LIMIT` (`recall/widget.py:{LINE_IMPORT}`).\n",
        "archive/OLD.md",
    )
    results = vc.collect(repo)
    assert [r.status for r in results] == [vc.FROZEN]
    # Still printed, with the same detail. Frozen means unmaintained, not invisible.
    assert f"`WIDGET_LIMIT` on line {LINE_WIDGET_LIMIT}" in results[0].detail
    _report, passed = vc.render(results, ceiling=0)
    assert passed


def test_a_frozen_documents_unverifiable_citations_do_not_load_the_ratchet(repo: Path) -> None:
    """The ceiling governs live documents only.

    A frozen document may not be edited, so counting its unverifiable citations in a number that
    "can only fall" is incoherent: nobody is permitted to lower that part of it. Left in, a new
    pre-registration with unanchored citations would push the count over and the only legal remedy
    would be raising the ceiling, which turns a ratchet into a rubber stamp.
    """
    write_doc(
        repo,
        f"Something over there (`recall/widget.py:{LINE_SCALED}`).\n",
        "preregistrations/2026-08-18-thing.md",
    )
    results = vc.collect(repo)
    assert [r.status for r in results] == [vc.FROZEN]
    # A ceiling of zero, and it still passes, because nothing unverifiable is live.
    assert vc.render(results, ceiling=0)[1] is True


def test_a_pre_registration_is_frozen_like_the_archive(repo: Path) -> None:
    """Standing instruction, 2026-08-18: never edit a number in a committed pre-registration,
    including a citation's line number. So the gate must never be able to demand one."""
    assert "docs/preregistrations/" in vc.FROZEN_PREFIXES
    write_doc(
        repo,
        f"The cap is `WIDGET_LIMIT` (`recall/widget.py:{LINE_IMPORT}`).\n",
        "preregistrations/2026-08-18-thing.md",
    )
    results = vc.collect(repo)
    assert [r.status for r in results] == [vc.FROZEN]
    assert vc.render(results, ceiling=0)[1] is True
    # Visible, though: frozen means unmaintained, not unreported.
    assert f"`WIDGET_LIMIT` on line {LINE_WIDGET_LIMIT}" in results[0].detail


def test_freezing_does_not_hide_a_stale_citation_from_the_report(repo: Path) -> None:
    write_doc(
        repo,
        f"The cap is `WIDGET_LIMIT` (`recall/widget.py:{LINE_IMPORT}`).\n",
        "archive/OLD.md",
    )
    report, _passed = vc.render(vc.collect(repo), ceiling=0)
    assert "FROZEN (1" in report
    assert "archive/OLD.md" in report


# --------------------------------------------------------------------------------------------
# The ratchet and the exit code
# --------------------------------------------------------------------------------------------


def test_the_unverifiable_ceiling_is_a_ratchet(repo: Path) -> None:
    write_doc(repo, f"Something over there (`recall/widget.py:{LINE_SCALED}`).\n")
    results = vc.collect(repo)
    assert [r.status for r in results] == [vc.UNVERIFIABLE]
    assert vc.render(results, ceiling=1)[1] is True
    assert vc.render(results, ceiling=0)[1] is False


def test_the_unverifiable_count_is_always_shown_even_when_the_roll_call_is_not(
    repo: Path,
) -> None:
    """Deferring the list must not become hiding the number.

    Within the ceiling the 53-entry roll-call is summarised rather than enumerated, so the one
    actionable line in a CI log is not buried under fifty that are not. The count stays in the
    summary either way -- that is the difference between a quieter report and a quieter gate.
    """
    write_doc(repo, f"Something over there (`recall/widget.py:{LINE_SCALED}`).\n")
    results = vc.collect(repo)

    quiet, passed = vc.render(results, ceiling=1)
    assert passed
    assert "UNVERIFIABLE=1" in quiet
    assert "UNVERIFIABLE (1" not in quiet

    loud, _ = vc.render(results, ceiling=1, list_unverifiable=True)
    assert "UNVERIFIABLE (1" in loud
    assert "docs/DESIGN.md" in loud

    # Over the ceiling the list appears without being asked for, because now it is actionable.
    over, failed = vc.render(results, ceiling=0)
    assert not failed
    assert "UNVERIFIABLE (1" in over


def test_an_absent_baseline_file_means_zero_and_not_unlimited(repo: Path) -> None:
    """A gate that passes because its own configuration went missing is not a gate."""
    assert vc.read_ceiling(repo) == 0


def test_the_baseline_file_is_read_past_comments(repo: Path) -> None:
    (repo / "docs" / vc.BASELINE_PATH.split("/")[-1]).write_text(
        "# why\n\n7\n", encoding="utf-8", newline="\n"
    )
    assert vc.read_ceiling(repo) == 7


def test_main_exits_1_on_a_stale_citation_and_0_when_clean(repo: Path, capsys) -> None:
    """The exit code is what CI reads, so the exit code is what gets asserted."""
    doc = write_doc(repo, f"The cap is `WIDGET_LIMIT` (`recall/widget.py:{LINE_WIDGET_LIMIT}`).\n")
    assert vc.main(["--root", str(repo), "--unverifiable-ceiling", "0"]) == 0

    doc.write_text(
        f"The cap is `WIDGET_LIMIT` (`recall/widget.py:{LINE_IMPORT}`).\n",
        encoding="utf-8",
        newline="\n",
    )
    assert vc.main(["--root", str(repo), "--unverifiable-ceiling", "0"]) == 1
    assert "STALE (1)" in capsys.readouterr().out


# --------------------------------------------------------------------------------------------
# Against this repository, which is the corpus the gate actually runs on
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live() -> list:
    return vc.collect(ROOT)


def test_this_repository_has_no_stale_citations(live: list) -> None:
    broken = [r for r in live if r.failed]
    assert not broken, "\n".join(
        f"{r.citation.doc}:{r.citation.doc_line}: {r.citation} -- {r.detail}" for r in broken
    )


def test_the_committed_ceiling_still_holds(live: list) -> None:
    unverifiable = [r for r in live if r.status == vc.UNVERIFIABLE]
    ceiling = vc.read_ceiling(ROOT)
    assert len(unverifiable) <= ceiling, (
        f"{len(unverifiable)} unverifiable citations against a ceiling of {ceiling}. "
        f"Anchor the new one; do not raise {vc.BASELINE_PATH}."
    )


def test_the_scan_actually_found_citations(live: list) -> None:
    """An empty corpus passes every assertion above it.

    If a glob or a regex breaks, `collect` returns [] and this whole file goes green while
    checking nothing at all. The floors are deliberately far below the real counts (150 citations,
    86 of them OK at the time of writing) so ordinary editing never trips them.
    """
    assert len(live) > 100
    assert sum(1 for r in live if r.status == vc.OK) > 50


def test_planting_a_stale_citation_in_a_real_document_is_detected(live: list) -> None:
    """The mutation this whole file exists for, on real documents and real source.

    Every citation this repository currently verdicts OK is displaced out of its enclosing scope,
    and the detection rate is asserted against a floor. This is the assertion an existence-only
    check scores **0%** on, and it is the reason the 0 STALE above is worth reading at all.

    **A rate and not "all of them", because the honest number is not 100%.** Measured 2026-08-18:
    61 of 86 detected, 70.9%. The 25 survivors are real and have a single cause -- displacing a
    citation past the end of a function often lands it near that function's CALL SITES, where its
    name legitimately appears again, so the anchor really is in scope at the new line. Asserting
    100% would mean either weakening the mutation until it passed, or special-casing the
    survivors, and both amount to writing the test around the answer.

    The floor is 55 against a measured 70.9, set the way this repository sets `--cov-fail-under`:
    low enough that ordinary editing never trips it, high enough that a checker which stopped
    reading line numbers would fail immediately.

    A displacement that stays INSIDE one function is deliberately not asserted at all. Measured
    the same day, a 40-line shift is caught in only 40% of cases, which falsified a pre-registered
    prediction of >=90%. See docs/preregistrations/2026-08-18-citation-anchor-verification.md.
    """
    #: Floor on the detection rate, in percent. See the docstring: measured 70.9.
    floor = 55
    checked = 0
    survived = []
    for result in live:
        if result.status != vc.OK:
            continue
        citation = result.citation
        source = (ROOT / citation.path).read_text(encoding="utf-8", errors="replace")
        total = len(source.splitlines())
        low, high = vc.scope_span(source, citation.start, total)
        # Land outside the enclosing scope, in whichever direction the file has room for.
        moved = high + vc.ANCHOR_WINDOW + 5
        if moved > total:
            moved = low - vc.ANCHOR_WINDOW - 5
        if not 1 <= moved <= total:
            continue
        checked += 1
        shifted = vc.Citation(
            doc=citation.doc,
            doc_line=citation.doc_line,
            path=citation.path,
            start=moved,
            end=moved,
        )
        if vc.verify(shifted, _context_for(citation), ROOT).status == vc.OK:
            survived.append(f"{citation} moved to :{moved} still reads OK")

    assert checked > 40, f"only {checked} citations were actually mutated"
    detected = checked - len(survived)
    rate = 100 * detected / checked
    assert rate >= floor, (
        f"only {detected}/{checked} ({rate:.1f}%) planted stale citations were detected, "
        f"against a floor of {floor}%. The check has stopped verifying line numbers.\n"
        + "\n".join(survived)
    )


def test_replacing_the_prose_with_a_nonexistent_identifier_never_reads_ok(live: list) -> None:
    """The other half of the mutation: break the ANCHOR instead of the line number.

    Every OK citation, re-verified against prose naming an identifier that occurs nowhere, must
    report UNVERIFIABLE. If any of them still read OK, some verdict is being reached without
    consulting the prose at all.
    """
    statuses = {
        vc.verify(r.citation, "`zzzznotarealidentifier`", ROOT).status
        for r in live
        if r.status == vc.OK
    }
    assert statuses == {vc.UNVERIFIABLE}


def _context_for(citation) -> str:
    lines = vc.strip_fenced_blocks(
        (ROOT / citation.doc).read_text(encoding="utf-8", errors="replace").splitlines()
    )
    return vc.context_lines(lines, citation.doc_line - 1)
