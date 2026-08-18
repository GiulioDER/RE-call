"""The documentation citation checker: that it catches drift, and that it stays quiet otherwise.

Both halves matter and the second is the one that decides whether the check survives. A citation
checker that cries wolf gets switched off, and then the coverage is gone rather than merely noisy.
That is not a hypothetical here: the FIRST implementation of this checker matched each citation
against a backticked anchor near it, and produced **33 findings on a tree whose citations had just
been repaired by hand**, nearly all of them correct citations flagged wrongly, because one
documentation line routinely carries several symbols and several citations with no reliable pairing
between them. It was replaced with the git line mapping the tests below exercise.

The failure this exists to prevent is real and recurring: one commit (`79a0d6ed`) shifted
`recall/index.py` by 26 lines and broke 23 citations across 5 documents at once.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "check_doc_citations", REPO / "scripts" / "check_doc_citations.py"
)
assert _spec and _spec.loader
citations = importlib.util.module_from_spec(_spec)
sys.modules["check_doc_citations"] = citations
_spec.loader.exec_module(citations)


# --- the arithmetic, which is where a wrong answer would be most confidently wrong ---------------

def test_a_line_before_every_hunk_does_not_move() -> None:
    """Edits later in a file cannot move a citation earlier in it."""
    assert citations.resolve(5, [(10, 0, 10, 26)]) == 5


def test_an_insertion_above_pushes_a_line_down() -> None:
    """The measured case: #381 inserted 26 lines and every citation below moved by 26."""
    assert citations.resolve(671, [(430, 0, 430, 26)]) == 697


def test_a_deletion_above_pulls_a_line_up() -> None:
    assert citations.resolve(100, [(10, 5, 10, 0)]) == 95


def test_several_hunks_accumulate() -> None:
    assert citations.resolve(500, [(10, 0, 10, 3), (100, 4, 103, 0), (200, 0, 199, 10)]) == 509


def test_a_line_inside_a_changed_hunk_is_unresolvable() -> None:
    """Stronger than a move: what was cited may not exist any more, so no number is offered."""
    assert citations.resolve(12, [(10, 5, 10, 5)]) is None


def test_a_line_at_the_last_position_of_a_hunk_is_still_inside_it() -> None:
    """An off-by-one here would report a confident wrong destination for an edited line."""
    assert citations.resolve(14, [(10, 5, 10, 5)]) is None
    assert citations.resolve(15, [(10, 5, 10, 5)]) == 15


# --- what counts as a citation -------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "see `recall/index.py:420`",
        "see `recall/index.py:420-478`",
        "see `recall_mcp/service.py:1659`",
        "see `recall/migrations/sql/0001_v08_baseline.sql:12`",
    ],
)
def test_a_repo_relative_citation_is_recognised(text: str) -> None:
    assert citations.CITATION.search(text) is not None


@pytest.mark.parametrize("text", ["see `cli.py:487`", "see `promotion.py:184`", "see `fix.py:24`"])
def test_a_bare_filename_is_not_treated_as_a_citation(text: str) -> None:
    """Documents use `cli.py:487` as shorthand for a module under discussion.

    Resolving those against the repository root reported "does not exist" for seven such
    references in the first implementation. A citation must carry a directory to be checkable, so
    the shorthand is left alone rather than failed.
    """
    assert citations.CITATION.search(text) is None


def test_a_prose_colon_is_not_a_citation() -> None:
    assert citations.CITATION.search("`recall/index.py` line 420") is None


# --- the policy file, which is what keeps frozen records out of the check ------------------------

def test_the_policy_declares_the_frozen_zone_and_its_marker_still_exists() -> None:
    """A marker that stops matching would silently un-freeze a registered prediction.

    `frozen_line` raises in that case rather than checking the zone, and this asserts the declared
    markers are all still present, so the raise is a real backstop and not the normal path.
    """
    exempt, zones = citations.load_policy()
    assert zones, "the frozen-zone declaration disappeared from docs/citation-policy.toml"
    for zone in zones:
        doc = REPO / zone["path"]
        assert doc.exists(), f"policy names a document that does not exist: {zone['path']}"
        boundary = citations.frozen_line(doc, zone["path"], zones)
        assert boundary is not None and boundary > 1
    assert exempt, "the exemption list disappeared"
    for rule in exempt:
        assert rule.get("reason"), f"exemption without a stated reason: {rule['path']}"


def test_a_vanished_marker_refuses_rather_than_passing() -> None:
    """The guard's own failure path, exercised: a stale policy must stop the run, not the check."""
    zone = {"path": "docs/citation-policy.toml", "marker": "## a marker that is not there"}
    with pytest.raises(SystemExit, match="frozen-zone marker"):
        citations.frozen_line(REPO / "docs" / "citation-policy.toml", "docs/citation-policy.toml",
                              [zone])


# --- the repository itself ------------------------------------------------------------------------

def test_every_citation_in_the_docs_still_resolves() -> None:
    """The gate. Fails when a commit moves code out from under a documented line number.

    Documents with uncommitted edits are skipped by the checker, so this passes locally while docs
    are being written and is only fully binding on a clean tree, which is what CI has.
    """
    findings, _skipped = citations.check()
    assert not findings, "\n" + "\n".join(f.render() for f in findings)
