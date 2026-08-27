"""A `§9f` in a docstring must still point at the section it was written about.

Lettered subsections are positional: inserting a new `### 9f.` re-letters every section after it,
and every `§9g`..`§9p` already written into a docstring silently starts pointing one section too
early. Nothing errors. The reference still resolves — to the wrong text — which is the same failure
shape as an artifact with no provenance: a plausible answer with no signal that it is wrong.

That is not hypothetical. Publishing the BEAM arm inserted `### 9f. Why 92.5 and our 0.444 are not
the same measurement`, and eleven citations across nine files went stale in one commit. Six of them
claimed §9f "established that a cosine threshold cannot gate abstention on BEAM" — a sentence §9f
has never contained.

So the fix cannot be "check the section exists": §9f *does* exist, that is the whole problem. Each
citation is pinned to a substring of the title it was written against, and re-lettering moves the
titles, so the pin breaks and someone has to look.

Scope: lettered labels only (`9h`, `10c`, ...). A bare `§9` names a whole numbered section, which
does not move when a subsection is inserted, and `§7` is ambiguous between the two documents.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = {
    "FINDINGS": ROOT / "results" / "FINDINGS.md",
    "RESULTS": ROOT / "results" / "RESULTS.md",
}

#: Directories whose prose cites the two result documents.
SCAN_ROOTS = ("benchmarks", "recall", "scripts", "tests", "docs")

#: Everything under `results/` EXCEPT the two documents that define the labels. The exclusion used
#: to cover the whole directory, on the rationale that "a cross-reference inside FINDINGS.md moves
#: together with the section it names" — true of FINDINGS.md and RESULTS.md, and false of the five
#: other markdown files that live beside them and cite those sections from outside.
SCAN_RESULTS_EXCEPT = {ROOT / "results" / "FINDINGS.md", ROOT / "results" / "RESULTS.md"}

#: label -> (document, substring that must appear in that section's title).
#: Add an entry when you add a citation — `test_every_cited_label_is_registered` enforces it.
EXPECTED: dict[str, tuple[str, str]] = {
    "2b": ("FINDINGS", "the old fitting rule was replaced"),
    # 5b, 7a and 9d were cited only from README.md, CHANGELOG.md and results/ARTIFACTS.md — all
    # three outside the old scan roots, so nothing could have caught them being re-lettered.
    "5b": ("FINDINGS", "the headline rate holds at full coverage"),
    "7a": ("RESULTS", "Retrieval depth curve"),
    "7b": ("RESULTS", "Abstention on the adversarial"),
    "9a": ("FINDINGS", "0.671 at k=5"),
    "9b": ("FINDINGS", "0.00 out of the box"),
    "9c": ("FINDINGS", "stronger entailment judge"),
    "9d": ("FINDINGS", "the paired head-to-head vs Mem0"),
    # Cited by SUITE-DESIGN.md and CHANGELOG.md as the source of the BEAM reproduce commands —
    # the evidence tier those documents now rest on, since the per-question dumps are gitignored.
    "9e": ("FINDINGS", "headline numbers actually are"),
    # Cited from docs/AGENT_MEMORY_FIELD_REVIEW.md, which rests on §9f for the argument that a
    # competitor's LoCoMo headline and ours are different quantities rather than two values of one.
    # This is the label the module docstring names as the cause of the original re-lettering
    # incident, so it is the one most worth keeping pinned.
    "9f": ("FINDINGS", "not the same measurement"),
    "9g": ("FINDINGS", "hosted embedder"),
    "9h": ("FINDINGS", "abstention category is a hallucination test"),
    "9i": ("FINDINGS", "the count rule does not pay"),
    "9j": ("FINDINGS", "Newest-wins dedup"),
    "9l": ("FINDINGS", "temporal_reasoning"),
    "9m": ("FINDINGS", "embedder-fragile"),
    "9n": ("FINDINGS", "regime sweep"),
    "10b": ("FINDINGS", "abstention layer failed here"),
    "10c": ("FINDINGS", "bounded domain"),
    # Cited from docs/preregistrations/2026-08-26-sdk-driver-equivalence.md, whose Result section
    # points at the published table for the driver-equivalence replication.
    "13f": ("RESULTS", "Driver-equivalence replication"),
}

_HEADER = re.compile(r"^#{2,4}\s+(\d+[a-z]?)\.\s+(.*)$", re.MULTILINE)
_LABEL = re.compile(r"§(\d+[a-z])")


def _titles(doc: str) -> dict[str, str]:
    """label -> section title, as the document currently defines them."""
    text = DOCS[doc].read_text(encoding="utf-8")
    return {m.group(1): m.group(2).strip() for m in _HEADER.finditer(text)}


def _scanned() -> list[Path]:
    """Every file whose lettered citations this suite is responsible for.

    Three sources, and the last two were previously blind spots: the repo-root markdown
    (README.md and CHANGELOG.md carry more §-citations than anything else in the tree, README's
    among them), and the non-FINDINGS/RESULTS markdown under `results/`.
    """
    paths: list[Path] = []
    for root in SCAN_ROOTS:
        base = ROOT / root
        if base.is_dir():
            paths.extend(base.rglob("*"))
    paths.extend(ROOT.glob("*.md"))
    paths.extend(p for p in (ROOT / "results").rglob("*.md")
                 if p.resolve() not in {q.resolve() for q in SCAN_RESULTS_EXCEPT})
    return paths


def _cited() -> dict[str, set[str]]:
    """label -> files citing it."""
    hits: dict[str, set[str]] = {}
    seen: set[Path] = set()
    for path in _scanned():
        if path.suffix not in {".py", ".md"} or not path.is_file():
            continue
        if path.resolve() in seen:
            continue
        seen.add(path.resolve())
        # This file names §9f and §9p to explain the failure it exists to catch. Those are
        # examples, not citations, and registering them would make the dead-entry check pass
        # for the wrong reason.
        if path.resolve() == Path(__file__).resolve():
            continue
        for match in _LABEL.finditer(path.read_text(encoding="utf-8")):
            hits.setdefault(match.group(1), set()).add(str(path.relative_to(ROOT)))
    return hits


def test_the_scan_finds_citations() -> None:
    """Guards against a refactor that moves the prose and leaves this suite vacuous."""
    assert len(_cited()) >= 10, "found too few lettered citations; the scan is probably broken"


@pytest.mark.parametrize("label", sorted(EXPECTED), ids=str)
def test_registered_label_still_carries_that_title(label: str) -> None:
    doc, needle = EXPECTED[label]
    titles = _titles(doc)
    assert label in titles, f"§{label} is cited but {doc}.md defines no section {label}"
    assert needle.lower() in titles[label].lower(), (
        f"§{label} is cited for {needle!r}, but {doc}.md §{label} is now "
        f"{titles[label]!r}. Sections were probably re-lettered: re-read every citation of "
        f"§{label} and repoint it, then update EXPECTED."
    )


def test_every_cited_label_is_registered() -> None:
    """A new citation must declare what it is citing, or it cannot be checked when letters move."""
    unregistered = {k: sorted(v) for k, v in _cited().items() if k not in EXPECTED}
    assert not unregistered, (
        f"lettered citations with no EXPECTED entry: {unregistered}. Add each one with a "
        f"substring of the title it was written against."
    )


def test_expected_has_no_dead_entries() -> None:
    """A pin nothing cites is a pin nobody maintains."""
    dead = sorted(set(EXPECTED) - set(_cited()))
    assert not dead, f"EXPECTED pins labels nothing cites any more: {dead}"
