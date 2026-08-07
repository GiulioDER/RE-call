"""Every advertised test count must be a floor the suite actually clears.

It said **890** while the suite collected **1,339** — stale by half, in two places (the badge and
the Engineering section). Nothing reported it, because a number in prose has no CI.

The fix is the shape, not the value. A point estimate is wrong the moment the next test lands, so
the README publishes `1,300+` and this asserts the suite clears it. Overstating fails; adding tests
does not. That makes the claim one that stays true instead of one that needs remembering.

Deliberately one-directional. Asserting the figure tracks the count closely would fail on every
growth spurt and get deleted, and understating is not a false claim — only claiming more tests than
exist is.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

#: The badge lives in `README.md` and the prose count in `docs/ENGINEERING.md`, which is where the
#: Engineering section moved on 2026-08-08 when the README was shortened. Both files are scanned for
#: both shapes rather than each being pinned to its current home: the guarantee this suite exists to
#: give is "every advertised count is a floor the suite clears", and pinning a shape to a file would
#: let the next move silently drop one from the check.
ADVERTISING_DOCS = (
    Path(__file__).resolve().parent.parent / "README.md",
    Path(__file__).resolve().parent.parent / "docs" / "ENGINEERING.md",
)

#: Below this, the run is a subset (`-k`, a single file) and its count says nothing about the
#: suite. Well under a full collection, well over any plausible targeted run.
_FULL_RUN_FLOOR = 500

_BADGE = re.compile(r"badge/tests-(\d+)%2B")
_PROSE = re.compile(r"\*\*([\d,]+)\+ tests")


def _advertised() -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    for doc in ADVERTISING_DOCS:
        text = doc.read_text(encoding="utf-8")
        found += [("badge", int(m.group(1))) for m in _BADGE.finditer(text)]
        found += [("prose", int(m.group(1).replace(",", ""))) for m in _PROSE.finditer(text)]
    return found


def test_the_readme_still_advertises_a_count() -> None:
    """If both mentions vanish this suite would pass by finding nothing to check."""
    kinds = {kind for kind, _ in _advertised()}
    assert kinds == {"badge", "prose"}, f"expected a badge and a prose count, found {kinds}"


def test_the_two_advertised_counts_agree() -> None:
    values = {value for _, value in _advertised()}
    assert len(values) == 1, f"badge and prose disagree: {sorted(values)}"


def test_the_suite_clears_the_advertised_floor(request: pytest.FixtureRequest) -> None:
    collected = len(request.session.items)
    if collected < _FULL_RUN_FLOOR:
        pytest.skip(f"partial run ({collected} tests) — this check needs a full collection")
    for kind, floor in _advertised():
        assert floor <= collected, (
            f"README {kind} advertises {floor}+ tests but the suite collects {collected}. "
            f"Lower the published floor — it is a claim, and right now it is false."
        )
