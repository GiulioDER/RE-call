"""Where a PostgreSQL stating rule says a source is: `recall.stated_facts._location`. No database.

On PostgreSQL a `RECALL_STATED_FACT_SOURCES` prefix names a place and is matched against each
memo's source (`StatingRule.by_source`), so the comparison must read a `file://` URI exactly as
the reader does (`recall.manifest.local_path_for`) and must not let a path's spelling move it
under a prefix. Review of PR 935, finding 4 (2026-10-18). Each test's red proof is in its docstring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from recall.stated_facts import StatingRule, _source_prefixes


def test_l1_a_file_uri_keeps_its_host() -> None:
    """Node tests/test_stated_facts_locations.py::test_l1_a_file_uri_keeps_its_host.

    `_location` dropped a file URI's authority, so `file://nas1/share/memory/door.md` compared as
    the local path `/share/memory/door.md` and the prefix `/share/memory` admitted a file on
    another host. Red against b7f5d9c03, the unchanged `_location` (recall/stated_facts.py):
    fails on "a file on another host was admitted as local".
    """
    rule = StatingRule(prefixes=("/share/memory",), tenant="t", by_source=True)
    assert not rule.admits("door.md", "file://nas1/share/memory/door.md"), "a file on another host was admitted as local"


def test_l2_a_source_cannot_climb_out_of_an_admitted_prefix() -> None:
    """Node tests/test_stated_facts_locations.py::test_l2_a_source_cannot_climb_out_of_an_admitted_prefix.

    The prefix test compared unnormalised paths, so `/srv/root/memory/../docs/x.md`, a file in
    `docs/`, started with the admitted `/srv/root/memory/`. Red against b7f5d9c03, the unchanged
    `_location` (recall/stated_facts.py): fails on "a source outside the prefix was admitted by
    its spelling". The last assertion is a control that passes before and after.
    """
    rule = StatingRule(prefixes=("/srv/root/memory",), tenant="t", by_source=True)
    for source in ("/srv/root/memory/../docs/x.md", "file:///srv/root/memory/../docs/x.md"):
        assert not rule.admits("x.md", source), f"a source outside the prefix was admitted by its spelling: {source}"
    assert rule.admits("x.md", "/srv/root/memory/./x.md")


@pytest.mark.skipif(sys.platform != "win32", reason="a drive letter in a file URI is a Windows path")
def test_l3_on_windows_a_path_prefix_admits_the_file_uri_of_its_memo(tmp_path: Path) -> None:
    """Node tests/test_stated_facts_locations.py::test_l3_on_windows_a_path_prefix_admits_the_file_uri_of_its_memo.

    `urlsplit` gives `file:///C:/x` the path `/C:/x`, while a path prefix resolves to `C:/x`, so on
    Windows no path prefix ever admitted the `as_uri()` sources desktop ingest and manifests store,
    and nothing stated, silently. Red on win32 against b7f5d9c03, the unchanged `_location`
    (recall/stated_facts.py): fails on "a path prefix refused the as_uri() source of its own memo".
    Skipped elsewhere, where the two spellings already agree.
    """
    memo = tmp_path / "memory" / "door.md"
    memo.parent.mkdir()
    memo.write_text("x", encoding="utf-8")
    rule = StatingRule(prefixes=_source_prefixes([str(tmp_path / "memory")], {}), tenant="t", by_source=True)
    assert rule.admits("door.md", memo.resolve().as_uri()), "a path prefix refused the as_uri() source of its own memo"
    assert rule.admits("door.md", str(memo.resolve()))
