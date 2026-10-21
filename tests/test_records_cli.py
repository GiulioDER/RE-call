"""`recall records` on a lite store (Task 1.2). PostgreSQL parity is K10 in
`tests/test_records_lookup.py`.

Invariants and the failure each one catches:
- C1 `--all --json` is a COMPLETE dump, however many pages it takes, because CCA's round-trip
  check reads every record past a cut as missing.
- C2 an anchor query prints each matching record with its match.
- C3 a lite file that does not exist lists nothing and is not created.
- C4 the command is classified for lite (served), and the generation route is refused by name.

Red proof, recorded 2026-10-21 on the deployment host inside the test bounds:
- C1 `tests/test_records_cli.py::test_all_is_a_complete_dump_across_pages`: `_lookup`'s paging
  loop mutated to stop after the first page: 4 of 9 records are printed and the equality
  assertion fails.
- C3 `...::test_a_missing_lite_file_lists_nothing_and_is_not_created`: the existence check in
  `_cmd_records` removed: the command exits on the absent file instead of listing nothing, and
  the "no records match" assertion fails.
- C2 `...::test_an_anchor_query_prints_each_match`: `--kind` not passed to the lookup: the false
  positive and the lesson are printed too.
- C4 `...::test_records_is_served_on_lite_and_refuses_the_generation_route`: `records` removed
  from `LITE_SERVED`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import recall.cli_commands.records_cmd as records_cmd
from recall.backends import LITE_REFUSED, LITE_SERVED
from tests.audit_memory_support import EXPECTED, REPO, copy_memos, lite_dsn, memo_name
from tests.lite_cli_support import _offline, run_cli  # noqa: F401  # _offline: autouse fixture


def _indexed(tmp_path: Path) -> list[str]:
    base = ["--embedder", "hashing", "--dsn", lite_dsn(tmp_path / "memory.db")]
    out = run_cli([*base, "index", str(copy_memos(tmp_path / "memos"))])
    assert "indexed" in out, out
    return base


def test_all_is_a_complete_dump_across_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """C1: with pages of four, nine records take three pages and all nine are printed."""
    base = _indexed(tmp_path)
    monkeypatch.setattr(records_cmd, "MAX_RECORDS", 4)
    envelope = json.loads(run_cli([*base, "records", "--all", "--json"]))
    assert envelope["truncated"] is False
    assert {memo_name(r["source"]): r["recall_record"] for r in envelope["records"]} == EXPECTED


def test_an_anchor_query_prints_each_match(tmp_path: Path) -> None:
    """C2."""
    base = _indexed(tmp_path)
    anchors = tmp_path / "anchors.json"
    anchors.write_text(json.dumps([EXPECTED["RE-call__0123456789ab__BUG-001.md"]["anchors"][0]]), encoding="utf-8")
    out = run_cli([*base, "records", "--repo", REPO, "--anchors", str(anchors), "--kind", "bug"])
    lines = [line for line in out.splitlines() if line.startswith("exact")]
    assert {line.split()[2] for line in lines} == {
        "RE-call:0123456789ab:BUG-001",
        "RE-call:222222222222:PERF-001~2",
        "RE-call:abcdef012345:BUG-002",
    }, out
    assert "caused 1" in next(line for line in lines if "BUG-001" in line), out
    refused = run_cli([*base, "records", "--anchors", str(anchors)])
    assert "EXIT" in refused and "needs `repo`" in refused, refused


def test_a_missing_lite_file_lists_nothing_and_is_not_created(tmp_path: Path) -> None:
    """C3."""
    path = tmp_path / "absent" / "memory.db"
    out = run_cli(["--dsn", lite_dsn(path), "records", "--all"])
    assert "no records match" in out, out
    assert not path.exists(), "a read-only lookup created the lite file"


def test_records_is_served_on_lite_and_refuses_the_generation_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C4."""
    assert "records" in LITE_SERVED and "records" not in LITE_REFUSED
    monkeypatch.setenv("RECALL_INDEX_MODE", "generation")
    out = run_cli(["--dsn", lite_dsn(tmp_path / "memory.db"), "records", "--all"])
    assert "EXIT" in out and "no generations" in out, out
