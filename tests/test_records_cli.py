"""`recall records` on a lite store and on PostgreSQL's generation route (Task 1.2). Legacy
PostgreSQL parity is K10 in `tests/test_records_lookup.py`.

Invariants and the failure each one catches:
- C1 `--all --json` is a COMPLETE dump read in ONE lookup, so one snapshot, because CCA's
  round-trip check reads every record a dump lacks as missing, and separate paged lookups on a
  legacy store (no generation id) could duplicate or skip records with `truncated` false.
- C2 an anchor query prints each matching record with its match.
- C3 a lite file that does not exist lists nothing and is not created.
- C4 the command is classified for lite (served), and the generation route is refused by name.
- C5 under RECALL_INDEX_MODE=generation on PostgreSQL it reads the active generation (DB suite).

Red proof, recorded on the deployment host inside the test bounds (C1 and C5 2026-10-22, the
rest 2026-10-21):
- C1 `tests/test_records_cli.py::test_all_is_one_complete_dump_from_one_snapshot`: `_lookup`
  mutated back to paging four records per lookup: "the dump took 3 lookups".
- C5 `...::test_the_generation_route_reads_the_active_generation`: the `route.uses_generation`
  branch disabled, so the legacy table is read: the generation id assertion fails.
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
from tests.audit_memory_support import EXPECTED, MEMOS, REPO, copy_memos, lite_dsn, memo_name
from tests.conftest import TEST_DSN, requires_db
from tests.lite_cli_support import _offline, run_cli  # noqa: F401  # _offline: autouse fixture
from tests.test_generations import (  # noqa: F401  # `manager` is a fixture, used by name
    _Embedder,
    _manifest,
    _pipeline,
    _reader,
    _ready,
    manager,
)


def _indexed(tmp_path: Path) -> list[str]:
    base = ["--embedder", "hashing", "--dsn", lite_dsn(tmp_path / "memory.db")]
    out = run_cli([*base, "index", str(copy_memos(tmp_path / "memos"))])
    assert "indexed" in out, out
    return base


def test_all_is_one_complete_dump_from_one_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """C1: every record, from ONE lookup. Separate paged lookups each read their own snapshot, and
    a legacy store has no generation id to tell them apart, so a write between pages duplicated or
    skipped records while `truncated` still read false."""
    base = _indexed(tmp_path)
    calls: list[dict[str, object]] = []
    real = records_cmd.lookup_records

    def counted(store: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return real(store, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(records_cmd, "lookup_records", counted)
    envelope = json.loads(run_cli([*base, "records", "--all", "--json"]))
    assert len(calls) == 1, f"the dump took {len(calls)} lookups, so {len(calls)} snapshots"
    assert envelope["truncated"] is False
    assert {memo_name(r["source"]): r["recall_record"] for r in envelope["records"]} == EXPECTED


@requires_db
def test_the_generation_route_reads_the_active_generation(
    manager, monkeypatch: pytest.MonkeyPatch  # noqa: F811  # the imported fixture
) -> None:
    """C5: `recall records` on PostgreSQL under RECALL_INDEX_MODE=generation."""
    name = "RE-call__0123456789ab__BUG-001.md"
    data = (MEMOS / name).read_bytes()
    manifest = _manifest(manager.tenant_id, data)
    generation = _ready(manager, manifest, _pipeline("model-a"), _reader(manifest, data), _Embedder(1))
    manager.promote(generation, unsafe_development=True)
    monkeypatch.setenv("RECALL_INDEX_MODE", "generation")
    out = run_cli(["--dsn", TEST_DSN, "--tenant", manager.tenant_id, "records", "--all", "--json"])
    envelope = json.loads(out)
    assert envelope["generation_id"] == generation, out
    assert [r["recall_record"] for r in envelope["records"]] == [EXPECTED[name]]


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
