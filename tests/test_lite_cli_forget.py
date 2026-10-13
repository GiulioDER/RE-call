"""`recall forget` on a `sqlite:///` DSN, and the same output as on PostgreSQL.

`forget` is the CLI's erasure path. Before this, a lite DSN reached `PgVectorStore` and died in
psycopg ("missing '=' after 'sqlite:///...' in connection info string"), so a lite user could not
delete a memory from the command line.

Invariants and the failure each one catches:
- K1 a dry run deletes nothing; `--yes` deletes the named source and reports a typo separately.
- K2 the generation route is refused on a lite DSN, by name, before the file is created.
- K3 a non-default `--table` on a lite DSN is refused: a lite file has one table, and opening it
  anyway would run the command against real memory (`recall --table scratch forget X --yes`
  erasing a real memo).
- K4 the same forget sequence prints the same lines on PostgreSQL and on lite.
- K5 forget on a missing lite path creates nothing (no directory, no file) and says so; it used
  to build a store there and report the source as a typo.
- K6 forget on a SQLite file that is not a lite store leaves it untouched (no tables, no meta,
  no WAL) and says so.
- K7 forget on a lite file that holds another tenant is refused by name, as a clean exit rather
  than a traceback, and deletes nothing.

Red proof: recorded in each test's docstring.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from recall.lite import LiteStore
from tests.conftest import TEST_DSN, requires_db
from tests.lite_cli_support import _offline, run_cli  # noqa: F401  # _offline: autouse fixture
from tests.test_lite_cli import _MEMOS, _dsn, _memos

_run = run_cli
#: The width `--embedder hashing` indexes at, so a test can open the store the CLI built.
HASHING_DIM = 64


def _store_path(tmp_path: Path) -> Path:
    """The file `_dsn(tmp_path)` names."""
    return tmp_path / "store" / "memory.db"


def _sources(tmp_path: Path) -> set[str]:
    with LiteStore(_store_path(tmp_path), dim=HASHING_DIM) as store:
        return set(store.source_content_hashes())


def test_k1_forget_on_a_lite_dsn(tmp_path: Path) -> None:
    """Red: `_cmd_forget` opening `PgVectorStore` again (the baseline) raises from psycopg, so the
    dry run assertion fails "forget did not run on the lite store"."""
    folder = _memos(tmp_path / "mem", _MEMOS)
    base = ["--embedder", "hashing", "--dsn", _dsn(tmp_path)]
    _run([*base, "index", str(folder)])
    rate = str((folder / "rate.md").resolve())
    assert rate in _sources(tmp_path)

    dry = _run([*base, "forget", rate])
    assert "DRY RUN: would forget 1 source(s)" in dry, f"forget did not run on the lite store: {dry}"
    assert rate in _sources(tmp_path), "the dry run deleted a memo"

    done = _run([*base, "forget", rate, "typo-source", "--yes"])
    assert "forgot 1 chunk(s) from 1 source(s)" in done, done
    assert "not found" in done and "typo-source" in done, done
    assert _sources(tmp_path) == {str((folder / "deploy.md").resolve())}, "forget left the memo or took another"


def test_k2_the_generation_route_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Red: the `lite_route_refusal` check removed from `_cmd_forget`; the command no longer names
    the missing generations (it fails inside the generation store instead)."""
    monkeypatch.setenv("RECALL_INDEX_MODE", "generation")
    out = _run(["--embedder", "hashing", "--dsn", _dsn(tmp_path), "forget", "x.md", "--yes"])
    assert "has no generations" in out, f"the generation route was not refused for forget: {out}"
    assert not _store_path(tmp_path).exists(), "the refused command still created the store"


def test_k3_a_scratch_table_is_refused_on_a_lite_dsn(tmp_path: Path) -> None:
    """Red: `open_legacy_store` ignoring `table` for a lite DSN again (the baseline): the forget runs
    against the real file and erases the memo."""
    folder = _memos(tmp_path / "mem", _MEMOS)
    base = ["--embedder", "hashing", "--dsn", _dsn(tmp_path)]
    _run([*base, "index", str(folder)])
    rate = str((folder / "rate.md").resolve())
    out = _run(["--embedder", "hashing", "--dsn", _dsn(tmp_path), "--table", "scratch", "forget", rate, "--yes"])
    assert "one table" in out and "scratch" in out, f"a scratch table was not refused on lite: {out}"
    assert "EXIT:" in out and "RAISED" not in out, f"the refusal reached the user as a traceback: {out}"
    assert rate in _sources(tmp_path), "a --table scratch forget erased a real memo"


def test_k5_a_missing_lite_path_is_not_created(tmp_path: Path) -> None:
    """Red, 2026-10-13, against the pre-fix `_cmd_forget` (no read-only check before
    `open_legacy_store`): the dry run builds a store there, failing "a dry run created the store"."""
    target = tmp_path / "typo" / "dir" / "memory.db"
    out = _run(["--embedder", "hashing", "--dsn", "sqlite:///" + target.as_posix(), "forget", "x.md"])
    assert not target.parent.exists(), f"a dry run created the store: {out}"
    assert "does not exist" in out and "EXIT:" in out, f"the missing file was not named: {out}"


def test_k6_a_foreign_sqlite_file_is_left_untouched(tmp_path: Path) -> None:
    """Red, 2026-10-13, against the pre-fix `_cmd_forget`: the dry run writes the lite schema and
    a meta row into the file, failing "a dry run wrote into a foreign file"."""
    target = tmp_path / "people.db"
    with sqlite3.connect(target) as conn:
        conn.execute("CREATE TABLE people (name TEXT)")
    conn.close()
    out = _run(["--embedder", "hashing", "--dsn", "sqlite:///" + target.as_posix(), "forget", "x.md"])
    conn = sqlite3.connect(target)
    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()
    assert tables == {"people"} and journal == "delete", f"a dry run wrote into a foreign file: {tables} {journal} {out}"
    assert "not a lite store" in out and "EXIT:" in out, f"the foreign file was not named: {out}"


def test_k7_another_tenant_is_refused_cleanly(tmp_path: Path) -> None:
    """Red, 2026-10-13, against the pre-fix code: `LiteStore` raises `LiteStoreError` for the
    tenant, which `recall.cli.main` let through as a traceback, failing "a traceback"."""
    folder = _memos(tmp_path / "mem", _MEMOS)
    base = ["--embedder", "hashing", "--dsn", _dsn(tmp_path)]
    _run([*base, "index", str(folder)])
    rate = str((folder / "rate.md").resolve())
    out = _run([*base, "--tenant", "other", "forget", rate, "--yes"])
    assert "EXIT:" in out and "RAISED" not in out, f"another tenant was refused with a traceback: {out}"
    assert "tenant" in out and "other" in out, out
    assert rate in _sources(tmp_path), "a forget for another tenant erased a memo"


@requires_db
def test_k4_forget_prints_the_same_on_postgres_and_lite(tmp_path: Path, cli_table: str) -> None:
    """Red: the lite `source_content_hashes` returning {} makes every lite source "not found", so
    the dry run line differs ("step dry run")."""
    folder = _memos(tmp_path / "mem", _MEMOS)
    rate = str((folder / "rate.md").resolve())
    backends = {
        "postgres": ["--embedder", "hashing", "--dsn", TEST_DSN, "--table", cli_table],
        "lite": ["--embedder", "hashing", "--dsn", _dsn(tmp_path)],
    }
    transcripts: dict[str, list[tuple[str, str]]] = {}
    for name, base in backends.items():
        _run([*base, "index", str(folder)])
        transcripts[name] = [
            ("dry run", _run([*base, "forget", rate, "typo-source"])),
            ("delete", _run([*base, "forget", rate, "typo-source", "--yes"])),
            ("again", _run([*base, "forget", rate, "--yes"])),
        ]
    for (step, want), (_, got) in zip(transcripts["postgres"], transcripts["lite"], strict=True):
        assert got == want, f"step {step}: postgres {want!r} != lite {got!r}"
    assert "forgot 1 chunk(s) from 1 source(s)" in dict(transcripts["lite"])["delete"]


def test_k8_a_refused_forget_builds_no_embedder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Erasure embeds nothing, so a forget refused on a lite DSN loads no model and sends no
    billed probe.

    Red, 2026-10-13, against the pre-fix `_cmd_forget` (the embedder was built before the lite
    refusals): fails on "a refused forget built an embedder".
    """
    from recall.cli_commands import index_search

    built: list[str] = []
    monkeypatch.setattr(index_search, "_make_embedder", lambda name: built.append(name))
    out = _run(["--embedder", "hashing", "--dsn", "sqlite:///" + (tmp_path / "missing.db").as_posix(), "forget", "x.md"])
    assert "does not exist" in out, out
    assert built == [], f"a refused forget built an embedder: {built}"
