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

Red proof: recorded in each test's docstring.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from recall.lite import LiteStore
from tests.conftest import TEST_DSN, requires_db
from tests.test_lite_cli import _MEMOS, _dsn, _memos, _run


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("RECALL_ENV", "RECALL_INDEX_MODE", "RECALL_SERVING_DSN", "RECALL_DSN", "RECALL_TRUST_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")


def _sources(tmp_path: Path) -> set[str]:
    with LiteStore(tmp_path / "store" / "memory.db", dim=64) as store:
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
    assert not (tmp_path / "store" / "memory.db").exists(), "the refused command still created the store"


def test_k3_a_scratch_table_is_refused_on_a_lite_dsn(tmp_path: Path) -> None:
    """Red: `open_legacy_store` ignoring `table` for a lite DSN again (the baseline): the forget runs
    against the real file and erases the memo."""
    folder = _memos(tmp_path / "mem", _MEMOS)
    base = ["--embedder", "hashing", "--dsn", _dsn(tmp_path)]
    _run([*base, "index", str(folder)])
    rate = str((folder / "rate.md").resolve())
    out = _run(["--embedder", "hashing", "--dsn", _dsn(tmp_path), "--table", "scratch", "forget", rate, "--yes"])
    assert "one table" in out and "scratch" in out, f"a scratch table was not refused on lite: {out}"
    assert rate in _sources(tmp_path), "a --table scratch forget erased a real memo"


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
