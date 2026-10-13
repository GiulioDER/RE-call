"""`recall scopes` on a `sqlite:///` DSN, with PostgreSQL's output.

`scopes` lists what a `--folder` or `--facet` filter can match. On a lite DSN it reached
PostgreSQL and died in psycopg, and `LiteStore` had neither `scope_inventory` nor
`scope_undeclared_count`.

Invariants and the failure each one catches:
- S1 folder and facet listings print the same lines on PostgreSQL and on lite for the same memos,
  root files, nested folders and facet-less memos included.
- S2 a lite file that does not exist yet lists nothing and is not created.
- S3 the generation route is refused on a lite DSN by name.

Red proof: recorded in each test's docstring.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests.conftest import TEST_DSN, requires_db
from tests.test_lite_cli import _run

_MEMOS = {
    "root.md": "---\ntype: decision\n---\n# Root\n\nA decision at the root of the memory.\n",
    "python/typing.md": "---\ntype: reference\n---\n# Typing\n\nPrefer explicit protocols.\n",
    "python/async.md": "---\ntype: decision\n---\n# Async\n\nUse anyio for structured concurrency.\n",
    "python/tools/ruff.md": "# Ruff\n\nNever run ruff format here; only ruff check.\n",
    "ops/deploy.md": "---\ntype: reference\n---\n# Deploy\n\nDeploys go out on Tuesdays.\n",
}


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("RECALL_ENV", "RECALL_INDEX_MODE", "RECALL_SERVING_DSN", "RECALL_DSN", "RECALL_TRUST_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")


def _memos(root: Path) -> Path:
    for name, text in _MEMOS.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _dsn(path: Path) -> str:
    return "sqlite:///" + path.as_posix()


@requires_db
def test_s1_scopes_print_the_same_on_postgres_and_lite(tmp_path: Path, cli_table: str) -> None:
    """Red: baseline `449aa0a1` raises from psycopg on the lite DSN. After the change, each alone:
    `LiteStore.scope_undeclared_count` returning 0 drops the facet listing's "declare no facet"
    line; `scope_inventory` sorting smallest first reorders the folder listing."""
    folder = _memos(tmp_path / "mem")
    backends = {
        "postgres": ["--embedder", "hashing", "--dsn", TEST_DSN, "--table", cli_table],
        "lite": ["--embedder", "hashing", "--dsn", _dsn(tmp_path / "memory.db")],
    }
    out: dict[str, dict[str, str]] = {}
    for name, base in backends.items():
        _run([*base, "index", str(folder)])
        out[name] = {
            dimension: _run([*base, "scopes", "--dimension", dimension]) for dimension in ("folder", "facet")
        }
    assert "python/tools" in out["postgres"]["folder"] and "reference" in out["postgres"]["facet"], out
    for dimension in ("folder", "facet"):
        want, got = out["postgres"][dimension], out["lite"][dimension]
        assert got == want, f"{dimension}: postgres {want!r} != lite {got!r}"


def test_s2_a_missing_lite_file_lists_nothing(tmp_path: Path) -> None:
    """Red: `_cmd_scopes` opening the store without the existence check creates the file."""
    path = tmp_path / "absent" / "memory.db"
    out = _run(["--dsn", _dsn(path), "scopes"])
    assert "no folder values in this corpus" in out, out
    assert not path.exists(), "listing scopes created a lite file"


def test_s3_the_generation_route_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Red: the `lite_route_refusal` check removed from `_cmd_scopes`: no refusal by name."""
    monkeypatch.setenv("RECALL_INDEX_MODE", "generation")
    out = _run(["--dsn", _dsn(tmp_path / "memory.db"), "scopes"])
    assert "has no generations" in out, f"the generation route was not refused: {out}"


def _indexed(tmp_path: Path, *flags: str) -> Path:
    path = tmp_path / "memory.db"
    _run(["--embedder", "hashing", "--dsn", _dsn(path), *flags, "index", str(_memos(tmp_path / "mem"))])
    return path


def test_s4_a_read_never_writes_a_file_that_is_not_a_lite_store(tmp_path: Path) -> None:
    """BUG-001 (P1). Red: opening the store at `inspect_lite_file(path).dim or 1` (the code before
    this fix) writes a lite meta (dim 1) and lite tables into an empty or foreign SQLite file, after
    which `recall index` at the real width is refused."""
    empty = tmp_path / "empty.db"
    empty.write_bytes(b"")
    foreign = tmp_path / "foreign.db"
    conn = sqlite3.connect(foreign)
    conn.execute("CREATE TABLE notes (x TEXT)")
    conn.commit()
    conn.close()
    for path in (empty, foreign):
        before = path.read_bytes()
        out = _run(["--dsn", _dsn(path), "scopes"])
        assert "EXIT" in out and "not a lite store" in out, out
        assert path.read_bytes() == before, f"listing scopes wrote into {path.name}"


def test_s5_a_corrupt_calibration_does_not_stop_the_listing(tmp_path: Path) -> None:
    """BUG-002. Red: the width came from an inspect that stops at a calibration that does not
    verify, so the code before this fix opened the store at width 1 and raised."""
    path = _indexed(tmp_path)
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO calibrations(created_at, payload) VALUES ('2099-01-01', 'garbage')")
    conn.commit()
    conn.close()
    out = _run(["--dsn", _dsn(path), "scopes"])
    assert "python/tools" in out and "RAISED" not in out, f"a corrupt calibration stopped the listing: {out}"


def test_s6_another_tenant_lists_nothing_as_on_postgres(tmp_path: Path) -> None:
    """BUG-003. PostgreSQL lists nothing for a tenant with no rows; lite now does the same instead
    of a `LiteStoreError` traceback (the code before this fix)."""
    path = _indexed(tmp_path, "--tenant", "memory")
    out = _run(["--dsn", _dsn(path), "scopes"])
    assert out.strip() == "no folder values in this corpus", f"another tenant did not list nothing: {out}"


def test_s7_listing_reads_metadata_not_vectors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PERF-001 and PERF-002. Red: with `LiteStore._load` (every embedding into numpy) and
    `inspect_lite_file` (every chunk text hashed) made to fail, the code before this fix failed."""
    path = _indexed(tmp_path)

    def _forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("scopes loaded vectors or hashed the corpus")

    monkeypatch.setattr("recall.lite.store.LiteStore._load", _forbidden)
    monkeypatch.setattr("recall.lite.inspect.inspect_lite_file", _forbidden)
    out = _run(["--dsn", _dsn(path), "scopes"])
    assert "python/tools" in out and "RAISED" not in out, f"scopes did more than read metadata: {out}"
