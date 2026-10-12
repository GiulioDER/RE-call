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
