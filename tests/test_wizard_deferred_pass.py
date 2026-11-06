"""The deferred pass over the second CCA hunt of the first-run code (PR 1211's deferred pool).

Each test names its finding (the pool id from commit 191dbda25 and the pass's DEF id), the
invariant, the failure it catches, and how it was proved red.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.lite_cli_support import _offline, run_cli  # noqa: F401  # _offline: autouse fixture

_REPO = Path(__file__).resolve().parents[1]
_LOCAL_DSN = "postgresql://u:p@127.0.0.1:1/recall"


# ----------------------------------------------------------------------------------------------
# ENV-004 (DEF-1): an empty RECALL_SERVING_DSN
# ----------------------------------------------------------------------------------------------


def test_an_empty_serving_dsn_falls_through_to_recall_dsn(tmp_path: Path) -> None:
    """ENV-004 (DEF-1). Invariant: with RECALL_SERVING_DSN set to "" (which the wizard writes into
    every server block and container) the CLI's default DSN is RECALL_DSN, as in the MCP server.
    Failure caught: the nested `os.environ.get` default treated "" as set, so a `recall` command
    inside a generated container connected with libpq's ambient defaults.
    Red proof: run against the pre-fix `recall/cli.py` (`DEFAULT_DSN`), the child printed "" and
    the equality assertion failed. A fresh interpreter, because the default is read at import."""
    env = {**os.environ, "RECALL_SERVING_DSN": "", "RECALL_DSN": "postgresql://app:pw@db:5432/recall"}
    env["PYTHONPATH"] = str(_REPO) + os.pathsep + env.get("PYTHONPATH", "")
    child = subprocess.run(
        [sys.executable, "-c", "import recall.cli as c; print(repr(c.DEFAULT_DSN))"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120, check=False,
    )
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == repr("postgresql://app:pw@db:5432/recall"), child.stdout


# ----------------------------------------------------------------------------------------------
# PERF-007 (DEF-2): the desktop configuration page's hardware probe
# ----------------------------------------------------------------------------------------------


def test_the_splade_check_probes_only_what_it_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    """PERF-007 (DEF-2). Invariant: the configuration page's SPLADE probe asks only the hardware
    probe and the CUDA VRAM probe, the two answers `_hardware_probe_done` reads. Failure caught:
    it ran `probe_system`, which also runs docker, wsl and systeminfo serially (up to about 88 s)
    and kept the checkbox disabled meanwhile.
    Red proof: a mutation of `recall.desktop.ui._splade_hardware_probe` returning
    `probe_system()` (the pre-fix body of `MainWindow._probe_hardware`) failed the `calls`
    assertion. The pre-fix module has no such function, and `MainWindow` needs PySide6, which the
    test host lacks, so the proof is by mutation."""
    import recall.wizard.probe as probe
    from recall.desktop import ui

    calls: list[str] = []

    def _record(name: str, value: object) -> Any:
        def probe_fn(*args: object, **kwargs: object) -> object:
            calls.append(name)
            return value
        return probe_fn

    hardware = SimpleNamespace(cuda_available=True)
    monkeypatch.setattr(probe, "probe_system", _record("system", SimpleNamespace(hardware=hardware, cuda_vram_bytes=1)))
    monkeypatch.setattr(probe, "probe_docker", _record("docker", (True, False)))
    monkeypatch.setattr(probe, "probe_wsl2", _record("wsl2", None))
    monkeypatch.setattr(probe, "probe_virtualization", _record("virtualization", None))
    monkeypatch.setattr(probe, "probe_hardware", _record("hardware", hardware))
    monkeypatch.setattr(probe, "probe_cuda_vram", _record("vram", 8 * 1024**3))

    result = ui._splade_hardware_probe()

    assert sorted(calls) == ["hardware", "vram"], calls
    assert result.hardware.cuda_available is True
    assert result.cuda_vram_bytes == 8 * 1024**3


# ----------------------------------------------------------------------------------------------
# PERF-010 (DEF-3): forget and withdraw look up only the named sources on PostgreSQL
# ----------------------------------------------------------------------------------------------


class _Conn:
    """Records every statement; answers each with `memo.md` present, so the request resolves."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, sql: str, params: object = None) -> Any:
        self.statements.append(" ".join(sql.split()))
        if "to_regclass" in sql:
            return SimpleNamespace(fetchone=lambda: (True,), fetchall=lambda: [(True,)])
        rows = [] if "manifest" in sql else [("memo.md", "digest")]
        return SimpleNamespace(fetchall=lambda: rows, fetchone=lambda: rows[0] if rows else None)


def _unscoped(statements: list[str]) -> list[str]:
    return [sql for sql in statements if "DISTINCT" in sql and "ANY(" not in sql]


def _forget_args() -> argparse.Namespace:
    return argparse.Namespace(
        dsn=_LOCAL_DSN, tenant="t", table="chunks", sources=["memo.md"], yes=False, vacuum=False,
        embedder="hashing", _runtime_route=None,
    )


def _quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("RECALL_SOURCE_POLICY_FILE", "RECALL_INDEX_MODE", "RECALL_EMBEDDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RECALL_ENV", "development")


def test_forget_on_postgresql_looks_up_only_the_named_sources(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """PERF-010 (DEF-3), legacy route. Invariant: `recall forget <source>` on a PostgreSQL store
    asks which of the NAMED sources exist, in SQL. Failure caught: it listed every distinct source
    of the tenant (with a jsonb extraction per row) to check one.
    Red proof: against the pre-fix `recall/store.py` (no `PgVectorStore.sources_present`) the CLI
    fell back to `source_content_hashes`, whose DISTINCT has no `ANY(`, and the `_unscoped`
    assertion failed."""
    from recall.cli_commands import index_search
    from recall.store import PgVectorStore

    _quiet(monkeypatch)
    conn = _Conn()

    class _Store(PgVectorStore):
        def __init__(self) -> None:
            self._table = "chunks"
            self._tenant = "t"

        def __enter__(self) -> _Store:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def check_schema(self) -> None:
            return None

        def _with_retry(self, fn: Any, *args: object, **kwargs: object) -> Any:
            return fn(conn)

    monkeypatch.setattr(index_search, "open_legacy_store", lambda *a, **k: _Store())
    monkeypatch.setattr(index_search, "_make_embedder", lambda name: SimpleNamespace(dim=8))
    index_search._cmd_forget(_forget_args())

    assert _unscoped(conn.statements) == [], conn.statements
    assert "would forget 1 source(s): memo.md" in capsys.readouterr().out


def test_forget_on_the_generation_route_looks_up_only_the_named_sources(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """PERF-010 (DEF-3), generation route. Invariant: the existence check of a forget on the
    generation route asks about the named sources only, across generations and in the adopted v0.8
    table. Failure caught: `sources_in_any_generation` and `sources_in_legacy_table` listed every
    source of the tenant.
    Red proof: against the pre-fix `recall/generation_store.py` and CLI call site, both DISTINCT
    statements lacked `ANY(` and the `_unscoped` assertion failed."""
    import recall.generation_store as generation_store
    from recall.cli_commands import index_search

    _quiet(monkeypatch)
    monkeypatch.setenv("RECALL_INDEX_MODE", "generation")
    conn = _Conn()
    real = generation_store.GenerationStore

    class _Store(real):  # type: ignore[misc, valid-type]
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._table = "chunks"
            self._tenant = "t"
            self._migration_target = "chunks"

        def __enter__(self) -> _Store:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def check_schema(self) -> None:
            return None

        def _with_retry(self, fn: Any, *args: object, **kwargs: object) -> Any:
            return fn(conn)

    monkeypatch.setattr(generation_store, "GenerationStore", _Store)
    monkeypatch.setattr(index_search, "_make_embedder", lambda name: SimpleNamespace(dim=8))
    index_search._cmd_forget(_forget_args())

    assert _unscoped(conn.statements) == [], conn.statements
    assert "would forget 1 source(s): memo.md" in capsys.readouterr().out


def test_a_generation_stores_source_lookup_reads_the_active_generation() -> None:
    """PERF-010 (DEF-3), the `GenerationStore.sources_present` override. Invariant: it asks
    `recall_chunks_v1` for the named `source_uri`s within the store's generation, and asks nothing
    for an empty request. Failure caught: the inherited `PgVectorStore.sources_present` filters a
    `source` column this table does not have and ignores the generation.
    Red proof: with the override deleted from `recall/generation_store.py`, the inherited method ran
    and the assertion on `source_uri = ANY(` failed."""
    import recall.generation_store as generation_store

    conn = _Conn()

    class _Store(generation_store.GenerationStore):
        def __init__(self) -> None:
            self._tenant = "t"

        def _generation_id(self) -> str:
            return "g1"

        def _with_retry(self, fn: Any, *args: object, **kwargs: object) -> Any:
            return fn(conn)

    store = _Store()
    assert store.sources_present([]) == set()
    assert conn.statements == []
    assert store.sources_present(["memo.md"]) == {"memo.md"}
    assert len(conn.statements) == 1, conn.statements
    statement = conn.statements[0]
    assert "source_uri = ANY(" in statement and "generation_id = %s" in statement, statement


# ----------------------------------------------------------------------------------------------
# CODE-017 (DEF-6): `recall wizard` and the global flags it ignores
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flag", "typed"),
    [
        ("--embedder", ["--embedder", "hashing"]),
        ("--serving-dsn", ["--serving-dsn", _LOCAL_DSN]),
        ("--serving-dsn", ["--dsn", _LOCAL_DSN]),
        ("--migration-dsn", ["--migration-dsn", _LOCAL_DSN]),
    ],
)
def test_the_wizard_refuses_global_flags_it_would_ignore(
    flag: str, typed: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CODE-017 (DEF-6). Invariant: `recall wizard` refuses an explicitly typed global
    `--embedder`, `--serving-dsn` (or `--dsn`) or `--migration-dsn`, naming the flag. Failure
    caught: they were silently ignored, so `recall --embedder voyage:voyage-4 wizard ...` built
    with the config's embedder and reported success.
    Red proof: against the pre-fix `recall/cli_commands/setup_wizard.py` (`_cmd_wizard`) every case
    went on to load the missing config and the 'does not accept' assertion failed."""
    monkeypatch.delenv("RECALL_EMBEDDER", raising=False)
    out = run_cli([*typed, "wizard", "--headless", "--config", str(tmp_path / "missing.json")])
    assert f"does not accept {flag}" in out, out


def test_the_wizard_ignores_an_embedder_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CODE-017 (DEF-6), negative control: RECALL_EMBEDDER and RECALL_MIGRATION_DSN in the
    environment are not typed flags, so the wizard does not refuse them and reaches its config."""
    monkeypatch.setenv("RECALL_EMBEDDER", "hashing")
    out = run_cli(["wizard", "--headless", "--config", str(tmp_path / "missing.json")])
    assert "does not accept" not in out, out
    assert "missing.json" in out, out


# ----------------------------------------------------------------------------------------------
# Owner decisions of 2026-11-06 on the pass's parked findings
# ----------------------------------------------------------------------------------------------


def test_project_keys_resolve_only_keys_named_like_the_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PERF-009 (DEF-14), owner decision 2026-11-06: pre-filter lexically. Invariant: under the
    ~/.claude.json lock, `_project_keys` resolves only keys whose last folder name matches the
    project's, and still finds every spelling of the project that does. Failure caught: it resolved
    every key, so one key on an offline network share could hold the lock past other writers'
    30 s deadline.
    Red proof: against the pre-fix `recall/wizard/wiring.py` (`_project_keys`) the two unrelated
    keys were resolved and the `resolved` assertion failed."""
    import pathlib

    from recall.wizard.wiring import _project_keys

    project = tmp_path / "proj"
    project.mkdir()
    spellings = {str(project), os.path.join(str(tmp_path), ".", "proj")}
    unrelated = [str(tmp_path / "elsewhere" / "other"), str(tmp_path / "share" / "offline")]
    projects: dict[str, object] = {key: {} for key in [*spellings, *unrelated]}

    resolved: list[str] = []
    real = pathlib.Path.resolve

    def recording(self: pathlib.Path, strict: bool = False) -> pathlib.Path:
        resolved.append(str(self))
        return real(self, strict=strict)

    monkeypatch.setattr(pathlib.Path, "resolve", recording)
    keys = _project_keys(projects, project)

    assert set(keys) == spellings, keys
    assert not set(unrelated) & set(resolved), resolved


def test_the_index_glob_help_no_longer_promises_code_chunking() -> None:
    """CODE-019 (DEF-8), owner decision 2026-11-06: remove the dead branch and correct the help.
    Invariant: `recall index --help` does not offer `'**/*.py'` as the way to index code, and names
    the command that chunks Python by symbol. Failure caught: the help suggested a glob whose only
    effect was to pick `chunk_code`, which `recall index` never reaches, since every non-markdown
    file is chunked from its extracted text blocks.
    Red proof: against the pre-fix `recall/cli_commands/index_search.py` (`--glob` help) the first
    assertion failed."""
    out = " ".join(run_cli(["index", "--help"]).split())
    assert "'**/*.py' for code" not in out, out
    assert "recall generation build --chunker code" in out, out
