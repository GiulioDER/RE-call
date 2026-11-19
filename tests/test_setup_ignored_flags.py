"""`recall setup` refuses the flags it would accept and then ignore (#1402, #1403, #1404, and
`setup --tenant`).

Each refusal test asserts that setup stops with a message naming the flag AND that neither setup
path ran. The controls show the same command without the flag (or with its default typed out)
still reaching setup, so a refusal that fired on everything would go red there.

Red proof, 2026-11-19, against master 63b7c3ea2 (`recall/cli_commands/setup_wizard.py`,
`_cmd_setup` before `_refuse_ignored_setup_flags` existed): every refusal case failed at
`assert refusal is not None`, because setup ran (the stub recorded a call) and returned. The
controls passed there too, as they should.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_LOCAL_DSN = "postgresql://recall:recall@localhost:5432/recall"


@pytest.fixture
def setup_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, object]]:
    """Replace both setup paths with recorders, so nothing is written or connected."""
    import recall.cli as cli
    import recall.lite.setup as lite_setup
    import recall.setup as setup

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_DOTENV_ERROR", None)
    for name in ("RECALL_EMBEDDER", "RECALL_INDEX_ROOT"):
        monkeypatch.delenv(name, raising=False)
    calls: list[dict[str, object]] = []

    def _full(**kw: object) -> dict[str, str]:
        calls.append({"path": "full", **kw})
        return {}

    def _lite(**kw: object) -> object:
        calls.append({"path": "lite", **kw})
        raise SystemExit("lite stub reached")

    monkeypatch.setattr(setup, "run_setup_wizard", _full)
    monkeypatch.setattr(lite_setup, "run_lite_setup", _lite)
    return calls


def _run(argv: list[str]) -> str | None:
    """Run the CLI; return the exit message, or None when it returned normally."""
    from recall.cli import main

    try:
        main(argv)
    except SystemExit as exc:
        return None if exc.code == "lite stub reached" else str(exc.code)
    return None


def _lite_argv(tmp_path: Path, *globals_: str) -> list[str]:
    return [*globals_, "--embedder", "hashing", "setup", "--lite", "--memory", str(tmp_path)]


# ------------------------------------------------------------------------------------------------
# #1402: the interview chooses the embedder, so a typed --embedder was dropped.
# ------------------------------------------------------------------------------------------------


def test_setup_refuses_a_typed_embedder(setup_calls: list[dict[str, object]]) -> None:
    """Invariant: `recall --embedder X setup` stops before the interview, naming --embedder.
    Failure caught: the interview's menu silently decided the embedder instead (#1402)."""
    refusal = _run([f"--dsn={_LOCAL_DSN}", "--embedder", "hashing", "setup"])
    assert refusal is not None, f"setup ran and ignored --embedder: {setup_calls!r}"
    assert "does not accept --embedder" in refusal, refusal
    assert setup_calls == []


def test_setup_ignores_an_embedder_from_the_environment(
    setup_calls: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: RECALL_EMBEDDER is the flag's default, not a typed flag, so setup still runs."""
    monkeypatch.setenv("RECALL_EMBEDDER", "hashing")
    assert _run([f"--dsn={_LOCAL_DSN}", "setup"]) is None
    assert [c["path"] for c in setup_calls] == ["full"]


def test_lite_setup_still_honours_a_typed_embedder(
    setup_calls: list[dict[str, object]], tmp_path: Path
) -> None:
    """Control: `--lite` does apply --embedder, so it must not be refused there."""
    assert _run(_lite_argv(tmp_path)) is None
    assert [c["path"] for c in setup_calls] == ["lite"]
    assert setup_calls[0]["embedder_name"] == "hashing"


# ------------------------------------------------------------------------------------------------
# #1403: a Lite store has one tenant and one table, so --tenant and --table were dropped.
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("flag", "value"), [("--tenant", "work"), ("--table", "experiment")])
def test_lite_setup_refuses_tenant_and_table(
    flag: str, value: str, setup_calls: list[dict[str, object]], tmp_path: Path
) -> None:
    """Invariant: `recall --tenant X setup --lite` and `recall --table X setup --lite` stop,
    naming the flag. Failure caught: Lite wrote RECALL_TENANT=default and its one table (#1403)."""
    refusal = _run(_lite_argv(tmp_path, flag, value))
    assert refusal is not None, f"setup --lite ran and ignored {flag}: {setup_calls!r}"
    assert f"does not accept {flag}" in refusal, refusal
    assert setup_calls == []


@pytest.mark.parametrize("flag", ["--serving-dsn", "--dsn"])
def test_lite_setup_refuses_a_typed_sqlite_dsn(
    flag: str, setup_calls: list[dict[str, object]], tmp_path: Path
) -> None:
    """Invariant: `recall --dsn sqlite:///X setup --lite` stops, naming the flag, before Lite runs.

    Failure caught (CCA BUG-003, run 20261119T182325Z): Lite builds and registers the store named
    by --store (default `.recall/memory.db`) and never reads a typed DSN, so
    `recall --dsn sqlite:///other.db setup --lite` set up a different store than the one named.
    A typed PostgreSQL DSN stays accepted and unused, as
    tests/test_lite_setup_boundaries.py::test_lite_dispatch_ignores_unused_postgres_credentials
    requires.

    Red proof, 2026-11-19, against `_refuse_ignored_setup_flags` at 2e546bec1 (the Lite branch
    checked only --tenant and --table): both cases failed at `assert refusal is not None`, with
    the Lite stub recorded as called.
    """
    store = tmp_path / "other.db"
    refusal = _run(_lite_argv(tmp_path, flag, f"sqlite:///{store.as_posix()}"))
    assert refusal is not None, f"setup --lite ran and ignored {flag}: {setup_calls!r}"
    assert "does not accept --serving-dsn" in refusal, refusal
    assert setup_calls == []


@pytest.mark.parametrize("flag", ["--serving-dsn", "--migration-dsn"])
def test_lite_setup_accepts_a_typed_postgres_dsn_it_does_not_use(
    flag: str, setup_calls: list[dict[str, object]], tmp_path: Path
) -> None:
    """Control: PostgreSQL credentials a Lite setup does not use are not refused.

    Red proof by the CI failure of 2026-11-19 (run 44608489215 on e7e54c66b): the refusal that
    fired on every typed DSN stopped this command with `does not accept --serving-dsn`, and the
    `--migration-dsn` case with `does not accept --migration-dsn`.
    """
    assert _run(_lite_argv(tmp_path, flag, _LOCAL_DSN)) is None
    assert [c["path"] for c in setup_calls] == ["lite"]


def test_lite_setup_accepts_a_dsn_from_the_environment(
    setup_calls: list[dict[str, object]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: the DSN flags default from the environment, so only a typed one is refused."""
    monkeypatch.setenv("RECALL_SERVING_DSN", _LOCAL_DSN)
    monkeypatch.setenv("RECALL_MIGRATION_DSN", _LOCAL_DSN)
    assert _run(_lite_argv(tmp_path)) is None
    assert [c["path"] for c in setup_calls] == ["lite"]


@pytest.mark.parametrize(("flag", "value"), [("--tenant", "default"), ("--table", "chunks")])
def test_lite_setup_accepts_the_default_tenant_and_table_typed_out(
    flag: str, value: str, setup_calls: list[dict[str, object]], tmp_path: Path
) -> None:
    """Control: typing the value Lite uses anyway is not ignored, so it is not refused."""
    assert _run(_lite_argv(tmp_path, flag, value)) is None
    assert [c["path"] for c in setup_calls] == ["lite"]


# ------------------------------------------------------------------------------------------------
# #1404: --memory, --store and --no-register are read only by the Lite path.
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flag", "typed"),
    [
        ("--memory", ["--memory", "notes"]),
        ("--store", ["--store", "memory.db"]),
        ("--no-register", ["--no-register"]),
    ],
)
def test_setup_refuses_lite_only_flags_without_lite(
    flag: str, typed: list[str], setup_calls: list[dict[str, object]]
) -> None:
    """Invariant: a Lite-only flag without --lite stops setup, naming the flag and --lite.
    Failure caught: the PostgreSQL interview ran and the flag did nothing (#1404)."""
    refusal = _run([f"--dsn={_LOCAL_DSN}", "setup", *typed])
    assert refusal is not None, f"setup ran and ignored {flag}: {setup_calls!r}"
    assert f"{flag} only with --lite" in refusal, refusal
    assert setup_calls == []


# ------------------------------------------------------------------------------------------------
# setup --tenant: only the scaffolded memory and Codex used the tenant; the seed, the Claude Code
# server and its hooks used `default`, so the registered server could not see that memory.
# ------------------------------------------------------------------------------------------------


def test_setup_refuses_a_tenant_it_would_only_half_apply(
    setup_calls: list[dict[str, object]],
) -> None:
    """Invariant: `recall --tenant X setup` stops before the interview, naming --tenant.
    Failure caught: memory went to X while the seed, the server and the hooks used `default`."""
    refusal = _run([f"--dsn={_LOCAL_DSN}", "--tenant", "work", "setup"])
    assert refusal is not None, f"setup ran with a half-applied tenant: {setup_calls!r}"
    assert "does not accept --tenant" in refusal, refusal
    assert setup_calls == []


def test_setup_keeps_honouring_table_and_the_default_tenant(
    setup_calls: list[dict[str, object]],
) -> None:
    """Control: the PostgreSQL interview applies --table everywhere it writes one, and the
    default tenant typed out is what it uses anyway, so neither is refused."""
    argv = [f"--dsn={_LOCAL_DSN}", "--table", "experiment", "--tenant", "default", "setup"]
    assert _run(argv) is None
    assert [c["path"] for c in setup_calls] == ["full"]
    assert setup_calls[0]["table"] == "experiment"
