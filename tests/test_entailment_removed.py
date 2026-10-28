"""The removed entailment judge: a configuration that still turns it on is refused, loudly.

Invariant: `RECALL_ENTAILMENT` set to a true (or unreadable) value refuses at every CLI command that
used to apply the judge (search, demo, code, quickstart), at `recall setup` before it rewrites a
`.env` holding it, and at MCP server startup, BEFORE any model, database or provider work; a false
value or an unset variable is accepted without a word. Failure mode caught: the judge's stricter filter disappearing
silently, so a deployment that relied on it is served wider results with nothing to say so, or a
leftover `RECALL_ENTAILMENT="0"` from the old setup wizard suddenly breaking a working install.

Red proof, 2026-10-01, each against the named production line with this file unchanged, each
failing at the named assertion, then restored and green:
- M1 `recall._env.refuse_removed_entailment` returning before its check (``return`` as the first
  statement): `test_true_and_unreadable_values_refuse_and_false_ones_pass` fails (``DID NOT
  RAISE``), and so do the CLI and MCP refusal tests.
- M2 the `_refuse_removed_entailment()` call removed from `index_search._cmd_search`:
  `test_cli_search_refuses_before_any_model_work` fails (the embedder sentinel is reached instead
  of the refusal).
- M3 the `refuse_removed_entailment(runtime_env)` call removed from the server lifespan:
  `test_mcp_startup_refuses_before_any_provider_work` fails (the provider sentinel is reached).
- M4 ``"0"`` removed from `recall._env._FALSE_VALUES`: `test_a_leftover_false_value_lets_the_cli_proceed`
  and `test_mcp_startup_accepts_a_leftover_false_value` fail (the refusal fires instead of the
  sentinel), and so does the value table test.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from recall._env import REMOVED_ENTAILMENT_MESSAGE, refuse_removed_entailment
from recall.cli import main


class _Reached(Exception):
    """Raised by a sentinel: execution got past the point the refusal must stop it at."""


def test_true_and_unreadable_values_refuse_and_false_ones_pass() -> None:
    for value in ("1", "true", "TRUE", " yes ", "on", "maybe"):
        with pytest.raises(ValueError, match="entailment judge was removed"):
            refuse_removed_entailment({"RECALL_ENTAILMENT": value})
    for env in ({}, {"RECALL_ENTAILMENT": "0"}, {"RECALL_ENTAILMENT": "false"},
                {"RECALL_ENTAILMENT": " Off "}, {"RECALL_ENTAILMENT": ""},
                {"RECALL_ENTAILMENT_MODEL": "cross-encoder/qnli-distilroberta-base"}):
        refuse_removed_entailment(env)


def _sentinel(*_args: object, **_kwargs: object) -> None:
    raise _Reached


def test_cli_search_refuses_before_any_model_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECALL_ENTAILMENT", "1")
    monkeypatch.setattr("recall.cli_commands.index_search._make_embedder", _sentinel)
    with pytest.raises(SystemExit) as raised:
        main(["search", "caching"])
    assert str(raised.value) == REMOVED_ENTAILMENT_MESSAGE


def test_a_leftover_false_value_lets_the_cli_proceed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECALL_ENTAILMENT", "0")
    monkeypatch.setattr("recall.cli_commands.index_search._make_embedder", _sentinel)
    with pytest.raises(_Reached):
        main(["search", "caching"])


def _drive_startup(env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    import recall_mcp.server as server
    from recall_mcp.settings import Settings

    monkeypatch.setattr(server, "provider_from_env", _sentinel)
    settings = Settings.from_env(env)

    async def _run() -> None:
        async with server._make_lifespan(None, settings=settings)(None):  # type: ignore[arg-type]
            pass

    asyncio.run(_run())


def test_mcp_startup_refuses_before_any_provider_work(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {"RECALL_ENTAILMENT": "on", "RECALL_EMBEDDER": "hashing"}
    with pytest.raises(ValueError, match="entailment judge was removed"):
        _drive_startup(env, monkeypatch)


def test_mcp_startup_accepts_a_leftover_false_value(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(_Reached):
        _drive_startup({"RECALL_ENTAILMENT": "0", "RECALL_EMBEDDER": "hashing"}, monkeypatch)


def test_quickstart_refuses_before_docker_or_the_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """`recall quickstart` applied the judge whenever RECALL_ENTAILMENT was on, so it refuses too.

    Red proof, 2026-10-28 (CCA of #849, STAKES-001): with the `_refuse_removed_entailment()` call
    removed from `setup_wizard._quickstart`, this fails because the Docker sentinel is reached
    (`_Reached`) instead of the refusal. Restored, green.
    """
    monkeypatch.setenv("RECALL_ENTAILMENT", "1")
    monkeypatch.setattr("recall.quickstart.docker_unavailable_reason", _sentinel)
    with pytest.raises(SystemExit) as raised:
        main(["quickstart"])
    assert str(raised.value) == REMOVED_ENTAILMENT_MESSAGE


@pytest.mark.parametrize("command", ["demo", "code"])
def test_demo_and_code_refuse_before_the_store_opens(
    command: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Red proof, 2026-10-28 (CCA of #849): with the `_refuse_removed_entailment()` call removed
    from `index_search._cmd_demo` (or `_cmd_code`), the matching case fails because the store
    sentinel is reached (`_Reached`) instead of the refusal. Restored, green.
    """
    monkeypatch.setenv("RECALL_ENTAILMENT", "1")
    monkeypatch.setattr("recall.cli_commands.index_search._make_embedder", lambda _name: object())
    monkeypatch.setattr("recall.cli_commands.index_search.PgVectorStore", _sentinel)
    with pytest.raises(SystemExit) as raised:
        main([command])
    assert str(raised.value) == REMOVED_ENTAILMENT_MESSAGE


def _old_wizard_env(tmp_path: Path, entailment: str) -> Path:
    from recall.setup import SETUP_BEGIN, SETUP_END

    path = tmp_path / ".env"
    path.write_text(
        f'{SETUP_BEGIN}\nRECALL_EMBEDDER="hashing"\nRECALL_ENTAILMENT="{entailment}"\n'
        f'RECALL_ENTAILMENT_MODEL="cross-encoder/qnli-distilroberta-base"\n{SETUP_END}\n',
        encoding="utf-8",
    )
    return path


def test_setup_refuses_to_rewrite_an_env_that_still_enables_the_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invariant: re-running `recall setup` on a `.env` an older wizard wrote with the judge on
    must not drop `RECALL_ENTAILMENT` silently (the managed block is rebuilt from the new answers
    alone), or the refusal above never fires for that deployment again. It refuses before any
    question and leaves the file as it was.

    Red proof, 2026-10-28 (CCA of #849, DAT-001): with the `_refuse_removed_entailment_in(env_path)`
    call removed from `recall.setup.run_setup_wizard`, this fails because the hardware-probe
    sentinel is reached (`_Reached`) instead of the refusal. Restored, green.
    """
    from recall.setup import run_setup_wizard

    path = _old_wizard_env(tmp_path, "1")
    before = path.read_text(encoding="utf-8")
    monkeypatch.setattr("recall.setup.probe_hardware", _sentinel)
    with pytest.raises(SystemExit) as raised:
        run_setup_wizard(dsn="postgresql://localhost/recall", env_path=path, input_fn=_sentinel)
    assert str(raised.value) == f"{path}: {REMOVED_ENTAILMENT_MESSAGE}"
    assert path.read_text(encoding="utf-8") == before


def test_recall_wizard_refuses_to_rewrite_an_env_that_still_enables_the_judge(
    tmp_path: Path,
) -> None:
    """`recall wizard` rebuilds the same managed block, so it must refuse the same way.

    Red proof, 2026-10-28 (CCA of #849, DAT-001, second route): with the
    `_refuse_removed_entailment_in(path)` call removed from `recall.setup._update_env_block`, this
    fails at `pytest.raises(SystemExit)` (DID NOT RAISE) and the key is gone from the file.
    Restored, green.
    """
    from recall.wizard.wiring import write_project_files

    _old_wizard_env(tmp_path, "1")
    before = (tmp_path / ".env").read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="entailment judge was removed"):
        write_project_files(
            project_root=tmp_path,
            dsn="postgresql://localhost/recall",
            embedder="hashing",
            memory_dir=tmp_path / "memory",
        )
    assert (tmp_path / ".env").read_text(encoding="utf-8") == before


def test_setup_proceeds_over_a_leftover_false_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: the `RECALL_ENTAILMENT="0"` every older wizard wrote must not stop a re-run."""
    from recall.setup import run_setup_wizard

    path = _old_wizard_env(tmp_path, "0")
    monkeypatch.setattr("recall.setup.probe_hardware", _sentinel)
    with pytest.raises(_Reached):
        run_setup_wizard(dsn="postgresql://localhost/recall", env_path=path, input_fn=_sentinel)
