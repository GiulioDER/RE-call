"""The removed entailment judge: a configuration that still turns it on is refused, loudly.

Invariant: `RECALL_ENTAILMENT` set to a true (or unreadable) value refuses at the CLI search path
and at MCP server startup, BEFORE any model, database or provider work; a false value or an unset
variable is accepted without a word. Failure mode caught: the judge's stricter filter disappearing
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
