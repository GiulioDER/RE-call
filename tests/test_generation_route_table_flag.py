"""The global `--table` flag on the generation route: refused, never dropped.

Invariant: under `RECALL_INDEX_MODE=generation` every generation is kept in `recall_chunks_v1`,
so a command that opens the generation store reads or erases that table whatever `--table`
names. A `--table` other than the default must therefore stop the command before any store is
opened. The failure this catches: `recall --table scratch forget --yes` erasing from the real
generation store, and `search`, `scopes`, `reasoning` and `records` answering from it, while the
operator believed they were working on a scratch table.

The store is replaced by a recorder that refuses to be opened, so the tests need no database; the
positive control proves the recorder is the store these commands reach.

Red proof, recorded 2026-11-19 on the deployment host, all three red in the intended assertion
and green again once restored:
- Baseline `origin/master` at 63b7c3ea2 (before the fix):
  `::test_a_scratch_table_is_refused_before_the_generation_store_opens` failed for all six
  commands on `assert ['recall_chunks_v1'] == []`, because `_cmd_remove`, `_cmd_search`,
  `_cmd_scopes`, `reasoning_cmd._cmd_reasoning` and `records_cmd._cmd_records` built
  `GenerationStore(args.dsn, ..., tenant=args.tenant)` and never read `args.table`. The twelve
  positive-control cases passed on that baseline, so the recorder is the store they open.
- Mutation of `RuntimeRoute.table_refusal`, its `not self.uses_generation` test removed:
  `::test_the_legacy_route_honours_the_table` failed on `assert "recall forget: --table
  'scratch' cannot be honoured ..." is None`.
- Mutation of `RuntimeRoute.table_refusal`, `table.casefold()` replaced by `table`:
  `::test_the_default_table_still_reaches_the_generation_store[Chunks-*]` failed for all six
  commands on `assert [] == ['recall_chunks_v1']`.
"""

from __future__ import annotations

import pytest

import recall.cli
import recall.generation_store
from recall.runtime_route import resolve_runtime_route
from tests.lite_cli_support import _offline, run_cli  # noqa: F401  # _offline: autouse fixture

# Nothing listens on port 1, and the store is replaced before anything could connect.
DSN = "postgresql://tester:unused@127.0.0.1:1/recall"

COMMANDS = {
    "forget": ["forget", "notes/a.md", "--yes"],
    "withdraw": ["withdraw", "notes/a.md", "--yes"],
    "search": ["search", "what did we decide?"],
    "scopes": ["scopes"],
    "reasoning": ["reasoning", "query", "what did we decide?"],
    "records": ["records", "--all"],
}

_OPENED = "OPENED THE GENERATION STORE"


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Generation route, and a generation store that records being opened and goes no further."""
    calls: list[str] = []

    class _Recorder:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append("recall_chunks_v1")
            raise SystemExit(_OPENED)

    monkeypatch.setenv("RECALL_INDEX_MODE", "generation")
    monkeypatch.delenv("RECALL_SOURCE_POLICY_FILE", raising=False)
    monkeypatch.setattr(recall.cli, "require_secure_dsn", lambda dsn: None)
    monkeypatch.setattr(recall.generation_store, "GenerationStore", _Recorder)
    return calls


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_a_scratch_table_is_refused_before_the_generation_store_opens(
    command: str, opened: list[str]
) -> None:
    out = run_cli(["--dsn", DSN, "--embedder", "hashing", "--table", "scratch", *COMMANDS[command]])
    assert opened == [], f"the generation store was opened despite --table 'scratch': {out}"
    assert "EXIT" in out and "--table 'scratch' cannot be honoured" in out, out
    assert "recall_chunks_v1" in out, out


@pytest.mark.parametrize("command", sorted(COMMANDS))
@pytest.mark.parametrize("table", ["chunks", "Chunks"])
def test_the_default_table_still_reaches_the_generation_store(
    command: str, table: str, opened: list[str]
) -> None:
    """Positive control: without a scratch table each command does open the recorder, so the
    refusal above is what stopped it. `Chunks` is the default table as PostgreSQL folds it."""
    out = run_cli(["--dsn", DSN, "--embedder", "hashing", "--table", table, *COMMANDS[command]])
    assert opened == ["recall_chunks_v1"], out
    assert _OPENED in out, out


def test_the_legacy_route_honours_the_table() -> None:
    """The refusal is the generation route's alone: the legacy route opens the named table."""
    legacy = resolve_runtime_route({"RECALL_INDEX_MODE": "legacy"})
    generation = resolve_runtime_route({"RECALL_INDEX_MODE": "generation"})
    assert legacy.table_refusal("scratch", "recall forget") is None
    assert generation.table_refusal("chunks", "recall forget") is None
    refusal = generation.table_refusal("scratch", "recall forget")
    assert refusal is not None and refusal.startswith("recall forget: --table 'scratch'"), refusal
