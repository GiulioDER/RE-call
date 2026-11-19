"""`recall generation`, `graph` and `calibration` refuse a scratch `--table` (#1421).

Invariant: on PostgreSQL these commands act on the tenant's generations and calibrations on either
route, and no `--table` selects those, so a `--table` other than the default must stop the command
before anything is opened. The failure this catches: `recall --table scratch generation rollback`
(or `gc`, `graph rebuild`, `calibration import`) changing the real store while the operator
believed they were working on a scratch table.

The generation manager, generation store and calibration repository are replaced by recorders that
refuse to be opened, so no database is needed; the positive control proves the recorder is what
each command opens.

Red proof, recorded 2026-11-19 on the deployment host, red in the intended assertion and green
again once restored:
- Baseline `origin/master` at 481cc8a56 (before the fix):
  `::test_a_scratch_table_is_refused_before_anything_opens` failed for all six commands on both
  routes, on `assert ['GenerationManager'] == []` (generation, graph rebuild),
  `['GenerationStore']` (graph status) and `['CalibrationRepository']` (calibration), because
  `_cmd_generation`, `_cmd_graph` and `_cmd_calibration` never read `args.table`. The twelve
  positive-control cases passed on that baseline.
- Mutation of `recall.runtime_route.generations_table_refusal`, `table.casefold()` replaced by
  `table`: `::test_the_default_table_still_reaches_the_store[*-Chunks]` failed for all six
  commands on `assert 0 == 1`.
"""

from __future__ import annotations

import pytest

import recall.calibration_v2
import recall.cli
import recall.generation_store
import recall.generations
from tests.lite_cli_support import _offline, run_cli  # noqa: F401  # _offline: autouse fixture

# Nothing listens on port 1, and every store is replaced before anything could connect.
DSN = "postgresql://tester:unused@127.0.0.1:1/recall"

COMMANDS = {
    "generation list": ["generation", "list"],
    "generation rollback": ["generation", "rollback"],
    "generation gc": ["generation", "gc"],
    "graph rebuild": ["graph", "rebuild", "--generation", "g1"],
    "graph status": ["graph", "status"],
    "calibration list": ["calibration", "list"],
}

_OPENED = "OPENED THE REAL STORE"

# Both routes: these commands ignore the route, so neither may let a scratch table through.
ROUTES = ["legacy", "generation"]


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every store these commands open records being opened and goes no further."""
    calls: list[str] = []

    def recorder(name: str) -> type:
        class _Recorder:
            def __init__(self, *args: object, **kwargs: object) -> None:
                calls.append(name)
                raise SystemExit(_OPENED)

        return _Recorder

    monkeypatch.delenv("RECALL_SOURCE_POLICY_FILE", raising=False)
    monkeypatch.setattr(recall.cli, "require_secure_dsn", lambda dsn: None)
    monkeypatch.setattr(recall.generations, "GenerationManager", recorder("GenerationManager"))
    monkeypatch.setattr(recall.generation_store, "GenerationStore", recorder("GenerationStore"))
    monkeypatch.setattr(recall.calibration_v2, "CalibrationRepository", recorder("CalibrationRepository"))
    return calls


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_a_scratch_table_is_refused_before_anything_opens(
    command: str, route: str, opened: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RECALL_INDEX_MODE", route)
    out = run_cli(["--dsn", DSN, "--embedder", "hashing", "--table", "scratch", *COMMANDS[command]])
    assert opened == [], f"the real store was opened despite --table 'scratch': {out}"
    assert "EXIT" in out and "--table 'scratch' cannot be honoured" in out, out


@pytest.mark.parametrize("table", ["chunks", "Chunks"])
@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_the_default_table_still_reaches_the_store(
    command: str, table: str, opened: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Positive control: without a scratch table each command does open its recorder, so the
    refusal above is what stopped it. `Chunks` is the default table as PostgreSQL folds it."""
    monkeypatch.setenv("RECALL_INDEX_MODE", "generation")
    out = run_cli(["--dsn", DSN, "--embedder", "hashing", "--table", table, *COMMANDS[command]])
    assert len(opened) == 1, out
    assert _OPENED in out, out
