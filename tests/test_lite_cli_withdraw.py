"""`recall withdraw` on a `sqlite:///` DSN and on PostgreSQL, beside `recall forget`.

Owner decision, 2026-10-22: forget keeps retired what a memo superseded; withdraw, for a memo that
was a mistake, gives it back. Both are the same preview-then-erase flow on the CLI.

Invariants and the failure each one catches:
- C1 on a lite DSN, `forget` keeps Paris retired, and `withdraw` of the forgotten Rome memo (found
  by the name it had, though no row of it is left) makes Paris current again.
- C2 the same forget-then-withdraw sequence prints the same lines on PostgreSQL and on lite.

Red proof: in each docstring.
"""

from __future__ import annotations

from pathlib import Path

from recall.lite import LiteStore
from tests.conftest import TEST_DSN, requires_db
from tests.lite_cli_support import _offline, run_cli  # noqa: F401  # _offline: autouse fixture
from tests.test_lite_cli import _dsn, _memos

HASHING_DIM = 64
MEMOS = {
    "paris.md": "# Home\n\nI live in Paris.\n",
    "rome.md": "---\nsupersedes: paris.md\n---\n# Home\n\nI moved to Rome.\n",
}


def _successor(tmp_path: Path, name: str) -> str | None:
    with LiteStore(tmp_path / "store" / "memory.db", dim=HASHING_DIM) as store:
        return store.supersession_graph().successor(name)


def test_c1_forget_keeps_retired_and_withdraw_gives_back_on_a_lite_dsn(tmp_path: Path) -> None:
    """Node tests/test_lite_cli_withdraw.py::test_c1_forget_keeps_retired_and_withdraw_gives_back_on_a_lite_dsn.

    Red, mutation of `_cmd_remove` (recall/cli_commands/index_search.py) erasing through
    `forget_memory` for a withdrawal too: a forget finds no row of the forgotten memo, and it fails
    on the receipt assertion ("no matching source(s) found").
    Red on master (`e5d87caba`): fails on "forget revived Paris" (the CLI's forget deleted Rome's
    claim with its chunks).
    """
    folder = _memos(tmp_path / "mem", MEMOS)
    base = ["--embedder", "hashing", "--dsn", _dsn(tmp_path)]
    run_cli([*base, "index", str(folder)])
    rome = str((folder / "rome.md").resolve())
    assert _successor(tmp_path, "paris.md") == "rome.md"

    run_cli([*base, "forget", rome, "--yes"])
    assert _successor(tmp_path, "paris.md") == "rome.md", "forget revived Paris"

    dry = run_cli([*base, "withdraw", rome])
    assert "DRY RUN: would withdraw 1 source(s)" in dry, dry
    assert _successor(tmp_path, "paris.md") == "rome.md", "the dry run withdrew"

    done = run_cli([*base, "withdraw", rome, "--yes"])
    assert "withdrew 0 chunk(s) from 1 source(s)" in done, done
    assert _successor(tmp_path, "paris.md") is None, "withdraw did not give Paris back"


@requires_db
def test_c2_withdraw_prints_the_same_on_postgres_and_lite(tmp_path: Path, cli_table: str) -> None:
    """Node tests/test_lite_cli_withdraw.py::test_c2_withdraw_prints_the_same_on_postgres_and_lite.

    Red, mutation of `LiteStore.recorded_sources_for_identifiers` (recall/lite/store.py) returning
    no record: the lite dry run calls the forgotten memo a typo, and it fails on "step withdraw dry
    run".
    """
    folder = _memos(tmp_path / "mem", MEMOS)
    rome = str((folder / "rome.md").resolve())
    backends = {
        "postgres": ["--embedder", "hashing", "--dsn", TEST_DSN, "--table", cli_table],
        "lite": ["--embedder", "hashing", "--dsn", _dsn(tmp_path)],
    }
    transcripts: dict[str, list[tuple[str, str]]] = {}
    for name, base in backends.items():
        run_cli([*base, "index", str(folder)])
        transcripts[name] = [
            ("forget", run_cli([*base, "forget", rome, "--yes"])),
            ("withdraw dry run", run_cli([*base, "withdraw", rome, "typo-source"])),
            ("withdraw", run_cli([*base, "withdraw", rome, "--yes"])),
        ]
    for (step, want), (_, got) in zip(transcripts["postgres"], transcripts["lite"], strict=True):
        assert got == want, f"step {step}: postgres {want!r} != lite {got!r}"
    assert "would withdraw 1 source(s)" in dict(transcripts["lite"])["withdraw dry run"]
