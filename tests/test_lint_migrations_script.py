"""scripts/lint_migrations.sh lints every new migration, the way the runner will execute it.

Invariants: a file whose name the runner would never apply is refused rather than silently ending
the scan; a `-- recall:transactional` file is linted with `--assume-in-transaction` and a
concurrent-index file without it; frozen migrations are not linted; and a squawk refusal fails the
script. Failure mode caught (CCA of #843, CODE-001): a stray `0026-draft.sql` broke the number
parse, bash abandoned the loop without tripping `set -e`, and the gate exited 0 with every later
migration unlinted.

No squawk and no database: the real script runs on a copied tree against a stub `squawk` that
records its arguments and exits with the code the test asks for.

Red proof, 2026-10-28:
- `test_a_name_the_runner_would_not_apply_fails_the_lint` against the script before this fix failed
  at `assert result.returncode != 0` (the script printed "nothing to lint" and exited 0).
- `test_each_file_is_linted_in_the_mode_the_runner_runs_it` by mutation (the two branches of the
  marker test swapped) failed at the assertion on the transactional call's arguments.
- `test_a_squawk_refusal_fails_the_lint` by mutation (`|| status=1` made `|| true`) failed at
  `assert result.returncode == 1`.
- `test_frozen_migrations_are_not_linted` by mutation (the FROZEN_THROUGH skip removed) failed at
  the assertion that 0025 was not passed to squawk.
Restored, all green.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash to run scripts/lint_migrations.sh")

_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$SQUAWK_CALLS"
exit "${SQUAWK_EXIT:-0}"
"""


def _tree(tmp_path: Path, migrations: dict[str, str]) -> Path:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "lint_migrations.sh", root / "scripts" / "lint_migrations.sh")
    (root / ".squawk.toml").write_text("excluded_rules = []\n", encoding="utf-8")
    sql = root / "recall" / "migrations" / "sql"
    sql.mkdir(parents=True)
    for name, text in migrations.items():
        (sql / name).write_bytes(text.encode("utf-8"))
    return root


def _run(tmp_path: Path, migrations: dict[str, str], *, squawk_exit: int = 0) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    root = _tree(tmp_path, migrations)
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    stub = stub_dir / "squawk"
    stub.write_bytes(_STUB.encode("utf-8"))
    stub.chmod(0o755)
    calls = tmp_path / "calls.txt"
    env = dict(os.environ)
    env["PATH"] = f"{stub_dir}{os.pathsep}{env.get('PATH', '')}"
    env["SQUAWK_CALLS"] = str(calls)
    env["SQUAWK_EXIT"] = str(squawk_exit)
    result = subprocess.run(
        [BASH, "scripts/lint_migrations.sh"], cwd=root, env=env, capture_output=True, text=True, timeout=60
    )
    lines = calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []
    return result, lines


_TRANSACTIONAL = "-- recall:transactional\nCREATE INDEX i ON t (x);\n"
_CONCURRENT = "-- recall:concurrent-index i_x\nCREATE INDEX CONCURRENTLY IF NOT EXISTS i_x ON t (x);\n"


def test_a_name_the_runner_would_not_apply_fails_the_lint(tmp_path: Path) -> None:
    # `-` sorts before `_`, so the stray name is reached before the real migration.
    result, calls = _run(
        tmp_path, {"0026-draft.sql": "SELECT 1;\n", "0026_real.sql": _TRANSACTIONAL}, squawk_exit=1
    )

    assert result.returncode != 0, result.stdout + result.stderr
    assert "0026-draft.sql" in result.stderr


def test_each_file_is_linted_in_the_mode_the_runner_runs_it(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, {"0026_tx.sql": _TRANSACTIONAL, "0027_ix.sql": _CONCURRENT})

    assert result.returncode == 0, result.stdout + result.stderr
    transactional = [c for c in calls if "0026_tx.sql" in c]
    concurrent = [c for c in calls if "0027_ix.sql" in c]
    assert len(transactional) == 1 and "--assume-in-transaction" in transactional[0]
    assert len(concurrent) == 1 and "--assume-in-transaction" not in concurrent[0]


def test_a_squawk_refusal_fails_the_lint(tmp_path: Path) -> None:
    result, _calls = _run(tmp_path, {"0026_tx.sql": _TRANSACTIONAL}, squawk_exit=1)

    assert result.returncode == 1


def test_frozen_migrations_are_not_linted(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, {"0025_old.sql": _TRANSACTIONAL, "0026_new.sql": _TRANSACTIONAL})

    assert result.returncode == 0
    assert not any("0025_old.sql" in c for c in calls)
    assert any("0026_new.sql" in c for c in calls)
