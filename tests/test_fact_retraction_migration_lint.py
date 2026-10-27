"""The real migration gate rejects ledger changes that scan under an exclusive table lock."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_fact_retraction_migrations_pass_lint() -> None:
    """Run the CI gate, including every new migration after its frozen checkpoint.

    Invariant: adding the retraction CHECKs must not scan existing ledger rows while holding
    the schema-change transaction's exclusive table lock. Red-proof baseline: the original
    0026_fact_ledger_retraction.sql adds both CHECKs without NOT VALID; the gate must reject
    those statements and this test must fail at the return-code assertion below. After repair,
    0026 installs the CHECKs as NOT VALID and 0028 validates them in a separate transaction.
    The parent audit records the the Linux test host baseline failure and repaired success in its proof receipt.
    """
    if shutil.which("squawk") is None:
        pytest.skip("migration lint requires squawk, installed by the CI analysis job")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", "scripts/lint_migrations.sh"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, (
        "the migration gate rejected the fact-ledger migration sequence:\n"
        f"{result.stdout}\n{result.stderr}"
    )
