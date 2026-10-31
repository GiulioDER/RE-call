"""BUG-002: a working daemon probe does not prove later inventory calls worked."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


pytestmark = pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="the offline Docker executable fixture requires a POSIX Bash environment",
)


@pytest.mark.parametrize("failure", ["listing", "inspect", "partial_inspect"])
def test_orphan_inventory_failures_are_not_clean(tmp_path: Path, failure: str) -> None:
    """Preserve partial rows while reporting failure instead of an empty successful scan.

    Red-proof baseline: 160a23a3f, same parameterized nodes. The intended red
    assertion is the nonzero exit status after the initial probe succeeds.
    Production targets: _orphan_candidates, _inspect_batch and cmd_orphans in
    scripts/session-db.sh. Each node also checks a healthy empty scan, so refusing
    every inventory cannot satisfy the test. the remote test runner receipts record actual outcomes.
    """
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    docker = binary_dir / "docker"
    docker.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >> "$ORPHAN_TEST_LOG"
case "$1" in
    ps)
        if [[ "$*" == *--filter* ]]; then
            case "$ORPHAN_TEST_MODE" in
                listing) exit 3 ;;
                inspect|partial_inspect) printf 'candidate\\n' ;;
            esac
        fi
        exit 0
        ;;
    inspect)
        if [ "$ORPHAN_TEST_MODE" = partial_inspect ]; then
            printf '%s||/surviving-orphan\\n' "$ORPHAN_TEST_MISSING"
        fi
        exit 3
        ;;
    *) exit 97 ;;
esac
""",
        encoding="utf-8",
        newline="\n",
    )
    docker.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / "scripts" / "session-db.sh"
    log = tmp_path / "calls.log"
    env = {
        **os.environ,
        "PATH": str(binary_dir) + os.pathsep + os.environ.get("PATH", ""),
        "ORPHAN_TEST_LOG": str(log),
        "ORPHAN_TEST_MISSING": str(tmp_path / "removed-checkout"),
    }

    def run(mode: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(script), "orphans"],
            cwd=tmp_path,
            env={**env, "ORPHAN_TEST_MODE": mode},
            capture_output=True,
            text=True,
            check=False,
        )

    healthy = run("empty")
    assert healthy.returncode == 0, healthy.stdout + healthy.stderr
    assert "no orphaned containers" in healthy.stdout
    log.write_text("", encoding="utf-8")

    failed = run(failure)
    output = failed.stdout + failed.stderr
    calls = log.read_text(encoding="utf-8").splitlines()
    assert calls[0] == "ps -q", "the daemon liveness probe must have succeeded first"
    assert any("--filter" in call for call in calls)
    assert failed.returncode != 0, output
    assert "incomplete orphan inventory" in output
    assert "no orphaned containers" not in output
    if failure == "partial_inspect":
        assert "ORPHAN surviving-orphan" in output, output
    if failure != "listing":
        assert any(call.startswith("inspect ") for call in calls)
