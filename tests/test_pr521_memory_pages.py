"""NUM-001: vm_stat supplies its page size; both platform sizes must report MiB.

Red proof: the pre-fix suite-preflight.sh reports 1024 for the 16384-byte
fixture and fails the 4096 MiB assertion. The 4096-byte fixture is the
unchanged control inside the same regression node.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX fallback fixture runs on Linux/macOS")
def test_vm_stat_report_uses_its_declared_page_size() -> None:
    script = Path(__file__).parents[1] / "scripts/suite-preflight.sh"
    bash = shutil.which("bash")
    awk = shutil.which("awk")
    assert bash and awk, "the POSIX fallback fixture requires bash and awk"
    env = dict(os.environ)
    env.pop("RECALL_SUITE_AVAIL_MB", None)
    for page_size, expected in ((4096, 1024), (16384, 4096)):
        wrapper = r'''
command() {
  if [ "$1" = -v ] && [ "$2" = powershell.exe ]; then return 1; fi
  builtin command "$@"
}
awk() {
  case "$1" in *MemAvailable*) return 0;; esac
  "$REAL_AWK" "$@"
}
vm_stat() {
  printf 'Mach Virtual Memory Statistics: (page size of %s bytes)\n' "$PAGE_SIZE"
  printf 'Pages free: 131072.\nPages inactive: 131072.\n'
}
source "$1" report
'''
        result = subprocess.run(
            [bash, "-c", wrapper, "fixture", str(script)],
            env={**env, "REAL_AWK": awk, "PAGE_SIZE": str(page_size)},
            text=True, capture_output=True, check=True,
        )
        assert f"available memory : {expected} MB" in result.stdout, result.stdout
        assert "workers it picks : 3" in result.stdout
