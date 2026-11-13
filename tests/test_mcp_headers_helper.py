"""`scripts/mcp_headers_helper.py`: one header set per run, a fresh session each time.

Red proof, 2026-11-12: with the helper's session mutated to a constant string,
`test_every_run_mints_a_new_session` failed on its inequality assert; with the empty-file check
removed, `test_an_unusable_token_file_exits_2_and_prints_nothing[empty]` failed on its exit code.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[1] / "scripts" / "mcp_headers_helper.py"
TOKEN = "t0k3n-" + "x" * 40


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HELPER)], env=env, capture_output=True, text=True, timeout=60
    )


def _env(token_file: Path | None, **extra: str) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("RECALL_MCP_")}
    if token_file is not None:
        env["RECALL_MCP_TOKEN_FILE"] = str(token_file)
    env.update(extra)
    return env


def test_prints_the_three_headers(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text(TOKEN + "\n", encoding="utf-8")
    result = _run(_env(token_file, RECALL_MCP_CLIENT_NAME="claude-code"))
    assert result.returncode == 0, result.stderr
    headers = json.loads(result.stdout)
    assert headers["Authorization"] == f"Bearer {TOKEN}"
    assert re.fullmatch(r"[0-9a-f]{32}", headers["X-Recall-Session"])
    assert headers["X-Recall-Client"] == "claude-code"


def test_every_run_mints_a_new_session(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text(TOKEN, encoding="utf-8")
    first, second = (json.loads(_run(_env(token_file)).stdout) for _ in range(2))
    assert first["X-Recall-Session"] != second["X-Recall-Session"]
    assert first["X-Recall-Client"] == "mcp-client"


@pytest.mark.parametrize("case", ["unset", "missing", "empty", "utf16"])
def test_an_unusable_token_file_exits_2_and_prints_nothing(tmp_path: Path, case: str) -> None:
    token_file = tmp_path / "token"
    if case == "empty":
        token_file.write_text("  \n", encoding="utf-8")
    if case == "utf16":
        # What Windows PowerShell 5.1's `>` writes.
        token_file.write_text(TOKEN, encoding="utf-16")
    result = _run(_env(None if case == "unset" else token_file))
    assert result.returncode == 2
    assert result.stdout == ""


def test_a_byte_order_mark_is_not_sent_inside_the_bearer(tmp_path: Path) -> None:
    """Invariant: a token file saved with a UTF-8 BOM yields the bare token.

    Failure mode caught: `read_text(encoding="utf-8").strip()` keeps U+FEFF, so the helper exits
    0 with `Bearer \\ufeff<token>` and the server answers 401 with nothing pointing at the file.

    Red proof, 2026-11-13, CCA deferred run 20261113T115607Z (FIX-001 for BUG-002):
    `python -m cca_checks tautology verify` ran this test, and the `utf16` case above, against
    the pre-fix helper; this one failed on the Authorization assert, the `utf16` case on the exit
    code (an uncaught UnicodeDecodeError exits 1).
    """
    token_file = tmp_path / "token"
    token_file.write_bytes(b"\xef\xbb\xbf" + TOKEN.encode() + b"\r\n")
    result = _run(_env(token_file))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["Authorization"] == f"Bearer {TOKEN}"
