"""Integer RECALL_* knobs read at server import time must be validated (ENV-002).

`recall_mcp.server` reads RECALL_PORT / RECALL_POOL_SIZE / RECALL_STATEMENT_TIMEOUT_MS with a
bare ``int()`` at import. A typo then crashes with ``invalid literal for int()`` that names no
variable (unlike the deliberately-validated RECALL_TRANSPORT right beside them), and nothing
bounds-checks the value: a negative RECALL_STATEMENT_TIMEOUT_MS reaches ``SET statement_timeout``
and 0 silently disables the pool-exhaustion cap the knob exists to enforce.

These are import-time reads, so they are exercised by importing the module in a subprocess with
the env under test — the honest path, not a re-implementation of the parsing.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest


def _import_server_with(**env_overrides: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update({k: str(v) for k, v in env_overrides.items()})
    return subprocess.run(
        [sys.executable, "-c", "import recall_mcp.server"],
        capture_output=True,
        text=True,
        env=env,
    )


def test_import_succeeds_with_valid_env():
    r = _import_server_with()
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("var", ["RECALL_PORT", "RECALL_POOL_SIZE", "RECALL_STATEMENT_TIMEOUT_MS"])
def test_non_int_knob_is_rejected_with_a_named_message(var):
    r = _import_server_with(**{var: "not-an-int"})
    assert r.returncode != 0, f"{var}=not-an-int should fail import"
    # Not just any ValueError: the message must name the variable (the pre-fix int() error says
    # only "invalid literal for int() with base 10: 'not-an-int'").
    assert f"{var}=" in r.stderr and "is not an integer" in r.stderr, r.stderr[-600:]


@pytest.mark.parametrize(
    "var,bad",
    [
        ("RECALL_PORT", "70000"),               # above the 65535 TCP maximum
        ("RECALL_POOL_SIZE", "0"),              # a zero-connection pool is nonsensical
        ("RECALL_STATEMENT_TIMEOUT_MS", "-5"),  # negative reaches SET statement_timeout
        ("RECALL_STATEMENT_TIMEOUT_MS", "0"),   # 0 disables the pool-exhaustion cap (fail-open)
    ],
)
def test_out_of_range_knob_is_rejected_at_import(var, bad):
    r = _import_server_with(**{var: bad})
    assert r.returncode != 0, f"{var}={bad} should be rejected at import, not accepted"
    assert f"{var}=" in r.stderr and "out of range" in r.stderr, r.stderr[-600:]
