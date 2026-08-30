#!/usr/bin/env python
"""Tests for scripts/session_memory_check.py.

Every one of these was mutation-tested: the check was broken on purpose and the named test watched
to go red. A guard nobody has watched fail has not been tested, and this guard's whole purpose is
to fail in a case that currently looks like success.

No network and no server: `check()` is driven through a fake `_call`, because what is under test is
the CLASSIFICATION, not the transport. The transport is exercised for real by
`scripts/session_memory_check.py --all` against the live corpus.

Run: python scripts/session_memory_check_tests.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import session_memory_check as smc  # noqa: E402

FAILURES: list[str] = []


def expect(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


def _envelope(payload: dict) -> dict:
    return {"result": {"content": [{"type": "text", "text": json.dumps(payload)}]}}


def _fake_call(stats: dict, search: dict):
    def call(entry, calls, timeout):  # noqa: ARG001 - signature must match the real one
        return {2: _envelope(stats), 3: _envelope(search)}
    return call


HEALTHY_STATS = {"chunks": 10285, "newest_indexed_at": "2026-08-30T18:29:51Z", "stale": False}
HEALTHY_SEARCH = {
    "trust_state": "trusted", "calibrated": True, "calibration_status": "certified",
    "abstained": False, "failure_code": None, "hits": [{"source": "a.md", "score": 0.7}],
}


def run(stats: dict, search: dict, monkeypatch_call=None) -> dict:
    original = smc._call
    smc._call = monkeypatch_call or _fake_call(stats, search)
    try:
        return smc.check("recall-memory", {"command": "x", "args": []}, "q", 5.0)
    finally:
        smc._call = original


# --- the four verdicts must be distinguishable ------------------------------------------------
# The entire value of this script is that DEAD, DEGRADED and QUIET are DIFFERENT things. Collapse
# any two and the session cannot tell "there is no memory" from "there is nothing recorded", which
# is the ambiguity it exists to remove.

expect("healthy corpus is OK", run(HEALTHY_STATS, HEALTHY_SEARCH)["verdict"] == smc.OK)

expect(
    "an unreachable server is DEAD",
    run({}, {}, monkeypatch_call=lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("connection refused")))["verdict"] == smc.DEAD,
)

expect(
    "an EMPTY corpus is DEAD, not OK",
    run({"chunks": 0}, HEALTHY_SEARCH)["verdict"] == smc.DEAD,
)

expect(
    "a degraded trust_state is DEGRADED, not OK",
    run(HEALTHY_STATS, {**HEALTHY_SEARCH, "trust_state": "degraded"})["verdict"] == smc.DEGRADED,
)

expect(
    "calibrated=false is DEGRADED even when trust_state says trusted",
    run(HEALTHY_STATS, {**HEALTHY_SEARCH, "calibrated": False})["verdict"] == smc.DEGRADED,
    "this is exactly what RECALL_TRUST_MODE=development produces",
)

expect(
    "a trusted corpus that returns nothing is QUIET, not OK and not DEAD",
    run(HEALTHY_STATS, {**HEALTHY_SEARCH, "hits": []})["verdict"] == smc.QUIET,
)

# --- the exit code is what session-open reads -------------------------------------------------
# QUIET must NOT fail the run: a legitimate abstention is the corpus working correctly, and a
# preflight that cries wolf on it gets skipped, which costs more than it saves.

def _exit_for(verdict: str) -> int:
    """Drive main() with a fixed verdict and return its exit code."""
    original = smc.check
    smc.check = lambda name, entry, canary, timeout: {  # noqa: ARG005
        "name": name, "verdict": verdict, "detail": "", "seconds": 0.1,
        "chunks": 1, "newest": "", "trust_state": "trusted",
        "calibration_status": "certified", "abstained": False, "hits": 0,
    }
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".mcp.json"
            path.write_text(
                json.dumps({"mcpServers": {"recall-memory": {"command": "x", "args": []}}}),
                encoding="utf-8",
            )
            return smc.main(["--config", str(path), "--quiet"])
    finally:
        smc.check = original


expect("OK exits 0", _exit_for(smc.OK) == 0)
expect("DEAD exits 1", _exit_for(smc.DEAD) == 1)
expect("DEGRADED exits 1", _exit_for(smc.DEGRADED) == 1)
expect(
    "QUIET exits 0",
    _exit_for(smc.QUIET) == 0,
    "a legitimate abstention is the corpus working; a preflight that cries wolf gets skipped",
)


def _main_exit(servers: dict) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / ".mcp.json"
        path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
        return smc.main(["--config", str(path), "--quiet"])


expect(
    "a config with NO recall server fails",
    _main_exit({"vps3-lite": {"type": "http", "url": "http://x/mcp"}}) == 1,
    "an empty memory layer must never read as a healthy session",
)

expect(
    "a missing config file fails",
    smc.main(["--config", "/definitely/not/here/.mcp.json", "--quiet"]) == 1,
)

# --- server selection --------------------------------------------------------------------------

_three = {
    "recall-memory": {"command": "ssh", "args": []},
    "recall-code": {"command": "ssh", "args": []},
    "recall-docs": {"command": "ssh", "args": []},
}
with tempfile.TemporaryDirectory() as _tmp:
    _p = Path(_tmp) / ".mcp.json"
    _p.write_text(json.dumps({"mcpServers": {**_three, "vps3-lite": {"type": "http"}}}), encoding="utf-8")
    expect("default checks only the memory server", set(smc.servers(_p, False)) == {"recall-memory"})
    expect("--all checks every recall server", set(smc.servers(_p, True)) == set(_three))
    expect("non-recall servers are never probed", "vps3-lite" not in smc.servers(_p, True))

# --- the one-line collapse ----------------------------------------------------------------------

_multi = RuntimeError("connection failed\n| - host: 'a', port: 1: timeout\n| - host: 'b': timeout\n+---")
expect("a multi-line error is collapsed to one line", "\n" not in smc._one_line(_multi))
expect("collapsing keeps the reason", "timeout" in smc._one_line(_multi))
expect("an empty message still names the failure", smc._one_line(RuntimeError("")) == "RuntimeError")

print()
if FAILURES:
    print(f"{len(FAILURES)} failure(s): {', '.join(FAILURES)}")
    sys.exit(1)
print("all checks passed")
