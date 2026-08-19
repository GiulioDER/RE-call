#!/usr/bin/env python3
"""Turn "the tests pass" into a receipt: what ran, in this session, with what counts.

Two modes, one file:

    --record    PostToolUse. If the command that just ran was a test run, parse its summary line
                and append a row to this session's receipt.
    (default)   PreToolUse. If the command is a `git push`, report what the receipt actually
                contains, as additionalContext.

## Why a receipt rather than a reminder

The push hook already printed "BUG-REVIEW REQUIRED". A reminder asks the reader to remember doing
something; it cannot tell "I ran the tests" from "the tests ran", and those have been different
things here before. The store carries the specific failures: a suite whose 514 skips meant the
database tests never ran at all and the pass count was meaningless, and a guard that passed only
because of a variable exported into the test's own environment. Both look like green.

So this reports **facts it observed**: the command, the counts parsed out of pytest's own summary
line, and how long ago. Absence is also a fact, and is reported as absence.

## Why it does not block the push

A blocking check fires on documentation pushes, dependency bumps and README typos, where running
a 12 minute suite is not the right answer. The over-block is how a guard gets deleted, which loses
the coverage entirely. This states the evidence and leaves the judgement where it belongs.

## Why the session id is part of the key

A receipt from another session is another session's evidence. The same rule that governs this
project's Docker containers governs this: a run you did not start is not yours to cite.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

STATE_DIR = os.path.join(os.path.expanduser("~"), ".claude", "hooks", "state")

# pytest's own summary line, in BOTH forms it is printed in.
#
# Decorated, from a default run:   "==== 411 passed, 22 skipped, 3 warnings in 705.19s ===="
# Bare, from `-q`:                 "411 passed, 22 skipped in 705.19s (0:11:45)"
#
# The bare form matters more than it looks. Matching only the decorated one made this hook
# blind to `python -m pytest tests/ -q`, which is the command recall's own CLAUDE.md
# prescribes, so `parse_counts` returned {} on every documented run. The reporter returns
# early on empty counts, which put BOTH real detections below an unreachable line: the
# "It was NOT green" warning, and the SKIP_ALARM that exists to catch the DB tests silently
# not running. The guard degraded to a nag that fires on every green run and stays silent on
# the false green it was written for. Measured 2026-08-18: 3 spurious nags in one session.
#
# The anchor on the bare form is the trailing "in <float>s", which is what separates a real
# summary from prose that happens to contain "3 passed". Adversarial cases that must NOT
# match are covered in the self-test below.
COUNT = re.compile(r"(\d+)\s+(passed|failed|skipped|error|errors|xfailed|xpassed|deselected)")
SUMMARY_LINE = re.compile(
    r"^(?:=+.*\b(?:passed|failed|error|no tests ran)\b.*=+"
    r"|(?:\d+\s+(?:passed|failed|skipped|errors?|xfailed|xpassed|deselected),?\s+)+in\s+[\d.]+s.*"
    r"|no tests ran in\s+[\d.]+s.*)$",
    re.MULTILINE,
)
# Skips at or above this are the documented false-green signature: the DB tests never ran.
SKIP_ALARM = 100

SEP = re.compile(r"(?:\|\||&&|[;|&\n])")


def state_path(session_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id or "unknown")[:80]
    return os.path.join(STATE_DIR, f"test-receipt-{safe}.jsonl")


WRAPPERS = {"env", "timeout", "nohup", "time", "xargs", "setsid", "nice", "sudo", "command"}


def command_tokens(segment: str) -> list[str]:
    """The tokens of one segment, starting at the real command word.

    Quoted text is dropped first, and env assignments, durations and wrappers are skipped, so
    `RECALL_TEST_DSN=x timeout 900 python -m pytest` starts at `python`. Returning the tokens
    from the command word onward is what makes COMMAND POSITION checkable: without it,
    `echo git push` reads as a push, which it is not.
    """
    unquoted = re.sub(r"'[^']*'|\"[^\"]*\"", " ", segment).strip()
    if not unquoted:
        return []
    tokens = unquoted.split()
    i = 0
    while i < len(tokens) and (
        re.match(r"^\w+=", tokens[i])
        or os.path.basename(tokens[i]) in WRAPPERS
        or re.fullmatch(r"[\d.]+[smhd]?", tokens[i])
    ):
        i += 1
    return tokens[i:]


def is_test_run(cmd: str) -> bool:
    """A command that RUNS tests. Command position only, and quoted text is data."""
    for segment in SEP.split(cmd):
        rest = command_tokens(segment)
        if not rest:
            continue
        head = os.path.basename(rest[0])
        if head.startswith("pytest"):
            return True
        if head.startswith("python") or head in {"py", "uv"}:
            if "-m" in rest and rest.index("-m") + 1 < len(rest):
                if rest[rest.index("-m") + 1].startswith("pytest"):
                    return True
            # The repo's own hook and guard suites are plain scripts, not pytest.
            if any(t.endswith("_tests.py") for t in rest[1:]):
                return True
    return False


def parse_counts(text: str) -> dict[str, int]:
    """Counts from pytest's summary line only.

    Scanning the WHOLE output would pick up numbers from failure messages and from any test that
    prints the word "passed", which is how a receipt starts lying. If no summary line is present,
    nothing is claimed.
    """
    m = SUMMARY_LINE.search(text or "")
    if not m:
        return {}
    line = (text or "")[m.start():m.end()]
    out: dict[str, int] = {}
    for count, label in COUNT.findall(line):
        key = "error" if label.startswith("error") else label
        out[key] = out.get(key, 0) + int(count)
    return out


def _strings(obj, depth=0):
    """Every string in a nested payload. The tool_response shape is not contractual."""
    if depth > 6:
        return
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v, depth + 1)


def record(payload: dict) -> None:
    cmd = ((payload.get("tool_input") or {}) or {}).get("command") or ""
    if not isinstance(cmd, str) or not is_test_run(cmd):
        return
    blob = "\n".join(_strings(payload.get("tool_response")))
    counts = parse_counts(blob)
    row = {
        "at": time.time(),
        "command": cmd[:400],
        "counts": counts,
        # No summary line means the run produced no verdict this can read: a crash, an
        # interrupt, or a truncated response. Recorded as such rather than as a zero.
        "verdict": "parsed" if counts else "no summary line found",
    }
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(state_path(payload.get("session_id", "")), "a", encoding="utf-8",
              newline="\n") as fh:
        fh.write(json.dumps(row) + "\n")


def is_push(cmd: str) -> bool:
    """`git push` at command position.

    `echo git push` and `git commit -m "push later"` are not pushes. The first is why this reads
    the command word rather than scanning for an adjacent pair of tokens anywhere in the line.
    """
    for segment in SEP.split(cmd):
        tokens = command_tokens(segment)
        if not tokens or os.path.basename(tokens[0]) not in {"git", "git.exe"}:
            continue
        # Skip git's own global options (`git -C dir push`) to reach the subcommand.
        i = 1
        while i < len(tokens):
            tok = tokens[i]
            if tok in {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"}:
                i += 2
                continue
            if tok.startswith("-"):
                i += 1
                continue
            break
        if i < len(tokens) and tokens[i] == "push":
            return True
    return False


def read_rows(session_id: str) -> list[dict]:
    path = state_path(session_id)
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def describe(rows: list[dict], now: float) -> str:
    if not rows:
        return (
            "TEST RECEIPT: no test run was recorded in this session. That is not an accusation, "
            "it is the state of the evidence: nothing here can support a claim that the suite is "
            "green. If you are about to say tests pass, run them first, or say plainly that you "
            "did not."
        )
    last = rows[-1]
    counts = last.get("counts") or {}
    mins = max(0, int((now - float(last.get("at", now))) // 60))
    if not counts:
        return (
            f"TEST RECEIPT: the last test command in this session ({last.get('command', '')[:90]}) "
            f"produced no readable summary line, {mins} min ago. A run with no verdict is not a "
            "pass. Re-run it and read the output before claiming anything about it."
        )
    parts = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    msg = (f"TEST RECEIPT: last test run in this session was {mins} min ago: {parts} "
           f"({last.get('command', '')[:90]}). Total recorded runs: {len(rows)}.")
    if counts.get("failed") or counts.get("error"):
        msg += " It was NOT green. Do not describe this branch as passing."
    elif counts.get("skipped", 0) >= SKIP_ALARM:
        # Plain ASCII deliberately: this string reaches a Windows console whose default codec
        # cannot encode most symbols, and a receipt that raises UnicodeEncodeError reports nothing.
        msg += (f" WARNING: {counts['skipped']} skips is the documented false-green signature in "
                "this "
                "repo: roughly 22 is healthy, several hundred means the DB-backed tests never "
                "ran and the pass count does not cover them.")
    return msg


# Cases the summary matcher must get right. Kept next to the regex because the regex is the whole
# guard: if it stops matching a real summary the hook reports "no verdict" on a green run, and if
# it starts matching prose it reports a verdict that nobody measured. Run with --selftest.
SELFTEST_MATCH = [
    ("decorated", "==== 411 passed, 22 skipped, 3 warnings in 705.19s ====",
     {"passed": 411, "skipped": 22}),
    ("-q simple", "141 passed in 98.12s (0:01:38)", {"passed": 141}),
    ("-q skips", "5455 passed, 22 skipped in 705.19s (0:11:45)", {"passed": 5455, "skipped": 22}),
    ("-q failing", "2 failed, 139 passed in 98.12s", {"failed": 2, "passed": 139}),
    ("-q false green", "120 passed, 340 skipped in 61.00s", {"passed": 120, "skipped": 340}),
    ("-q errors", "1 error in 2.10s", {"error": 1}),
    ("-q no tests", "no tests ran in 0.03s", {}),
]
SELFTEST_REJECT = [
    ("prose", "the log says 3 passed and moved on"),
    ("echoed counter", "staging dirs before=15 after=15"),
    ("progress dots", "........................ [ 51%]"),
    ("assertion text", "AssertionError: 5 passed checks expected"),
    ("coverage row", "recall/index.py   282   14   95%"),
]


def selftest() -> int:
    bad = 0
    for name, text, want in SELFTEST_MATCH:
        got = parse_counts(text)
        if not SUMMARY_LINE.search(text) or got != want:
            bad += 1
            print(f"FAIL match  {name}: want {want}, got {got}")
    for name, text in SELFTEST_REJECT:
        if SUMMARY_LINE.search(text):
            bad += 1
            print(f"FAIL reject {name}: matched a line that is not a summary")
    print(f"selftest: {len(SELFTEST_MATCH)} match cases, {len(SELFTEST_REJECT)} reject cases, "
          f"{bad} failed")
    return 1 if bad else 0


def main() -> int:
    if "--selftest" in sys.argv[1:]:
        # Before the stdin read, which would otherwise block when run by hand.
        return selftest()
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    try:
        if not isinstance(payload, dict):
            return 0
        if "--record" in sys.argv[1:]:
            record(payload)
            return 0
        cmd = ((payload.get("tool_input") or {}) or {}).get("command") or ""
        if not isinstance(cmd, str) or not is_push(cmd):
            return 0
        message = describe(read_rows(payload.get("session_id", "")), time.time())
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "additionalContext": message,
            }
        }))
    except Exception:  # noqa: BLE001 - never break a push over a receipt
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
