#!/usr/bin/env python3
"""Refuse to run a measurement while its pre-registration is still uncommitted.

The standing rule is: write the prediction down, **commit it**, then measure. The commit is not
ceremony. An uncommitted pre-registration has no timestamp anyone can trust, including the person
who wrote it, so a prediction edited after the number arrives is indistinguishable from one that
was always there. Until now that rule was advice, and the store records two ways it failed while
being followed in spirit: two predictions that contradicted each other six lines apart, and a
registered check that went unreported through two pre-registrations.

## Exactly what this enforces, and what it cannot

It enforces **"commit the record before you run the thing"**, which is mechanically checkable: if
the pre-registration directory has uncommitted changes and you are launching a measurement, that
is an unambiguous violation and the call is denied.

It does **not** enforce "a pre-registration exists for this specific measurement". Nothing can:
matching a shell command to the hypothesis it tests is not decidable from the command line. A
clean tree with no relevant record therefore passes, and says so in a note rather than pretending
otherwise. A guard that claims more coverage than it has is worse than no guard, because the claim
is what gets remembered.

## Why command position, and not a substring match

`head scripts/run_locomo_arms.sh`, `git commit -m "fix run_gap_parallel.sh"` and a heredoc
documenting a benchmark all contain the name of a measurement. None of them runs one. A guard that
blocks you from reading or describing a benchmark is a guard you switch off within the hour, and a
switched-off guard protects nothing, so the token has to appear where a command actually goes:
either as the command word itself, or as the script/module argument of an interpreter.

Quoted strings and heredoc bodies are stripped first, for the same reason.

## Fails OPEN

An unparseable payload, a missing git, a repository this does not understand: all allow the call.
The escape hatch is the bare, unquoted word ALLOW_UNREGISTERED_MEASUREMENT, and using it is
reported rather than silent, because an escape nobody can see is an escape nobody audits.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys

ESCAPE = "ALLOW_UNREGISTERED_MEASUREMENT"

# Where a pre-registration lives in this project. Both spellings are real: the newer per-question
# records under docs/, and the original benchmark-wide one that predates them.
PREREG_PATHS = ("docs/preregistrations", "benchmarks/PREREGISTRATION.md")

SEP = re.compile(r"(?:\|\||&&|[;|&\n])")
ENV_PREFIX = re.compile(r"^(?:\w+=\S+\s+)*")
WRAPPERS = {"env", "xargs", "timeout", "nice", "ionice", "stdbuf", "taskset", "setsid",
            "sudo", "command", "nohup", "time", "exec", "doas"}
LEADING_NOISE = {"then", "do", "else", "elif", "fi", "done", "!", "{", "(",
                 "if", "while", "until", "case", "select", "in", "esac"}
KEY_VALUE = re.compile(r"^\w+=")

INTERPRETERS = {"python", "python3", "python.exe", "py", "bash", "sh", "zsh", "dash", "uv"}

# A script whose basename says it measures something. Grounded in this tree's actual entry points
# (scripts/run_*_arms.sh, scripts/ablate_*.py, scripts/score_pairs.py, benchmarks/*), not invented.
MEASUREMENT_BASENAME = re.compile(
    r"""^(?:run|score|ablate|bench|eval|probe)_        # run_locomo_arms.sh, ablate_*.py, score_pairs.py
      | _(?:arms|bench|eval|score|sweep|study)[._]     # *_arms.sh, *_sweep.py
      | benchmark                                      # anything spelling it out
    """,
    re.VERBOSE,
)
# Module paths that ARE the measurement harness.
MEASUREMENT_MODULE = re.compile(r"^(?:recall\.eval|benchmarks)(?:\.|$)")
# `recall.cli calibration calibrate` publishes a calibration, which is a measured claim.
CALIBRATE = ("calibration", "calibrate")


def deny(reason: str) -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }))
    sys.exit(0)


def note(message: str) -> None:
    print(json.dumps({"systemMessage": message}))


def strip_heredocs(cmd: str) -> str:
    """Drop heredoc BODIES: they are data being written, not commands being run."""
    out: list[str] = []
    lines = cmd.split("\n")
    i = 0
    while i < len(lines):
        out.append(lines[i])
        consumed_to = None
        for m in re.finditer(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1", lines[i]):
            delim = m.group(2)
            for j in range(i + 1, len(lines)):
                if lines[j].strip() == delim:
                    consumed_to = j
                    break
            break
        i = (consumed_to if consumed_to is not None else i) + 1
    return "\n".join(out)


def segments(cmd: str) -> list[str]:
    return [s.strip() for s in SEP.split(strip_heredocs(cmd)) if s.strip()]


def tokenise(segment: str) -> list[str]:
    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return segment.split()


def _base(tok: str) -> str:
    return os.path.basename(tok.strip("\"'").replace("\\", "/"))


def strip_leading_noise(tokens: list[str]) -> list[str]:
    """Advance past shell keywords, env assignments and wrappers to the real command word."""
    i = 0
    after_flag = False
    while i < len(tokens):
        tok = tokens[i]
        if tok in LEADING_NOISE or _base(tok) in WRAPPERS:
            after_flag = False
            i += 1
            continue
        if tok.startswith("-"):
            after_flag = True
            i += 1
            continue
        if KEY_VALUE.match(tok) or re.fullmatch(r"[\d.]+[smhd]?", tok) or after_flag:
            after_flag = False
            i += 1
            continue
        break
    return tokens[i:]


def is_measurement(cmd: str) -> tuple[bool, str]:
    """Return (verdict, what_matched). Only COMMAND positions are considered."""
    # Quoted text is data. A commit message naming a benchmark must not trip this.
    for segment in segments(cmd):
        unquoted = re.sub(r"'[^']*'|\"[^\"]*\"", " ", segment)
        tokens = strip_leading_noise(tokenise(ENV_PREFIX.sub("", unquoted.strip())))
        if not tokens:
            continue
        head = _base(tokens[0])
        rest = tokens[1:]

        # 1. The script is the command word: `./scripts/run_locomo_arms.sh`
        if MEASUREMENT_BASENAME.search(head):
            return True, head

        # 2. An interpreter running it: `python scripts/ablate_x.py`, `bash scripts/run_x.sh`
        if head in INTERPRETERS or head.startswith("python"):
            skip = False
            for i, tok in enumerate(rest):
                if skip:
                    skip = False
                    continue
                if tok == "-m" and i + 1 < len(rest):
                    if MEASUREMENT_MODULE.match(rest[i + 1]):
                        return True, rest[i + 1]
                    if rest[i + 1] in {"recall.cli", "recall"} and all(
                        v in rest for v in CALIBRATE
                    ):
                        return True, "recall.cli calibration calibrate"
                    skip = True
                    continue
                if tok.startswith("-"):
                    continue
                if MEASUREMENT_BASENAME.search(_base(tok)):
                    return True, _base(tok)

        # 3. pytest pointed at the benchmark tree.
        if head.startswith("pytest") or "pytest" in rest[:2]:
            for tok in rest:
                if tok.startswith("benchmarks") or "/benchmarks" in tok.replace("\\", "/"):
                    return True, tok
    return False, ""


def repo_root(cwd: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", cwd, "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def dirty_preregs(root: str) -> list[str]:
    """Uncommitted changes under the pre-registration paths, staged or not, tracked or not."""
    present = [p for p in PREREG_PATHS if os.path.exists(os.path.join(root, p))]
    if not present:
        return []
    try:
        out = subprocess.run(["git", "-C", root, "status", "--porcelain", "--", *present],
                             capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    try:
        if not isinstance(payload, dict):
            return 0
        tool_input = payload.get("tool_input")
        cmd = (tool_input or {}).get("command") if isinstance(tool_input, dict) else ""
        if not isinstance(cmd, str) or not cmd:
            return 0

        measuring, matched = is_measurement(cmd)
        if not measuring:
            return 0

        unquoted = re.sub(r"'[^']*'|\"[^\"]*\"", " ", cmd)
        if ESCAPE in re.split(r"[\s#;|&]+", unquoted):
            note(f"preregistration-guard: measurement ALLOWED via {ESCAPE} ({matched})")
            return 0

        cwd = payload.get("cwd") or os.getcwd()
        root = repo_root(cwd)
        if not root:
            return 0

        dirty = dirty_preregs(root)
        if dirty:
            listed = "\n  ".join(dirty[:10])
            deny(
                f"This launches a measurement ({matched}) while the pre-registration is still "
                f"uncommitted:\n  {listed}\n\n"
                "An uncommitted prediction has no timestamp anyone can trust, so it cannot be "
                "told apart from one written after the number arrived. Commit the record first "
                "(stage by pathspec), then run this. If the prediction is already committed and "
                "these are unrelated edits, commit or stash them. If this genuinely is not a "
                f"registered measurement, add the bare word {ESCAPE} to the command."
            )

        note(
            f"preregistration-guard: measurement ({matched}) allowed, pre-registration tree is "
            "clean. Note this checks only that nothing is UNCOMMITTED; it cannot verify a record "
            "exists for this specific question."
        )
    except Exception as exc:  # noqa: BLE001 - fail open, but say so
        print(json.dumps({
            "systemMessage": f"preregistration-guard errored and allowed the command: {exc!r}"
        }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
