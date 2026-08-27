#!/usr/bin/env python3
"""Tests for the write-time recall hook, which runs before EVERY tool call in a session.

The whole risk surface is that this hook is invisible when it works and catastrophic when it
misbehaves: it sits in front of every `Write`, `Edit` and `Bash` in a live A/B, and any way it can
deny, crash, or hang changes the outcome it exists to measure. So every test here is about ONE
question: can this hook affect a session other than by adding context?

Nothing needs a database, a corpus or a model — retrieval is injected, so the tests pin the hook's
own decisions rather than the retriever's.

Mutation-tested 2026-08-27 by `scripts/write_time_hook_mutations.py`, eight ways, all eight
killed.
These are the MEASURED reds, not the predicted ones, and they are wider than I expected because a
hook that mis-extracts its payload or starts denying breaks several tests at once:

    `payload_of` returns the command for Write            -> payload_extraction (both fields),
                                                            can_never_deny, broken_trace, + RAISED
    MIN_QUERY_CHARS ignored (fires on any length)         -> short_payloads_are_ignored
    the unconfigured-DSN early return removed             -> unconfigured_is_silent
    the `except Exception` around search() removed        -> RAISED (uncaught into the session,
                                                            which is the defect exactly)
    `additionalContext` swapped for `permissionDecision`  -> can_never_deny (all four checks),
                                                            broken_trace, vocabulary_not_applied
    the session identity dropped from the trace           -> every_trace_line_names_its_session
                                                            (session_id, cwd and the stamp)
    a non-object event is accepted (json.load only)       -> malformed_stdin_is_survivable
                                                            (RAISED on `[]` and `3`)
    trace() lets OSError escape                           -> RAISED

Re-measure with `python scripts/write_time_hook_mutations.py`; it copies the hook, mutates the
copy, and reports any survivor. A surviving mutation means this table is not evidence.
"""

from __future__ import annotations

import io
import json
import pathlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import write_time_recall_hook as hook  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL {name}: {detail}")


def run(event: dict, env: dict, monkey_search=None) -> str:
    """Drive main() with a fake stdin and a controlled environment; return what it printed."""

    import os

    saved_env = dict(os.environ)
    saved_stdin, saved_stdout = sys.stdin, sys.stdout
    saved_search = hook.search
    try:
        os.environ.clear()
        os.environ.update(env)
        sys.stdin = io.StringIO(json.dumps(event))
        sys.stdout = io.StringIO()
        if monkey_search is not None:
            hook.search = monkey_search
        hook.main()
        return sys.stdout.getvalue()
    finally:
        hook.search = saved_search
        sys.stdin, sys.stdout = saved_stdin, saved_stdout
        os.environ.clear()
        os.environ.update(saved_env)


HIT = [("python-write-text-crlf-churn.md", "pass newline='\\n' to write_text", 0.83)]


def test_payload_extraction() -> None:
    """The query must be the text being COMMITTED, per tool. A wrong field queries nothing."""

    check("Write uses content",
          hook.payload_of("Write", {"content": "abc", "command": "rm -rf /"}) == "abc")
    check("Edit uses new_string",
          hook.payload_of("Edit", {"new_string": "xyz", "content": ""}) == "xyz")
    check("Bash uses command",
          hook.payload_of("Bash", {"command": "git add -A"}) == "git add -A")
    check("an unrelated tool yields nothing",
          hook.payload_of("Read", {"file_path": "a.py"}) == "")


def test_short_payloads_are_ignored() -> None:
    """A payload too short to carry an identifier retrieves whatever shares a common word."""

    out = run({"tool_name": "Bash", "tool_input": {"command": "ls"}},
              {"RECALL_HOOK_DSN": "postgresql://x/y"}, lambda q, d: HIT)
    check("a 2-character command does not search", out == "", f"printed {out!r}")


def test_unconfigured_is_silent() -> None:
    """Without a DSN the session must behave EXACTLY as if the hook were absent."""

    out = run({"tool_name": "Write", "tool_input": {"content": "x" * 200}}, {},
              lambda q, d: HIT)
    check("no DSN means no output at all", out == "", f"printed {out!r}")


def test_a_retrieval_failure_is_silent() -> None:
    """A corpus that is down must not break the session it is measuring."""

    def explode(query: str, dsn: str):
        raise RuntimeError("connection refused")

    out = run({"tool_name": "Write", "tool_input": {"content": "x" * 200}},
              {"RECALL_HOOK_DSN": "postgresql://x/y"}, explode)
    check("a retrieval error prints nothing and does not raise", out == "", f"printed {out!r}")


def test_the_hook_can_never_deny() -> None:
    """⛔ The load-bearing test. Denying a write would change outcomes for reasons unrelated to
    memory quality and make the registered endpoint uninterpretable."""

    out = run({"tool_name": "Write", "tool_input": {"content": "version_file.write_text(x)" * 9}},
              {"RECALL_HOOK_DSN": "postgresql://x/y"}, lambda q, d: HIT)
    check("it produced output on a hit", out.strip() != "", "expected an injection")
    parsed = json.loads(out)
    block = parsed.get("hookSpecificOutput", {})
    check("the output carries additionalContext", "additionalContext" in block, f"got {block}")
    check("the output carries NO permissionDecision",
          "permissionDecision" not in block, f"got {block}")
    check("the memo text reaches the context",
          "write_text" in block.get("additionalContext", ""), "memo body missing")
    check("no deny anywhere in the payload", "deny" not in out.lower(), f"printed {out!r}")


def test_no_hits_prints_nothing() -> None:
    """An empty result must not inject an empty banner: that is pure context cost for no signal."""

    out = run({"tool_name": "Write", "tool_input": {"content": "y" * 200}},
              {"RECALL_HOOK_DSN": "postgresql://x/y"}, lambda q, d: [])
    check("zero hits means zero output", out == "", f"printed {out!r}")


def test_a_broken_trace_is_not_fatal() -> None:
    """A hook that dies because its logging failed would change the run it is measuring."""

    out = run({"tool_name": "Write", "tool_input": {"content": "z" * 200}},
              {"RECALL_HOOK_DSN": "postgresql://x/y",
               # A parent that IS a file, so mkdir raises NotADirectoryError. An embedded NUL
               # would be the obvious choice and is not available: os.environ rejects it.
               "RECALL_HOOK_TRACE": str(Path(__file__).resolve() / "sub" / "trace.jsonl")},
              lambda q, d: HIT)
    check("a broken trace destination still injects", "additionalContext" in out,
          f"printed {out!r}")


def test_malformed_stdin_is_survivable() -> None:
    """The hook is fed by a client; a malformed event must not raise into the session."""

    for payload in ('not json at all', '[]', '3', '"x"', 'null'):
        _malformed(payload)


def _malformed(payload: str) -> None:
    """`[]` and `3` are VALID json without a `.get`, so a decode-error-only guard let an
    AttributeError escape into the session. Well-formed but wrong is the likelier malformation."""

    import os

    saved = dict(os.environ)
    saved_stdin, saved_stdout = sys.stdin, sys.stdout
    try:
        os.environ.clear()
        sys.stdin = io.StringIO(payload)
        sys.stdout = io.StringIO()
        code = hook.main()
        # Captured BEFORE stdout is restored, and checked AFTER. Calling check() here would print
        # its own "ok" line into the StringIO and discard it, so the test would look unrun.
        printed = sys.stdout.getvalue()
    finally:
        sys.stdin, sys.stdout = saved_stdin, saved_stdout
        os.environ.clear()
        os.environ.update(saved)
    check(f"malformed stdin {payload!r} returns 0 and prints nothing",
          code == 0 and printed == "", f"code={code} printed={printed!r}")


def test_vocabulary_is_recorded_not_applied() -> None:
    """The trigger is an ANNOTATION. If it ever gated an injection, the A/B would silently be
    measuring the gated variant instead of the one registered."""

    vocab = {"write_text"}
    check("it fires on a matching token",
          hook.vocabulary_would_fire("version_file.write_text(x)", vocab))
    check("it does not fire otherwise", not hook.vocabulary_would_fire("print('hello')", vocab))

    # And the injection happens regardless of what the trigger says.
    out = run({"tool_name": "Write", "tool_input": {"content": "print('hello')" * 20}},
              {"RECALL_HOOK_DSN": "postgresql://x/y"}, lambda q, d: HIT)
    check("a payload the trigger would reject is STILL injected",
          "additionalContext" in out, "the trigger gated an injection")


def test_every_trace_line_names_its_session() -> None:
    """Stage A's endpoint is injections PER SESSION, and every session of the run appends to ONE
    trace file. A record without a session key makes that endpoint uncomputable: the file holds a
    correct total that no per-session number can be recovered from, which is worse than an empty
    file because it looks like data.

    Both keys are asserted for a reason. `session_id` is the client's, so it is the true grouping;
    `cwd` is the harness's per-session sandbox, so it is the one the harness can join on without
    knowing what id the client chose. Losing either costs an analysis step.
    """

    import tempfile

    for label, monkey, expect_error in (
        ("on an injection", lambda q, d: HIT, False),
        ("on a retrieval failure", _raise, True),
    ):
        with tempfile.TemporaryDirectory() as tmp:
            destination = pathlib.Path(tmp) / "trace.jsonl"
            run(
                {
                    "tool_name": "Write",
                    "tool_input": {"content": "print('hello')" * 20},
                    "session_id": "sess-abc",
                    "cwd": "/sandbox/task-7",
                },
                {"RECALL_HOOK_DSN": "postgresql://x/y", "RECALL_HOOK_TRACE": str(destination)},
                monkey,
            )
            lines = [json.loads(line) for line in
                     destination.read_text(encoding="utf-8").splitlines() if line.strip()]
            check(f"a line was written {label}", len(lines) == 1, f"{len(lines)} lines")
            if not lines:
                continue
            record = lines[0]
            check(f"session_id is carried {label}", record.get("session_id") == "sess-abc",
                  repr(record.get("session_id")))
            check(f"cwd is carried {label}", record.get("cwd") == "/sandbox/task-7",
                  repr(record.get("cwd")))
            check(f"the line is timestamped {label}", bool(record.get("at")), repr(record.get("at")))
            check(f"the record still says what it was {label}",
                  ("error" in record) is expect_error, sorted(record))


def _raise(query, dsn):
    raise RuntimeError("no database")


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    print(f"write-time recall hook: {len(tests)} test groups\n")
    for test in tests:
        print(test.__name__)
        test()
        print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for failure in FAILURES:
            print(f"  {failure}")
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
