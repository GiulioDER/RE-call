#!/usr/bin/env python3
"""Tests for the harness's config-dir isolation, which is what makes any arm comparable.

`ClaudeExecConfig.config_dir` exists so a session can run ONE named hook without losing the
isolation `--bare` provides. That combination is not documented anywhere, so it was measured
against the live CLI (2.1.238) before the field was added:

    `--bare` alone                      hooks SKIPPED even when supplied via `--settings`;
                                        the init event reported 7 plugins loaded
    `CLAUDE_CONFIG_DIR` + no `--bare`   the hook FIRED; 0 plugins, 0 MCP servers

So this is not a weakening of `--bare`. It is stricter on plugins, and it admits exactly what the
named directory contains.

⛔ The risk this file exists to cover: a harness that silently admits a SECOND hook, or silently
drops the one it was given, would corrupt a run without failing. Every test below is about that.

Mutation-tested 2026-08-27 by `scripts/agent_ab_config_dir_mutations.py`, four ways, all killed.
Re-measure with that script; a surviving mutation means this file is not evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.agent_ab.claude_exec import ClaudeExecConfig  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL {name}: {detail}")


def test_bare_and_config_dir_are_refused_together() -> None:
    """Both set means "give me a hook" AND "skip hooks". Guessing is how an arm measures something
    nobody registered, so it must raise rather than prefer one."""

    try:
        ClaudeExecConfig(config_dir="/tmp/cfg", bare=True, strict_mcp_config=False)
    except ValueError as error:
        check("bare + config_dir raises", "mutually exclusive" in str(error), str(error))
    else:
        check("bare + config_dir raises", False, "it was accepted")


def test_config_dir_turns_bare_off_in_the_command() -> None:
    """`--bare` in the argv would skip the very hook the directory exists to supply."""

    config = ClaudeExecConfig(config_dir="/tmp/cfg", bare=False, strict_mcp_config=False)
    argv = config.command("hello")
    check("no --bare in the command", "--bare" not in argv, f"argv={argv}")

    default = ClaudeExecConfig(strict_mcp_config=False)
    check("the DEFAULT arm still passes --bare", "--bare" in default.command("hello"),
          "the default must be unchanged, or every prior result moves")


def test_the_default_is_unchanged() -> None:
    """Every result in this lane was produced with these defaults. If they moved, the new field
    changed runs it was never supposed to touch."""

    config = ClaudeExecConfig(strict_mcp_config=False)
    check("bare defaults to True", config.bare is True)
    check("config_dir defaults to None", config.config_dir is None)


def test_isolation_reaches_the_subprocess_environment() -> None:
    """The field is inert unless CLAUDE_CONFIG_DIR actually reaches the child, and it must not be
    overridable by the caller's own env block."""

    import inspect

    from benchmarks.agent_ab import claude_exec

    source = inspect.getsource(claude_exec.run_claude_case)
    check("run_claude_case sets CLAUDE_CONFIG_DIR", 'CLAUDE_CONFIG_DIR' in source,
          "the field would be inert")

    env_index = source.find("config.env.items()")
    cfg_index = source.find('environment["CLAUDE_CONFIG_DIR"]')
    check("it is set AFTER config.env, so a caller cannot silently override the isolation",
          env_index != -1 and cfg_index != -1 and cfg_index > env_index,
          f"env at {env_index}, config_dir at {cfg_index}")


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    print(f"agent_ab config-dir isolation: {len(tests)} test groups\n")
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
