"""Block a commit or push that a concurrent session's branch switch would send to the wrong place.

Reads a PreToolUse hook payload on stdin; prints a `deny` decision or nothing.

WHY THIS EXISTS (measured, 2026-07-31, GiulioDER/mem-bench):
  1. `git checkout -b fix/devils-advocate-findings`   -> on the new branch
  2. ... a lot of editing ...
  3. another session in the SAME clone switched the working tree back to `master`
  4. `git commit`                                     -> landed on master
  5. `git push -u origin fix/devils-advocate-findings`-> pushed the branch's STALE ref,
                                                         i.e. master's old tip, under the
                                                         branch name. Reported success.
  6. `gh pr create`                                   -> "No commits between master and
                                                         fix/devils-advocate-findings"

Every individual command succeeded. The only symptom was an error four steps later, from a
different tool. Two checks would have caught it at step 4 and step 5 respectively:

  COMMIT: refuse when HEAD is on the repo's default branch (or detached). Work that was meant
          for a feature branch must not silently become a commit on master.
  PUSH:   refuse `git push <remote> <branch>` when <branch> is not the branch currently checked
          out. That combination is how a stale local ref gets published under a name you believe
          you are on. A TAG is exempt: it names an object directly, there is no working tree it can
          disagree with, and refusing one broke every release (see `_is_tag_only`).

FAIL-OPEN, deliberately. This is an ergonomics guard, not a security boundary. Not a repo, no
HEAD, an unparseable command -> allow. An unexpected exception -> allow, but say so, because a
guard that silently stopped guarding is the failure mode this whole session was about.

ESCAPE HATCH: put `RACE_GUARD_OK` anywhere in the command (e.g. a trailing `# RACE_GUARD_OK`)
and both checks are skipped. Intentional commits to a default branch stay possible; they just
have to be intentional.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys

ESCAPE = "RACE_GUARD_OK"


def _git(args: list[str], cwd: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _to_os_path(path: str) -> str:
    """MSYS/Git-Bash `/c/Users/x` -> `C:/Users/x` (and `/cygdrive/c/...` likewise).

    Commands here are written for Git Bash, so their paths are MSYS-style, but this hook runs
    under a Windows Python for which `/c/Users/...` is not a directory at all. Without this the
    guard found no repo, returned "nothing to check", and allowed everything -- passing its own
    pipe-test by doing nothing, which is the exact shape of a guard that cannot fire.
    """
    if os.name != "nt":
        return path
    m = re.match(r"^/(?:cygdrive/)?([A-Za-z])(/.*)?$", path)
    if m:
        return f"{m.group(1).upper()}:{m.group(2) or '/'}"
    return path


def _target_dir(payload: dict, cmd: str) -> str:
    """Where the command will actually run.

    Commands in this setup are routinely `cd /c/Users/.../repo && git commit ...`, so the hook's
    own cwd is the session directory and NOT the repository being committed to. Ignoring the
    leading `cd` would make the guard inspect the wrong repo -- it would read the worktree's
    branch and cheerfully allow a commit onto another clone's master.
    """
    base = _to_os_path(payload.get("cwd") or os.getcwd())
    m = re.match(r"""\s*cd\s+("[^"]+"|'[^']+'|[^\s&;|]+)""", cmd)
    if not m:
        return base
    target = _to_os_path(m.group(1).strip("\"'"))
    if re.match(r"^(/|[A-Za-z]:)", target):
        return target
    return os.path.join(base, target)


def _default_branch(cwd: str) -> str | None:
    ref = _git(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], cwd)
    if ref:
        return ref.split("/", 1)[1] if "/" in ref else ref
    for name in ("master", "main"):
        if _git(["show-ref", "--verify", "--quiet", f"refs/heads/{name}"], cwd) is not None:
            return name
    return None


def _git_invocations(cmd: str) -> list[list[str]] | None:
    """Every real git invocation in `cmd`, as token lists starting at the subcommand.

    None means "could not parse", and the caller falls back to the regex path below.

    Text matching was not good enough, and the failure was observed rather than theorised. On
    2026-08-16 this guard blocked an append to a memory index because the PROSE being written
    contained the words of a push command inside a heredoc, and it parsed a fragment of that
    sentence as a branch name. The parser in git_add_pathspec_guard.py already strips heredoc
    bodies, splits on shell operators, respects quoting and descends into `bash -c`, so a mention
    of a command inside data is data. Reused rather than reimplemented.
    """
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import git_add_pathspec_guard as parser
    except Exception:  # noqa: BLE001 - fall back rather than stop guarding
        return None
    try:
        found: list[list[str]] = []
        for segment in parser.segments(parser.strip_heredocs(cmd)):
            found.extend(parser.git_commands(segment))
        return found
    except Exception:  # noqa: BLE001
        return None


def _pushed_branch_tokens(tokens: list[str]) -> str | None:
    """The branch in an already-parsed `push` invocation, or None when there is nothing to check.

    Same exclusions as the regex version below, which this replaces on the parsed path.
    """
    args = tokens[1:]
    if any(t in ("--delete", "-d", "--tags", "--mirror", "--all") for t in args):
        return None
    positional = [t for t in args if not t.startswith("-")]
    if len(positional) < 2:
        return None
    branch = positional[1]
    if branch == "HEAD" or ":" in branch:
        return None
    return branch


def _pushed_branch(cmd: str) -> str | None:
    """The branch name in `git push [flags] <remote> <branch>`, or None when there is not one.

    Returns None -- meaning "nothing to check" -- for every form where comparing against the
    current branch would be wrong rather than merely unnecessary: a bare `git push`, a bare
    `HEAD`, a refspec (`HEAD:master`), a deletion, or a tag push. Guessing at those would block
    legitimate work, and a guard that blocks legitimate work gets removed.
    """
    m = re.search(r"\bgit\s+push\b(.*)", cmd)
    if not m:
        return None
    tail = m.group(1)
    # Stop at a shell operator: `git push origin x && something-else` must not read the tail.
    tail = re.split(r"&&|\|\||[;|]", tail)[0]
    try:
        tokens = shlex.split(tail)
    except ValueError:
        return None
    if any(t in ("--delete", "-d", "--tags", "--mirror", "--all") for t in tokens):
        return None
    positional = [t for t in tokens if not t.startswith("-")]
    if len(positional) < 2:
        return None
    branch = positional[1]
    # `git push origin HEAD` pushes the branch you are ON, by construction. git resolves HEAD at
    # push time rather than looking up a local ref literally named "HEAD", so the stale-ref hazard
    # this guard exists for cannot arise. Comparing the string "HEAD" against the current branch
    # never matches, which refused the single most common way to push a feature branch on every
    # attempt. Measured 2026-08-16: `git push origin HEAD` and `git push -u origin HEAD` both denied.
    if branch == "HEAD":
        return None
    if ":" in branch:  # explicit refspec -- the author is being deliberate
        return None
    return branch


def _is_tag_only(ref: str, cwd: str) -> bool:
    """Whether `ref` names a tag and NOT also a branch.

    ⛔ **A tag is not a branch, and the hazard this guard exists for does not apply to one.**
    `git push origin v1.2.3` publishes an object the author just named; there is no working tree it
    could silently disagree with, and it is how every signed release in these repositories is made.
    The push check refused all of them, because it compares the pushed NAME against the checked out
    branch and a tag never matches. Measured 2026-08-22 during the v0.9.8 release, where it fired
    twice and both overrides were the operator's own -- which is the real cost: a guard that cries
    wolf on a routine action trains the reach for the escape hatch that the guard exists to prevent.

    ⚠️ A name that is BOTH a tag and a branch stays guarded. git itself refuses an ambiguous
    `push origin <name>`, so waving it through would assert something git does not agree with.
    """
    is_tag = _git(["rev-parse", "--verify", "--quiet", f"refs/tags/{ref}"], cwd) is not None
    is_branch = _git(["rev-parse", "--verify", "--quiet", f"refs/heads/{ref}"], cwd) is not None
    return is_tag and not is_branch


def _deny(reason: str) -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }))
    sys.exit(0)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    cmd = (payload.get("tool_input") or {}).get("command") or ""
    if not cmd or ESCAPE in cmd:
        return 0

    # Parse first, and only fall back to matching raw text if the parser is unavailable. The
    # fallback is strictly the old behaviour, so a missing parser degrades to what shipped before
    # rather than to no guard at all.
    invocations = _git_invocations(cmd)
    if invocations is None:
        wants_commit = re.search(r"\bgit\s+commit\b", cmd) is not None
        wants_push = re.search(r"\bgit\s+push\b", cmd) is not None
        push_tokens = None
    else:
        wants_commit = any(inv and inv[0] == "commit" for inv in invocations)
        push_invocations = [inv for inv in invocations if inv and inv[0] == "push"]
        wants_push = bool(push_invocations)
        push_tokens = push_invocations[0] if push_invocations else None
    if not (wants_commit or wants_push):
        return 0

    cwd = _target_dir(payload, cmd)
    if not os.path.isdir(cwd) or _git(["rev-parse", "--git-dir"], cwd) is None:
        return 0

    current = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    if not current:
        return 0

    if wants_commit:
        if current == "HEAD":
            _deny(
                f"BLOCKED: HEAD is DETACHED in {cwd}. A commit here belongs to no branch and is "
                f"unreachable once you check anything else out.\n"
                f"Check `git status`, then `git checkout -b <name>` to keep the work.\n"
                f"Override with a trailing `# {ESCAPE}` if this is deliberate."
            )
        default = _default_branch(cwd)
        if default and current == default:
            _deny(
                f"BLOCKED: you are on '{current}', the default branch of {cwd}.\n"
                f"This guard exists because another session sharing this clone can switch the "
                f"working tree out from under you between `checkout -b` and `commit` -- which is "
                f"exactly how work meant for a feature branch landed on master on 2026-07-31, "
                f"and was only noticed four steps later when `gh pr create` said 'No commits "
                f"between master and <branch>'.\n"
                f"Run `git branch --show-current` to confirm, then `git checkout -b <name>`.\n"
                f"Override with a trailing `# {ESCAPE}` if committing to {current} is intended."
            )

    if wants_push:
        branch = _pushed_branch_tokens(push_tokens) if push_tokens else _pushed_branch(cmd)
        if branch and branch != current and not _is_tag_only(branch, cwd):
            # `^{commit}` because an ANNOTATED tag's own hash is the TAG OBJECT, not the commit,
            # so the two lines below compared different kinds of thing and printed a mismatch that
            # did not exist. It is a no-op for a branch, which is the only ref that reaches here
            # now, and it keeps the message honest for the ambiguous tag-and-branch case.
            local = _git(["rev-parse", "--short", f"{branch}^{{commit}}"], cwd) or "?"
            head = _git(["rev-parse", "--short", "HEAD"], cwd) or "?"
            _deny(
                f"BLOCKED: pushing '{branch}' while '{current}' is checked out in {cwd}.\n"
                f"  {branch} -> {local}\n"
                f"  HEAD ({current}) -> {head}\n"
                f"git pushes the LOCAL REF named '{branch}', not your working tree. If a "
                f"concurrent session moved refs, that ref can be stale and the push reports "
                f"success while publishing the wrong commit -- observed 2026-07-31.\n"
                f"Either `git checkout {branch}` first, or point it at your work with "
                f"`git branch -f {branch} HEAD`.\n"
                f"Override with a trailing `# {ESCAPE}` if pushing another branch is intended."
            )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - fail open, but visibly
        print(json.dumps({
            "systemMessage": f"git-clone-race-guard errored and allowed the command: {exc!r}"
        }))
        raise SystemExit(0) from None
