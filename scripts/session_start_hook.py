#!/usr/bin/env python3
"""SessionStart hook: claim this workspace, refuse a shared one, generate MCP config.

Why this exists as a hook and not as a command the model is asked to run.
Measured 2026-08-16 in this repository: **21 worktrees carried 2 claims**, and
six checkouts were dirty, including the main one. `scripts/session-open.sh` was
correct and simply never invoked, because invoking it depended on the model
remembering to before it touched anything. A hook does not forget.

Three jobs, in the order that matters:

1. **Refuse a shared workspace.** The main checkout is shared by definition:
   every worktree resolves its objects through it, and it is the directory every
   session reaches by habit. A worktree another live session claims is the same
   hazard one level down. Neither can be vetoed (SessionStart cannot stop a
   session from starting), so both are reported as loudly as this channel allows.
2. **Claim it**, so the *next* session is told whose it is, rather than
   discovering the collision through a lost edit.
3. **Generate `.mcp.json` where the repo knows how**, because the client reads
   that file at startup and nothing later in the session can write it in time.

Project-agnostic on purpose. Where a repo ships `scripts/session-space.sh` this
defers to it, so the repo stays the authority on its own policy. Where it does
not, the same two rules are enforced from here. That asymmetry is the whole
point: the protocol used to exist only in checkouts new enough to carry the
scripts, and those were never the checkouts causing the damage.

**The failure directions are not symmetric, and this file is written around
that.** A guard that wrongly says "claimed" costs somebody their uncommitted
work. A guard that wrongly says "refused" costs a moment's confusion. So:
cannot-tell counts as alive, a check that did not run is never reported as a
check that passed, and every path that ends in "claimed" has actually written
the claim. Findings from the bug audit of 2026-08-16 that this encodes:

- Liveness is asked BEFORE the age test, not after. An unreadable timestamp on a
  claim whose process is running is not a stale claim.
- The claim is taken with O_EXCL, so two sessions racing an unclaimed worktree
  cannot both be told they hold it.
- "The script refused" is decided by its exit code, never by whether it happened
  to print anything.
- Timeouts kill the process tree. Killing only the direct child left a grandchild
  holding the pipe and a 3 second timeout took 45 seconds.

Install (registered under SessionStart in ~/.claude/settings.json):

    python "<path to this file>"

It never breaks a session. Every failure path degrades to a message.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

#: Hard ceiling on any single command. A session start that hangs behind a hook
#: is worse than a session start with no guard.
TIMEOUT = 20

#: Ceiling on the hook as a whole. Individual timeouts do not compose: a dozen
#: commands at 20s each is four minutes in front of the user. Once this is spent,
#: remaining steps are skipped and say so rather than running unbounded.
BUDGET = 25.0
_STARTED = time.monotonic()

#: Returned instead of a real exit code when a command could not be launched, or
#: was killed for exceeding its timeout. Kept distinct from any real status,
#: because "the guard refused" and "the guard never ran" are opposite facts and
#: only one of them is about this workspace.
LAUNCH_FAILED = -1


def stale_hours() -> int:
    """Claim age at which an abandoned worktree may be taken over.

    Parsed here rather than at import: anything that can raise at module level
    sits outside the only exception handler this script has, and a hook that
    exits 1 with a traceback because an env var says "12h" has broken the one
    promise in the docstring.
    """
    raw = os.environ.get("RECALL_CLAIM_STALE_HOURS", "12")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 12


def budget_left() -> float:
    return BUDGET - (time.monotonic() - _STARTED)


def find_bash() -> str | None:
    """Absolute path to Git Bash, or None.

    Bare "bash" is WRONG on this machine and fails in a way that reads as a
    refusal. Windows CreateProcess searches System32 *before* PATH, and
    System32\\bash.exe is the WSL launcher, so subprocess(["bash", ...]) reaches
    WSL and dies with `execvpe(/bin/bash) failed` even though shutil.which finds
    Git Bash on PATH. Measured directly, 2026-08-16.

    So: resolve an absolute path, anchored on the git executable that is
    certainly present, and refuse any candidate living under System32.
    """
    env_bash = os.environ.get("GIT_BASH")
    if env_bash and Path(env_bash).is_file():
        return env_bash

    candidates: list[Path] = []
    git_exe = shutil.which("git")
    if git_exe:
        # .../Git/cmd/git.exe or .../Git/mingw64/bin/git.exe -> .../Git
        for parent in Path(git_exe).resolve().parents:
            if (parent / "bin" / "bash.exe").is_file():
                candidates.append(parent / "bin" / "bash.exe")
                break
    candidates += [
        Path(r"C:\Program Files\Git\bin\bash.exe"),
        Path(r"C:\Program Files (x86)\Git\bin\bash.exe"),
    ]
    which_bash = shutil.which("bash")
    if which_bash:
        candidates.append(Path(which_bash))

    for c in candidates:
        try:
            if c.is_file() and "system32" not in str(c).lower():
                return str(c)
        except OSError:
            continue
    return None


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the child AND its descendants.

    proc.kill() ends only the direct child. Measured 2026-08-16: a bash script
    whose body was `sleep 45`, run with timeout=3, took 45.2 seconds, because
    bash died and `sleep` kept the inherited stdout pipe open, so the read never
    finished. The timeout was enforced against the wrong process.
    """
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=5,
                check=False,
            )
        else:
            proc.kill()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def run(args, cwd=None, timeout=TIMEOUT, env=None):
    """Run a command. Returns (rc, stdout, stderr), all stripped. Never raises.

    stdout and stderr are kept APART. Merging them corrupts every caller that
    parses the result: git warns on stderr while exiting 0 (ambiguous refname,
    CRLF conversion, an unreadable directory during status), and a merged
    `rev-parse --show-toplevel` then yields a "path" with a warning and a
    newline embedded in it.
    """
    if timeout is None or timeout > budget_left():
        timeout = max(1.0, min(timeout or TIMEOUT, budget_left()))
    if budget_left() <= 0:
        return LAUNCH_FAILED, "", "hook time budget exhausted"
    try:
        proc = subprocess.Popen(
            args,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,  # a child that prompts must fail, not hang
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return LAUNCH_FAILED, "", "could not launch"
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, (out or "").strip(), (err or "").strip()
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            out, err = proc.communicate(timeout=5)
        except (subprocess.TimeoutExpired, ValueError, OSError):
            out, err = "", ""
        return LAUNCH_FAILED, (out or "").strip(), "timed out"
    except (OSError, ValueError):
        _kill_tree(proc)
        return LAUNCH_FAILED, "", "communication failed"


def git(cwd, *args, timeout=TIMEOUT):
    """Git, with stdout only. Warnings on stderr must not enter parsed values."""
    env = dict(
        os.environ,
        GIT_TERMINAL_PROMPT="0",  # never block a session start on a credential prompt
        GIT_OPTIONAL_LOCKS="0",
    )
    rc, out, _ = run(["git", *args], cwd=cwd, timeout=timeout, env=env)
    return rc, out


def run_script(script: Path, *args, cwd=None, env=None):
    """Run a shell script through Git Bash. Returns (rc, message).

    LAUNCH_FAILED means the script never really ran, which is NOT a refusal.
    WSL's bash exits nonzero with its own diagnostic rather than failing to
    launch, so a bare exit code cannot tell the two apart and the signature is
    checked explicitly.
    """
    bash = find_bash()
    if not bash:
        return LAUNCH_FAILED, "no Git Bash found"
    rc, out, err = run([bash, str(script), *args], cwd=cwd, env=env)
    message = "\n".join(x for x in (out, err) if x).strip()
    if rc != 0 and ("execvpe(/bin/bash)" in message or "WSL (" in message):
        return LAUNCH_FAILED, message
    return rc, message


def process_alive(pid: str) -> bool:
    """Is this pid running? "Cannot tell" counts as ALIVE.

    `kill -0` is wrong on this platform and wrong in the dangerous direction:
    CLAUDE_PID is a Windows pid and Git Bash's kill only knows its own process
    table, so it reported a running claude.exe as dead. The two errors are not
    symmetric. A false alive costs somebody a deliberate release; a false dead
    hands away a worktree that a live session is mid-edit in.

    Deliberately matches scripts/session-space.sh, including the empty-pid case,
    so the two mechanisms cannot disagree about who holds a worktree: an EMPTY
    pid is "no holder recorded" and falls through to the age test, while a
    non-empty pid that cannot be parsed or checked is alive.
    """
    if not pid:
        return False  # nothing recorded; the age test decides
    if not pid.isdigit():
        return True  # unparseable: cannot tell
    if os.name == "nt":
        rc, out, _ = run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], timeout=10)
        if rc != 0:
            return True  # cannot tell
        return pid in out
    rc, _, _ = run(["ps", "-p", pid], timeout=10)
    if rc == LAUNCH_FAILED:
        return True  # `ps` missing or timed out: cannot tell
    return rc == 0


def read_claim(claim_file: Path) -> dict:
    """Parse a claim file. Unreadable for ANY reason yields {}.

    errors="replace" and the ValueError arm matter: a claim written under a
    legacy code page, or a torn concurrent write splitting a multibyte
    character, raises UnicodeDecodeError, which is a ValueError and not an
    OSError. Uncaught it escapes to the top-level handler, and the hook then
    produces no guard, no claim and no output at all.
    """
    try:
        text = claim_file.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return {}
    out = {}
    for line in text.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def claim_is_stale(claim: dict) -> bool:
    """May this claim be taken over?

    **Liveness first.** Asking the timestamp first inverts the rule this file is
    built on: a claim whose `claimed_epoch` is missing or truncated but whose
    process is running belongs to a live session, and calling it stale hands away
    a worktree somebody is working in. Reproduced against a running claude.exe
    during the audit of 2026-08-16.
    """
    if process_alive(claim.get("pid", "")):
        return False
    when = claim.get("claimed_epoch", "")
    if not when.isdigit():
        return True  # no live process and no readable timestamp
    return (int(time.time()) - int(when)) > stale_hours() * 3600


def claim_body(session_id: str, pid: str, branch: str, root: Path) -> str:
    return "session={}\npid={}\nbranch={}\nworktree={}\nclaimed_at={}\nclaimed_epoch={}\n".format(
        session_id or "unknown",
        pid,
        branch,
        root.as_posix(),
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        int(time.time()),
    )


#: Rounds of 50ms to wait for the claim lock, and how many of those to tolerate a
#: lock directory carrying no holder pid before treating it as an orphan.
LOCK_ROUNDS = 20
LOCK_NOPID_ROUNDS = 10


def acquire_claim_lock(lock_dir: Path, pid: str) -> bool:
    """Take the claim lock, breaking one whose holder is demonstrably gone.

    `os.mkdir` is the atomic primitive, mirroring scripts/session-space.sh so the
    two implementations really do match rather than merely claim to.

    **Why a lock at all, when the create was already O_EXCL.** An exclusive
    *create* only decides who may make a file that does not exist. Taking over a
    STALE claim is a compare-and-swap on a file that does, and no single
    filesystem call does that. Fixing only the create left the takeover racing
    through `os.replace`, and two sessions taking over one stale claim were both
    told they had won, 3 trials out of 3. Faking it by unlinking first is worse:
    the second racer's unlink deletes the WINNER's freshly written claim.

    A lock must be reclaimable or it becomes the only permanent failure state in
    the mechanism, so it records its holder and a dead holder's lock is broken.
    "Cannot tell" counts as alive here too.
    """
    for i in range(LOCK_ROUNDS):
        try:
            os.mkdir(lock_dir)
            try:
                (lock_dir / "pid").write_text(pid, encoding="utf-8")
            except OSError:
                pass
            return True
        except FileExistsError:
            owner = ""
            try:
                owner = (lock_dir / "pid").read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                pass
            if (owner and not process_alive(owner)) or (not owner and i >= LOCK_NOPID_ROUNDS):
                shutil.rmtree(lock_dir, ignore_errors=True)
                continue
            time.sleep(0.05)
        except OSError:
            return False
    return False


def write_claim(claim_file: Path, body: str) -> bool:
    """Write the claim body. Only ever called while holding the lock.

    Temp-then-replace, so a reader never sees a torn file and a crash cannot
    leave the zero-byte claim that used to wedge the worktree. The temp is
    removed on failure rather than left behind in the git directory.
    """
    tmp = claim_file.with_name(f"{claim_file.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(body, encoding="utf-8", newline="\n")
        os.replace(tmp, claim_file)
        return True
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def builtin_guard(root: Path, git_dir: Path, is_main: bool, session_id: str, pid: str):
    """The same two rules, for a repo that ships no session-space script.

    Returns (blocked_message_or_None, notes).
    """
    if is_main:
        _, branch = git(root, "rev-parse", "--abbrev-ref", "HEAD")
        _, dirty = git(root, "status", "--porcelain")
        return (
            f"This is the repository's MAIN checkout ({root}), on branch "
            f"{branch or '?'} and {'DIRTY' if dirty else 'clean'}. Every session "
            "shares this index and working tree, so one session stages another's "
            "files and a rebase moves HEAD under a session that is mid-edit. "
            "Create an isolated worktree off origin/master before editing:\n"
            "  git worktree add -b <branch> .claude/worktrees/<name> origin/master",
            [],
        )

    claim_file = git_dir / "claude-session-claim"
    lock_dir = claim_file.with_name(claim_file.name + ".lock")
    if not acquire_claim_lock(lock_dir, pid):
        return (
            "Another session is claiming this worktree right now, so this one did not take "
            f"it. If that is wrong, the lock is at {lock_dir} and "
            "`scripts/session-space.sh release --force` clears it.",
            [],
        )
    try:
        # Read, decide and write all INSIDE the lock. The decision is about a file
        # another session may be rewriting, so a check taken before the lock is a
        # check about the past, and that is exactly what let two sessions both
        # take over one stale claim.
        claim = read_claim(claim_file)
        holder = claim.get("session", "")
        # Existence is a question about the FILE, not about whether it parsed.
        # Asking `bool(claim)` wedged a worktree permanently: a zero-byte or
        # unparseable claim took the create path, the create failed because the
        # file was right there, and every later session was refused with a
        # permissions diagnosis that no permission change could clear. This hook
        # can produce exactly that file by being killed mid-write.
        existed = claim_file.exists()
        if holder and holder != session_id and not claim_is_stale(claim):
            alive = "alive" if process_alive(claim.get("pid", "")) else "no longer running"
            return (
                f"This worktree is CLAIMED by another session ({holder}, since "
                f"{claim.get('claimed_at', '?')}, pid {claim.get('pid', '?')} {alive}). "
                "Do not edit here: that session may be mid-rebase, and a shared index "
                "loses one side silently. Take your own worktree off origin/master.",
                [],
            )

        _, branch = git(root, "rev-parse", "--abbrev-ref", "HEAD")
        if write_claim(claim_file, claim_body(session_id, pid, branch, root)):
            notes = []
            if existed and holder and holder != session_id:
                notes.append(f"took over a stale claim from {holder}")
            return None, notes
        return (
            f"Could not write the claim file ({claim_file}). This workspace is "
            "NOT claimed, so the next session will be told it is free. Fix the "
            "permissions or work somewhere else.",
            [],
        )
    finally:
        shutil.rmtree(lock_dir, ignore_errors=True)


#: Which remote MCP servers belong to which project, keyed by the origin
#: `owner/repo` slug, which is the only project identifier stable across
#: worktrees (a worktree's directory name is not even its branch). Lives beside
#: the secrets file, outside every repo, because the server list is a host
#: inventory and that is disclosure on its own.
MCP_POLICY = Path.home() / ".claude" / "mcp-projects.json"
MCP_SECRETS = Path(os.environ.get("RECALL_MCP_SECRETS", "")) if os.environ.get(
    "RECALL_MCP_SECRETS") else Path.home() / ".claude" / "recall-mcp-secrets.json"


def repo_slug(root: Path) -> str | None:
    """`owner/repo` from the origin remote, or None."""
    rc, url = git(root, "remote", "get-url", "origin")
    if rc != 0 or not url:
        return None
    url = url.strip()
    # Taking the last two segments of ANY origin turned a local path into a
    # plausible identity: `C:/Users/gde00/Documents/recall` became
    # `Documents/recall`. That identity selects which servers get written, and
    # the written file carries bearer tokens, so a fabricated slug is a
    # credential-placement bug waiting for a non-empty `default` policy.
    if re.match(r"^file://", url, re.I):
        return None                                        # a path dressed as a URL
    if not (re.match(r"^(https?|ssh|git)://", url, re.I)      # https://host/owner/repo
            or re.match(r"^[^/\\@]+@[^/\\:]+:", url)):        # git@host:owner/repo
        return None
    m = re.search(r"[:/]([^/:]+/[^/]+?)(?:\.git)?/?$", url)
    return m.group(1) if m else None


def generate_mcp_generic(root: Path, slug: str | None) -> tuple[str, str]:
    """Write `.mcp.json` for a project that ships no generator of its own.

    Measured 2026-08-16: **4 of 41 checkouts** across both projects had an
    `.mcp.json`, so 37 of 41 sessions started with no MCP servers at all. That,
    not reliability, is why the servers go unused. recall ships
    `scripts/session-mcp.sh`; sentiment-agent ships nothing, and it is the
    project whose corpus those servers actually are.

    The server list is **curated per project, not maximal**. Loading everything
    everywhere costs a standing context budget for tools that are never called
    (`mcp-pg-ops` is 38 tool definitions against 2 calls ever), and it puts one
    project's code search in front of a session working on another, which does
    not error: it answers confidently about the wrong repository.

    Returns (action, message) for the report and the log.
    """
    if not slug:
        return "no-origin", "no origin remote, so no project identity to select servers by"
    try:
        policy = json.loads(MCP_POLICY.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "no-policy", f"no server policy at {MCP_POLICY}"
    if not isinstance(policy, dict):
        return "policy-malformed", f"{MCP_POLICY} is not a JSON object"
    projects = policy.get("projects")
    wanted = projects.get(slug) if isinstance(projects, dict) else None
    if wanted is None:
        wanted = policy.get("default") or []
    if not isinstance(wanted, list):
        return "policy-malformed", f"the server list for {slug} is not a list"
    if not wanted:
        return "none-configured", f"no servers configured for {slug}"

    # Refuse BEFORE writing, never after. The file carries bearer tokens and
    # internal host addresses; checking afterwards would mean warning about a
    # file that already exists on disk with real credentials in it.
    rc, _ = git(root, "check-ignore", "-q", ".mcp.json")
    if rc != 0 and rc != 1:
        # Three outcomes, not two: ignored, not ignored, and could-not-tell.
        # Collapsing the third into the second asserts a fact never established
        # and sends the reader to edit a .gitignore that is already correct.
        return "ignore-check-failed", (
            "could not verify whether .mcp.json is gitignored, so nothing was written. "
            "The file would carry bearer tokens and internal host addresses, and an "
            "unverified ignore rule is not a verified one."
        )
    if rc == 1:
        return "refused-not-ignored", (
            ".mcp.json is NOT gitignored in this checkout, so it was not written. "
            "It would carry bearer tokens and internal host addresses. "
            "Add '.mcp.json' to .gitignore first."
        )
    try:
        raw = json.loads(MCP_SECRETS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "no-secrets", f"no secrets file at {MCP_SECRETS}"
    secrets = raw.get("servers") if isinstance(raw, dict) else None
    if not isinstance(secrets, dict):
        return "secrets-malformed", f"{MCP_SECRETS} has no 'servers' object"

    servers, missing = {}, []
    for name in wanted:
        cfg = secrets.get(name)
        if not isinstance(cfg, dict) or not cfg.get("url"):
            missing.append(name)
            continue
        entry = {"type": "http", "url": cfg["url"]}
        if cfg.get("token"):
            entry["headers"] = {"Authorization": f"Bearer {cfg['token']}"}
        servers[name] = entry
    if not servers:
        return "no-servers-resolved", f"none of {wanted} are in the secrets file"

    try:
        with (root / ".mcp.json").open("w", encoding="utf-8", newline="\n") as fh:
            json.dump({"mcpServers": servers}, fh, indent=2)
            fh.write("\n")
    except OSError as exc:
        return "write-failed", f"could not write .mcp.json: {exc}"
    note = f"wrote .mcp.json for {slug}: {', '.join(sorted(servers))}"
    if missing:
        note += f" (not in secrets file: {', '.join(missing)})"
    return "generated", note


def mcp_measurement_note(detail: str) -> str:
    """The one line that turns every session start into a data point.

    Whether a `.mcp.json` written here is early enough for the client to read it
    is still an open, pre-registered question, so this states both outcomes
    rather than asserting the predicted one.
    """
    return (
        f"MCP  {detail}. **This session's own MCP tool list is the measurement**: if those "
        "servers are absent, the client had already read the missing file and they arrive "
        "only on the next session in this checkout. Record which it was in "
        "docs/preregistrations/2026-08-16-sessionstart-hook-mcp-ordering.md"
    )


def trunk_ref(root: Path) -> str | None:
    """The remote trunk, never the local ref.

    The local `master` here is routinely stale, and a stale base looks exactly
    like a regression. No fetch: a hook must not put the network on the
    session-start path, so this compares against whatever was last fetched.
    """
    rc, head = git(root, "rev-parse", "--abbrev-ref", "origin/HEAD", timeout=10)
    if rc == 0 and head and "/" in head:
        return head
    for candidate in ("origin/master", "origin/main"):
        rc, _ = git(root, "rev-parse", "--verify", "-q", candidate, timeout=10)
        if rc == 0:
            return candidate
    return None


def rules_drift(root: Path, ref: str | None) -> list:
    """Is the project's own instruction file the current one?

    Measured 2026-08-16 across both projects on this machine: **6 of 41
    checkouts** carried current project rules. In recall, 17 of 21 worktrees had
    no CLAUDE.md at all, because the file landed on master on 2026-08-15 and
    every older worktree predates it. A session in one of those runs with none of
    the rules it believes it has, and nothing anywhere says so.

    This REPORTS and never repairs, for two reasons. The content of a rules file
    is loaded by the harness from disk before this hook runs, so injecting it
    here would mean acting on rules that are not in the file I can see. And a
    local difference is not always staleness: an ablation arm is a deliberate
    difference, and a hook that pushed you to "fix" it would quietly end the
    experiment. So absence is stated plainly, and difference is stated neutrally.
    """
    if not ref:
        return []
    lines = []
    names = ["CLAUDE.md"]

    # Follow @-imports from BOTH the local file and the trunk's copy, and take
    # the union. Reading only the local one is subtly wrong in the case that
    # matters most: a checkout stale enough to predate an import line would not
    # list the file it is missing, so precisely the worst-off checkouts would be
    # told nothing. Reading only the trunk misses a local-only import.
    sources = []
    local_root_rules = root / "CLAUDE.md"
    if local_root_rules.is_file():
        try:
            sources.append(local_root_rules.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            pass
    rc, trunk_text = git(root, "show", f"{ref}:CLAUDE.md")
    if rc == 0 and trunk_text:
        sources.append(trunk_text)

    for text in sources:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("@") and stripped.endswith(".md") and " " not in stripped:
                candidate = stripped[1:]
                if candidate not in names and len(names) < 6:
                    names.append(candidate)

    for name in names:
        rc, remote_hash = git(root, "rev-parse", f"{ref}:{name}")
        if rc != 0 or not remote_hash:
            continue  # not tracked on the trunk; there is nothing to be stale against
        local = root / name
        if not local.is_file():
            _, missed = git(root, "log", "--oneline", f"HEAD..{ref}", "--", name)
            n = len([x for x in missed.splitlines() if x.strip()])
            lines.append(
                f"rules  {name} is ABSENT from this checkout but tracked on {ref}"
                + (f" ({n} commit(s) to it not here)" if n else "")
                + f". You are working without it. Read it without switching branch: "
                f"git show {ref}:{name}"
            )
            continue
        rc, local_hash = git(root, "hash-object", str(local))
        if rc == 0 and local_hash and local_hash != remote_hash:
            _, missed = git(root, "log", "--oneline", f"HEAD..{ref}", "--", name)
            n = len([x for x in missed.splitlines() if x.strip()])
            lines.append(
                f"rules  {name} differs from {ref}"
                + (f", which has {n} commit(s) to it that this checkout lacks" if n else "")
                + f". Could be a behind checkout, local edits, or a deliberate variant. "
                f"Compare before assuming: git diff {ref} -- {name}"
            )
    return lines


def orient(root: Path) -> list:
    """Branch, position against the remote trunk, and foreign changes."""
    lines = []
    _, branch = git(root, "rev-parse", "--abbrev-ref", "HEAD")
    _, head = git(root, "log", "--oneline", "-1")
    lines.append(f"branch  {branch}   head  {head}")

    ref = trunk_ref(root)
    if ref:
        _, behind = git(root, "rev-list", "--count", f"HEAD..{ref}")
        _, ahead = git(root, "rev-list", "--count", f"{ref}..HEAD")
        if behind.isdigit() and ahead.isdigit():
            note = "   <- rebase before blaming a test" if behind != "0" else ""
            lines.append(f"vs {ref}  {behind} behind, {ahead} ahead{note}")

    _, dirty = git(root, "status", "--porcelain")
    n = len([x for x in dirty.splitlines() if x.strip()])
    if n:
        lines.append(f"uncommitted  {n} file(s); do not stage what you did not write")
    return lines


def implements_claim(script: Path) -> bool:
    """Does this session-space.sh understand `claim`?

    A repo of the same shape but an older vintage prints its usage and exits 2 on
    an unknown subcommand, and that usage text would otherwise be rendered to the
    model as the body of WORKSPACE REFUSED, on every single session start.
    """
    try:
        return "claim)" in script.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return False


def build_report(payload: dict, state: dict) -> str | None:
    """The whole hook, minus IO. Returns the additionalContext, or None.

    `state` is filled in as it goes, so the log records what happened even on a
    path that returns early or raises.
    """
    # A compaction is the same session in the same workspace. Re-running the
    # guard there is noise in a context window that has just been compressed.
    if payload.get("source") == "compact":
        state["outcome"] = "skipped-compact"
        return None

    cwd = payload.get("cwd") or os.getcwd()
    session_id = payload.get("session_id") or ""
    pid = os.environ.get("CLAUDE_PID", str(os.getpid()))

    rc, top = git(cwd, "rev-parse", "--show-toplevel")
    if rc != 0 or not top:
        state["outcome"] = "not-a-git-repo"
        return None
    root = Path(top)

    rc_a, git_dir = git(cwd, "rev-parse", "--absolute-git-dir")
    rc_b, common = git(cwd, "rev-parse", "--git-common-dir")
    if rc_a != 0 or not git_dir:
        state["outcome"] = "no-git-dir"
        return None
    git_dir_p = Path(git_dir)
    if not git_dir_p.is_absolute():
        state["outcome"] = "git-dir-not-absolute"
        return None
    # In the main checkout these resolve to the same directory; in a linked
    # worktree they differ. That is the whole test, and it depends on no path
    # convention. --git-common-dir can come back relative, so resolve it.
    common_p = (Path(cwd) / common).resolve() if rc_b == 0 and common else git_dir_p
    try:
        is_main = git_dir_p.resolve() == common_p
    except OSError:
        is_main = False
    state["is_main"] = is_main

    blocked = None
    notes: list[str] = []
    extra: list[str] = []
    space_sh = root / "scripts" / "session-space.sh"

    if space_sh.is_file() and implements_claim(space_sh):
        # The repo ships its own policy, so defer to it. Pass the session id
        # through the variable it reads: a hook is given it on stdin, not in env.
        env = dict(os.environ, CLAUDE_CODE_SESSION_ID=session_id, CLAUDE_PID=pid)
        rc, message = run_script(space_sh, "claim", cwd=str(root), env=env)
        if rc == LAUNCH_FAILED:
            # Bash is the optimisation, not the rule. Falling through keeps both
            # rules enforced on a machine where the script cannot run, instead of
            # reporting a refusal that is really an interpreter problem and has
            # nothing to do with this workspace.
            blocked, notes = builtin_guard(root, git_dir_p, is_main, session_id, pid)
            first = message.splitlines()[0][:80] if message else "no bash"
            notes.append(f"built-in guard; could not run session-space.sh: {first}")
        elif rc != 0:
            # Decided by the EXIT CODE, never by whether it printed anything. An
            # empty message here used to be falsy downstream and announced the
            # workspace as claimed when the script had refused it.
            blocked = message or f"scripts/session-space.sh refused with exit {rc} and no diagnostic."
        else:
            notes.append("scripts/session-space.sh")
    else:
        blocked, notes = builtin_guard(root, git_dir_p, is_main, session_id, pid)
        if not blocked and space_sh.is_file():
            notes.append("built-in guard; session-space.sh does not implement `claim`")
        elif not blocked:
            notes.append("built-in guard; this repo ships no session-space.sh")
    state["outcome"] = "refused" if blocked else "claimed"

    # MCP config. The client reads .mcp.json when the session starts, so this is
    # the only point in a session at which writing it can possibly matter.
    # Whether it is early ENOUGH is a separate, measured question, and until it
    # is answered the message below states both outcomes rather than guessing.
    mcp_sh = root / "scripts" / "session-mcp.sh"
    mcp_json = root / ".mcp.json"
    state["mcp_json_existed_before_hook"] = mcp_json.exists()
    state["mcp_action"] = "already-present" if mcp_json.exists() else "n/a"
    if not mcp_json.exists():
        if blocked:
            state["mcp_action"] = "skipped-workspace-refused"
            extra.append("MCP  .mcp.json not generated (workspace refused first)")
        elif mcp_sh.is_file():
            # The repo ships a generator, so it decides its own server list.
            rc, message = run_script(mcp_sh, cwd=str(root))
            tail = message.splitlines()[-1] if message else "ok"
            if rc == 0:
                state["mcp_action"] = "generated"
                extra.append(mcp_measurement_note(tail))
            elif rc == LAUNCH_FAILED:
                state["mcp_action"] = "launch-failed"
                extra.append(
                    "MCP  .mcp.json MISSING and session-mcp.sh could not be run "
                    f"({tail[:120]}). No MCP tools this session."
                )
            else:
                state["mcp_action"] = "failed"
                extra.append(f"MCP  session-mcp.sh FAILED: {tail[:300]}")
        else:
            # No generator in the repo: use the global per-project policy. This
            # is the path that reaches sentiment-agent's 20 checkouts, which
            # ship nothing and are where the qwen servers are the right corpus.
            action, message = generate_mcp_generic(root, repo_slug(root))
            state["mcp_action"] = action
            if action == "generated":
                extra.append(mcp_measurement_note(message))
            elif action in ("refused-not-ignored", "write-failed", "no-secrets"):
                extra.append(f"MCP  NOT written: {message}")
            # Anything else (no policy entry for this project, no origin) is the
            # ordinary case for a repo that is simply not meant to have servers,
            # and saying so on every session start would be noise.

    # Is the instruction file I was given the current one? Reported whether or
    # not the workspace was refused: a refused workspace still tells you
    # something about the checkout you landed in.
    extra += rules_drift(root, trunk_ref(root))

    # Containers, only when there is something to say. A hook that reports
    # "nothing found" on every session start trains you to skip reading it.
    rc, out, _ = run(
        ["docker", "ps", "--format", "{{.Names}} {{.Ports}}", "--filter", "label=recall.checkout"],
        timeout=10,
    )
    if rc == 0 and out:
        extra.append(
            "containers  "
            + "; ".join(out.splitlines())
            + "   (one you did not start in this session is someone else's)"
        )

    if blocked:
        body = ["WORKSPACE REFUSED. Fix this before editing anything.", "", blocked]
    else:
        suffix = f" ({'; '.join(notes)})" if notes else ""
        body = [f"Workspace claimed for this session.{suffix}"]
        body += orient(root)
    body += extra
    if budget_left() <= 0:
        body.append("(hook time budget exhausted; some checks were skipped)")
    return "\n".join(body)


#: Append-only record of every session start. Its purpose is one open question:
#: whether a `.mcp.json` written by this hook is early enough for the client to
#: read it, or lands one session too late. That cannot be measured by asking the
#: hook, only by comparing what it did against what the resulting session could
#: actually call, and a session with nothing written down cannot be compared.
#: Pre-registration:
#: docs/preregistrations/2026-08-16-sessionstart-hook-mcp-ordering.md
LOG = Path.home() / ".claude" / "session-start.log"


def log_line(payload: dict, state: dict) -> None:
    """One row per session start, on EVERY path including the early returns.

    Rotation goes through a temp file and an atomic replace: a read-truncate-
    write racing with a concurrent appender loses whatever was written in the
    window, and this log's whole purpose is to be counted later.
    """
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        if LOG.exists() and LOG.stat().st_size > 512_000:
            keep = LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-2000:]
            tmp = LOG.with_name(LOG.name + f".{os.getpid()}.tmp")
            tmp.write_text("\n".join(keep) + "\n", encoding="utf-8", newline="\n")
            os.replace(tmp, LOG)
        row = {
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session": (payload.get("session_id") or "")[:8] if isinstance(payload, dict) else "",
            "source": payload.get("source") if isinstance(payload, dict) else None,
            "cwd": payload.get("cwd") if isinstance(payload, dict) else None,
            "elapsed_s": round(time.monotonic() - _STARTED, 2),
        }
        row.update(state)
        with LOG.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row) + "\n")
    except (OSError, ValueError, TypeError):
        pass


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    if not isinstance(payload, dict):
        # Valid JSON of the wrong shape. Without this, payload.get raises and the
        # top-level handler turns a broken contract into total silence.
        log_line({}, {"outcome": "unusable-payload"})
        return 0

    state: dict = {"outcome": "error"}
    try:
        context = build_report(payload, state)
    except Exception as exc:  # noqa: BLE001 - degrade to a message, never to silence
        state["outcome"] = "error"
        state["error"] = f"{type(exc).__name__}: {exc}"[:200]
        context = (
            "The workspace guard could not run to completion "
            f"({state['error']}). Treat this workspace as UNVERIFIED: confirm you are "
            "not in the main checkout and not in a worktree another session holds."
        )
    finally:
        log_line(payload, state)

    if context:
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": context,
            }
        }))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001 - a guard must never break session start
        sys.exit(0)
