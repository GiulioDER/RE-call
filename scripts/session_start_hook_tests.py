"""Regression tests for the SessionStart workspace hook.

Each test names the audit finding it pins. A test that cannot fail is not a
test, so every guard here is exercised with the input that used to defeat it.
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

HOOK = str(Path(__file__).resolve().parent / "session_start_hook.py")
DEPLOYED = Path.home() / ".claude" / "hooks" / "session_start_workspace.py"
SCRATCH = Path(tempfile.gettempdir()) / "recall-hooktests"

results = []


def load():
    spec = importlib.util.spec_from_file_location("hook", HOOK)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {detail}" if detail else ""))


def sh(*args, cwd=None):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True)


def new_repo(name, worktree=True, rules=None):
    """A throwaway repo with an optional linked worktree.

    `rules` commits a CLAUDE.md and points refs/remotes/origin/master at the
    result, so drift can be tested without a real remote.
    """
    base = SCRATCH / name
    if base.exists():
        subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", str(base)], capture_output=True)
    base.mkdir(parents=True)
    sh("git", "init", "-q", ".", cwd=base)
    sh("git", "config", "user.email", "t@t", cwd=base)
    sh("git", "config", "user.name", "t", cwd=base)
    (base / "a.txt").write_text("x")
    sh("git", "add", "a.txt", cwd=base)
    if rules is not None:
        (base / "CLAUDE.md").write_text(rules, newline="\n")
        sh("git", "add", "CLAUDE.md", cwd=base)
    sh("git", "commit", "-qm", "init", cwd=base)
    sh("git", "update-ref", "refs/remotes/origin/master", "HEAD", cwd=base)
    if worktree:
        sh("git", "worktree", "add", "-q", "-b", "wt", "./wt", "HEAD", cwd=base)
        return base, base / "wt"
    return base, base


def claim_path(wt: Path):
    r = sh("git", "rev-parse", "--absolute-git-dir", cwd=wt)
    return Path(r.stdout.strip()) / "claude-session-claim"


def report(m, cwd, session="S1", source="startup"):
    """One session start, as the hook process would run it.

    The hook's time budget is measured from module import (`_STARTED`), because
    in production every session start is a fresh process. Tests load the module
    once and call this many times, so without the reset the calls SHARE one 25s
    budget: on a slow runner a later call finds it spent, git is never asked,
    and the call returns None. That was CI run 36000505874 attempt 1 (82s for
    this file, where attempt 2 on the same commit took 31s).
    """
    m._STARTED = time.monotonic()
    state = {}
    return m.build_report({"session_id": session, "cwd": str(cwd), "source": source}, state), state


def seen(out, st):
    """What the hook said and why, for a check's detail line. Safe on None."""
    first = out.splitlines()[0][:100] if out else None
    return f"outcome={st.get('outcome')!r} out={first!r}"


# ---------------------------------------------------------------- BUG-001
def test_live_pid_unreadable_epoch_is_not_stale():
    """A live process with a truncated claim must NOT be taken over."""
    m = load()
    base, wt = new_repo("bug001")
    cf = claim_path(wt)
    cf.write_text(f"session=OTHER\npid={live_windows_pid()}\n", encoding="utf-8")  # no epoch
    out, _ = report(m, wt, session="ME")
    ok = "WORKSPACE REFUSED" in out and "OTHER" in out
    check("BUG-001 live pid + unreadable epoch is REFUSED", ok, out.splitlines()[0])

    # Control: the same claim with a DEAD pid and an old epoch must be takeoverable,
    # otherwise the test above would pass for a guard that refuses everything.
    cf.write_text("session=OTHER\npid=999999999\nclaimed_epoch=1\n", encoding="utf-8")
    out2, _ = report(m, wt, session="ME")
    ok2 = "WORKSPACE REFUSED" not in out2 and "took over a stale claim" in out2
    check("BUG-001 control: dead pid + old epoch IS taken over", ok2, out2.splitlines()[0])


# ---------------------------------------------------------------- BUG-002
def test_concurrent_claim_has_one_winner():
    """Two sessions racing an unclaimed worktree: exactly one may be told it won."""
    base, wt = new_repo("bug002")

    def attempt(sid):
        mm = load()
        out, _ = report(mm, wt, session=sid)
        return out

    with ThreadPoolExecutor(max_workers=2) as ex:
        outs = list(ex.map(attempt, ["AAA", "BBB"]))
    winners = sum(1 for o in outs if "Workspace claimed for this session" in o)
    check(
        "BUG-002 exactly one winner in a concurrent claim",
        winners == 1,
        f"winners={winners}; " + " | ".join(o.splitlines()[0][:70] for o in outs),
    )


# ---------------------------------------------------------------- BUG-003
def test_timeout_bounds_wall_clock():
    """Killing the child must kill the grandchild holding the pipe."""
    m = load()
    script = SCRATCH / "slow.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("#!/usr/bin/env bash\nsleep 45\n", newline="\n")
    bash = m.find_bash()
    t0 = time.monotonic()
    rc, out, err = m.run([bash, str(script)], timeout=3)
    dt = time.monotonic() - t0
    check(
        "BUG-003 a 3s timeout returns in under 12s (was 45.2s)",
        dt < 12 and rc == m.LAUNCH_FAILED,
        f"elapsed={dt:.1f}s rc={rc} err={err!r}",
    )


# ---------------------------------------------------------------- BUG-004
def test_silent_nonzero_script_is_a_refusal():
    """session-space.sh exiting 1 with no output must not read as success."""
    m = load()
    base, wt = new_repo("bug004")
    scripts = wt / "scripts"
    scripts.mkdir()
    # must contain 'claim)' or the contract probe rejects it before we get here
    (scripts / "session-space.sh").write_text(
        "#!/usr/bin/env bash\ncase $1 in claim) exit 1 ;; esac\nexit 1\n", newline="\n"
    )
    out, _ = report(m, wt)
    ok = "WORKSPACE REFUSED" in out and "no diagnostic" in out
    check("BUG-004 silent non-zero exit is a REFUSAL", ok, out.splitlines()[0])


# ---------------------------------------------------------------- BUG-005
def test_non_utf8_claim_does_not_silence_the_hook():
    m = load()
    base, wt = new_repo("bug005")
    cf = claim_path(wt)
    cf.write_bytes(b"session=OTHER\npid=999999999\nclaimed_epoch=1\nworktree=C:/Jos\xe9/x\n")
    try:
        out, _ = report(m, wt, session="ME")
        ok = out is not None and len(out) > 0
        detail = out.splitlines()[0]
    except Exception as exc:
        ok, detail = False, f"raised {type(exc).__name__}: {exc}"
    check("BUG-005 non-UTF-8 claim file does not crash the guard", ok, detail)


# ---------------------------------------------------------------- BUG-006
def test_bad_env_var_does_not_crash_at_import():
    env = dict(os.environ, RECALL_CLAIM_STALE_HOURS="12h")
    p = subprocess.run(
        [sys.executable, HOOK],
        input='{"session_id":"x","cwd":"' + str(SCRATCH).replace("\\", "/") + '","source":"startup"}',
        capture_output=True, text=True, env=env,
    )
    check(
        "BUG-006 unparseable RECALL_CLAIM_STALE_HOURS exits 0",
        p.returncode == 0 and "Traceback" not in p.stderr,
        f"rc={p.returncode} stderr={p.stderr[:120]!r}",
    )


# ---------------------------------------------------------------- BUG-008
def test_cannot_tell_liveness_is_alive():
    m = load()
    m.run = lambda *a, **k: (m.LAUNCH_FAILED, "", "could not launch")
    check(
        "BUG-008 an unrunnable liveness check reports ALIVE",
        m.process_alive("4242") is True,
        "",
    )
    m2 = load()
    check("BUG-008 empty pid falls through to the age test (matches the shell)",
          m2.process_alive("") is False, "")
    m3 = load()
    check("BUG-008 unparseable pid is ALIVE", m3.process_alive("abc") is True, "")


# ---------------------------------------------------------------- BUG-012
def test_non_object_payload_is_logged_not_silent():
    log = Path.home() / ".claude" / "session-start.log"
    before = log.read_text(encoding="utf-8").count("unusable-payload") if log.exists() else 0
    p = subprocess.run([sys.executable, HOOK], input="[]", capture_output=True, text=True)
    after = log.read_text(encoding="utf-8").count("unusable-payload") if log.exists() else 0
    check(
        "BUG-012 a JSON array payload is recorded, not silent",
        p.returncode == 0 and after == before + 1,
        f"rc={p.returncode} log rows {before}->{after}",
    )


# ---------------------------------------------------------------- BUG-015
def test_foreign_session_space_falls_back():
    """A same-named script that does not implement `claim` must not refuse."""
    m = load()
    base, wt = new_repo("bug015")
    scripts = wt / "scripts"
    scripts.mkdir()
    (scripts / "session-space.sh").write_text(
        "#!/usr/bin/env bash\necho 'usage: something else' >&2\nexit 2\n", newline="\n"
    )
    out, _ = report(m, wt)
    ok = "WORKSPACE REFUSED" not in out and "does not implement" in out
    check("BUG-015 a foreign session-space.sh falls back to the built-in guard", ok,
          out.splitlines()[0])


# ---------------------------------------------------------------- rules drift
def test_rules_drift():
    """The measured failure: 17 of 21 recall worktrees had no CLAUDE.md at all."""
    m = load()
    base, wt = new_repo("rules", rules="# rules\n\nsome standing instruction\n")

    out, _ = report(m, wt, session="R1")
    check("rules: an identical CLAUDE.md says nothing", "rules  " not in out,
          out.splitlines()[-1][:80])

    (wt / "CLAUDE.md").unlink()
    out, _ = report(m, wt, session="R1")
    ok = "rules  CLAUDE.md is ABSENT" in out and "git show origin/master:CLAUDE.md" in out
    check("rules: an absent CLAUDE.md is reported with the command to read it", ok,
          [x for x in out.splitlines() if x.startswith("rules")][:1])

    (wt / "CLAUDE.md").write_text("# rules\n\nlocally changed\n", newline="\n")
    out, _ = report(m, wt, session="R1")
    line = [x for x in out.splitlines() if x.startswith("rules")]
    ok = bool(line) and "differs from origin/master" in line[0] and "ABSENT" not in line[0]
    check("rules: a differing CLAUDE.md is reported as differing, not absent", ok, line[:1])

    # An ablation arm is a deliberate difference. The report must describe, not
    # instruct: a hook that pushed a repair here would end the experiment.
    ok = bool(line) and "deliberate variant" in line[0] and "fix" not in line[0].lower()
    check("rules: a difference is reported neutrally, with no repair instruction", ok, line[:1])

    base2, wt2 = new_repo("norules")  # no CLAUDE.md tracked anywhere
    out, _ = report(m, wt2, session="R2")
    check("rules: a repo with no tracked CLAUDE.md says nothing", "rules  " not in out)


def test_rules_drift_follows_imports():
    m = load()
    base, wt = new_repo("rulesimport", rules="# rules\n\n@CLAUDE-lessons.md\n")
    (base / "CLAUDE-lessons.md").write_text("# lessons\n", newline="\n")
    sh("git", "add", "CLAUDE-lessons.md", cwd=base)
    sh("git", "commit", "-qm", "lessons", cwd=base)
    sh("git", "update-ref", "refs/remotes/origin/master", "HEAD", cwd=base)
    # The worktree was cut BEFORE the lessons commit, so it lacks the file: the
    # exact shape of the 11 sentiment-agent checkouts missing CLAUDE-lessons.md.
    out, _ = report(m, wt, session="R3")
    check("rules: an @-imported file that is absent here is reported",
          "CLAUDE-lessons.md is ABSENT" in out,
          [x for x in out.splitlines() if x.startswith("rules")][:2])

    # The case that matters most: a checkout so stale its CLAUDE.md predates the
    # import line cannot name the file it is missing. Imports must therefore be
    # read from the trunk too, or the worst-off checkouts are told nothing.
    (wt / "CLAUDE.md").write_text("# rules\n\nold copy, no import line\n", newline="\n")
    out, _ = report(m, wt, session="R3")
    check("rules: an import the STALE local file does not mention is still checked",
          "CLAUDE-lessons.md is ABSENT" in out,
          [x for x in out.splitlines() if x.startswith("rules")][:2])


# ---------------------------------------------------------------- mcp policy
def _mcp_fixture(tmp: Path, slug, wanted, gitignored=True):
    """A repo with an origin, a policy file and a secrets file, all throwaway."""
    m = load()
    base, wt = new_repo("mcp-" + slug.split("/")[-1])
    sh("git", "remote", "add", "origin", f"https://example.invalid/{slug}.git", cwd=base)
    if gitignored:
        (base / ".gitignore").write_text(".mcp.json\n", newline="\n")
        sh("git", "add", ".gitignore", cwd=base)
        sh("git", "commit", "-qm", "ignore", cwd=base)
        (wt / ".gitignore").write_text(".mcp.json\n", newline="\n")
    policy = tmp / "policy.json"
    policy.write_text(json.dumps({"projects": {slug: wanted}, "default": []}), newline="\n")
    secrets = tmp / "secrets.json"
    secrets.write_text(json.dumps({"servers": {
        "qwen-mcp": {"url": "http://h.invalid:49555/mcp", "token": "T"},
        "vps3-lite": {"url": "http://h.invalid:49556/mcp"},
        "code-rag": {"url": "http://h.invalid:8765/mcp", "token": "T"},
    }}), newline="\n")
    m.MCP_POLICY = policy
    m.MCP_SECRETS = secrets
    # The generator records approval with RE-call's own script, which writes the
    # REAL ~/.claude.json. A test must never reach it: point it at a path that
    # does not exist, and stub `run` where a test needs the call itself.
    m.MCP_APPROVE = tmp / "no-such-approve.py"
    return m, base, wt


def test_mcp_generic_generator():
    tmp = SCRATCH / "mcpcfg"
    tmp.mkdir(parents=True, exist_ok=True)

    m, base, wt = _mcp_fixture(tmp, "GiulioDER/sentiment-agent", ["qwen-mcp", "vps3-lite"])
    check("mcp: the origin slug is the project key",
          m.repo_slug(wt) == "GiulioDER/sentiment-agent", m.repo_slug(wt))

    action, msg = m.generate_mcp_generic(wt, m.repo_slug(wt))
    written = json.loads((wt / ".mcp.json").read_text(encoding="utf-8"))
    names = sorted(written["mcpServers"])
    check("mcp: writes exactly the curated server list",
          action == "generated" and names == ["qwen-mcp", "vps3-lite"], f"{action}: {names}")

    check("mcp: a token becomes an Authorization header",
          written["mcpServers"]["qwen-mcp"]["headers"]["Authorization"] == "Bearer T")
    check("mcp: a server with no token gets no auth header",
          "headers" not in written["mcpServers"]["vps3-lite"])
    check("mcp: an uncurated server is NOT written",
          "code-rag" not in written["mcpServers"])

    # The refusal that must fire: this file carries bearer tokens.
    m2, base2, wt2 = _mcp_fixture(tmp, "GiulioDER/leaky-repo", ["qwen-mcp"], gitignored=False)
    action, msg = m2.generate_mcp_generic(wt2, m2.repo_slug(wt2))
    check("mcp: REFUSES to write when .mcp.json is not gitignored",
          action == "refused-not-ignored" and not (wt2 / ".mcp.json").exists(), msg[:70])

    # A project with no entry must be silent, not an error.
    m3, base3, wt3 = _mcp_fixture(tmp, "GiulioDER/unlisted", [])
    action, _ = m3.generate_mcp_generic(wt3, m3.repo_slug(wt3))
    check("mcp: a project with no configured servers writes nothing",
          action == "none-configured" and not (wt3 / ".mcp.json").exists(), action)


# ------------------------------------------------- second audit, 2026-08-16 (post-merge)
def test_concurrent_takeover_of_a_stale_claim_has_one_winner():
    """AUDIT-2 BUG-001: the O_EXCL fix covered only the UNCLAIMED case.

    Racing a STALE claim went through os.replace, which is atomic but not
    exclusive, so both sessions were told they had won, 3 trials out of 3.
    """
    base, wt = new_repo("audit2-001")
    cf = claim_path(wt)
    for trial in range(3):
        cf.write_text("session=GONE\npid=999999999\nclaimed_epoch=1\n", encoding="utf-8")

        def attempt(sid):
            return report(load(), wt, session=sid)[0]

        with ThreadPoolExecutor(max_workers=2) as ex:
            outs = list(ex.map(attempt, [f"A{trial}", f"B{trial}"]))
        winners = sum(1 for o in outs if "Workspace claimed for this session" in o)
        if winners != 1:
            check(f"BUG-001 stale-claim race has one winner (trial {trial})", False,
                  f"winners={winners}")
            return
    check("BUG-001 stale-claim race has exactly one winner (3 trials)", True)


def test_unparseable_claim_does_not_wedge_the_worktree():
    """AUDIT-2 BUG-002: `existed = bool(claim)` wedged a worktree permanently.

    A zero-byte claim took the create path, the create failed EEXIST because the
    file was right there, and every later session was refused with a permissions
    diagnosis that no permission change could clear. The hook produces exactly
    that file if it is killed between the create and the write.
    """
    m = load()
    base, wt = new_repo("audit2-002")
    cf = claim_path(wt)
    for label, content in (("zero-byte", b""), ("garbage", b"garbage\n"), ("blank", b"   \n")):
        cf.write_bytes(content)
        out, st = report(m, wt, session="ME")
        ok = "WORKSPACE REFUSED" not in out and st.get("outcome") == "claimed"
        if not ok:
            check(f"BUG-002 an unparseable claim ({label}) does not wedge", False,
                  out.splitlines()[0][:90])
            return
    check("BUG-002 zero-byte/garbage/blank claims all recover", True)


def test_ignore_check_that_cannot_run_is_not_reported_as_unignored():
    """AUDIT-2 BUG-004: could-not-tell was collapsed into 'not ignored'."""
    tmp = SCRATCH / "mcpcfg2"
    tmp.mkdir(parents=True, exist_ok=True)
    m, base, wt = _mcp_fixture(tmp, "GiulioDER/sentiment-agent", ["qwen-mcp"])
    real_git = m.git
    m.git = lambda root, *a, **k: (
        (m.LAUNCH_FAILED, "") if a[:1] == ("check-ignore",) else real_git(root, *a, **k))
    action, msg = m.generate_mcp_generic(wt, "GiulioDER/sentiment-agent")
    ok = action == "ignore-check-failed" and not (wt / ".mcp.json").exists()
    check("BUG-004 an unrunnable ignore check is not reported as 'not gitignored'", ok,
          f"{action}: {msg[:70]}")


def test_malformed_config_cannot_erase_the_workspace_verdict():
    """AUDIT-2 BUG-005: wrong-shape JSON raised and replaced a REFUSED verdict."""
    tmp = SCRATCH / "mcpcfg3"
    tmp.mkdir(parents=True, exist_ok=True)
    for label, policy, secrets in (
        ("policy is a list", "[]", '{"servers":{}}'),
        ("projects is a list", '{"projects":[]}', '{"servers":{}}'),
        ("secrets is a list", '{"projects":{"GiulioDER/x":["qwen-mcp"]}}', "[]"),
        ("server entry is a string",
         '{"projects":{"GiulioDER/x":["qwen-mcp"]}}', '{"servers":{"qwen-mcp":"nope"}}'),
    ):
        m, base, wt = _mcp_fixture(tmp, "GiulioDER/x", ["qwen-mcp"])
        m.MCP_POLICY.write_text(policy, newline="\n")
        m.MCP_SECRETS.write_text(secrets, newline="\n")
        try:
            action, _ = m.generate_mcp_generic(wt, "GiulioDER/x")
        except Exception as exc:
            check(f"BUG-005 malformed config ({label}) does not raise", False,
                  f"{type(exc).__name__}: {exc}")
            return
        if action == "generated":
            check(f"BUG-005 malformed config ({label}) writes nothing", False, action)
            return
    check("BUG-005 four malformed config shapes all degrade without raising", True)


def test_repo_slug_rejects_a_local_path_origin():
    """AUDIT-2 BUG-008: any path yielded a plausible slug, which selects servers."""
    m = load()
    base, wt = new_repo("audit2-008")
    bad = ["C:/Users/gde00/Documents/recall", "/c/Users/gde00/Documents/sentiment_agent",
           "file:///c/repos/recall", "../sibling/repo"]
    good = {"https://github.com/GiulioDER/RE-call.git": "GiulioDER/RE-call",
            "git@github.com:GiulioDER/sentiment-agent.git": "GiulioDER/sentiment-agent"}
    for url in bad:
        sh("git", "remote", "remove", "origin", cwd=base)
        sh("git", "remote", "add", "origin", url, cwd=base)
        got = m.repo_slug(wt)
        if got is not None:
            check("BUG-008 a local-path origin yields no slug", False, f"{url} -> {got}")
            return
    for url, want in good.items():
        sh("git", "remote", "remove", "origin", cwd=base)
        sh("git", "remote", "add", "origin", url, cwd=base)
        got = m.repo_slug(wt)
        if got != want:
            check("BUG-008 control: a real hosting URL still resolves", False, f"{url} -> {got}")
            return
    check("BUG-008 local paths rejected, hosting URLs still resolve", True)


# ---------------------------------------------------------------- core behaviour
def live_windows_pid():
    """A pid that is certainly alive, in the numbering tasklist understands.

    NOT os.getpid() under Git Bash, and not $$: those are MSYS pids that
    tasklist cannot see, which silently turns "is this claim live?" into "no".
    Python's own pid IS a Windows pid when run by the Windows interpreter.
    """
    return str(os.getpid()) if os.name == "nt" else str(os.getpid())


def test_core_paths():
    """Built from throwaway repos, so this runs in any checkout on any machine."""
    m = load()
    base, wt = new_repo("core")

    out, st = report(m, base, session="anyone")
    check("core: MAIN checkout is refused",
          "WORKSPACE REFUSED" in (out or "") and "MAIN checkout" in (out or ""), seen(out, st))

    out, st = report(m, wt, session="S1")
    check("core: an unclaimed worktree is accepted",
          "Workspace claimed for this session" in (out or ""), seen(out, st))

    out, st = report(m, wt, session="S2")
    check("core: a worktree held by a live session is refused",
          "WORKSPACE REFUSED" in (out or "") and "CLAIMED by another session" in (out or ""),
          seen(out, st))

    # Red proof, 2026-09-24: mutating builtin_guard's
    # `if holder and holder != session_id and not claim_is_stale(claim)` to
    # `if holder and not claim_is_stale(claim)` fails THIS check with
    # outcome='refused' out='WORKSPACE REFUSED. ...' (it used to raise TypeError
    # on None instead, which hid which of the two it was).
    out, st = report(m, wt, session="S1")
    check("core: the holder is still accepted on a second start",
          "Workspace claimed for this session" in (out or "") and st.get("outcome") == "claimed",
          seen(out, st))

    out, st = report(m, wt, session="S1", source="compact")
    check("core: compaction is silent",
          out is None and st.get("outcome") == "skipped-compact", seen(out, st))

    out, st = report(m, tempfile.gettempdir(), session="x")
    check("core: a non-repo is silent",
          out is None and st.get("outcome") == "not-a-git-repo", seen(out, st))


def test_git_that_never_answers_is_not_silent():
    """A git call that did not complete is not "this is not a repository".

    Invariant: when git never answers (timeout, spent budget, not launchable)
    the hook says the workspace is UNVERIFIED. The failure mode: it logged
    `not-a-git-repo` and printed NOTHING, so a session in a worktree someone
    else holds was told nothing either. Found through CI run 36000505874, where
    the shared test budget ran out before the holder's second start.

    Red proof, 2026-09-24, node `session_start_hook_tests.py::
    test_git_that_never_answers_is_not_silent`:
    - baseline: the pre-fix hook at 87ff3093 fails the first check with
      outcome='not-a-git-repo' out=None (its second check cannot run there,
      since `git_full` does not exist yet, so the mutation below proves it);
    - mutation: deleting the `if rc == LAUNCH_FAILED: return unverified(...)`
      after `--show-toplevel` in build_report fails the first check the same way;
    - mutation: deleting the same branch after `--absolute-git-dir` fails the
      second check with outcome='no-git-dir' out=None.
    """
    m = load()
    base, wt = new_repo("git-never-answers")

    # Spent budget: run() refuses to launch anything, as on the CI runner.
    m._STARTED = time.monotonic() - m.BUDGET - 1
    state = {}
    out = m.build_report({"session_id": "S1", "cwd": str(wt), "source": "startup"}, state)
    check("git that never answers (show-toplevel) is reported, not silent",
          out is not None and "UNVERIFIED" in out and state.get("outcome") == "git-unavailable",
          seen(out, state))

    # The second probe on its own: only --absolute-git-dir fails to launch.
    real = m.git_full

    def flaky(cwd, *args, **kw):
        if "--absolute-git-dir" in args:
            return m.LAUNCH_FAILED, "", "timed out"
        return real(cwd, *args, **kw)

    m.git_full = flaky
    try:
        out, st = report(m, wt, session="S1")
    finally:
        m.git_full = real
    check("git that never answers (absolute-git-dir) is reported, not silent",
          out is not None and "UNVERIFIED" in out and st.get("outcome") == "git-unavailable",
          seen(out, st))

    # Control: with git answering, a real non-repo is still silent, so the
    # checks above cannot pass for a hook that simply always speaks.
    out, st = report(m, tempfile.gettempdir(), session="x")
    check("git that never answers, control: an answered non-repo stays silent",
          out is None and st.get("outcome") == "not-a-git-repo", seen(out, st))


# ------------------------------------------- ported from the deployed copy, 2026-09-24
def test_a_demonstrably_dead_holder_waits_minutes_not_hours():
    """A recorded pid that Windows says is gone releases after the short grace.

    Invariant: a claim whose pid is RECORDED and not alive is stale after
    `dead_grace_minutes()` (15 by default), not after `stale_hours()` (12). A
    claim with NO pid still waits the full twelve hours, because there the
    timestamp is the only evidence. The failure mode: a worktree refused for
    eight more hours behind a holder already known to be dead (audit of
    2026-08-16), which is what the source did before this port.

    Red proof, 2026-09-24, node `session_start_hook_tests.py::
    test_a_demonstrably_dead_holder_waits_minutes_not_hours`:
    - baseline: the pre-port source at 94ec62b3 fails "a dead holder 20 minutes
      old is taken over" with outcome='refused' out='WORKSPACE REFUSED. ...';
    - mutation: `return age > dead_grace_minutes() * 60` to
      `return age > stale_hours() * 3600` in claim_is_stale fails the same check
      the same way;
    - mutation: `if pid:` to `if True:` in claim_is_stale fails "a claim with NO
      pid still waits the full timeout" with stale=True.
    """
    m = load()
    base, wt = new_repo("dead-grace")
    cf = claim_path(wt)
    now = int(time.time())

    cf.write_text(f"session=GONE\npid=999999999\nclaimed_epoch={now - 20 * 60}\n",
                  encoding="utf-8")
    out, st = report(m, wt, session="ME")
    check("dead grace: a dead holder 20 minutes old is taken over",
          "took over a stale claim" in (out or "") and st.get("outcome") == "claimed",
          seen(out, st))

    # Control: inside the grace the same dead holder still holds, so the check
    # above cannot pass for a guard that takes over every dead claim at once.
    cf.write_text(f"session=GONE\npid=999999999\nclaimed_epoch={now - 5 * 60}\n",
                  encoding="utf-8")
    out, st = report(m, wt, session="ME2")
    check("dead grace, control: a dead holder 5 minutes old is still refused",
          "WORKSPACE REFUSED" in (out or ""), seen(out, st))

    # No pid recorded: process_alive("") is False meaning "nothing recorded",
    # not "dead", so the grace must NOT apply.
    stale = m.claim_is_stale({"session": "X", "claimed_epoch": str(now - 20 * 60)})
    check("dead grace: a claim with NO pid still waits the full timeout",
          stale is False, f"stale={stale}")


def test_dead_grace_minutes_is_parsed_defensively():
    """An unparseable or zero grace must neither crash the hook nor disable the grace.

    Red proof, 2026-09-24: mutating dead_grace_minutes' `return max(1, int(raw))`
    to `return int(raw)` fails this check with {'15m': 15, '0': 0, '7': 7}. The
    pre-port source has no dead_grace_minutes, so it cannot be the baseline.
    """
    m = load()
    got = {}
    old = os.environ.get("RECALL_CLAIM_DEAD_GRACE_MINUTES")
    try:
        for raw in ("15m", "0", "7"):
            os.environ["RECALL_CLAIM_DEAD_GRACE_MINUTES"] = raw
            try:
                got[raw] = m.dead_grace_minutes()
            except ValueError as exc:
                got[raw] = f"raised {exc}"
    finally:
        if old is None:
            os.environ.pop("RECALL_CLAIM_DEAD_GRACE_MINUTES", None)
        else:
            os.environ["RECALL_CLAIM_DEAD_GRACE_MINUTES"] = old
    check("dead grace: '15m' falls back to 15, '0' clamps to 1, '7' is 7",
          got == {"15m": 15, "0": 1, "7": 7}, str(got))


def test_stdio_servers_are_stamped_with_positive_identity():
    """Every stdio server line carries this checkout's mark and a safe session id.

    Invariant: `{client_mark}` becomes `<hostname>-<sha256(root)[:8]>`, and
    `{session_id}` becomes the session id only when it matches
    `_SAFE_SESSION_ID`; anything else becomes a fresh `recall-session-<hex>`.
    The failure mode: an unstamped server that no session-end cleanup can close
    by positive identity, or a payload-controlled string spliced into a command.

    Red proof, 2026-09-24 (render_stdio is new, so by mutation only):
    - `if session_id and _SAFE_SESSION_ID.match(session_id)` to `if session_id`
      fails "an unsafe session id is replaced" with
      RECALL_MCP_SESSION_ID=x; rm -rf ~;
    - `.replace("{client_mark}", mark)` to `.replace("{client_mark}", "")` fails
      "the client mark is the host and a hash of the root" with
      RECALL_MCP_CLIENT= (and the exact-dict check beside it).
    """
    import hashlib
    import socket
    m = load()
    root = SCRATCH / "stdio-root"
    tmpl = {"command": "ssh", "args": ["h", "RECALL_MCP_CLIENT={client_mark}",
                                       "RECALL_MCP_SESSION_ID={session_id}"]}
    got = m.render_stdio(tmpl, root, "abc-123")
    mark = f"{socket.gethostname()}-{hashlib.sha256(str(root).encode()).hexdigest()[:8]}"
    check("stdio: the client mark is the host and a hash of the root",
          got["args"][1] == f"RECALL_MCP_CLIENT={mark}", got["args"][1])
    check("stdio: a safe session id is stamped as given",
          got == {"type": "stdio", "command": "ssh",
                  "args": ["h", f"RECALL_MCP_CLIENT={mark}", "RECALL_MCP_SESSION_ID=abc-123"]},
          str(got))

    bad = m.render_stdio(tmpl, root, "x; rm -rf ~")
    sid = bad["args"][2].split("=", 1)[1]
    check("stdio: an unsafe session id is replaced, never spliced in",
          sid.startswith("recall-session-") and "rm" not in sid, bad["args"][2])


def test_stdio_servers_need_no_secrets_file_and_are_approved():
    """A policy `stdio` entry is written without the secrets file, then approved.

    Invariant: a server defined in the policy's `stdio` table carries no secret,
    so a missing secrets file must not stop it, and the file just written is
    handed to `session_mcp_approve.py --root <root> --from-mcp-json <file>`.
    The failure mode: "no secrets file" silently meaning "no recall-memory",
    and a written server left pending approval, which a session sees as no tools.

    Red proof, 2026-09-24, node `session_start_hook_tests.py::
    test_stdio_servers_need_no_secrets_file_and_are_approved`. The pre-port
    source is NOT valid proof here: it rejects the third argument with a
    TypeError, which is a signature, not a behaviour. So by mutation:
    - `if not servers: return "no-secrets"` to `if True: ...` fails "a stdio
      server is written without a secrets file" with action 'no-secrets';
    - `servers[name] = render_stdio(...)` to `pass` fails the same check the
      same way;
    - deleting `note += f"; {approve_mcp(root)}"` fails "the written file is
      handed to the approval script" with calls=[];
    - `return "approved" if rc == 0` to `if True` fails "a failed approval says
      FAILED" (the message ends '; approved').
    """
    tmp = SCRATCH / "mcpstdio"
    tmp.mkdir(parents=True, exist_ok=True)
    m, base, wt = _mcp_fixture(tmp, "GiulioDER/RE-call", ["recall-memory", "qwen-mcp"])
    m.MCP_POLICY.write_text(json.dumps({
        "projects": {"GiulioDER/RE-call": ["recall-memory", "qwen-mcp"]},
        "stdio": {"recall-memory": {"command": "ssh",
                                    "args": ["vps", "RECALL_MCP_CLIENT={client_mark}"]}},
    }), newline="\n")
    m.MCP_SECRETS = tmp / "no-such-secrets.json"
    approve = tmp / "approve.py"
    approve.write_text("# stand-in; never executed, run() is stubbed\n")
    m.MCP_APPROVE = approve
    real_run = m.run
    calls = []

    def stub(rc, err=""):
        def fake_run(args, **kw):
            if str(approve) in [str(a) for a in args]:
                calls.append([str(a) for a in args])
                return rc, "", err
            return real_run(args, **kw)
        return fake_run

    m.run = stub(0)
    action, msg = m.generate_mcp_generic(wt, "GiulioDER/RE-call", "S-1")
    mcp_json = wt / ".mcp.json"
    written = json.loads(mcp_json.read_text(encoding="utf-8"))["mcpServers"] if (
        mcp_json.exists()) else {}
    check("stdio: a stdio server is written without a secrets file",
          action == "generated" and written.get("recall-memory", {}).get("type") == "stdio"
          and "qwen-mcp" not in written, f"{action}: {msg[:90]}")
    want_tail = ["--root", str(wt), "--from-mcp-json", str(mcp_json)]
    check("stdio: the written file is handed to the approval script",
          len(calls) == 1 and calls[0][-4:] == want_tail and msg.endswith("; approved"),
          f"calls={calls} msg=...{msg[-40:]!r}")

    # A failing approval is reported as a failure, never as approved.
    m.run = stub(2, "boom")
    mcp_json.unlink(missing_ok=True)
    action, msg = m.generate_mcp_generic(wt, "GiulioDER/RE-call", "S-1")
    check("stdio: a failed approval says FAILED",
          action == "generated" and msg.endswith("approval FAILED: boom"), msg[-60:])

    # A missing approval script is named, not silently skipped.
    m.run = real_run
    m.MCP_APPROVE = tmp / "no-such-approve.py"
    mcp_json.unlink(missing_ok=True)
    action, msg = m.generate_mcp_generic(wt, "GiulioDER/RE-call", "S-1")
    check("stdio: a missing approval script is reported",
          action == "generated" and "not approved:" in msg and msg.endswith("is missing"),
          msg[-80:])


# ---------------------------------------- per-launch identity for stdio servers, 2026-09-26
LAUNCH_ENV = ("RECALL_MCP_LAUNCH_SHELL", "RECALL_MCP_SSH", "GIT_BASH")
#: The production shape of the policy's `recall-memory` entry, with the host and paths made up.
REMOTE = ("cd ~/srv && export RECALL_TENANT=memory RECALL_MCP_CLIENT={client_mark} && "
          "RECALL_MCP_SESSION_ID={session_id} && exec ~/venv/bin/python -m recall_mcp.server")


@contextmanager
def launch_env(**values):
    """Set or clear the launch overrides for one block, and put them back after."""
    saved = {k: os.environ.get(k) for k in LAUNCH_ENV}
    try:
        for k in LAUNCH_ENV:
            os.environ.pop(k, None)
        for k, v in values.items():
            os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _stdio_policy(m, slug, tmpl):
    m.MCP_POLICY.write_text(json.dumps({"projects": {slug: ["recall-memory"]},
                                        "stdio": {"recall-memory": tmpl}}), newline="\n")


def test_ssh_stdio_servers_are_launched_with_a_launch_id():
    """A policy ssh server is written through the shared launch line, stamps intact.

    Invariant: `generate_mcp_generic` writes an ssh stdio server as
    `<shell> -c MCP_LAUNCH recall-mcp-launch <ssh> <host> <remote>`, with the
    remote command still carrying `RECALL_MCP_CLIENT` and `RECALL_MCP_SESSION_ID`
    (which both session-end cleanups read back out of `.mcp.json`). The failure
    mode: a server with no launch ID, which the sweep judges by the per-checkout
    identity, so a leaked one reads as held while any session of the project is
    open.

    Red proof, 2026-09-26, node `session_start_hook_tests.py::
    test_ssh_stdio_servers_are_launched_with_a_launch_id`:
    - baseline: the pre-change `scripts/session_start_hook.py` at abfc1369
      FAILED "the server runs through the launch shell and the shared launch
      line" with command='ssh', args ['-o', 'BatchMode=yes', 'vps2', ...]. The
      stamps and quiet-report checks passed there, as they should: they pin what
      must survive the change;
    - mutation: `servers[name], why = with_launch_id(render_stdio(...))` to
      `servers[name], why = render_stdio(...), ""` fails the same check the same
      way.
    """
    tmp = SCRATCH / "mcplaunch"
    tmp.mkdir(parents=True, exist_ok=True)
    slug = "GiulioDER/agent-memory-bench"
    m, base, wt = _mcp_fixture(tmp, slug, ["recall-memory"])
    _stdio_policy(m, slug, {"command": "ssh", "args": ["-o", "BatchMode=yes", "vps2", REMOTE]})
    with launch_env(RECALL_MCP_LAUNCH_SHELL="C:/fake/bash.exe",
                    RECALL_MCP_SSH="C:/fake/ssh.exe"):
        action, msg = m.generate_mcp_generic(wt, slug, "S-1")
    mcp_json = wt / ".mcp.json"
    got = json.loads(mcp_json.read_text(encoding="utf-8"))["mcpServers"].get("recall-memory", {})
    args = got.get("args", [])
    check("stdio launch: the server runs through the launch shell and the shared launch line",
          action == "generated" and got.get("command") == "C:/fake/bash.exe"
          and args[:3] == ["-c", m.MCP_LAUNCH, "recall-mcp-launch"]
          and args[3:5] == ["C:/fake/ssh.exe", "vps2"] and len(args) == 6,
          f"{action}: command={got.get('command')!r} args[:5]={args[:5]!r}")
    remote = args[-1] if args else ""
    check("stdio launch: the remote command keeps its stamps, the session id readable back",
          remote.startswith("cd ~/srv && ") and "RECALL_MCP_CLIENT=" in remote
          and "{client_mark}" not in remote
          and m.mcp_config_session_id(mcp_json) == "S-1", remote[:120])
    check("stdio launch: a wrapped server adds nothing to the report", "no launch ID" not in msg,
          msg[-120:])


def test_the_generated_launch_mints_an_id_the_sweep_can_read():
    """Run the written command for real: each launch gets its own ID, in the sweep's form.

    Invariant: executing the entry `with_launch_id` returns, under the bash
    `launch_shell()` really resolves, hands ssh `-o BatchMode=yes <host>` and a
    remote command beginning `export RECALL_MCP_LAUNCH_ID=<digits>-<digits>-<digits>`,
    a different ID on every launch, which `session_mcp_sweep.local_launch_ids`
    (the consumer that decides held or orphaned) reads back. ssh is a stub that
    prints its argv, so nothing leaves this machine. The failure mode: a launch
    line that is written but not expanded, or not unique, which looks the same
    in `.mcp.json` and leaves every server unrecognisable to the sweep.

    Red proof, 2026-09-26, node `session_start_hook_tests.py::
    test_the_generated_launch_mints_an_id_the_sweep_can_read`, by mutating
    `MCP_LAUNCH`, the production line that mints the ID:
    - the export single-quoted (`'export RECALL_MCP_LAUNCH_ID=... && '"$3"`), so
      the ID is not expanded locally, fails "each launch carries exactly one ID
      the sweep reads" with ids=[set(), set()]: the unexpanded
      `${EPOCHSECONDS:-0}` is not digits, which is the point of the sweep's
      pattern;
    - the ID replaced by the constant `1-1-1` fails "two launches get two
      different IDs" with [{'1-1-1'}, {'1-1-1'}];
    - `-o BatchMode=yes` deleted from the line fails "ssh gets BatchMode, the
      host, and the remote command intact" with the argv starting 'vps2 export'.
    """
    m = load()
    shell = m.launch_shell()
    check("stdio launch: a bash is resolved on this machine, and it is not System32's WSL",
          bool(shell) and "system32" not in shell.lower(), repr(shell))
    if not shell:
        return
    fake = SCRATCH / "fake-ssh"
    fake.write_text("#!/bin/sh\nprintf '%s\\n' \"$*\"\n", newline="\n")
    tmpl = {"command": "ssh", "args": ["-o", "BatchMode=yes", "vps2", REMOTE]}
    with launch_env(RECALL_MCP_SSH=fake.as_posix()):
        entry, why = m.with_launch_id(m.render_stdio(tmpl, SCRATCH / "launch-root", "S-2"))
    outs = []
    for _ in range(2):
        r = subprocess.run([entry["command"], *entry["args"]], capture_output=True, text=True,
                           timeout=30, stdin=subprocess.DEVNULL)
        outs.append(r.stdout.strip())
    spec = importlib.util.spec_from_file_location(
        "sweep", str(Path(HOOK).with_name("session_mcp_sweep.py")))
    sweep = importlib.util.module_from_spec(spec)
    sys.modules["sweep"] = sweep  # dataclasses look their module up while it executes
    spec.loader.exec_module(sweep)
    ids = [sweep.local_launch_ids(o) for o in outs]
    check("stdio launch: each launch carries exactly one ID the sweep reads",
          why == "" and all(len(i) == 1 for i in ids), f"why={why!r} ids={ids} out={outs[0][:90]!r}")
    check("stdio launch: two launches get two different IDs",
          len(ids[0] | ids[1]) == 2, str(ids))
    check("stdio launch: ssh gets BatchMode, the host, and the remote command intact",
          all(o.startswith("-o BatchMode=yes vps2 export RECALL_MCP_LAUNCH_ID=")
              and " && cd ~/srv && export RECALL_TENANT=memory RECALL_MCP_CLIENT=" in o
              and o.endswith("RECALL_MCP_SESSION_ID=S-2 && exec ~/venv/bin/python -m "
                             "recall_mcp.server")
              for o in outs), outs[0][:160])


def test_windows_never_guesses_a_bare_bash():
    """On Windows only a Git for Windows bash qualifies; none found means unwrapped, and said.

    Invariant: with no override, `launch_shell()` on Windows is `find_bash()`
    (which refuses System32) mapped from Git's `bin/bash.exe` launcher to the
    real `usr/bin/bash.exe` beside it, never whatever a bare `bash` resolves to,
    and None when `find_bash()` finds nothing. `generate_mcp_generic` then
    writes the policy entry unchanged and names the reason in its report. The
    failure mode: ssh run inside WSL with none of this machine's keys, which a
    client shows as a server with no tools.

    Red proof, 2026-09-26, node `session_start_hook_tests.py::
    test_windows_never_guesses_a_bare_bash`, by mutating `launch_shell()`:
    - `bash = find_bash()` to `bash = shutil.which("bash")` (the obvious
      "simplification") fails "no Git Bash means None, never the WSL bash on
      PATH" with 'C:/Windows/System32/bash.exe', and the unwrapped-server checks
      with the server written as that WSL bash;
    - the `bin` to `usr/bin` mapping disabled fails "Git's bin/bash.exe launcher
      is mapped to the real usr/bin/bash.exe" with '.../fake-git/bin/bash.exe';
    - `if unlaunched:` to `if False:` in `generate_mcp_generic` fails "an
      unwrapped server is reported", the report naming no launch ID.
    """
    if os.name != "nt":
        print("SKIP  windows bash resolution: not Windows")
        return
    m = load()
    git_root = SCRATCH / "fake-git"
    for rel in ("usr/bin/bash.exe", "bin/bash.exe"):
        (git_root / rel).parent.mkdir(parents=True, exist_ok=True)
        (git_root / rel).write_text("")
    with launch_env(GIT_BASH=str(git_root / "bin" / "bash.exe")):
        found = m.launch_shell()
    check("stdio launch: Git's bin/bash.exe launcher is mapped to the real usr/bin/bash.exe",
          found == (git_root / "usr" / "bin" / "bash.exe").as_posix(), repr(found))

    real_which = m.shutil.which
    wsl = "C:/Windows/System32/bash.exe"
    try:
        m.shutil.which = lambda name, *a, **k: wsl if name == "bash" else real_which(name)
        m.find_bash = lambda: None
        with launch_env():
            none = m.launch_shell()
        check("stdio launch: no Git Bash means None, never the WSL bash on PATH",
              none is None, repr(none))

        tmp = SCRATCH / "mcpnobash"
        tmp.mkdir(parents=True, exist_ok=True)
        slug = "GiulioDER/agent-memory-bench"
        m2, base, wt = _mcp_fixture(tmp, slug, ["recall-memory"])
        m2.find_bash = lambda: None
        _stdio_policy(m2, slug, {"command": "ssh",
                                 "args": ["-o", "BatchMode=yes", "vps2", REMOTE]})
        with launch_env():
            action, msg = m2.generate_mcp_generic(wt, slug, "S-1")
        got = json.loads((wt / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]
        check("stdio launch: with no Git Bash the server is written as the policy wrote it",
              got["recall-memory"]["command"] == "ssh"
              and got["recall-memory"]["args"][:3] == ["-o", "BatchMode=yes", "vps2"],
              str(got["recall-memory"])[:100])
        check("stdio launch: an unwrapped server is reported",
              "no launch ID" in msg and "no Git Bash found" in msg, msg[-140:])
    finally:
        m.shutil.which = real_which


def test_ssh_options_the_wrapper_would_drop_leave_the_server_as_written():
    """An ssh entry `MCP_LAUNCH` cannot express is left alone and named; a non-ssh one is silent.

    Invariant: `MCP_LAUNCH` passes only `-o BatchMode=yes`, so an entry with
    any other ssh option is returned unchanged with a reason, and a non-ssh
    command is returned unchanged with no report. The failure mode: a `-p 2222`
    or `-i key` silently dropped, which connects to the wrong port or with the
    wrong key and looks like a dead server.

    Red proof, 2026-09-26, node `session_start_hook_tests.py::
    test_ssh_options_the_wrapper_would_drop_leave_the_server_as_written`, by
    disabling the options check in `with_launch_id()` (`if False and any(...)`):
    fails "an ssh option the wrapper cannot pass keeps the entry as written"
    with the entry rewritten to the launch shell, `-p 2222` gone and why=''.
    """
    m = load()
    with launch_env(RECALL_MCP_LAUNCH_SHELL="C:/fake/bash.exe", RECALL_MCP_SSH="C:/fake/ssh.exe"):
        odd = {"type": "stdio", "command": "ssh", "args": ["-p", "2222", "vps2", "cmd"]}
        got, why = m.with_launch_id(dict(odd))
        check("stdio launch: an ssh option the wrapper cannot pass keeps the entry as written",
              got == odd and "-p" in why, f"{got} why={why!r}")
        plain = {"type": "stdio", "command": "python", "args": ["-m", "srv"]}
        got, why = m.with_launch_id(dict(plain))
        check("stdio launch: a non-ssh server is untouched", got == plain and why ==
              "not an ssh server", f"{got} why={why!r}")
        bare = {"type": "stdio", "command": "ssh", "args": ["vps2", "cmd"]}
        got, why = m.with_launch_id(dict(bare))
        check("stdio launch: an entry with no options still gets the launch line",
              why == "" and got["args"][3:] == ["C:/fake/ssh.exe", "vps2", "cmd"], str(got))


def test_bare_ssh_is_the_system_openssh_on_windows():
    """The wrapper runs the ssh the client would have run, and keeps an explicit one.

    Invariant: on Windows, `launch_ssh("ssh")` is System32's OpenSSH when it
    exists, and an explicit path is returned as written. The failure mode: the
    wrapper, which runs inside Git Bash, picking up Git's own ssh, with a
    different config, agent and known_hosts from every launch before it.

    Red proof, 2026-09-26, node `session_start_hook_tests.py::
    test_bare_ssh_is_the_system_openssh_on_windows`, by disabling the Windows
    branch of `launch_ssh()` (`if False:`): fails "a bare ssh is System32's
    OpenSSH" with 'ssh'.
    """
    if os.name != "nt":
        print("SKIP  system OpenSSH: not Windows")
        return
    m = load()
    system = Path(os.environ.get("SystemRoot") or r"C:\Windows") / "System32" / "OpenSSH" / "ssh.exe"
    if not system.is_file():
        print(f"SKIP  system OpenSSH: {system} is not installed here")
        return
    with launch_env():
        bare, explicit = m.launch_ssh("ssh"), m.launch_ssh("D:/tools/ssh.exe")
    check("stdio launch: a bare ssh is System32's OpenSSH", bare == system.as_posix(), repr(bare))
    check("stdio launch: an explicit ssh is kept as written", explicit == "D:/tools/ssh.exe",
          repr(explicit))


def test_session_mcp_sh_imports_the_launch_line_rather_than_copying_it():
    """Structural control, not a behaviour test: one definition, two generators.

    `scripts/session-mcp.sh` must build its servers with this hook's
    `launch_stdio` and hold no copy of the launch line, or the two generators
    can drift into launching servers the sweep reads differently. The behaviour
    itself was checked by running both versions of `session-mcp.sh` and
    comparing the `.mcp.json` they wrote: byte-identical on 2026-09-26. Goes red
    if the old `LAUNCH = (...)` block is restored (checked that day).
    """
    text = Path(HOOK).with_name("session-mcp.sh").read_text(encoding="utf-8")
    check("stdio launch: session-mcp.sh imports launch_stdio and keeps no copy of the line",
          "from session_start_hook import launch_stdio" in text
          and "RECALL_MCP_LAUNCH_ID=${EPOCHSECONDS" not in text
          and "return launch_stdio(" in text)


def test_a_cwd_that_is_not_a_directory_is_its_own_outcome():
    """A cwd Python cannot resolve is neither "not a repository" nor "git broke".

    Invariant: a payload cwd that is not a directory logs `cwd-not-a-directory`
    and stays silent. The failure mode, on the base this ports onto: git cannot
    launch in a missing directory, so without this check the row is decided by
    how that launch fails, and the difference between a bad path and a broken
    git is lost (hit in the 2026-08-16 audit by a POSIX-style path on Windows).

    Red proof, 2026-09-24: the pre-port source at 94ec62b3 fails this check with
    outcome='git-unavailable' out='The workspace guard could not run to
    completion ...', so a bad path told the session its workspace was
    UNVERIFIED; mutating `if not os.path.isdir(cwd):` to `if False:` fails it
    the same way.
    """
    m = load()
    missing = SCRATCH / "no" / "such" / "directory"
    out, st = report(m, missing, session="x")
    check("a cwd that is not a directory is logged as such and stays silent",
          out is None and st.get("outcome") == "cwd-not-a-directory", seen(out, st))


def test_deployed_copy_matches_this_source(deployed=DEPLOYED):
    """The deployed hook is the one that runs; this file is only its source.

    Before 2026-09-24 they had drifted in BOTH directions: the deployed copy had
    the dead-pid grace, stdio servers, approval and `cwd-not-a-directory`, which
    the source lacked, while the source had the session-id refresh, which the
    deployed copy lacked. Nothing reported either.

    Skipped where nothing is deployed, which is every CI runner. On failure it
    says which way the drift runs, because the repair differs: a line only in
    the deployed copy must be PORTED here first, since copying this file over it
    would delete live behaviour.

    Red proof, 2026-09-24: against the real deployed copy on the day of the
    port, it failed with "28 line(s) only in the source, 5 only in the deployed
    copy" (the session-id refresh the deployed copy still lacked). Green against
    a byte copy of this source; SKIP against a path that does not exist.
    """
    if not deployed.exists():
        print(f"SKIP  deployed copy matches the source: {deployed} does not exist on this "
              "machine, as on every CI runner, so there is nothing to compare")
        return
    have = deployed.read_text(encoding="utf-8")
    want = Path(HOOK).read_text(encoding="utf-8")
    same = have == want
    detail = ""
    if not same:
        import difflib
        diff = list(difflib.unified_diff(want.splitlines(), have.splitlines(), lineterm="", n=0))
        only_src = sum(1 for d in diff if d.startswith("-") and not d.startswith("---"))
        only_dep = sum(1 for d in diff if d.startswith("+") and not d.startswith("+++"))
        # A changed line counts on both sides, so a non-zero deployed count is
        # not proof of a deployed-only feature; it is the prompt to read the diff.
        detail = (f"{deployed} differs from {HOOK}: {only_src} line(s) only in the source, "
                  f"{only_dep} only in the deployed copy. Read the diff first: port anything "
                  "the deployed copy does that the source does not, then copy the source over it")
    check("the deployed hook matches this source", same, detail)


if __name__ == "__main__":
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for fn in [
        test_core_paths,
        test_git_that_never_answers_is_not_silent,
        test_live_pid_unreadable_epoch_is_not_stale,
        test_concurrent_claim_has_one_winner,
        test_timeout_bounds_wall_clock,
        test_silent_nonzero_script_is_a_refusal,
        test_non_utf8_claim_does_not_silence_the_hook,
        test_bad_env_var_does_not_crash_at_import,
        test_cannot_tell_liveness_is_alive,
        test_non_object_payload_is_logged_not_silent,
        test_foreign_session_space_falls_back,
        test_rules_drift,
        test_mcp_generic_generator,
        test_concurrent_takeover_of_a_stale_claim_has_one_winner,
        test_unparseable_claim_does_not_wedge_the_worktree,
        test_ignore_check_that_cannot_run_is_not_reported_as_unignored,
        test_malformed_config_cannot_erase_the_workspace_verdict,
        test_repo_slug_rejects_a_local_path_origin,
        test_rules_drift_follows_imports,
        test_a_demonstrably_dead_holder_waits_minutes_not_hours,
        test_dead_grace_minutes_is_parsed_defensively,
        test_stdio_servers_are_stamped_with_positive_identity,
        test_stdio_servers_need_no_secrets_file_and_are_approved,
        test_ssh_stdio_servers_are_launched_with_a_launch_id,
        test_the_generated_launch_mints_an_id_the_sweep_can_read,
        test_windows_never_guesses_a_bare_bash,
        test_ssh_options_the_wrapper_would_drop_leave_the_server_as_written,
        test_bare_ssh_is_the_system_openssh_on_windows,
        test_session_mcp_sh_imports_the_launch_line_rather_than_copying_it,
        test_a_cwd_that_is_not_a_directory_is_its_own_outcome,
        test_deployed_copy_matches_this_source,
    ]:
        try:
            fn()
        except Exception as exc:
            check(f"{fn.__name__} (harness)", False, f"raised {type(exc).__name__}: {exc}")
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    sys.exit(1 if failed else 0)
