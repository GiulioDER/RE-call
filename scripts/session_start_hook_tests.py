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
from pathlib import Path

HOOK = str(Path(__file__).resolve().parent / "session_start_hook.py")
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
    state = {}
    return m.build_report({"session_id": session, "cwd": str(cwd), "source": source}, state), state


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

    out, _ = report(m, base, session="anyone")
    check("core: MAIN checkout is refused",
          "WORKSPACE REFUSED" in out and "MAIN checkout" in out)

    out, _ = report(m, wt, session="S1")
    check("core: an unclaimed worktree is accepted",
          "Workspace claimed for this session" in out, out.splitlines()[0])

    out, _ = report(m, wt, session="S2")
    check("core: a worktree held by a live session is refused",
          "WORKSPACE REFUSED" in out and "CLAIMED by another session" in out)

    out, _ = report(m, wt, session="S1")
    check("core: the holder is still accepted on a second start",
          "Workspace claimed for this session" in out)

    out, st = report(m, wt, session="S1", source="compact")
    check("core: compaction is silent", out is None and st["outcome"] == "skipped-compact")

    out, st = report(m, tempfile.gettempdir(), session="x")
    check("core: a non-repo is silent", out is None and st["outcome"] == "not-a-git-repo")


if __name__ == "__main__":
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for fn in [
        test_core_paths,
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
    ]:
        try:
            fn()
        except Exception as exc:
            check(f"{fn.__name__} (harness)", False, f"raised {type(exc).__name__}: {exc}")
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    sys.exit(1 if failed else 0)
