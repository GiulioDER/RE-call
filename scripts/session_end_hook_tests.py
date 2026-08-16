"""Regression tests for the SessionEnd workspace hook.

Every test here names the defect it pins. The ones that matter are refusals, and
each has a control, because a hook that refuses everything would otherwise pass
the whole file.

Two lessons from the audit of the first version are built into the shape of this
file. The container test asserts **the exact argv the hook passed**, not what a
stub chose to return: asserting the stub's return value tested the stub. And the
log is redirected to a temp file via RECALL_SESSION_LOG, so these tests neither
read another session's rows nor append noise to a log meant to be counted later.
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HOOK = str(Path(__file__).resolve().parent / "session_end_hook.py")
SCRATCH = Path(tempfile.gettempdir()) / "recall-endtests"
TEST_LOG = SCRATCH / "session-end.log"

results = []


def load():
    sys.path.insert(0, str(Path(HOOK).parent))
    spec = importlib.util.spec_from_file_location("endhook", HOOK)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {detail}" if detail else ""))


def sh(*args, cwd=None):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True)


def new_repo(name):
    base = SCRATCH / name
    if base.exists():
        subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", str(base)], capture_output=True)
    base.mkdir(parents=True)
    sh("git", "init", "-q", ".", cwd=base)
    sh("git", "config", "user.email", "t@t", cwd=base)
    sh("git", "config", "user.name", "t", cwd=base)
    (base / "a.txt").write_text("x")
    sh("git", "add", "a.txt", cwd=base)
    sh("git", "commit", "-qm", "init", cwd=base)
    sh("git", "worktree", "add", "-q", "-b", "wt", "./wt", "HEAD", cwd=base)
    return base, base / "wt"


def git_dir_of(wt: Path) -> Path:
    return Path(sh("git", "rev-parse", "--absolute-git-dir", cwd=wt).stdout.strip())


def run_hook(payload: dict, env_extra=None):
    """Invoke the hook as the harness does, with the log redirected."""
    env = dict(os.environ, RECALL_SESSION_LOG=str(TEST_LOG))
    env.update(env_extra or {})
    before = TEST_LOG.read_text(encoding="utf-8").count("\n") if TEST_LOG.exists() else 0
    p = subprocess.run([sys.executable, HOOK], input=json.dumps(payload),
                       capture_output=True, text=True, env=env)
    rows = TEST_LOG.read_text(encoding="utf-8").splitlines() if TEST_LOG.exists() else []
    return p, (json.loads(rows[-1]) if len(rows) > before else None)


# ------------------------------------------------- AUDIT-3 BUG-003: positive identity
def test_release_requires_positive_identity():
    m = load()
    base, wt = new_repo("end-identity")
    gd = git_dir_of(wt)
    cf = gd / "claude-session-claim"

    for label, holder, sid in (
        ("another session's claim", "SOMEONE-ELSE", "ME"),
        ("an EMPTY session id", "SOMEONE-ELSE", ""),
        ("a claim with no session= line", None, "ME"),
    ):
        cf.write_text(("" if holder is None else f"session={holder}\n") + "pid=1234\n",
                      encoding="utf-8")
        msg = m.release_claim(gd, sid, "999")
        if not (cf.exists() and "left in place" in msg):
            check(f"BUG-003 refuses to release: {label}", False, msg)
            return
    check("BUG-003 refuses to release on any unproven ownership (3 shapes)", True)

    cf.write_text("session=ME\npid=1234\n", encoding="utf-8")
    msg = m.release_claim(gd, "ME", "999")
    check("BUG-003 control: our OWN claim IS released",
          not cf.exists() and msg == "released", msg)


# ------------------------------------------------- AUDIT-3 BUG-004: never break a live lock
def test_never_breaks_a_live_lock():
    m = load()
    base, wt = new_repo("end-livelock")
    gd = git_dir_of(wt)
    cf = gd / "claude-session-claim"
    cf.write_text("session=ME\npid=1234\n", encoding="utf-8")
    lock = gd / "claude-session-claim.lock"
    lock.mkdir()
    (lock / "pid").write_text(str(os.getpid()), encoding="utf-8")  # certainly alive
    msg = m.release_claim(gd, "ME", "999")
    check("BUG-004 a lock held by a LIVE process is not broken",
          cf.exists() and "another session holds the claim lock" in msg, msg)
    check("BUG-004 the live lock still exists afterwards", lock.exists())

    # Control: a lock whose holder is dead must be reclaimable, or the above
    # would pass for a hook that can never acquire a lock at all.
    (lock / "pid").write_text("999999999", encoding="utf-8")
    msg = m.release_claim(gd, "ME", "999")
    check("BUG-004 control: a DEAD holder's lock is broken and the claim released",
          not cf.exists() and msg == "released", msg)


# ------------------------------------------------- AUDIT-3 BUG-013: assert the argv, not the stub
def test_container_filter_is_exactly_this_checkout():
    m = load()
    base, wt = new_repo("end-filter")
    calls = []

    def fake_run(args, cwd=None, timeout=10, env=None, budget_left=None):
        calls.append(list(args))
        if args[:3] == ["docker", "ps", "-aq"]:
            return (0, "deadbeefcafe", "")
        return (0, "", "")

    m.run = fake_run
    status, detail = m.remove_own_container(wt)
    query = next((c for c in calls if c[:3] == ["docker", "ps", "-aq"]), [])
    expected = f"label=recall.checkout={str(wt).replace(chr(92), '/')}"
    check("BUG-013 the docker filter argv is exactly this checkout's label",
          expected in query, f"expected {expected!r} in {query!r}")
    check("BUG-013 the query is filtered at all (no bare docker ps)",
          "--filter" in query, str(query))
    removes = [c for c in calls if c[:3] == ["docker", "rm", "-f"]]
    check("removal is by id, exactly once",
          len(removes) == 1 and removes[0][3] == "deadbeefcafe", str(removes))
    check("status reports removed", status == "removed" and "deadbeef" in detail, detail)


def test_docker_unavailable_is_not_reported_as_none():
    """BUG-007: 'could not check' must not read as 'nothing to clean up'."""
    m = load()
    base, wt = new_repo("end-dockerdown")
    m.run = lambda *a, **k: (m.LAUNCH_FAILED, "", "could not launch")
    status, detail = m.remove_own_container(wt)
    check("BUG-007 an unrunnable docker is 'unknown', not 'none'",
          status == "unknown" and "NOT checked" in detail, f"{status}: {detail}")
    m2 = load()
    m2.run = lambda *a, **k: (m2.NOT_ATTEMPTED, "", "budget")
    status, detail = m2.remove_own_container(wt)
    check("BUG-007 an unattempted query is 'not-attempted', not 'none'",
          status == "not-attempted", f"{status}: {detail}")


# ------------------------------------------------- AUDIT-3 BUG-001 / BUG-002: gating and order
def test_container_is_not_touched_when_the_claim_is_someone_elses():
    """BUG-001: a session refused the workspace must not tear down the holder's DB."""
    base, wt = new_repo("end-notowner")
    gd = git_dir_of(wt)
    (gd / "claude-session-claim").write_text("session=OTHER\npid=1234\n", encoding="utf-8")
    p, row = run_hook({"session_id": "ME", "cwd": str(wt), "reason": "other"})
    check("BUG-001 a non-owner does not touch the container",
          row and row.get("container") == "skipped"
          and row.get("outcome") == "closed-not-owner", str(row)[:130])
    check("BUG-001 the other session's claim survives",
          (gd / "claude-session-claim").exists())


def test_container_removed_before_claim_released():
    """BUG-002: releasing first lets the next session's new container match."""
    m = load()
    base, wt = new_repo("end-order")
    gd = git_dir_of(wt)
    (gd / "claude-session-claim").write_text("session=ME\npid=1234\n", encoding="utf-8")
    order = []
    real_release = m.release_claim
    m.remove_own_container = lambda root: (order.append("container"), ("none", "stub"))[1]
    m.release_claim = lambda *a, **k: (order.append("claim"), real_release(*a, **k))[1]
    os.environ["RECALL_SESSION_LOG"] = str(TEST_LOG)
    m.main.__globals__["remove_own_container"] = m.remove_own_container
    m.main.__globals__["release_claim"] = m.release_claim
    sys.stdin = None  # main reads via read_payload; patch it instead
    m.read_payload = lambda timeout=5.0: {"session_id": "ME", "cwd": str(wt)}
    m.main.__globals__["read_payload"] = m.read_payload
    m.main()
    check("BUG-002 the container is removed BEFORE the claim is released",
          order == ["container", "claim"], str(order))


# ------------------------------------------------- AUDIT-3 BUG-010 / BUG-008
def test_clear_does_not_tear_anything_down():
    base, wt = new_repo("end-clear")
    gd = git_dir_of(wt)
    cf = gd / "claude-session-claim"
    cf.write_text("session=ME\npid=1234\n", encoding="utf-8")
    p, row = run_hook({"session_id": "ME", "cwd": str(wt), "reason": "clear"})
    check("BUG-010 reason=clear is skipped entirely",
          row and row.get("outcome") == "skipped-clear", str(row)[:110])
    check("BUG-010 the claim survives a clear", cf.exists())


def test_always_writes_a_row():
    """BUG-008: a non-string session_id produced no row at all."""
    for label, payload in (("non-string session_id", {"session_id": 123, "cwd": str(SCRATCH)}),
                           ("null session_id", {"session_id": None, "cwd": str(SCRATCH)}),
                           ("empty payload", {})):
        p, row = run_hook(payload)
        if p.returncode != 0 or row is None:
            check(f"BUG-008 a row is always written ({label})", False,
                  f"rc={p.returncode} row={row}")
            return
    check("BUG-008 a row is written for every payload shape (3 shapes)", True)


def test_unusable_cwd_is_not_reported_as_not_a_repo():
    p, row = run_hook({"session_id": "X", "cwd": "/c/definitely/not/here", "reason": "other"})
    check("an unusable cwd is cwd-unusable, not not-a-git-repo",
          row and row.get("outcome") == "cwd-unusable", str(row)[:110])
    p, row = run_hook({"session_id": "X", "cwd": str(SCRATCH), "reason": "other"})
    check("control: a real non-repo directory still reports not-a-git-repo",
          row and row.get("outcome") == "not-a-git-repo", str(row)[:110])


# ------------------------------------------------- AUDIT-4: the three cases that let P0s ship
def test_no_claim_file_means_no_teardown():
    """BUG-102: 'no claim' was read as 'mine', and the MAIN checkout never has one.

    session-space.sh refuses the main checkout before it would write a claim, so
    the shared checkout is permanently in this state. An auditor proved the old
    code force-removed a real container labelled for a different session.
    """
    m = load()
    base, wt = new_repo("end-noclaim")
    assert not (git_dir_of(wt) / "claude-session-claim").exists()
    calls = []

    def fake_run(args, cwd=None, timeout=10, env=None, budget_left=None):
        calls.append(list(args))
        return (0, "deadbeefcafe", "") if args[:3] == ["docker", "ps", "-aq"] else (0, "", "")

    m.run = fake_run
    m.main.__globals__["run"] = fake_run
    m.read_payload = lambda timeout=5.0: {"session_id": "SOME-OTHER", "cwd": str(wt)}
    m.main.__globals__["read_payload"] = m.read_payload
    os.environ["RECALL_SESSION_LOG"] = str(TEST_LOG)
    m.main()
    removes = [c for c in calls if c[:3] == ["docker", "rm", "-f"]]
    row = json.loads(TEST_LOG.read_text(encoding="utf-8").splitlines()[-1])
    check("BUG-102 an absent claim performs ZERO docker rm",
          not removes, f"{len(removes)} rm call(s): {removes}")
    check("BUG-102 and it says ownership was not established",
          row.get("outcome") == "closed-not-owner"
          and "no claim" in str(row.get("container_detail", "")), str(row)[:130])


def test_unreadable_payload_touches_nothing():
    """BUG-103: an unread payload lost `reason`, defeating the clear-skip."""
    m = load()
    base, wt = new_repo("end-badpayload")
    calls = []
    m.run = lambda *a, **k: (calls.append(list(a[0])), (0, "", ""))[1]
    m.main.__globals__["run"] = m.run
    m.read_payload = lambda timeout=5.0: {"_stdin": "timed out"}
    m.main.__globals__["read_payload"] = m.read_payload
    os.environ["RECALL_SESSION_LOG"] = str(TEST_LOG)
    m.main()
    row = json.loads(TEST_LOG.read_text(encoding="utf-8").splitlines()[-1])
    check("BUG-103 an unreadable payload performs no docker call at all",
          not calls and row.get("outcome") == "payload-unreadable", f"{calls} {row}"[:130])


def test_missing_common_module_still_writes_a_row():
    """BUG-101: the install copies single files; two are needed, so this happens."""
    import shutil
    lone = SCRATCH / "lone-install"
    if lone.exists():
        shutil.rmtree(lone, ignore_errors=True)
    lone.mkdir(parents=True)
    # Exactly the install shape: the hook alone, WITHOUT session_hook_common.py.
    shutil.copy(HOOK, lone / "session_end_workspace.py")
    log = lone / "row.log"
    p = subprocess.run([sys.executable, str(lone / "session_end_workspace.py")],
                       input=json.dumps({"session_id": "X", "cwd": str(SCRATCH)}),
                       capture_output=True, text=True,
                       env=dict(os.environ, RECALL_SESSION_LOG=str(log)))
    row = json.loads(log.read_text(encoding="utf-8").splitlines()[-1]) if log.exists() else None
    check("BUG-101 a missing session_hook_common exits 0 and STILL writes a row",
          p.returncode == 0 and row is not None
          and row.get("outcome") == "no-common-module",
          f"rc={p.returncode} row={str(row)[:90]} stderr={p.stderr[:60]!r}")


def test_claim_survives_a_failed_container_teardown():
    """BUG-105: releasing after a failed teardown strands an unattributable container."""
    m = load()
    base, wt = new_repo("end-failedteardown")
    gd = git_dir_of(wt)
    (gd / "claude-session-claim").write_text("session=ME\npid=1234\n", encoding="utf-8")
    m.remove_own_container = lambda root: ("failed", "could not remove abc123")
    m.main.__globals__["remove_own_container"] = m.remove_own_container
    m.read_payload = lambda timeout=5.0: {"session_id": "ME", "cwd": str(wt)}
    m.main.__globals__["read_payload"] = m.read_payload
    os.environ["RECALL_SESSION_LOG"] = str(TEST_LOG)
    m.main()
    row = json.loads(TEST_LOG.read_text(encoding="utf-8").splitlines()[-1])
    check("BUG-105 a failed teardown leaves the claim in place",
          (gd / "claude-session-claim").exists()
          and "left in place" in str(row.get("claim", "")), str(row)[:130])


def test_git_unavailable_is_not_reported_as_not_a_repo():
    """BUG-104: 'git would not run' is not a fact about the directory."""
    m = load()
    base, wt = new_repo("end-nogit")
    m.git = lambda *a, **k: (m.LAUNCH_FAILED, "")
    m.main.__globals__["git"] = m.git
    m.read_payload = lambda timeout=5.0: {"session_id": "ME", "cwd": str(wt)}
    m.main.__globals__["read_payload"] = m.read_payload
    os.environ["RECALL_SESSION_LOG"] = str(TEST_LOG)
    m.main()
    row = json.loads(TEST_LOG.read_text(encoding="utf-8").splitlines()[-1])
    check("BUG-104 an unrunnable git is 'git-unavailable', not 'not-a-git-repo'",
          row.get("outcome") == "git-unavailable", str(row)[:120])


def test_survey_reports_but_does_not_tidy():
    m = load()
    base, wt = new_repo("end-survey")
    (wt / "dirty.txt").write_text("uncommitted work someone else may own")
    s = m.survey(wt)
    check("uncommitted work is counted", s["uncommitted"] == 1, str(s))
    check("uncommitted work is LEFT on disk", (wt / "dirty.txt").exists())
    after = sh("git", "status", "--porcelain", cwd=wt).stdout
    check("nothing was staged", "A  " not in after and "M  " not in after, after.strip()[:60])
    m2 = load()
    m2.git = lambda *a, **k: (m2.LAUNCH_FAILED, "")
    dead = m2.survey(wt)
    check("BUG-107 EVERY survey field is None when git fails, including the list",
          all(v is None for v in dead.values()), str(dead))


if __name__ == "__main__":
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if TEST_LOG.exists():
        TEST_LOG.unlink()
    for fn in (test_release_requires_positive_identity,
               test_never_breaks_a_live_lock,
               test_container_filter_is_exactly_this_checkout,
               test_docker_unavailable_is_not_reported_as_none,
               test_container_is_not_touched_when_the_claim_is_someone_elses,
               test_container_removed_before_claim_released,
               test_clear_does_not_tear_anything_down,
               test_always_writes_a_row,
               test_unusable_cwd_is_not_reported_as_not_a_repo,
               test_no_claim_file_means_no_teardown,
               test_unreadable_payload_touches_nothing,
               test_missing_common_module_still_writes_a_row,
               test_claim_survives_a_failed_container_teardown,
               test_git_unavailable_is_not_reported_as_not_a_repo,
               test_survey_reports_but_does_not_tidy):
        try:
            fn()
        except Exception as exc:
            check(f"{fn.__name__} (harness)", False, f"raised {type(exc).__name__}: {exc}")
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    sys.exit(1 if failed else 0)
