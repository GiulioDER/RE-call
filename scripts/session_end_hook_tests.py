"""Regression tests for the SessionEnd workspace hook.

The two that matter are refusals: this hook must not release a claim it does not
own, and must not remove a container it did not start. Both are exercised with
the input that would defeat a careless implementation, and each has a control so
a hook that refuses everything cannot pass.
"""

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HOOK = str(Path(__file__).resolve().parent / "session_end_hook.py")
SCRATCH = Path(tempfile.gettempdir()) / "recall-endtests"

results = []


def load():
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


def test_releases_only_its_own_claim():
    m = load()
    base, wt = new_repo("end-claim")
    gd = git_dir_of(wt)
    cf = gd / "claude-session-claim"

    # THE REFUSAL: a claim held by another session must survive.
    cf.write_text("session=SOMEONE-ELSE\npid=1234\nclaimed_epoch=1\n", encoding="utf-8")
    msg = m.release_claim(gd, "ME")
    check("a claim owned by ANOTHER session is left in place",
          cf.exists() and "left in place" in msg, msg)

    # The control: our own claim must actually be released, or the test above
    # would pass for a hook that never releases anything.
    cf.write_text("session=ME\npid=1234\nclaimed_epoch=1\n", encoding="utf-8")
    msg = m.release_claim(gd, "ME")
    check("control: our OWN claim is released", not cf.exists() and msg == "released", msg)

    check("no claim to release is not an error",
          m.release_claim(gd, "ME") == "no claim to release")


def test_release_also_clears_a_stranded_lock():
    """A lock outliving the claim blocks the next session, as it once did."""
    m = load()
    base, wt = new_repo("end-lock")
    gd = git_dir_of(wt)
    cf = gd / "claude-session-claim"
    cf.write_text("session=ME\npid=1234\nclaimed_epoch=1\n", encoding="utf-8")
    lock = gd / "claude-session-claim.lock"
    lock.mkdir()
    (lock / "pid").write_text("999999999")
    m.release_claim(gd, "ME")
    check("releasing also clears a stranded lock", not lock.exists() and not cf.exists())


def test_removes_only_its_own_container():
    """Selection is by the recall.checkout LABEL, never by a name pattern."""
    m = load()
    base, wt = new_repo("end-container")
    calls = []

    def fake_run(args, cwd=None, timeout=10):
        calls.append(args)
        if args[:3] == ["docker", "ps", "-aq"]:
            # Only a container labelled with THIS checkout is ever returned.
            wanted = f"label=recall.checkout={str(wt).replace(chr(92), '/')}"
            return (0, "deadbeefcafe") if wanted in args else (0, "")
        return (0, "")

    m.run = fake_run
    msg = m.remove_own_container(wt)
    filters = [a for c in calls for a in c if str(a).startswith("label=")]
    check("the container query filters on recall.checkout for THIS checkout",
          any("recall.checkout=" in f and "end-container" in f for f in filters),
          str(filters))
    check("a labelled container is removed", "removed" in msg and "deadbeef" in msg, msg)
    removes = [c for c in calls if c[:3] == ["docker", "rm", "-f"]]
    check("exactly one container removed, by id", len(removes) == 1, str(removes))

    # THE REFUSAL: a different checkout's container must not be touched.
    calls.clear()
    other = wt.parent / "some-other-checkout"
    msg = m.remove_own_container(other)
    removes = [c for c in calls if c[:3] == ["docker", "rm", "-f"]]
    check("another checkout's container is NOT removed",
          not removes and "none started by this checkout" in msg, msg)


def test_survey_reports_but_does_not_tidy():
    m = load()
    base, wt = new_repo("end-survey")
    (wt / "dirty.txt").write_text("uncommitted work someone else may own")
    s = m.survey(wt)
    check("uncommitted work is counted", s["uncommitted"] == 1, str(s))
    check("uncommitted work is LEFT on disk", (wt / "dirty.txt").exists())
    after = sh("git", "status", "--porcelain", cwd=wt).stdout
    check("nothing was staged", "A  " not in after and "M  " not in after, after.strip()[:60])


def test_unusable_cwd_is_not_reported_as_not_a_repo():
    """A path we cannot enter and a path that is not a repo are different facts.

    An MSYS-style `/c/...` cwd arrives as a real string Windows cannot resolve.
    Calling that "not a git repo" hid a skipped cleanup behind a plausible
    outcome, which is the same error shape this whole guard exists to avoid.
    """
    log = Path.home() / ".claude" / "session-end.log"
    payload = json.dumps({"session_id": "X", "cwd": "/c/definitely/not/here", "reason": "clear"})
    p = subprocess.run([sys.executable, HOOK], input=payload, capture_output=True, text=True)
    last = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    check("an unusable cwd is recorded as cwd-unusable, not not-a-git-repo",
          p.returncode == 0 and last.get("outcome") == "cwd-unusable", str(last)[:110])

    # Control: a real directory that genuinely is not a repo still says so.
    payload = json.dumps({"session_id": "X", "cwd": str(SCRATCH), "reason": "clear"})
    subprocess.run([sys.executable, HOOK], input=payload, capture_output=True, text=True)
    last = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    check("control: a real non-repo directory still reports not-a-git-repo",
          last.get("outcome") == "not-a-git-repo", str(last)[:110])


def test_never_raises_and_always_logs():
    log = Path.home() / ".claude" / "session-end.log"
    before = len(log.read_text(encoding="utf-8").splitlines()) if log.exists() else 0
    for payload in ('{"session_id":"x","cwd":"' + str(SCRATCH).replace("\\", "/") + '"}',
                    "[]", "not json at all", ""):
        p = subprocess.run([sys.executable, HOOK], input=payload,
                           capture_output=True, text=True)
        if p.returncode != 0 or "Traceback" in p.stderr:
            check("every payload shape exits 0 without a traceback", False,
                  f"payload={payload[:20]!r} rc={p.returncode} {p.stderr[:80]!r}")
            return
    after = len(log.read_text(encoding="utf-8").splitlines()) if log.exists() else 0
    check("every payload shape exits 0 without a traceback", True)
    check("a row is written even for unusable payloads", after >= before + 4,
          f"log rows {before} -> {after}")


if __name__ == "__main__":
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for fn in (test_releases_only_its_own_claim,
               test_release_also_clears_a_stranded_lock,
               test_removes_only_its_own_container,
               test_survey_reports_but_does_not_tidy,
               test_unusable_cwd_is_not_reported_as_not_a_repo,
               test_never_raises_and_always_logs):
        try:
            fn()
        except Exception as exc:
            check(f"{fn.__name__} (harness)", False, f"raised {type(exc).__name__}: {exc}")
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    sys.exit(1 if failed else 0)
