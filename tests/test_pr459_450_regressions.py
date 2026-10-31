"""Regressions for PRs 459 through 450, exercised on the baseline 3b3d875f8.

The red receipt records each node and its intended assertion. Transport doubles
never invoke Docker or a model. Process probes own their children and use fake CLIs.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
POSIX = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process and shell behavior")


def test_stdio_credentials_stay_outside_retained_artifacts(tmp_path):
    """SEC-001: baseline returns the credential-bearing retained destination."""
    from benchmarks.agent_ab.recall_server import StdioRecallSpec

    artifacts = tmp_path / "artifacts"
    retained = artifacts / "recall-mcp.json"
    credential = "fixture-secret"
    spec = StdioRecallSpec(dsn=f"postgresql://fixture:{credential}@localhost/db", cwd=ROOT)
    live = spec.write_mcp_config(retained)
    try:
        assert live.parent != artifacts, "live credentials must not be retained artifacts"
        assert credential in live.read_text(encoding="utf-8")
        assert credential not in retained.read_text(encoding="utf-8")
        if sys.platform != "win32":
            assert live.stat().st_mode & 0o077 == 0
            assert live.parent.stat().st_mode & 0o077 == 0
    finally:
        close = getattr(spec, "close", None)
        if close is not None:
            close()
    assert not live.exists(), "credential configuration must disappear at run cleanup"


def test_gate_cleans_stdio_config_on_exception(tmp_path, monkeypatch):
    """SEC-001: the gate's existing null context retains its credential configuration."""
    from benchmarks.agent_ab.recall_server import StdioRecallSpec
    from scripts import agent_ab_gate as gate

    written = []

    async def ready(self):
        return {"handshake_ms": 0, "tool_count": 1, "search_ms": 0, "abstained": False}

    async def fail_session(row, variant, config):
        written.append(Path(config.mcp_config))
        raise RuntimeError("fixture session failed")

    monkeypatch.setattr(StdioRecallSpec, "check", ready)
    monkeypatch.setattr(gate, "run_claude_case", fail_session)
    monkeypatch.setattr(sys, "argv", ["gate", "--workdir", str(tmp_path)])
    with pytest.raises(RuntimeError, match="fixture session failed"):
        asyncio.run(gate.main())
    assert written and not written[0].exists(), "the gate must close live configuration on error"


@POSIX
def test_checker_timeout_preserves_its_caller(tmp_path):
    """BUG-001: an isolated outer probe is SIGKILLed by the baseline killpg."""
    probe = (
        "import sys; from pathlib import Path; "
        "from benchmarks.agent_ab.checkers._run import run_bounded; "
        "r=run_bounded([sys.executable,'-c','import time; time.sleep(5)'],"
        f"cwd=Path({str(tmp_path)!r}),timeout_s=0.1); "
        "assert r.timed_out; print('caller survived')"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True,
        timeout=8, start_new_session=True,
    )
    assert result.returncode == 0, "a checker timeout must not kill its caller"
    assert "caller survived" in result.stdout


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat")
        if state.is_file() and state.read_text().split()[2] == "Z":
            return False
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _kill_recorded(path: Path) -> None:
    if not path.exists():
        return
    for pid in json.loads(path.read_text()):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@POSIX
@pytest.mark.parametrize("cancel", [False, True], ids=["timeout", "cancellation"])
def test_claude_adapter_reaps_spawned_work(tmp_path, monkeypatch, cancel):
    """BUG-004: baseline waits on descendant pipes or leaves a cancelled CLI alive."""
    from benchmarks.agent_ab import claude_exec
    from benchmarks.agent_ab.schema import RECALL_ON

    pids = tmp_path / "pids.json"
    fake = tmp_path / "fake_cli.py"
    fake.write_text(
        "import json,os,subprocess,sys,time\n"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(5)'])\n"
        f"open({str(pids)!r},'w').write(json.dumps([os.getpid(),child.pid]))\n"
        "time.sleep(5)\n", encoding="utf-8",
    )
    monkeypatch.setattr(
        claude_exec.ClaudeExecConfig, "command", lambda self, prompt: [sys.executable, str(fake)]
    )
    config = claude_exec.ClaudeExecConfig(timeout_s=0.25, strict_mcp_config=False)

    async def run():
        task = asyncio.create_task(
            claude_exec.run_claude_case({"task_id": "fixture", "user_input": "fixture"}, RECALL_ON, config)
        )
        if cancel:
            for _ in range(200):
                if pids.exists():
                    break
                await asyncio.sleep(0.01)
            task.cancel()
        try:
            await task
        except (claude_exec.ClaudeTranscriptError, asyncio.CancelledError):
            pass

    started = time.monotonic()
    try:
        asyncio.run(run())
        elapsed = time.monotonic() - started
        recorded = json.loads(pids.read_text())
        assert elapsed < 3, "a timeout must not wait for surviving descendant pipes"
        assert not any(_alive(pid) for pid in recorded), "timeout/cancellation must reap the CLI tree"
    finally:
        _kill_recorded(pids)


@pytest.mark.parametrize("parent_exited,assignment_fails", [(True, False), (False, False), (False, True)])
def test_windows_adapter_owns_work_before_launch(monkeypatch, parent_exited, assignment_fails):
    """BUG-004: baseline neither owns Windows descendants nor gates their launch."""
    from benchmarks.agent_ab import claude_exec
    from benchmarks.agent_ab.schema import RECALL_ON

    events = []

    class Job:
        def __init__(self):
            events.append("job")

        def command(self, command):
            return ["fixture-wrapper", *command]

        def assign(self, pid):
            events.append("assign")
            if assignment_fails:
                raise OSError("fixture job assignment refused")

        def terminate(self):
            events.append("terminate")

        def close(self):
            events.append("close")

    class Writer:
        def write(self, data):
            assert data == b"1" and "assign" in events
            events.append("release")

        async def drain(self):
            pass

        def close(self):
            pass

    class Process:
        pid = 1234
        returncode = None
        stdin = Writer()

        async def communicate(self):
            if "terminate" in events or "direct-kill" in events:
                return b"", b""
            if parent_exited:
                self.returncode = 0
            raise asyncio.TimeoutError

        def kill(self):
            events.append("direct-kill")

        async def wait(self):
            return 0

    async def launch(*command, **kwargs):
        events.append("launch")
        return Process()

    monkeypatch.setattr(claude_exec, "sys", SimpleNamespace(platform="win32", executable=sys.executable), raising=False)
    monkeypatch.setattr(claude_exec, "WindowsJob", Job, raising=False)
    monkeypatch.setattr(claude_exec.ClaudeExecConfig, "command", lambda self, prompt: ["fixture-cli"])
    monkeypatch.setattr(claude_exec.asyncio, "create_subprocess_exec", launch)
    error = None
    try:
        asyncio.run(claude_exec.run_claude_case(
            {"task_id": "fixture", "user_input": "fixture"}, RECALL_ON,
            claude_exec.ClaudeExecConfig(strict_mcp_config=False),
        ))
    except (OSError, claude_exec.ClaudeTranscriptError) as failure:
        error = type(failure).__name__
    if assignment_fails:
        assert error == "OSError", "failed ownership must refuse the CLI launch"
        assert "release" not in events and "direct-kill" in events
    else:
        assert "terminate" in events, "owned work must be terminated even after its direct parent exits"
        assert events.index("assign") < events.index("release")
        assert error == "ClaudeTranscriptError"
    assert events[-1] == "close"


def test_windows_job_native_contract(monkeypatch):
    """BUG-004 control: mutation clears the native kill-on-close flag; no kernel executes here."""
    import ctypes
    from benchmarks.agent_ab import _windows_job as jobs

    seen = {}
    closed = []

    class Function:
        def __init__(self, callback):
            self.callback = callback

        def __call__(self, *args):
            return self.callback(*args)

    def limits(handle, kind, pointer, size):
        value = ctypes.cast(pointer, ctypes.POINTER(jobs._ExtendedLimits)).contents
        seen.update(handle=handle, kind=kind, size=size, flags=value.BasicLimitInformation.LimitFlags)
        return 1

    def opened(access, inherit, pid):
        seen.update(access=access, inherit=inherit, pid=pid)
        return 202

    api = SimpleNamespace(
        CreateJobObjectW=Function(lambda *args: 101),
        SetInformationJobObject=Function(limits),
        OpenProcess=Function(opened),
        AssignProcessToJobObject=Function(lambda job, process: seen.update(assigned=(job, process)) or 1),
        TerminateJobObject=Function(lambda job, code: seen.update(terminated=job) or 1),
        CloseHandle=Function(lambda handle: closed.append(handle) or 1),
    )
    monkeypatch.setattr(jobs, "sys", SimpleNamespace(platform="win32", executable=sys.executable))
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: api, raising=False)
    job = jobs.WindowsJob()
    job.assign(1234)
    job.terminate()
    job.close()
    job.close()
    assert seen["flags"] == 0x2000, "the job must kill every member when its owner closes"
    assert seen["kind"] == 9 and seen["access"] == 0x0101 and seen["pid"] == 1234
    assert seen["assigned"] == (101, 202) and seen["terminated"] == 101
    assert closed == [202, 101]
    assert api.CreateJobObjectW.restype is ctypes.c_void_p
    if ctypes.sizeof(ctypes.c_void_p) == 8:
        assert seen["size"] == 144


def test_windows_job_wrapper_gate(tmp_path):
    """BUG-004 control: mutation removes the release gate and starts unowned work."""
    marker = tmp_path / "launched"
    command = [sys.executable, "-I", "-S", str(ROOT / "benchmarks/agent_ab/_windows_job.py"),
               "--child", sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"]
    result = subprocess.run(command, input=b"", capture_output=True, timeout=5)
    assert not marker.exists(), "the CLI must not start before its parent releases the owned wrapper"
    assert result.returncode == 2


@pytest.mark.parametrize("kind", ["empty", "primary", "control", "cost"])
def test_small_sample_analysis_is_written(tmp_path, monkeypatch, kind):
    """BUG-003: baseline KeyError/None formatting must become an insufficiency report."""
    from benchmarks.agent_ab.schema import RECALL_OFF, RECALL_ON, SessionRecord
    from scripts import agent_ab_analyze_tasks as analyzer

    records = []
    count = 0 if kind == "empty" else 1 if kind == "primary" else 6
    for i in range(count + (kind == "control")):
        for variant in (RECALL_ON, RECALL_OFF):
            meta = {"locus": "both" if i == count else "memory_only", "base_task_id": "fixture"}
            if variant == RECALL_ON:
                meta["recall_tools_available"] = ["mcp__recall__recall_search"]
            records.append(SessionRecord(
                task_id=f"fixture#{i}", variant=variant, success=variant == RECALL_ON,
                input_tokens=100 if kind == "cost" and i == 0 else None, metadata=meta,
            ))
    monkeypatch.setattr(analyzer, "load", lambda run_id: (records, tmp_path))
    monkeypatch.setattr(sys, "argv", ["analyze", "--run-id", "fixture"])
    failure = None
    try:
        code = analyzer.main()
    except (KeyError, TypeError) as error:
        failure = type(error).__name__
        code = None
    assert failure is None, f"insufficient samples must be reportable, got {failure}"
    assert code == 0
    payload = json.loads((tmp_path / "analysis.json").read_text())
    if kind in {"empty", "primary"}:
        assert payload["primary_per_pair"]["p_value"] is None
    if kind == "empty":
        assert payload["primary_per_task"]["n_tasks"] == 0
    if kind == "control":
        assert payload["control"]["p_value"] is None
    if kind == "cost":
        assert payload["cost"]["input_tokens"]["delta_mean"] is None


@pytest.mark.parametrize("trials,balanced", [(1024, False), (1024, True), (2048, False)])
def test_public_mcnemar_keeps_large_probabilities_bounded(trials, balanced):
    """NUM-001: baseline explicit float conversion overflows before bounded division."""
    from benchmarks.agent_ab.stats import mcnemar_exact

    on = [True] * trials
    if balanced:
        on[trials // 2:] = [False] * (trials // 2)
    off = [not value for value in on]
    value = None
    try:
        value, _, _ = mcnemar_exact(on, off)
    except OverflowError:
        pass
    assert value is not None, "valid paired counts must not overflow a binomial intermediate"
    assert math.isfinite(value) and 0 <= value <= 1
    if balanced:
        assert value == 1


def _workspace(tmp_path):
    work = tmp_path / "case"
    shutil.copytree(ROOT / "benchmarks/agent_ab/workspaces/ts-separator-canary/tree", work)
    return work


@pytest.mark.parametrize("kind", ["collection", "setup", "timeout", "uncollected"])
def test_canary_refuses_apparatus_failures(tmp_path, monkeypatch, kind):
    """BUG-005: baseline awards success for every non-green mutant transport result."""
    from benchmarks.agent_ab.checkers import ts_separator_canary as canary
    from benchmarks.agent_ab.checkers._run import Completed

    work = _workspace(tmp_path)
    calls = 0

    def run(command, **kwargs):
        nonlocal calls
        calls += 1
        mutant = calls == 2
        if not mutant:
            xml = '<testsuite><testcase classname="fixture" name="test_separator"/></testsuite>'
        elif kind in {"collection", "setup"}:
            xml = '<testsuite><testcase classname="fixture" name="test_separator"><error message="apparatus"/></testcase></testsuite>'
        else:
            xml = '<testsuite/>'
        for arg in command:
            if str(arg).startswith("--junitxml="):
                Path(str(arg).split("=", 1)[1]).write_text(xml)
        return Completed(2 if mutant else 0, "fixture output", "", 0.01, mutant and kind == "timeout")

    monkeypatch.setattr(canary, "run_bounded", run)
    verdict = canary.check(work)
    assert not verdict.passed, "apparatus failure is not behavioral red proof"


def test_canary_refuses_real_mutant_collection_error(tmp_path):
    """BUG-005: replacing a submitted module removes its added export, not its behavior."""
    from benchmarks.agent_ab.checkers import ts_separator_canary as canary

    work = _workspace(tmp_path)
    module = work / "recall/normalise.py"
    module.write_text(module.read_text() + "\nEXTRA = 1\n")
    (work / "tests/test_normalise.py").write_text(
        "from recall.normalise import EXTRA\ndef test_extra():\n    assert EXTRA == 1\n"
    )
    verdict = canary.check(work)
    assert not verdict.passed, "a mutant ImportError must not receive regression-proof credit"


def test_canary_refuses_assertion_name_inside_another_exception(tmp_path):
    """BUG-005: an exception's prose is not its actual assertion failure type."""
    from benchmarks.agent_ab.checkers import ts_separator_canary as canary

    work = _workspace(tmp_path)
    (work / "tests/test_normalise.py").write_text(
        'from recall.normalise import normalise\ndef test_separator():\n'
        '    if normalise("a" + chr(0x2028) + "b") != "a b":\n'
        '        raise ValueError("not an AssertionError")\n'
    )
    assert not canary.check(work).passed, "exception prose must not masquerade as assertion red"


def test_canary_accepts_genuine_behavioral_red(tmp_path):
    """BUG-005 control: red-proved by mutating the production assertion classifier to False."""
    from benchmarks.agent_ab.checkers import ts_separator_canary as canary

    work = _workspace(tmp_path)
    tests = work / "tests/test_normalise.py"
    tests.write_text(tests.read_text() + '\ndef test_separator():\n    assert normalise("a" + chr(0x2028) + "b") == "a b"\n')
    assert canary.check(work).passed


def _linux_checkout(tmp_path, monkeypatch, name):
    root = tmp_path / name
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/linux-check.sh", root / "scripts/linux-check.sh")
    (root / "pyproject.toml").write_text('[project]\nname="fixture"\ndependencies=[]\n')
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
    binary = tmp_path / "bin"
    binary.mkdir(exist_ok=True)
    docker = binary / "docker"
    docker.write_text(
        '#!/usr/bin/env python3\nimport json,os,sys\na=sys.argv[1:]\n'
        'with open(os.environ["DOCKER_LOG"],"a") as f: f.write(json.dumps(a)+"\\n")\n'
        'if a and a[0]=="run" and any("test -f /venv/.linux-check-ready" in x for x in a):\n'
        ' with open(os.environ["DOCKER_LOG"],"a") as f: f.write(json.dumps(["readiness_pathconv",os.environ.get("MSYS_NO_PATHCONV")])+"\\n")\n'
        'if a and a[0]=="inspect":\n print(os.environ.get("DOCKER_OWNER","foreign")); sys.exit(0)\n'
        'if a and a[0]=="run" and any(".linux-check-ready" in x for x in a):\n'
        ' if any("test -f" in x for x in a): sys.exit(int(os.environ.get("DOCKER_NOT_READY","0")))\n'
        'sys.exit(0)\n'
    )
    docker.chmod(0o755)
    log = tmp_path / f"{name}.docker.jsonl"
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("DOCKER_LOG", str(log))
    monkeypatch.setenv("DOCKER_OWNER", str(root.resolve()))
    return root, log


def _linux(root, *args):
    return subprocess.run(
        ["bash", str(root / "scripts/linux-check.sh"), *args],
        capture_output=True, text=True, timeout=10,
    )


def _docker_calls(log):
    return [json.loads(line) for line in log.read_text().splitlines()]


@POSIX
def test_linux_database_identity_belongs_to_checkout(tmp_path, monkeypatch):
    """STAKES-001: baseline --down in both checkouts deletes the same global name."""
    names = []
    for name in ("one", "two"):
        root, log = _linux_checkout(tmp_path, monkeypatch, name)
        assert _linux(root, "--down").returncode == 0
        removed = [a[-1] for a in _docker_calls(log) if a[:2] == ["rm", "-f"]]
        assert removed, "the owned fixture container should be removed"
        names.append(removed[0])
    assert names[0] != names[1], "different checkouts must never address the same test database"


@POSIX
def test_linux_down_refuses_foreign_database(tmp_path, monkeypatch):
    """STAKES-001 formal proof: baseline removes a container despite a foreign owner."""
    root, log = _linux_checkout(tmp_path, monkeypatch, "foreign")
    monkeypatch.setenv("DOCKER_OWNER", "another-checkout")
    result = _linux(root, "--down")
    assert result.returncode != 0, "foreign ownership must be an explicit refusal"
    assert not any(a[:2] == ["rm", "-f"] for a in _docker_calls(log))


@POSIX
def test_linux_existing_volume_requires_readiness(tmp_path, monkeypatch):
    """ENV-001: a preexisting partially initialized volume must be bootstrapped again."""
    root, log = _linux_checkout(tmp_path, monkeypatch, "partial")
    monkeypatch.setenv("DOCKER_NOT_READY", "1")
    assert _linux(root, "tests/", "-q").returncode == 0
    assert any("uv pip install" in " ".join(a) for a in _docker_calls(log)), "volume existence is not readiness"


@POSIX
def test_linux_cache_probe_preserves_container_paths(tmp_path, monkeypatch):
    """ENV-001: the Docker readiness probe must inherit the existing MSYS guard."""
    root, log = _linux_checkout(tmp_path, monkeypatch, "path-conversion")
    monkeypatch.delenv("MSYS_NO_PATHCONV", raising=False)
    assert _linux(root, "tests/", "-q").returncode == 0
    guards = [a[1] for a in _docker_calls(log) if a[0] == "readiness_pathconv"]
    assert guards == ["1"], "container paths in the readiness probe need MSYS_NO_PATHCONV=1"


@POSIX
def test_linux_dependencies_select_a_new_cache(tmp_path, monkeypatch):
    """ENV-001: baseline cache identity ignores a changed dependency specification."""
    root, log = _linux_checkout(tmp_path, monkeypatch, "requirements")
    names = []
    for dependency in ("one", "two"):
        (root / "pyproject.toml").write_text(f'[project]\nname="fixture"\ndependencies=["{dependency}"]\n')
        assert _linux(root, "tests/", "-q").returncode == 0
        volumes = [a[-1] for a in _docker_calls(log) if a[:2] == ["volume", "inspect"]]
        assert volumes
        names.append(volumes[-1])
    assert names[0] != names[1], "dependency changes must invalidate the cached environment"
