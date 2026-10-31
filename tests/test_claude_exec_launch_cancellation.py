"""Cancellation must retain subprocess ownership while its launch is pending.

PR1109 CI exposed the existing cancellation cleanup assertion on the merged
PR1105 adapter. This deterministic version delays delivery of the real process
handle after the fake CLI has spawned a child. On the pre-fix merged tree,
cancelling that await bypasses the communicate error handler and the final
live-process assertion fails. The fixed adapter must finish the protected
launch, terminate its owned tree, and preserve cancellation.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.agent_ab import claude_exec
from benchmarks.agent_ab.schema import RECALL_ON


def _alive(pid: int) -> bool:
    try:
        status = Path(f"/proc/{pid}/stat")
        if status.is_file() and status.read_text().split()[2] == "Z":
            return False
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group ownership")
def test_cancellation_during_launch_reaps_the_real_process_tree(tmp_path, monkeypatch):
    """Target run_claude_case's launch await; red is surviving CLI/child PIDs."""
    pids = tmp_path / "owned-pids.json"
    cli = tmp_path / "fake_cli.py"
    cli.write_text(
        "import json,os,subprocess,sys,time\n"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'])\n"
        f"open({str(pids)!r},'w').write(json.dumps([os.getpid(),child.pid]))\n"
        "time.sleep(20)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        claude_exec.ClaudeExecConfig, "command", lambda self, prompt: [sys.executable, str(cli)]
    )
    create = asyncio.create_subprocess_exec

    async def exercise():
        ready, release = asyncio.Event(), asyncio.Event()

        async def delayed_handle(*args, **kwargs):
            process = await create(*args, **kwargs)
            for _ in range(500):
                if pids.exists():
                    break
                await asyncio.sleep(0.01)
            assert pids.exists(), "the real CLI must spawn its child before cancellation"
            ready.set()
            await release.wait()
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_handle)
        task = asyncio.create_task(
            claude_exec.run_claude_case(
                {"task_id": "owned-launch", "user_input": "fixture"},
                RECALL_ON,
                claude_exec.ClaudeExecConfig(timeout_s=5, strict_mcp_config=False),
            )
        )
        await asyncio.wait_for(ready.wait(), timeout=8)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=8)

    try:
        asyncio.run(exercise())
        assert not any(_alive(pid) for pid in json.loads(pids.read_text())), (
            "cancellation during launch must terminate the owned CLI and its child"
        )
    finally:
        if pids.exists():
            for pid in json.loads(pids.read_text()):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def test_cancellation_during_windows_launch_kills_the_unassigned_wrapper(monkeypatch):
    """Original launch await leaves the unassigned wrapper alive; the kill assertion is red."""
    events = []

    class Job:
        def command(self, command):
            return ["fixture-wrapper", *command]

        def assign(self, pid):
            events.append("assigned")

        def terminate(self):
            events.append("job-terminated")

        def close(self):
            events.append("job-closed")

    class Process:
        pid = 1234
        returncode = None

        def kill(self):
            events.append("wrapper-killed")

        async def communicate(self):
            events.append("drained")
            return b"", b""

    async def exercise():
        ready, release = asyncio.Event(), asyncio.Event()

        async def launch(*args, **kwargs):
            ready.set()
            await release.wait()
            return Process()

        monkeypatch.setattr(claude_exec.asyncio, "create_subprocess_exec", launch)
        task = asyncio.create_task(
            claude_exec.run_claude_case(
                {"task_id": "windows-launch", "user_input": "fixture"},
                RECALL_ON,
                claude_exec.ClaudeExecConfig(strict_mcp_config=False),
            )
        )
        await ready.wait()
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    monkeypatch.setattr(claude_exec, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(claude_exec, "WindowsJob", Job)
    monkeypatch.setattr(claude_exec.ClaudeExecConfig, "command", lambda self, prompt: ["fixture-cli"])
    asyncio.run(exercise())
    assert "wrapper-killed" in events, "an empty job cannot terminate its unassigned wrapper"
    assert "assigned" not in events, "cancellation must not release the CLI launch"
    assert "drained" in events and events[-1] == "job-closed"
