"""Launch the preregistered one task Codex and RE-call smoke comparison."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.agent_ab import (
    CodexExecConfig,
    RECALL_OFF,
    RECALL_ON,
    make_codex_runner,
    run_paired,
    summarize_pairs,
    write_jsonl,
)

RUN_ID = "smoke-2026-08-19-rerun-03"
PROMPT = """Inspect the repository at the current working directory.
Do not modify files or use web search. If RE-call memory is available, first make exactly one
RE-call search for the phrase `paired Codex benchmark adapter`. Do not quote or reproduce any
memory content in your answer. If RE-call is unavailable, continue without it.
Return exactly three short lines:
1. The repository name.
2. One existing file under benchmarks/agent_ab.
3. Whether pyproject.toml exists in the repository root.
"""


def _codex_home(source_home: Path, root: Path, name: str, *, with_recall: bool) -> Path:
    home = root / name
    home.mkdir()
    auth = source_home / "auth.json"
    if auth.exists():
        shutil.copy2(auth, home / "auth.json")
    elif not os.environ.get("CODEX_API_KEY"):
        raise RuntimeError("Codex authentication is unavailable")
    config = [
        'model = "gpt-5.6-luna"',
        'model_reasoning_effort = "high"',
        'service_tier = "priority"',
        "",
    ]
    if with_recall:
        ssh_executable = _ssh_executable()
        ssh_config = str(Path.home() / ".ssh" / "config").replace("\\", "/")
        config.extend(
            [
                "[mcp_servers.recall-memory]",
                f'command = "{ssh_executable.replace(chr(92), "/")}"',
                f'args = ["-T", "-o", "BatchMode=yes", "-F", "{ssh_config}", "vps2", "cd ~/recall-repos && set -a && . ./.env && set +a && RECALL_TENANT=memory RECALL_EMBEDDER=voyage:voyage-4 RECALL_INDEX_ROOT=/home/sentiment/recall-repos/memory exec .venv/bin/python -m recall_mcp.server"]',
                "enabled = true",
                "startup_timeout_sec = 30",
                "",
            ]
        )
    (home / "config.toml").write_text("\n".join(config), encoding="utf-8")
    return home


def _codex_executable() -> str:
    configured = os.environ.get("CODEX_EXECUTABLE")
    if configured:
        return configured
    candidates = sorted(
        (Path.home() / "AppData" / "Local" / "OpenAI" / "Codex" / "bin").glob(
            "*/codex.exe"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return str(candidates[0])
    discovered = shutil.which("codex")
    if discovered:
        return discovered
    raise RuntimeError("Codex CLI executable was not found")


def _ssh_executable() -> str:
    configured = os.environ.get("RECALL_SSH_EXECUTABLE")
    if configured:
        return configured
    candidate = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "OpenSSH" / "ssh.exe"
    if candidate.exists():
        return str(candidate)
    discovered = shutil.which("ssh")
    if discovered:
        return discovered
    raise RuntimeError("OpenSSH executable was not found")


async def _run() -> dict[str, object]:
    worktree = Path(__file__).resolve().parents[1]
    source_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    artifact_root = worktree / "benchmarks" / "artifacts" / "codex_ab" / RUN_ID
    artifact_root.mkdir(parents=True, exist_ok=True)
    task = {
        "task_id": "smoke-readonly-001",
        "prompt": PROMPT,
        "cwd": str(worktree),
        "metadata": {"run_id": RUN_ID, "purpose": "wiring_smoke"},
    }
    with tempfile.TemporaryDirectory(prefix="codex-ab-") as temporary:
        temporary_root = Path(temporary)
        on_home = _codex_home(source_home, temporary_root, "on", with_recall=True)
        off_home = _codex_home(source_home, temporary_root, "off", with_recall=False)
        executable = _codex_executable()
        configs = {
            RECALL_ON: CodexExecConfig(
                executable=executable,
                cwd=worktree,
                env={"CODEX_HOME": str(on_home)},
                sandbox="danger-full-access",
                timeout_s=600,
            ),
            RECALL_OFF: CodexExecConfig(
                executable=executable,
                cwd=worktree,
                env={"CODEX_HOME": str(off_home)},
                sandbox="danger-full-access",
                timeout_s=600,
            ),
        }
        records = await run_paired([task], make_codex_runner(configs))
    write_jsonl(artifact_root / "records.jsonl", records)
    summary = summarize_pairs(records)
    (artifact_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "run_id": RUN_ID,
        "artifact_root": str(artifact_root),
        "summary": summary,
    }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(_run()), indent=2, sort_keys=True))
