"""Does a config dir isolate as well as `--bare`, or does the developer's CLAUDE.md leak in?

The nonce probe answered its own question and raised a worse one: the agent quoted this machine's
CLAUDE.md back, at 94,980 input tokens for a one-file task. The stage A streams show no leak
markers, but the same probe proved the stream does not record system-reminders at all, so absence
there is not evidence. The token count is, because a leaked CLAUDE.md is tens of thousands of them.

One identical trivial prompt, no tools, one turn, under four conditions: each isolation mode
inside a repo sandbox and outside the repository, because the two leaks have different causes.
The comparison is the measurement; the absolute numbers move with what the leaked files say.
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.agent_ab.claude_exec import ClaudeExecConfig, run_claude_case  # noqa: E402
from benchmarks.agent_ab.schema import RECALL_OFF  # noqa: E402

SCRATCH = Path(__file__).resolve().parents[1] / "benchmarks" / "artifacts" / "isolation-probe"
PROMPT = (
    "Answer in one line. Do you have a document titled 'User-level notes' or a section about "
    "'No dash as punctuation' anywhere in your context? Reply LEAKED if you do, CLEAN if you "
    "do not."
)


async def one(label: str, *, bare: bool, config_dir: Path | None, cwd: Path) -> None:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        # The subscription token answers 401 and exits 0, so an unset key would print four
        # plausible rows of nonsense rather than failing.
        raise SystemExit("OPENROUTER_API_KEY is not set")
    config = ClaudeExecConfig(
        model="anthropic/claude-haiku-4.5",
        cwd=cwd,
        timeout_s=180.0,
        env={"ANTHROPIC_BASE_URL": "https://openrouter.ai/api",
             "ANTHROPIC_AUTH_TOKEN": key,
             "ANTHROPIC_API_KEY": ""},
        bare=bare,
        config_dir=config_dir,
        strict_mcp_config=False,
        allowed_tools=(),
    )
    record = await run_claude_case(
        {"task_id": label, "user_input": PROMPT}, RECALL_OFF, config
    )
    response = (record.response or "").strip().replace("\n", " ")[:120]
    print(f"{label:24s} in={record.input_tokens:>8}  {response.encode('ascii', 'replace').decode()}")


async def main() -> int:
    work = SCRATCH / "iso-work"
    work.mkdir(parents=True, exist_ok=True)
    empty = SCRATCH / "iso-config"
    empty.mkdir(parents=True, exist_ok=True)
    (empty / "settings.json").write_text("{}\n", encoding="utf-8", newline="\n")

    # 1. What every earlier result in this lane ran under.
    await one("bare, outside repo", bare=True, config_dir=None, cwd=work)
    # 2. What the hook run must use, in the same directory.
    await one("config-dir, outside", bare=False, config_dir=empty, cwd=work)
    # 3. The same, in a REPO sandbox, which is where the real sessions run.
    sandbox = Path(__file__).resolve().parents[1] / "benchmarks"
    if sandbox.is_dir():
        await one("config-dir, in sandbox", bare=False, config_dir=empty, cwd=sandbox)
        await one("bare, in sandbox", bare=True, config_dir=None, cwd=sandbox)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
