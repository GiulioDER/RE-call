"""Codex payload adapter for the shared RE-call hook implementation.

The adapter covers exactly five events: session start, pre-compact, pre-tool-use, prompt submit
and session end. Codex has no post-tool event, so the general knowledge FAILURE path
(`post-tool-use-failure`) is Claude only and is deliberately not accepted here.

It marks every delegated run with `RECALL_HOOK_CLIENT=codex`, which is how the general knowledge
event log tells the two clients apart. The knowledge home, the log and the on/off switch are NOT
redirected with `RECALL_HOOK_CONFIG_HOME`: they are resolved from the project config home, so Claude
and Codex share one corpus, one log and one switch.
"""
from __future__ import annotations

import io
import json
import os
import sys


def _normalise(payload: object, event: str) -> dict[str, object]:
    data = dict(payload) if isinstance(payload, dict) else {}
    if event == "user-prompt-submit" and not data.get("prompt"):
        for key in ("user_prompt", "text", "message", "input"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                data["prompt"] = value
                break
    if not data.get("cwd"):
        data["cwd"] = os.getcwd()
    return data


def main(argv: list[str] | None = None) -> int:
    """Run one Codex hook while preserving the shared fail-open contract."""
    args = list(sys.argv[1:] if argv is None else argv)
    event = args[0] if args else ""
    if event not in {
        "session-start",
        "pre-compact",
        "pre-tool-use",
        "user-prompt-submit",
        "session-end",
    }:
        return 0
    try:
        # Codex frames one JSON payload per hook invocation. Read the complete frame: truncating a
        # large but valid prompt/tool payload would turn it into invalid JSON and silently lose the
        # memory check. The hook remains fail-open if the client sends malformed or unreasonable
        # input.
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        if event == "session-end":
            # Codex does not expose CLAUDE_PID. Its hook process and its local MCP child are both
            # descended from the Codex client process, so that concrete parent PID is the session
            # identity used by the shared cleanup path. A marker is included for transports whose
            # command line carries the client token instead of a usable parent chain.
            parent_pid = str(os.getppid())
            payload["_client_pid"] = parent_pid
            payload["_client_mark"] = os.environ.get("RECALL_MCP_CLIENT", f"codex-{parent_pid}")
            payload["_client_session_id"] = (
                os.environ.get("RECALL_MCP_SESSION_ID") or payload.get("session_id") or ""
            )
        from recall.codex import codex_integration_dir

        previous_config_home = os.environ.get("RECALL_HOOK_CONFIG_HOME")
        previous_project_home = os.environ.get("RECALL_PROJECT_CONFIG_HOME")
        previous_client = os.environ.get("RECALL_HOOK_CLIENT")
        os.environ["RECALL_HOOK_CONFIG_HOME"] = str(codex_integration_dir())
        from recall_hooks import main as recall_main

        previous_argv = sys.argv
        previous_stdin = sys.stdin
        try:
            sys.argv = ["recall-codex-hooks", event]
            os.environ["RECALL_HOOK_CLIENT"] = "codex"
            sys.stdin = io.StringIO(json.dumps(_normalise(payload, event)))
            return int(recall_main())
        finally:
            sys.argv = previous_argv
            sys.stdin = previous_stdin
            if previous_config_home is None:
                os.environ.pop("RECALL_HOOK_CONFIG_HOME", None)
            else:
                os.environ["RECALL_HOOK_CONFIG_HOME"] = previous_config_home
            if previous_project_home is None:
                os.environ.pop("RECALL_PROJECT_CONFIG_HOME", None)
            else:
                os.environ["RECALL_PROJECT_CONFIG_HOME"] = previous_project_home
            if previous_client is None:
                os.environ.pop("RECALL_HOOK_CLIENT", None)
            else:
                os.environ["RECALL_HOOK_CLIENT"] = previous_client
    except (KeyboardInterrupt, SystemExit):
        return 0
    except BaseException:
        # Memory hints never deny a Codex action or prevent a prompt.
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
