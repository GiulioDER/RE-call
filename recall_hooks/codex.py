"""Codex payload adapter for the shared RE-call hook implementation."""
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
        from recall.codex import codex_integration_dir

        previous_config_home = os.environ.get("RECALL_HOOK_CONFIG_HOME")
        previous_project_home = os.environ.get("RECALL_PROJECT_CONFIG_HOME")
        os.environ["RECALL_HOOK_CONFIG_HOME"] = str(codex_integration_dir())
        from recall_hooks import main as recall_main

        previous_argv = sys.argv
        previous_stdin = sys.stdin
        try:
            sys.argv = ["recall-codex-hooks", event]
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
    except (KeyboardInterrupt, SystemExit):
        return 0
    except BaseException:
        # Memory hints never deny a Codex action or prevent a prompt.
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
