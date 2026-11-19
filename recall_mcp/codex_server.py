"""Launch the RE-call MCP server from Codex's protected integration config."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _config_path() -> Path:
    raw = os.environ.get("RECALL_CODEX_CONFIG", "").strip()
    if raw:
        return Path(raw).expanduser()
    codex_home = os.environ.get("CODEX_HOME", "").strip()
    root = Path(codex_home).expanduser() if codex_home else Path.home() / ".codex"
    return root / "re-call" / "recall-hook.json"


def main() -> int:
    from recall._env import refuse_current_directory_executables

    refuse_current_directory_executables()
    path = _config_path()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"recall-codex-mcp: cannot read {path}: {exc}", file=sys.stderr)
        return 2
    if not isinstance(document, dict) or not str(document.get("dsn", "")).strip():
        print(f"recall-codex-mcp: {path} has no usable dsn", file=sys.stderr)
        return 2
    # The other choices `recall setup` made for this server (reranking, the reasoning arm, a cloud
    # embedder's key). The server never reads `.env`, so without this block they never reached it
    # (issues 1393, 1394). Refused when malformed rather than half applied.
    server_env = document.get("server_env", {})
    if not isinstance(server_env, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in server_env.items()
    ):
        print(f"recall-codex-mcp: {path} has a server_env that is not an object of strings", file=sys.stderr)
        return 2
    # The installer-owned file is the source of truth. Codex inherits the user's environment, which
    # may contain stale RECALL_* values from another project; set these explicitly or the server can
    # silently serve the wrong corpus.
    core = ("RECALL_SERVING_DSN", "RECALL_TENANT", "RECALL_EMBEDDER", "RECALL_TABLE", "RECALL_TRANSPORT")
    keys = tuple(dict.fromkeys((*server_env, *core)))
    previous = {key: os.environ.get(key) for key in keys}
    previous_mark = os.environ.get("RECALL_MCP_CLIENT")
    try:
        # Before the five keys below, so `server_env` cannot redirect the corpus.
        os.environ.update(server_env)
        os.environ["RECALL_SERVING_DSN"] = str(document["dsn"])
        os.environ["RECALL_TENANT"] = str(document.get("tenant", "default"))
        os.environ["RECALL_EMBEDDER"] = str(document.get("embedder", "fastembed"))
        os.environ["RECALL_TABLE"] = str(document.get("table", "chunks"))
        os.environ["RECALL_TRANSPORT"] = "stdio"
        # Codex does not provide CLAUDE_PID. The parent is the concrete Codex client process, so
        # stamp its identity into the server environment for the SessionEnd hook to match.
        os.environ.setdefault("RECALL_MCP_CLIENT", f"codex-{os.getppid()}")
        from recall_mcp.server import main as server_main

        server_main()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if previous_mark is None:
            os.environ.pop("RECALL_MCP_CLIENT", None)
        else:
            os.environ["RECALL_MCP_CLIENT"] = previous_mark
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
