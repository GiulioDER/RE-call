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
    path = _config_path()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"recall-codex-mcp: cannot read {path}: {exc}", file=sys.stderr)
        return 2
    if not isinstance(document, dict) or not str(document.get("dsn", "")).strip():
        print(f"recall-codex-mcp: {path} has no usable dsn", file=sys.stderr)
        return 2
    # The installer-owned file is the source of truth. Codex inherits the user's environment, which
    # may contain stale RECALL_* values from another project; set these explicitly or the server can
    # silently serve the wrong corpus.
    keys = ("RECALL_SERVING_DSN", "RECALL_TENANT", "RECALL_EMBEDDER", "RECALL_TABLE")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["RECALL_SERVING_DSN"] = str(document["dsn"])
        os.environ["RECALL_TENANT"] = str(document.get("tenant", "default"))
        os.environ["RECALL_EMBEDDER"] = str(document.get("embedder", "fastembed"))
        os.environ["RECALL_TABLE"] = str(document.get("table", "chunks"))
        from recall_mcp.server import main as server_main

        server_main()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
