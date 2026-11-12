"""Print the HTTP headers a client sends to an authenticated RE-call MCP server, as JSON.

For Claude Code's `headersHelper` (run at session start and on reconnect) and Codex's
`http_headers_helper`. Each run mints a fresh `X-Recall-Session`, so the server can attribute
every call to the session that made it (see docs/AUTH.md, "Attributing calls to a session").

    RECALL_MCP_TOKEN_FILE   file holding the bearer token (required; keep it mode 600)
    RECALL_MCP_CLIENT_NAME  label sent as X-Recall-Client (default: mcp-client)

The token is read from a file, never from an argument or the environment, so it reaches neither
the process table nor a shell history. On any problem the script exits 2 with a message on
stderr and prints nothing on stdout, so a client never sends a half-built header set.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path


def main() -> int:
    path = os.environ.get("RECALL_MCP_TOKEN_FILE", "").strip()
    if not path:
        print("RECALL_MCP_TOKEN_FILE is not set", file=sys.stderr)
        return 2
    try:
        token = Path(path).read_text(encoding="utf-8").strip()
    except OSError as exc:
        print(f"cannot read RECALL_MCP_TOKEN_FILE: {exc.strerror}", file=sys.stderr)
        return 2
    if not token:
        print("RECALL_MCP_TOKEN_FILE is empty", file=sys.stderr)
        return 2
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Recall-Session": uuid.uuid4().hex,
        "X-Recall-Client": os.environ.get("RECALL_MCP_CLIENT_NAME", "").strip() or "mcp-client",
    }
    print(json.dumps(headers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
