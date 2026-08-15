#!/usr/bin/env bash
# Generate this checkout's .mcp.json.
#
# Why generated rather than committed: the internal servers authenticate with bearer tokens, and
# this repository is public (PyPI, GitHub, the MCP registry). `.mcp.json` is gitignored and built
# here from `~/.claude/recall-mcp-secrets.json`, so a token has no path into a commit even if
# somebody stages the whole tree. `.mcp.json.example` is the committed shape, with placeholders.
#
# Each git worktree is its own project root as far as the MCP client is concerned, so every
# worktree needs its own copy. Run this once per checkout.
#
# Usage:
#   scripts/session-mcp.sh            # write .mcp.json for this checkout
#   scripts/session-mcp.sh --check    # report what would be written, write nothing

set -euo pipefail

SECRETS="${RECALL_MCP_SECRETS:-$HOME/.claude/recall-mcp-secrets.json}"
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
OUT="$ROOT/.mcp.json"

# The corpus the dogfood server serves. Deliberately NOT a session container and NOT the shared
# `recall-db-1`: it is a long-lived, read-mostly index of recall's own docs, and the test suite
# must never point at it. Rebuild it with:
#   docker run -d --name recall-dogfood -e POSTGRES_USER=recall -e POSTGRES_PASSWORD=recall \
#     -e POSTGRES_DB=recall -p 127.0.0.1:5433:5432 -v recall_dogfood_pgdata:/var/lib/postgresql \
#     pgvector/pgvector:pg18
#   RECALL_MIGRATION_DSN=$DOGFOOD_DSN RECALL_DSN=$DOGFOOD_DSN python -m recall.cli schema apply
#   RECALL_DSN=$DOGFOOD_DSN RECALL_EMBEDDER=fastembed python -m recall.cli index docs
DOGFOOD_DSN="${RECALL_DOGFOOD_DSN:-postgresql://recall:recall@127.0.0.1:5433/recall}"

if [ ! -f "$SECRETS" ]; then
    cat >&2 <<EOF
session-mcp: no secrets file at $SECRETS

Create it with the bearer tokens for the internal servers:

  {
    "qwen-mcp":   "...",
    "qwen-vps3":  "...",
    "code-rag":   "...",
    "mcp-pg-ops": "..."
  }

Keep it outside the repository. recall is public.
EOF
    exit 1
fi

if [ "${1:-}" = "--check" ]; then
    echo "session-mcp: would write $OUT"
    echo "  secrets:      $SECRETS"
    echo "  dogfood DSN:  $DOGFOOD_DSN"
    python - "$SECRETS" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
have = [k for k in ("qwen-mcp", "qwen-vps3", "code-rag", "mcp-pg-ops") if d.get(k)]
missing = [k for k in ("qwen-mcp", "qwen-vps3", "code-rag", "mcp-pg-ops") if not d.get(k)]
print("  tokens found: " + (", ".join(have) or "none"))
if missing:
    print("  MISSING:      " + ", ".join(missing))
PY
    exit 0
fi

SECRETS="$SECRETS" OUT="$OUT" ROOT="$ROOT" DOGFOOD_DSN="$DOGFOOD_DSN" python <<'PY'
import json, os

secrets = json.load(open(os.environ["SECRETS"], encoding="utf-8"))


def bearer(name):
    token = secrets.get(name)
    if not token:
        raise SystemExit(f"session-mcp: no token for {name!r} in the secrets file")
    return {"Authorization": f"Bearer {token}"}


config = {
    "mcpServers": {
        # Recall serving its own documentation. This is the one server whose corpus is actually
        # this project, so it is the one worth reaching for first when the question is "what does
        # recall already do / already decide about X".
        "recall": {
            "type": "stdio",
            "command": "python",
            "args": ["-m", "recall_mcp.server"],
            "cwd": os.environ["ROOT"],
            "env": {
                "RECALL_DSN": os.environ["DOGFOOD_DSN"],
                "RECALL_EMBEDDER": "fastembed",
                # This index is a developer convenience over our own docs, bound to no tenant
                # generation and calibrated against nothing. Strict mode correctly refuses it.
                # Production must never be configured this way.
                "RECALL_TRUST_MODE": "development",
            },
        },
        # The remaining servers index /opt/sentiment_agent and query the sentiment_agent
        # database. They are reachable and useful, but their corpus is NOT recall: `code_search`
        # here will not find recall's source, and `db_query_ro` does not reach recall's tables.
        "code-rag": {
            "type": "http",
            "url": "http://100.91.148.25:8765/mcp",
            "headers": bearer("code-rag"),
        },
        "qwen-mcp": {
            "type": "http",
            "url": "http://100.91.148.25:49555/mcp",
            "headers": bearer("qwen-mcp"),
        },
        "qwen-vps3": {
            "type": "http",
            "url": "http://100.119.123.49:49555/mcp",
            "headers": bearer("qwen-vps3"),
        },
        "vps3-lite": {
            "type": "http",
            "url": "http://100.119.123.49:49556/mcp",
        },
        "mcp-pg-ops": {
            "type": "http",
            "url": "http://100.91.148.25:18003/mcp",
            "headers": bearer("mcp-pg-ops"),
        },
    }
}

out = os.environ["OUT"]
# newline="\n": this repo is eol=lf and Python would otherwise write CRLF on Windows.
with open(out, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(config, fh, indent=2)
    fh.write("\n")
print(f"session-mcp: wrote {out} ({len(config['mcpServers'])} servers)")
PY

# A generated file carrying tokens must be ignored before it exists, not after. If this ever
# fires, the ignore rule has been lost and the next `git add` would stage credentials.
if ! git -C "$ROOT" check-ignore -q .mcp.json 2>/dev/null; then
    echo "session-mcp: WARNING .mcp.json is NOT gitignored in this checkout." >&2
    echo "session-mcp: it now contains bearer tokens. Add '.mcp.json' to .gitignore before staging." >&2
    exit 1
fi
echo "session-mcp: confirmed .mcp.json is gitignored"
