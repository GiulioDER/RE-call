#!/usr/bin/env bash
# Generate this checkout's .mcp.json.
#
# Why generated rather than committed: this repository is public (PyPI, GitHub, the MCP registry),
# and the internal servers are described by both a bearer token and a host address. **Both are
# disclosure.** A host inventory with no credentials still tells a reader which machines exist and
# what runs on them, so neither the URLs nor the tokens live in the tree. They come from
# `~/.claude/recall-mcp-secrets.json`, and `.mcp.json` is gitignored.
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

# The corpus the dogfood servers serve. Deliberately NOT a session container and NOT the shared
# `recall-db-1`: it is a long-lived, read-mostly index of this project's own docs and memory, and
# the test suite must never point at it. Rebuild:
#   docker run -d --name recall-dogfood -e POSTGRES_USER=recall -e POSTGRES_PASSWORD=recall \
#     -e POSTGRES_DB=recall -p 127.0.0.1:5433:5432 -v recall_dogfood_pgdata:/var/lib/postgresql \
#     pgvector/pgvector:pg18
#   RECALL_MIGRATION_DSN=$DSN RECALL_DSN=$DSN python -m recall.cli schema apply
#   RECALL_DSN=$DSN RECALL_EMBEDDER=fastembed python -m recall.cli index docs
# Index each tenant SEPARATELY. Re-indexing prunes sources that have vanished from disk, so
# pointing both corpora at one tenant deletes the other.
DOGFOOD_DSN="${RECALL_DOGFOOD_DSN:-postgresql://recall:recall@127.0.0.1:5433/recall}"

# The .mcp.json this writes carries secrets. Refuse to create it at all unless the ignore rule is
# already in place: checking afterwards would mean warning about a file that already exists on disk
# with real credentials in it.
if ! git -C "$ROOT" check-ignore -q .mcp.json 2>/dev/null; then
    echo "session-mcp: REFUSING to write .mcp.json — it is not gitignored in this checkout." >&2
    echo "session-mcp: the generated file carries bearer tokens and internal host addresses," >&2
    echo "session-mcp: and this repository is public. Add '.mcp.json' to .gitignore first." >&2
    exit 1
fi

if [ ! -f "$SECRETS" ]; then
    cat >&2 <<EOF
session-mcp: no secrets file at $SECRETS

Create it, OUTSIDE this repository:

  {
    "servers": {
      "code-rag":   { "url": "http://HOST:PORT/mcp", "token": "..." },
      "qwen-mcp":   { "url": "http://HOST:PORT/mcp", "token": "..." },
      "qwen-vps3":  { "url": "http://HOST:PORT/mcp", "token": "..." },
      "vps3-lite":  { "url": "http://HOST:PORT/mcp" },
      "mcp-pg-ops": { "url": "http://HOST:PORT/mcp", "token": "..." }
    }
  }

A server with no "token" is configured without an Authorization header.
EOF
    exit 1
fi

# Only two forms are accepted, and anything else is refused rather than ignored. Falling through
# on an unrecognised argument means `session-mcp.sh --dry-run` and `session-mcp.sh --help` both
# WRITE a file full of credentials, which is the opposite of what either name promises.
case "${1:-}" in
    ""|--check) ;;
    *)
        echo "usage: scripts/session-mcp.sh [--check]" >&2
        echo "  (no args)  write .mcp.json for this checkout" >&2
        echo "  --check    report what would be written, write nothing" >&2
        exit 2
        ;;
esac

if [ "${1:-}" = "--check" ]; then
    echo "session-mcp: would write $OUT"
    echo "  secrets:      $SECRETS"
    echo "  dogfood DSN:  $DOGFOOD_DSN"
    echo "  .mcp.json is gitignored: yes"
    SECRETS="$SECRETS" python <<'PY'
import json, os
d = json.load(open(os.environ["SECRETS"], encoding="utf-8")).get("servers", {})
print(f"  remote servers configured: {len(d)}")
for name, cfg in sorted(d.items()):
    print(f"    - {name} (auth: {'yes' if cfg.get('token') else 'none'})")
PY
    exit 0
fi

SECRETS="$SECRETS" OUT="$OUT" ROOT="$ROOT" DOGFOOD_DSN="$DOGFOOD_DSN" python <<'PY'
import json, os

secrets = json.load(open(os.environ["SECRETS"], encoding="utf-8"))
remote = secrets.get("servers")
if not remote:
    raise SystemExit("session-mcp: the secrets file has no 'servers' object")

root, dsn = os.environ["ROOT"], os.environ["DOGFOOD_DSN"]


def dogfood(tenant=None):
    """One stdio server against the dogfood corpus, optionally scoped to a tenant."""
    env = {"RECALL_DSN": dsn, "RECALL_EMBEDDER": "fastembed"}
    if tenant:
        env["RECALL_TENANT"] = tenant
    return {
        "type": "stdio",
        "command": "python",
        "args": ["-m", "recall_mcp.server"],
        "cwd": root,
        "env": env,
    }


servers = {
    # This project's own docs, and its own memory store. The only servers whose corpus is this
    # repository. Note their recall_search currently refuses: the MCP server passes no TrustPolicy,
    # so it is strict regardless of RECALL_TRUST_MODE, and an uncalibrated corpus is INDEX_NOT_READY.
    "recall": dogfood(),
    "recall-memory": dogfood("memory"),
}

for name, cfg in remote.items():
    url = cfg.get("url")
    if not url:
        raise SystemExit(f"session-mcp: server {name!r} has no url")
    entry = {"type": "http", "url": url}
    if cfg.get("token"):
        entry["headers"] = {"Authorization": f"Bearer {cfg['token']}"}
    servers[name] = entry

out = os.environ["OUT"]
# newline="\n": this repo is eol=lf and Python would otherwise write CRLF on Windows.
with open(out, "w", encoding="utf-8", newline="\n") as fh:
    json.dump({"mcpServers": servers}, fh, indent=2)
    fh.write("\n")
print(f"session-mcp: wrote {out} ({len(servers)} servers)")
PY
