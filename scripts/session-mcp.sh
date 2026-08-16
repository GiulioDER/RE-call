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

if [ ! -f "$SECRETS" ] && [ -n "${RECALL_MCP_INCLUDE_REMOTE:-}" ]; then
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
try:
    d = json.load(open(os.environ["SECRETS"], encoding="utf-8")).get("servers", {})
except OSError:
    d = {}
included = bool(os.environ.get("RECALL_MCP_INCLUDE_REMOTE"))
print("  would write: recall, recall-memory (this project's own corpus)")
print(f"  remote servers available: {len(d)}, included: {'YES' if included else 'no'}")
for name, cfg in sorted(d.items()):
    print(f"    - {name} (auth: {'yes' if cfg.get('token') else 'none'})")
if not included:
    print("  their corpus is /opt/sentiment_agent, not this repository.")
    print("  set RECALL_MCP_INCLUDE_REMOTE=1 to add them anyway.")
PY
    exit 0
fi

SECRETS="$SECRETS" OUT="$OUT" ROOT="$ROOT" DOGFOOD_DSN="$DOGFOOD_DSN" python <<'PY'
import json, os

# Only the remote half needs the secrets file, so a checkout without one still
# gets the two local dogfood servers instead of nothing. Requiring it
# unconditionally survived the change that made the remote servers opt-in, and
# turned "no secrets file" into "no MCP at all".
want_remote = bool(os.environ.get("RECALL_MCP_INCLUDE_REMOTE"))
remote = {}
if want_remote:
    try:
        remote = json.load(open(os.environ["SECRETS"], encoding="utf-8")).get("servers") or {}
    except OSError:
        raise SystemExit(f"session-mcp: no secrets file at {os.environ['SECRETS']}") from None
    if not remote:
        raise SystemExit("session-mcp: the secrets file has no 'servers' object")

root, dsn = os.environ["ROOT"], os.environ["DOGFOOD_DSN"]


def dogfood(tenant=None):
    """One stdio server against the dogfood corpus, optionally scoped to a tenant.

    RECALL_TRUST_MODE is set HERE and not left to the operator's shell. An MCP stdio server
    launched with an explicit `env` block does not inherit exported variables, so a corpus that
    searches fine from the terminal still answered INDEX_NOT_READY through the client. These two
    servers are local, uncalibrated, developer-only corpora, which is exactly the case the relaxed
    mode exists for; nothing else should be configured this way.
    """
    env = {
        "RECALL_DSN": dsn,
        "RECALL_EMBEDDER": "fastembed",
        "RECALL_TRUST_MODE": "development",
    }
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
    # repository. They run with RECALL_TRUST_MODE=development, set in the env block above, because
    # these corpora are uncalibrated and bound to no generation; a strict server refuses them with
    # INDEX_NOT_READY, which is correct for anything but a local dogfood index.
    "recall": dogfood(),
    "recall-memory": dogfood("memory"),
}

# The internal servers are OFF by default here, and that is a deliberate reversal.
#
# Their corpus is /opt/sentiment_agent: `code_search` does not search recall, and `db_query_ro`
# does not reach recall's tables. Asked about this repository they do not error, they answer
# confidently about the wrong one, which is the most expensive failure mode on this list. They also
# cost standing context in every session: `mcp-pg-ops` alone is 38 tool definitions against 2 calls
# ever, measured over 147,119 tool calls.
#
# They remain one variable away for the sessions that genuinely want them, and they are configured
# by default in sentiment-agent's own checkouts, where that corpus IS the project:
#
#   RECALL_MCP_INCLUDE_REMOTE=1 scripts/session-mcp.sh
if want_remote:
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

# Writing the file is only half of it. A project-scoped server sits at "pending approval" until
# the CLIENT has recorded the approval for this directory, and a non-interactive session can
# never answer that prompt. Measured 2026-08-17: 306 tracked projects on this machine, zero with
# an approved .mcp.json server, while `claude mcp list` reported both recall servers as
# "⏸ Pending approval" with the file sitting on disk in front of it.
#
# So the correct fix for "the servers never load" is not to write the file EARLIER, it is to
# approve it. Only the names cross into the client config; the definitions and the secrets stay
# here. See scripts/session_mcp_approve.py for what this deliberately does not do.
#
# Never fatal: a session whose servers stay pending is degraded, not stopped.
python "$ROOT/scripts/session_mcp_approve.py" --root "$ROOT" --from-mcp-json "$OUT" || \
    echo "session-mcp: approval step failed; servers will stay pending" >&2
