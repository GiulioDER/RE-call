#!/usr/bin/env bash
# Generate this checkout's .mcp.json.
#
# Why generated rather than committed: this repository is public (PyPI, GitHub, the MCP registry),
# and a server is described by both a bearer token and a host address. **Both are disclosure.** A
# host inventory with no credentials still tells a reader which machines exist and what runs on
# them, so no URL, host, account name or absolute path lives in the tree. They come from
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

# WHERE THE CORPORA ARE, AND WHY THE SERVER RUNS THERE RATHER THAN HERE
#
# 🔁 Rewritten 2026-08-30. This script used to point every recall server at a local
# `recall-dogfood` container on 127.0.0.1:5433 with RECALL_TRUST_MODE=development. All three parts
# were wrong by then, and each one had been documented as wrong in CLAUDE.md before this file was
# read again:
#
#   1. The container was removed on 2026-08-25 ("Nothing should be pointed at 5433"). Every server
#      this script wrote died on ConnectionRefused.
#   2. An MCP `env` block REPLACES the environment rather than extending it, so a local server
#      launched with only RECALL_* keys never had PATH or APPDATA and died on
#      `ModuleNotFoundError: anyio` before it could reach the database anyway.
#   3. RECALL_TRUST_MODE=development is "actively wrong" for a certified corpus: it marks a trusted
#      answer `degraded` and forces `calibrated` false, and because a relaxed gate never errors,
#      nothing reports it.
#
# The corpora now live on one host and are certified there, so the server runs where they are, over
# ssh stdio, with strict trust expressed by the ABSENCE of RECALL_TRUST_MODE. Strict-by-omission is
# deliberate: setting that variable to any string at all is how a corpus ends up served relaxed
# while the config claims otherwise.

# The .mcp.json this writes carries host addresses. Refuse to create it at all unless the ignore
# rule is already in place: checking afterwards would mean warning about a file that already exists
# on disk with real credentials in it.
if ! git -C "$ROOT" check-ignore -q .mcp.json 2>/dev/null; then
    echo "session-mcp: REFUSING to write .mcp.json — it is not gitignored in this checkout." >&2
    echo "session-mcp: the generated file carries bearer tokens and internal host addresses," >&2
    echo "session-mcp: and this repository is public. Add '.mcp.json' to .gitignore first." >&2
    exit 1
fi

if [ ! -f "$SECRETS" ]; then
    cat >&2 <<EOF
session-mcp: no secrets file at $SECRETS

Create it, OUTSIDE this repository. The "recall_corpus" object is REQUIRED -- it is what tells this
script where recall's own corpora are served, and without it a session gets no memory at all:

  {
    "recall_corpus": {
      "ssh_host": "HOST",
      "python":   "/path/to/venv/bin/python",
      "env_file": "/path/to/.env",
      "workdir":  "/path/to/checkout",
      "tenants": {
        "recall-memory": { "tenant": "memory",  "embedder": "voyage:voyage-4" }
      }
    },
    "servers": {
      "vps3-lite":  { "url": "http://HOST:PORT/mcp" },
      "mcp-pg-ops": { "url": "http://HOST:PORT/mcp", "token": "..." }
    }
  }

Each tenant needs the embedder its ACTIVE generation was built with. A wrong choice among models of
the same dimension does NOT error: pgvector computes a cosine over whatever produced the vectors and
returns a confidently ranked list that means nothing. Read them from the corpus rather than guessing:

  select tenant_id, pipeline_identity->>'embedder' from recall_generations where state='active';

A server under "servers" with no "token" is configured without an Authorization header.
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
    echo "  .mcp.json is gitignored: yes"
    SECRETS="$SECRETS" python <<'PY'
import json, os
d = json.load(open(os.environ["SECRETS"], encoding="utf-8"))
corpus = d.get("recall_corpus") or {}
tenants = corpus.get("tenants") or {}
remote = d.get("servers", {})
included = bool(os.environ.get("RECALL_MCP_INCLUDE_REMOTE"))
if tenants:
    print(f"  recall servers ({len(tenants)}, this project's own corpora, strict trust):")
    for name, cfg in sorted(tenants.items()):
        print(f"    - {name} -> tenant {cfg.get('tenant')} via {cfg.get('embedder')}")
else:
    print("  recall servers: NONE — 'recall_corpus.tenants' is missing or empty.")
    print("  a session written from this will have no memory. See the secrets template above.")
print(f"  remote servers available: {len(remote)}, included: {'YES' if included else 'no'}")
for name, cfg in sorted(remote.items()):
    print(f"    - {name} (auth: {'yes' if cfg.get('token') else 'none'})")
if not included:
    print("  their corpus is /opt/sentiment_agent, not this repository.")
    print("  set RECALL_MCP_INCLUDE_REMOTE=1 to add them anyway.")
PY
    exit 0
fi

SECRETS="$SECRETS" OUT="$OUT" python <<'PY'
import json, os

secrets = json.load(open(os.environ["SECRETS"], encoding="utf-8"))
want_remote = bool(os.environ.get("RECALL_MCP_INCLUDE_REMOTE"))

corpus = secrets.get("recall_corpus") or {}
tenants = corpus.get("tenants") or {}
if not tenants:
    raise SystemExit(
        "session-mcp: the secrets file has no 'recall_corpus.tenants'.\n"
        "session-mcp: REFUSING to write a config with no memory servers in it. A session that\n"
        "session-mcp: starts without them cannot tell 'recall is down' from 'recall found\n"
        "session-mcp: nothing', and will quietly work from whatever it happens to remember."
    )
missing = [k for k in ("ssh_host", "python", "env_file", "workdir") if not corpus.get(k)]
if missing:
    raise SystemExit(f"session-mcp: recall_corpus is missing {', '.join(missing)}")


def corpus_server(tenant, embedder):
    """One recall MCP server, run where the corpus lives, over ssh stdio.

    Two things here are load-bearing and neither is obvious from the shape:

    `set -a; . env_file; set +a` sources the credentials the embedder needs (the Voyage key), which
    a non-interactive ssh command does NOT get from a login profile.

    `unset RECALL_TRUST_MODE` makes strict the DEFAULT rather than a hope. These corpora are
    certified and bound to an active generation, so strict is correct for all of them; leaving the
    variable to whatever the remote environment holds is how a certified corpus ends up answering
    `degraded` with nothing raising.
    """
    remote_cmd = (
        f"cd {corpus['workdir']} && "
        f"set -a && . {corpus['env_file']} && set +a && "
        f"unset RECALL_TRUST_MODE && "
        f"RECALL_ENV=production RECALL_TENANT={tenant} RECALL_EMBEDDER={embedder} "
        f"{corpus['python']} -m recall_mcp.server"
    )
    return {
        "type": "stdio",
        "command": "ssh",
        # BatchMode: fail immediately rather than blocking a session start on a password prompt
        # nobody is there to answer.
        "args": ["-o", "BatchMode=yes", corpus["ssh_host"], remote_cmd],
    }


# This project's own corpora: its memory store, its generated code index, its docs. The only
# servers whose corpus is this repository.
servers = {
    name: corpus_server(cfg["tenant"], cfg["embedder"])
    for name, cfg in sorted(tenants.items())
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
    remote = secrets.get("servers") or {}
    if not remote:
        raise SystemExit("session-mcp: the secrets file has no 'servers' object")
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
for name in sorted(tenants):
    print(f"  {name}: tenant {tenants[name]['tenant']}, strict trust")
PY

# Writing the file is only half of it. A project-scoped server sits at "pending approval" until
# the CLIENT has recorded the approval for this directory, and an interactive session that never
# answers the prompt leaves it there forever. Measured 2026-08-17: 306 tracked projects on this
# machine, zero with an approved .mcp.json server, while `claude mcp list` reported both recall
# servers as "⏸ Pending approval" with the file sitting on disk in front of it.
#
# So the correct fix for "the servers never load" is not to write the file EARLIER, it is to
# approve it. Only the names cross into the client config; the definitions and the secrets stay
# here. See scripts/session_mcp_approve.py for what this deliberately does not do.
#
# Never fatal: a session whose servers stay pending is degraded, not stopped.
python "$ROOT/scripts/session_mcp_approve.py" --root "$ROOT" --from-mcp-json "$OUT" || \
    echo "session-mcp: approval step failed; servers will stay pending" >&2
