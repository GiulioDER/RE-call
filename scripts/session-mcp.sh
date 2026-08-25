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

# Where this project's own corpora actually live, and why the server runs THERE.
#
# 🔁 Corrected 2026-08-25. This used to generate two stdio servers against a local
# `recall-dogfood` container on 127.0.0.1:5433, with `RECALL_TRUST_MODE=development` hardcoded and
# a comment calling those corpora "uncalibrated". Every part of that had stopped being true, and
# it failed in the direction that HIDES a working system:
#
#   - the container was not running, and nothing said so. `session-open.sh` reported
#     ".mcp.json present (2 servers)" and a session read that as healthy;
#   - the corpora are on VPS2, not here;
#   - they are CERTIFIED, not uncalibrated. Measured 2026-08-25 against the live database: all
#     three tenants resolve `certified`, and a STRICT server answers `trust_state=trusted,
#     calibrated=true`. The relaxed mode was a workaround for a condition that had since been
#     fixed, and leaving it on downgraded a trusted corpus to `degraded` on every query.
#
# The corpus postgres on VPS2 listens on 127.0.0.1:55432 ONLY (loopback), and the memory and code
# tenants are embedded with hosted Voyage models whose API key lives in VPS2's own `.env`. So the
# server runs ON VPS2 over ssh stdio: corpus, key and models are already there, no tunnel is
# needed, and `.mcp.json` carries NO secret at all because it sources that `.env` on the far side.
#
# ⚠️ The serving checkout must sit at the DATABASE's migration level, which is why this is a
# variable rather than `engine`. Measured 2026-08-25: the database is at 0016, while
# `~/recall-repos/engine` is recall 0.9.6 and knows only up to 0014, so the server refuses to
# start against it with `SchemaTooNew: table 'chunks' has unknown schema migration(s)
# ['0015','0016']`. That refusal is correct and loud. It is also why a stale default here costs a
# session its tools, silently, from the client's point of view.
VPS2_HOST="${RECALL_VPS2_HOST:-vps2}"
VPS2_CHECKOUT="${RECALL_VPS2_CHECKOUT:-~/recall-repos/graph-annotations-6d3aeb28}"
VPS2_PYTHON="${RECALL_VPS2_PYTHON:-~/recall-repos/.venv/bin/python}"
VPS2_ENV_FILE="${RECALL_VPS2_ENV:-~/recall-repos/.env}"

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
    echo "  VPS2 host:    $VPS2_HOST"
    echo "  VPS2 checkout:$VPS2_CHECKOUT"
    echo "  .mcp.json is gitignored: yes"
    SECRETS="$SECRETS" python <<'PY'
import json, os
try:
    d = json.load(open(os.environ["SECRETS"], encoding="utf-8")).get("servers", {})
except OSError:
    d = {}
included = bool(os.environ.get("RECALL_MCP_INCLUDE_REMOTE"))
docs = bool(os.environ.get("RECALL_MCP_INCLUDE_DOCS"))
print("  would write: recall-memory, recall-code (this project's own corpora, on VPS2)")
print(f"  docs tenant included: {'YES' if docs else 'no'}"
      f"{'' if docs else '  (bge-large goes resident on VPS2; RECALL_MCP_INCLUDE_DOCS=1)'}")
print(f"  remote servers available: {len(d)}, included: {'YES' if included else 'no'}")
for name, cfg in sorted(d.items()):
    print(f"    - {name} (auth: {'yes' if cfg.get('token') else 'none'})")
if not included:
    print("  their corpus is /opt/sentiment_agent, not this repository.")
    print("  set RECALL_MCP_INCLUDE_REMOTE=1 to add them anyway.")
PY
    exit 0
fi

SECRETS="$SECRETS" OUT="$OUT" ROOT="$ROOT" VPS2_HOST="$VPS2_HOST" \
VPS2_CHECKOUT="$VPS2_CHECKOUT" VPS2_PYTHON="$VPS2_PYTHON" VPS2_ENV_FILE="$VPS2_ENV_FILE" \
python <<'PY'
import json, os, shlex

# Only the remote half needs the secrets file, so a checkout without one still
# gets the recall servers instead of nothing. Requiring it unconditionally
# survived the change that made the remote servers opt-in, and turned "no
# secrets file" into "no MCP at all".
want_remote = bool(os.environ.get("RECALL_MCP_INCLUDE_REMOTE"))
remote = {}
if want_remote:
    try:
        remote = json.load(open(os.environ["SECRETS"], encoding="utf-8")).get("servers") or {}
    except OSError:
        raise SystemExit(f"session-mcp: no secrets file at {os.environ['SECRETS']}") from None
    if not remote:
        raise SystemExit("session-mcp: the secrets file has no 'servers' object")

host = os.environ["VPS2_HOST"]
checkout = os.environ["VPS2_CHECKOUT"]
python_bin = os.environ["VPS2_PYTHON"]
env_file = os.environ["VPS2_ENV_FILE"]


def vps2(tenant, embedder):
    """One stdio server run ON VPS2, against the certified generation for `tenant`.

    Three things are deliberate and each one was a measured failure before it was a rule.

    **The embedder is passed per tenant, never defaulted.** The three tenants are all 1024
    dimensions and all three are DIFFERENT models (memory `voyage:voyage-4`, code
    `voyage:voyage-code-3`, docs `BAAI/bge-large-en-v1.5`). pgvector computes a cosine between any
    two 1024-vectors without complaint, so the wrong embedder here does not raise, it returns a
    confidently ranked list that means nothing. The old config sent `RECALL_EMBEDDER=fastembed`
    (bge-small, 384d) to every server, which at least failed loudly on width; a 1024d mismatch
    would not have.

    **RECALL_ENV=production is required for the generation path.** Without it the server uses the
    legacy store, which knows nothing about generations, so every search reports
    `generation=None, calibration_status=missing` and a strict policy refuses it. That symptom
    reads exactly like a broken calibration and has been misdiagnosed as one.

    **RECALL_TRUST_MODE is NOT set, so the strict default stands.** These corpora are certified;
    relaxing the gate here would mark a trusted answer `degraded` and drop `calibrated` to false.

    The secrets stay on VPS2: the command sources that host's own `.env`, so nothing sensitive is
    written into `.mcp.json`. `exec` replaces the shell so the server is the process ssh owns, and
    a client that closes the pipe kills the server rather than orphaning it behind a live shell.
    """
    inner = (
        f"cd {checkout} && set -a && . {env_file} && set +a && "
        f"export RECALL_TENANT={shlex.quote(tenant)} "
        f"RECALL_EMBEDDER={shlex.quote(embedder)} RECALL_ENV=production && "
        f"unset RECALL_TRUST_MODE && exec {python_bin} -m recall_mcp.server"
    )
    return {
        "type": "stdio",
        "command": "ssh",
        # BatchMode: a server that blocks on a passphrase prompt is a server the client waits on
        # forever. Fail immediately instead, so the tool list is visibly short rather than late.
        "args": ["-o", "BatchMode=yes", host, inner],
    }


servers = {
    # This project's own corpora, on VPS2, each bound to a certified calibration. These are the
    # only servers whose corpus is this repository. Measured 2026-08-25, all three resolve
    # `certified`; `recall-memory` was additionally driven end to end and answered
    # `trust_state=trusted, calibrated=true` with the strict default in force.
    "recall-memory": vps2("memory", "voyage:voyage-4"),
    "recall-code": vps2("re-call-code-gen", "voyage:voyage-code-3"),
}

# The docs tenant is OFF by default, and the reason is VPS2's memory rather than its usefulness.
# `re-call-docs` is embedded with `BAAI/bge-large-en-v1.5`, a local ONNX model that goes resident
# in EVERY stdio server session on that host, which also runs the live trading services. The two
# servers above are hosted-API embedders and load no model at all. Turn it on deliberately:
#
#   RECALL_MCP_INCLUDE_DOCS=1 scripts/session-mcp.sh
#
# ⚠️ Its active generation was promoted 2026-08-20 and its calibration is bound to that
# generation, so it is certified but it does NOT contain docs written since. Certification binds
# to a generation, never to what is on disk today.
if os.environ.get("RECALL_MCP_INCLUDE_DOCS"):
    servers["recall"] = vps2("re-call-docs", "BAAI/bge-large-en-v1.5")

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
