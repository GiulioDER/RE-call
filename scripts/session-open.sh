#!/usr/bin/env bash
# Open a session in this checkout.
#
# Read-only except for writing `.mcp.json`. It does NOT start a database: most sessions never run
# the DB-backed suite, and starting a container for them is how containers get stranded. It tells
# you the one command to run when you do need one.

set -uo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

say() { printf '\n\033[1m%s\033[0m\n' "$1"; }

say "Checkout"
printf '  path    %s\n' "$ROOT"
# Ask git for the branch. The directory name is not the branch, and acting on the directory name
# has put commits on the wrong branch here before.
printf '  branch  %s\n' "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '(not a git repo)')"
printf '  head    %s\n' "$(git log --oneline -1 2>/dev/null || echo '-')"

say "Position against origin/master"
if git rev-parse --verify -q origin/master >/dev/null; then
    # Against origin/master, never the local ref: the local one here is routinely stale, and a
    # stale base looks exactly like a regression.
    behind="$(git rev-list --count HEAD..origin/master 2>/dev/null || echo '?')"
    ahead="$(git rev-list --count origin/master..HEAD 2>/dev/null || echo '?')"
    printf '  %s commit(s) behind, %s ahead\n' "$behind" "$ahead"
    if [ "$behind" != "0" ] && [ "$behind" != "?" ]; then
        printf '  note: behind origin/master. Rebase before blaming a test.\n'
    fi
else
    printf '  origin/master not found (run `git fetch origin`)\n'
fi

say "Uncommitted changes"
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    git status --short | head -20
    printf '  (another session may share this checkout; do not stage what you did not write)\n'
else
    printf '  clean\n'
fi

say "MCP config"
if [ -f "$ROOT/.mcp.json" ]; then
    # Path via argv, not interpolated into a Python literal: a path containing a quote or ending
    # in a backslash would otherwise produce a syntax error reported as an unknown server count.
    printf '  .mcp.json present (%s servers)\n' \
        "$(MCP_PATH="$ROOT/.mcp.json" python -c \
            'import json,os;print(len(json.load(open(os.environ["MCP_PATH"],encoding="utf-8"))["mcpServers"]))' \
            2>/dev/null || echo '?')"
else
    printf '  generating...\n'
    # The generator refuses when .mcp.json is not gitignored, and that refusal is the one thing in
    # this report that must not scroll past as an indented line. Stop the session on it.
    if ! bash "$ROOT/scripts/session-mcp.sh" 2>&1 | sed 's/^/  /'; then
        printf '\n  SESSION SETUP FAILED: see the message above. Not continuing.\n'
        exit 1
    fi
fi

say "Database"
bash "$ROOT/scripts/session-db.sh" status 2>&1 | sed 's/^/  /'
printf '  start one only if you will run the DB suite:\n'
printf '    eval "$(scripts/session-db.sh up)"\n'

say "Stranded containers"
bash "$ROOT/scripts/session-db.sh" orphans 2>&1 | sed 's/^/  /'

printf '\n'
