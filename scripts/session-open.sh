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

# FIRST, before reporting anything else. There is no point describing the branch and the containers
# of a workspace this session must not be working in, and a refusal buried under four sections of
# report is a refusal that gets scrolled past. Twenty worktrees, five of them dirty, and the main
# checkout itself dirty on a two-month-old branch is what this is for.
say "Workspace"
# Fail CLOSED when the guard itself is missing. A checkout old enough not to carry
# `session-space.sh` is exactly the checkout most likely to be a stale shared one, so "the guard is
# absent" must not read as "the guard passed".
if [ ! -f "$ROOT/scripts/session-space.sh" ]; then
    printf '  scripts/session-space.sh is missing from this checkout.\n'
    printf '  That is itself a warning: this checkout is behind, and stale checkouts are the\n'
    printf '  shared ones. Update it, or create a fresh worktree off origin/master, before working.\n\n'
    exit 1
fi
if ! bash "$ROOT/scripts/session-space.sh" check 2>&1 | sed 's/^/  /'; then
    exit 1
fi
bash "$ROOT/scripts/session-space.sh" claim 2>&1 | sed 's/^/  /'

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

# ⚠️ The line above counts ENTRIES IN A FILE. It is not a health check, and it reads exactly like
# one: it printed a truthful, reassuring server count at the start of every session for days while
# every server in that file was dead. So actually talk to the memory server before the session
# starts trusting it.
#
# This is the one preflight worth the seconds it costs, because its failure mode is silent. A dead
# memory server and a memory server with nothing to say produce the same thing from inside a
# session -- an empty result -- and only one of them means "there is nothing recorded about this".
# Without this check a session quietly re-derives decisions that were already made and re-asks
# questions that were already answered.
#
# Skip with RECALL_SKIP_MEMORY_CHECK=1 when offline; expect to be misremembering things if you do.
say "Memory"
if [ -n "${RECALL_SKIP_MEMORY_CHECK:-}" ]; then
    printf '  skipped (RECALL_SKIP_MEMORY_CHECK set); treat every empty recall result as unknown\n'
elif ! python "$ROOT/scripts/session_memory_check.py" --quiet 2>&1; then
    # Not fatal: a session with no memory is degraded, not stopped, and refusing to open would
    # strand anyone whose network is down. The banner the check prints is the point.
    printf '  continuing WITHOUT a usable memory layer\n'
fi

say "Database"
bash "$ROOT/scripts/session-db.sh" status 2>&1 | sed 's/^/  /'
printf '  start one only if you will run the DB suite:\n'
printf '    eval "$(scripts/session-db.sh up)"\n'

say "Stranded containers"
bash "$ROOT/scripts/session-db.sh" orphans 2>&1 | sed 's/^/  /'

printf '\n'
