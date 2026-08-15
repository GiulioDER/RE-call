#!/usr/bin/env bash
# Close a session in this checkout.
#
# Removes this checkout's own database container and reports what is left behind. It removes
# nothing else: another session's container, the shared `recall-db-1` and the `recall-dogfood`
# corpus are all left alone, because this script cannot know whether somebody is mid-run against
# them.
#
#   scripts/session-close.sh            # tear down this checkout's DB, report the rest
#   scripts/session-close.sh --keep-db  # report only, leave the container up

set -uo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

say() { printf '\n\033[1m%s\033[0m\n' "$1"; }

say "Uncommitted work in this checkout"
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    git status --short | head -30
    printf '\n  Nothing here is committed for you. Stage by pathspec; `git add -A` is blocked.\n'
else
    printf '  clean\n'
fi

say "Unpushed commits"
if git rev-parse --abbrev-ref '@{u}' >/dev/null 2>&1; then
    count="$(git rev-list --count '@{u}..HEAD' 2>/dev/null || echo 0)"
    if [ "$count" != "0" ]; then
        git log --oneline '@{u}..HEAD' | head -20
    else
        printf '  none\n'
    fi
else
    printf '  no upstream set for %s\n' "$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
fi

say "This checkout's database"
if [ "${1:-}" = "--keep-db" ]; then
    bash "$ROOT/scripts/session-db.sh" status 2>&1 | sed 's/^/  /'
    printf '  left running at your request\n'
else
    bash "$ROOT/scripts/session-db.sh" down 2>&1 | sed 's/^/  /'
fi

say "Containers this script will not touch"
docker ps --format '{{.Names}}\t{{.Ports}}' 2>/dev/null \
    | grep -Ev "recall-sess-$(printf '%s' "$ROOT" | sha256sum | cut -c1-8)" \
    | sed 's/^/  /' || printf '  none running\n'
printf '  Another session may be mid-run against these. Remove them only if you know otherwise.\n'

say "Stranded containers"
bash "$ROOT/scripts/session-db.sh" orphans 2>&1 | sed 's/^/  /'

printf '\n'
