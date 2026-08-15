#!/usr/bin/env bash
# Close a session in this checkout.
#
# Removes this checkout's own database container, and sweeps ORPHANS: containers whose checkout
# has been deleted, which no session can ever reclaim because the session that made them is gone.
# Everything else is reported and left alone. Another session's live container, the shared
# `recall-db-1` and the `recall-dogfood` corpus are never touched, because this script cannot know
# whether somebody is mid-run against them, and a 12 minute suite killed at minute 9 reads exactly
# like a code failure.
#
# An orphan is only removed once it is proved idle. Its checkout being gone means nobody can start
# new work against it; it does not prove nobody is connected right now, so the sweep asks the
# database and skips any container with a live client. A container is cheap. Somebody's in-flight
# run is not.
#
#   scripts/session-close.sh                 # tear down this checkout's DB, sweep orphans
#   scripts/session-close.sh --keep-db       # leave this checkout's container up
#   scripts/session-close.sh --keep-orphans  # report orphans, remove none
#   scripts/session-close.sh --dry-run       # report everything, remove nothing

set -uo pipefail

KEEP_DB=0
KEEP_ORPHANS=0
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --keep-db)      KEEP_DB=1 ;;
        --keep-orphans) KEEP_ORPHANS=1 ;;
        --dry-run)      DRY_RUN=1; KEEP_DB=1; KEEP_ORPHANS=1 ;;
        *)
            echo "usage: scripts/session-close.sh [--keep-db] [--keep-orphans] [--dry-run]" >&2
            exit 2
            ;;
    esac
done

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

say() { printf '\n\033[1m%s\033[0m\n' "$1"; }

# Clients connected to this container's `recall` database, excluding the backend answering the
# question. Anything other than a clean integer is treated as "busy": an unreachable or
# still-starting container must not be read as idle, because that reading is what authorises
# deleting it.
_client_count() {
    local id out state
    id="$1"
    # A container that is not running has no clients by definition, and `docker exec` against it
    # fails in exactly the same way an unreachable one does. Without this branch every stopped
    # orphan reads as "could not ask" and is skipped forever, which is the state that produced the
    # 23-hour-old stopped orphan this sweep exists to clear.
    state="$(docker inspect "$id" --format '{{.State.Running}}' 2>/dev/null)"
    [ "$state" = "false" ] && { printf '0'; return; }
    [ "$state" != "true" ] && { printf 'busy'; return; }
    out="$(docker exec "$id" psql -U recall -d recall -tAc \
        "select count(*)-1 from pg_stat_activity where datname='recall';" 2>/dev/null | tr -d '[:space:]')"
    case "$out" in
        ''|*[!0-9]*) printf 'busy' ;;
        *)           printf '%s' "$out" ;;
    esac
}

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
if [ "$KEEP_DB" -eq 1 ]; then
    bash "$ROOT/scripts/session-db.sh" status 2>&1 | sed 's/^/  /'
    printf '  left running at your request\n'
else
    bash "$ROOT/scripts/session-db.sh" down 2>&1 | sed 's/^/  /'
fi

# Ask session-db.sh for the id rather than re-deriving it. Two copies of a derivation drift, and
# the drift is silent here: this filter would list this checkout's own container under "will not
# touch" moments after `down` had in fact removed it.
say "Containers this script will not touch"
docker ps --format '{{.Names}}\t{{.Ports}}' 2>/dev/null \
    | grep -Ev "recall-sess-$(bash "$ROOT/scripts/session-db.sh" id)" \
    | sed 's/^/  /' || printf '  none running\n'
printf '  Another session may be mid-run against these. Remove them only if you know otherwise.\n'

# A pre-registration left at "predicted, not yet measured" is the one artefact that rots into a
# falsehood if nobody comes back to it: the prediction stays, the result never lands, and the next
# reader cannot tell an abandoned experiment from a pending one.
if [ -d "$ROOT/docs/preregistrations" ]; then
    pending="$(grep -rl "Status:.*predicted, not yet measured" "$ROOT/docs/preregistrations" 2>/dev/null)"
    if [ -n "$pending" ]; then
        say "Pre-registrations still unmeasured"
        printf '%s\n' "$pending" | sed "s|^$ROOT/||; s/^/  /"
        printf '  Append the result and flip the status, or say plainly that it was abandoned.\n'
    fi
fi

say "Stranded containers"
orphan_report="$(bash "$ROOT/scripts/session-db.sh" orphans 2>&1)"
printf '%s\n' "$orphan_report" | sed 's/^/  /'

# `orphans` prints "ORPHAN <name>  (checkout gone: <path>)". Take the name from column 2.
orphan_names="$(printf '%s\n' "$orphan_report" | awk '/^ORPHAN /{print $2}')"

if [ -n "$orphan_names" ]; then
    if [ "$KEEP_ORPHANS" -eq 1 ]; then
        say "Orphan sweep"
        if [ "$DRY_RUN" -eq 1 ]; then
            printf '  --dry-run: nothing removed. Without it, each idle orphan above is removed.\n'
        else
            printf '  --keep-orphans: nothing removed.\n'
        fi
    else
        say "Orphan sweep"
        removed=0
        skipped=0
        while read -r name; do
            [ -z "$name" ] && continue
            clients="$(_client_count "$name")"
            if [ "$clients" = "busy" ]; then
                printf '  SKIP    %-32s could not be asked whether it is idle\n' "$name"
                skipped=$((skipped + 1))
            elif [ "$clients" -gt 0 ] 2>/dev/null; then
                printf '  SKIP    %-32s %s client(s) still connected\n' "$name" "$clients"
                skipped=$((skipped + 1))
            elif docker rm -f "$name" >/dev/null 2>&1; then
                printf '  REMOVED %-32s idle, and its checkout is gone\n' "$name"
                removed=$((removed + 1))
            else
                printf '  SKIP    %-32s removal failed\n' "$name"
                skipped=$((skipped + 1))
            fi
        done <<EOF
$orphan_names
EOF
        printf '  %s removed, %s left alone.\n' "$removed" "$skipped"
        [ "$skipped" -gt 0 ] && printf '  A skipped orphan is still in use. Re-run later rather than forcing it.\n'
    fi
fi

printf '\n'
