#!/usr/bin/env bash
# Close a session in this checkout.
#
# Removes this checkout's own database container, and sweeps ORPHANS: containers whose checkout
# has been deleted, which no session can ever reclaim because the session that made them is gone.
# Everything else is reported and left alone. Another session's live container and the shared
# `recall-db-1` are never touched, because this script cannot know whether somebody is mid-run
# against them, and a 12 minute suite killed at minute 9 reads exactly like a code failure.
# (🔁 2026-08-25: `recall-dogfood` was named here too and no longer exists.)
#
# Two conditions gate every removal, and neither is the fall-through:
#
#   1. The container carries this tooling's own `recall.session` label. Compose containers whose
#      working directory has vanished are REPORTED and never removed: that filter matches every
#      compose project on the daemon, and from Git Bash a WSL-side path always reads as gone.
#   2. It reports zero client backends on two reads a few seconds apart. That is not a proof, and
#      the sweep does not claim one: it is the difference between idle and idle-at-the-instant-I-
#      looked. The suite opens and closes connections per use, so a live run sits at zero for long
#      stretches. A container is cheap; somebody's in-flight run is not.
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
    # 25-hour-old stopped orphan this sweep exists to clear.
    state="$(docker inspect "$id" --format '{{.State.Running}}' 2>/dev/null)"
    [ "$state" = "false" ] && { printf '0'; return; }
    [ "$state" != "true" ] && { printf 'busy'; return; }
    # Not scoped to `datname='recall'`. A client attached to any other database in the same
    # container is still a client, and scoping the count hid it.
    out="$(docker exec "$id" psql -U recall -d postgres -tAc \
        "select count(*) from pg_stat_activity where pid <> pg_backend_pid() and backend_type = 'client backend';" \
        2>/dev/null | tr -d '[:space:]')"
    case "$out" in
        ''|*[!0-9]*) printf 'busy' ;;
        *)           printf '%s' "$out" ;;
    esac
}

# Positive identity, required before anything is deleted.
#
# `session-db.sh orphans` deliberately reports three populations: containers this tooling started,
# compose containers whose working directory has vanished, and compose containers whose directory
# survives without a `.git` entry. Reporting the last two is useful. **Deleting them is not ours to
# do.** That filter matches every compose project on the daemon. The sweep therefore removes only
# what carries this tooling's own label, and merely prints the rest.
#
# The WSL hazard this comment used to describe is now handled upstream: a container started inside
# WSL records `/home/...`, whose root Git Bash cannot see, and `orphans` prints those as CHECK
# rather than ORPHAN. The awk below harvests `^ORPHAN ` only, so a CHECK line reaches the report
# and never reaches the sweep. Both halves matter; do not widen that pattern.
_is_ours() {
    local id owner
    id="$1"
    owner="$(docker inspect "$id" --format '{{index .Config.Labels "recall.session"}}' 2>/dev/null)"
    [ -n "$owner" ]
}

# An instantaneous count of zero is not proof that nobody is using a database. The suite opens and
# closes a connection per use, so a 12 minute run sits at zero clients for long stretches, and both
# shared services on this machine report zero right now. Two reads a few seconds apart is still not
# a proof, but it is the difference between "idle" and "idle at the instant I happened to look".
_idle_twice() {
    local id first second
    id="$1"
    first="$(_client_count "$id")"
    [ "$first" = "0" ] || return 1
    sleep 3
    second="$(_client_count "$id")"
    [ "$second" = "0" ]
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
        # Every branch below either SKIPS or removes, and removal is reached only by satisfying
        # two positive conditions in order. Deletion must never be the fall-through case: the
        # previous draft removed whenever the value was neither "busy" nor numerically positive,
        # so an integer comparison that ERRORED read as "idle" rather than as "could not tell".
        while read -r name; do
            [ -z "$name" ] && continue
            if ! _is_ours "$name"; then
                printf '  REPORT  %-32s not started by this tooling, left alone\n' "$name"
                skipped=$((skipped + 1))
            elif _idle_twice "$name"; then
                if docker rm -f "$name" >/dev/null 2>&1; then
                    printf '  REMOVED %-32s idle on two reads, checkout gone\n' "$name"
                    removed=$((removed + 1))
                else
                    printf '  SKIP    %-32s removal failed\n' "$name"
                    skipped=$((skipped + 1))
                fi
            else
                printf '  SKIP    %-32s in use, or could not be asked\n' "$name"
                skipped=$((skipped + 1))
            fi
        done <<EOF
$orphan_names
EOF
        printf '  %s removed, %s left alone.\n' "$removed" "$skipped"
        [ "$skipped" -gt 0 ] && printf '  Left-alone entries are in use, unreachable, or not ours. Re-run later; never force them.\n'
    fi
fi

say "Workspace"
if [ "$DRY_RUN" -eq 1 ]; then
    bash "$ROOT/scripts/session-space.sh" whose 2>&1 | sed 's/^/  /'
    printf '  --dry-run: claim not released.\n'
else
    # Released last, after everything above has reported. A claim dropped early would let another
    # session move in while this one is still tearing its container down.
    bash "$ROOT/scripts/session-space.sh" release 2>&1 | sed 's/^/  /'
fi

printf '\n'
