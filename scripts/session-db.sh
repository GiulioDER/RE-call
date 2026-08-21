#!/usr/bin/env bash
# Session-scoped PostgreSQL for the recall test suite.
#
# Why this exists, in one sentence: the test suite DROPs tables, so two sessions pointed at one
# container delete each other's tables mid-run and report failures that describe nothing about
# the code under test.
#
# The rule this script implements:
#
#   Every session that runs the DB-backed suite starts its OWN container and removes it when
#   done. No session ever runs pytest against the shared `recall-db-1` on port 5432.
#
# The container is named and labelled after the checkout it belongs to, so `down` can never
# remove a container belonging to a different session, and `orphans` can find containers whose
# checkout has since been deleted or reduced to an empty directory.
#
# Usage:
#   eval "$(scripts/session-db.sh up)"   # start + export RECALL_TEST_DSN into this shell
#   scripts/session-db.sh status
#   scripts/session-db.sh down
#   scripts/session-db.sh orphans        # list containers whose checkout is gone or a remnant
#
# `up` prints an `export` line on stdout and all human-facing text on stderr, so `eval` gets
# exactly the assignment and nothing else.

set -euo pipefail

IMAGE="${RECALL_DB_IMAGE:-pgvector/pgvector:pg18}"
LABEL_KEY="recall.session"

# The session id is derived from the checkout path, not from a random value and not from the
# directory's basename. A worktree's directory name is not its branch and can repeat across
# clones; the absolute path is the thing that is actually unique to this checkout.
# The spelling must be normalised before it is hashed. `git rev-parse --show-toplevel` returns
# `C:/Users/...` while `pwd` in this shell returns `/c/Users/...`, and those hash differently: an
# invocation that fell back to `pwd` would give the same checkout a SECOND identity, a second
# container on a second port, and a `down` that cannot see the first. That is exactly the stranding
# this script exists to clean up, manufactured by the script itself.
_checkout_root() {
    local root
    if root="$(git rev-parse --show-toplevel 2>/dev/null)" && [ -n "$root" ]; then
        printf '%s' "$root"
    elif command -v cygpath >/dev/null 2>&1; then
        cygpath -m "$(pwd -P)"
    else
        pwd -P
    fi
}

_session_id() {
    # Lowercase hex, stable across runs, short enough to read in `docker ps`.
    _checkout_root | tr -d '\n' | sha256sum | cut -c1-8
}

_container() { printf 'recall-sess-%s' "$(_session_id)"; }

#: Window the derived ports live in. Kept as constants because CLAUDE.md documents the range and a
#: reader reserving a port for something else needs the real bound, not an approximation.
PORT_BASE=5400
PORT_SPREAD=400
PORT_SCAN=120

# Deterministic starting point so a session reclaims its own port across restarts, then a linear
# scan because determinism alone does not guarantee the port is free.
#
# `/dev/tcp` failing means "refused" here, but it also means "this bash has no /dev/tcp", and the
# two are indistinguishable from the exit status alone. Treating the second as "free" would make
# every session pick the same port. `_assert_probe_works` rules it out once, against a port we know
# is busy, rather than letting the probe fail open.
_assert_probe_works() {
    local busy
    busy="$(docker ps --format '{{.Ports}}' 2>/dev/null |
        grep -oE '127\.0\.0\.1:[0-9]+' | head -1 | cut -d: -f2)"
    [ -z "$busy" ] && return 0  # nothing listening to test against; nothing to prove
    if ! (echo > "/dev/tcp/127.0.0.1/$busy") 2>/dev/null; then
        echo "session-db: /dev/tcp reports port $busy free while docker publishes it." >&2
        echo "session-db: the free-port probe is unreliable in this shell; refusing to guess." >&2
        return 1
    fi
    return 0
}

_pick_port() {
    local sid start p
    sid="$(_session_id)"
    start=$(( PORT_BASE + (0x${sid} % PORT_SPREAD) ))
    for (( p = start; p < start + PORT_SCAN; p++ )); do
        if ! (echo > "/dev/tcp/127.0.0.1/$p") 2>/dev/null; then
            printf '%s' "$p"
            return 0
        fi
    done
    echo "session-db: no free port in ${start}..$((start + PORT_SCAN - 1))" >&2
    return 1
}

_running_port() {
    docker inspect "$(_container)" \
        --format '{{range $p, $c := .NetworkSettings.Ports}}{{range $c}}{{.HostPort}}{{end}}{{end}}' \
        2>/dev/null || true
}

cmd_up() {
    local name port dsn attempt
    name="$(_container)"

    port="$(_running_port)"
    if [ -n "$port" ]; then
        echo "session-db: reusing $name on port $port" >&2
    else
        _assert_probe_works || return 1
        # A stopped container with our name would make `docker run` fail on the name collision.
        docker rm -f "$name" >/dev/null 2>&1 || true

        # Picking a free port and binding it are two steps, and nothing reserves the port in
        # between. Two checkouts starting at once can both see the same port free, and the loser
        # gets "port is already allocated". Retry rather than abort: aborting under `set -e` inside
        # `eval "$(...)"` produces an empty eval, which exits 0, so the caller would carry on with
        # no RECALL_TEST_DSN and every DB test would skip while the run reported success.
        for attempt in 1 2 3 4 5; do
            port="$(_pick_port)" || return 1
            if docker run -d \
                --name "$name" \
                --label "${LABEL_KEY}=$(_session_id)" \
                --label "recall.checkout=$(_checkout_root)" \
                -e POSTGRES_USER=recall \
                -e POSTGRES_PASSWORD=recall \
                -e POSTGRES_DB=recall \
                -p "127.0.0.1:${port}:5432" \
                "$IMAGE" >/dev/null 2>&1
            then
                echo "session-db: started $name on port $port" >&2
                break
            fi
            docker rm -f "$name" >/dev/null 2>&1 || true
            if [ "$attempt" -eq 5 ]; then
                echo "session-db: could not bind a free port after 5 attempts" >&2
                return 1
            fi
            echo "session-db: port $port was taken, retrying" >&2
            sleep 1
        done
    fi

    # Readiness must be proved on the PUBLISHED PORT, not on the container's unix socket.
    # `pg_isready -U recall` over the socket answers during initdb, when the entrypoint runs a
    # temporary server with listen_addresses='' that the host port cannot reach. The session
    # container has no volume, so initdb runs on every `up` and that window is entered every time.
    # Answering early is the worst case: the caller starts pytest, gets a refusal, and the suite
    # skips every DB test while exiting 0.
    local i
    for i in $(seq 1 90); do
        if docker exec "$name" pg_isready -h 127.0.0.1 -U recall >/dev/null 2>&1 &&
           (echo > "/dev/tcp/127.0.0.1/$port") 2>/dev/null
        then
            break
        fi
        if [ "$i" -eq 90 ]; then
            echo "session-db: $name never accepted a TCP connection on $port" >&2
            return 1
        fi
        sleep 1
    done

    dsn="postgresql://recall:recall@127.0.0.1:${port}/recall"
    echo "session-db: ready" >&2
    printf 'export RECALL_TEST_DSN=%s\n' "$dsn"
}

cmd_down() {
    local name
    name="$(_container)"
    if docker inspect "$name" >/dev/null 2>&1; then
        docker rm -f "$name" >/dev/null
        echo "session-db: removed $name" >&2
    else
        echo "session-db: nothing to remove for this checkout" >&2
    fi
}

cmd_status() {
    local name port
    name="$(_container)"
    port="$(_running_port)"
    if [ -n "$port" ]; then
        echo "session-db: $name is up on 127.0.0.1:$port"
        echo "  export RECALL_TEST_DSN=postgresql://recall:recall@127.0.0.1:${port}/recall"
    else
        echo "session-db: $name is not running"
    fi
}

# A container whose checkout no longer exists cannot be reclaimed by the session that made it,
# because that session is gone. The checkout path is recorded on the container itself, which is
# the only reliable way to find these after the fact.
#
# Three kinds are reported, because all three have actually happened here:
#   - containers this script started, labelled `recall.checkout`
#   - containers `docker compose up` started from a worktree, which Compose labels with
#     `com.docker.compose.project.working_dir`. Deleting the worktree leaves these running, and
#     nothing else ever looks for them.
#   - either of those whose directory still exists but is no longer a checkout.
#
# The third kind is why `[ -d ]` is not the test. Measured 2026-08-20: five containers from
# `interesting-cannon-7e684c` sat exited for ten hours while this command printed "no orphaned
# containers", because the worktree had been removed and its empty directory left behind. A
# directory alone cannot be told apart from a live checkout; the presence of a `.git` entry can,
# and it is the same test in the main checkout (where `.git` is a directory) and in a linked
# worktree (where it is a file).
#
# The same measurement found 30 of 33 containers on the machine were compose stacks rather than
# session containers, so the compose half of this command is the half that does the work.
#
# ⚠️ Do NOT reach for `git -C "$dir" rev-parse` as the remnant test. Git walks UPWARD, so inside
# `<main>/.claude/worktrees/<remnant>` it finds the parent repository and answers `true` for a
# directory that is empty. Verified against the remnant above.
_norm_path() {
    printf '%s' "$1" | tr '\\' '/' | tr '[:upper:]' '[:lower:]' | sed 's|/\{1,\}$||'
}

# The main checkout of this repository: the parent of the shared git directory, which resolves to
# the same path from a linked worktree as from the main checkout itself.
_main_checkout() {
    local common
    common="$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" || return 1
    [ -n "$common" ] || return 1
    dirname "$common"
}

# Scope matters more here than the remnant test does, and it fails in the dangerous direction.
# `com.docker.compose.project.working_dir` matches EVERY compose container on the machine,
# including projects that have nothing to do with recall, and an ORPHAN line reads as an
# instruction to `docker rm -f` something. A live stack belonging to another project must never
# earn one, so the remnant test runs only on paths this repository can vouch for: its own session
# containers, anything under the main checkout, and the `.claude/worktrees/` convention, which
# covers worktrees made outside the main checkout as `Documents/.claude/worktrees/` was.
#
# A checkout that is simply GONE is still reported whatever its path, because a directory that is
# not there belongs to nobody.
_is_ours() {
    local norm main
    [ "${2:-}" = "session" ] && return 0
    norm="$(_norm_path "$1")"
    case "$norm" in */.claude/worktrees/*) return 0 ;; esac
    main="$(_main_checkout)" || return 1
    [ -n "$main" ] || return 1
    main="$(_norm_path "$main")"
    [ "$norm" = "$main" ] && return 0
    case "$norm" in "$main"/*) return 0 ;; esac
    return 1
}

#: Windows resolves no path longer than 260 characters, and `[ -d ]` cannot distinguish "not
#: there" from "too long to look at". Found while writing these tests: a fixture repository under a
#: deep temp directory made every LIVE worktree report `checkout gone`, which is the dangerous
#: direction, since an ORPHAN line invites `docker rm -f` on a checkout somebody is using. Anything
#: near the limit is reported as unverifiable instead, and the caller is told to look itself.
PATH_LIMIT=240

# The first component of an absolute path, which is the question "can this shell see that
# filesystem at all?" rather than "is this particular directory there?".
#
# A container started inside WSL records `/home/...` or `/mnt/...`, and from Git Bash neither root
# exists, so `[ -d ]` says gone and a running stack reads as an orphan from one namespace over.
# `session-close.sh` documents that and defends against it by refusing to remove anything without
# recall's own label; the report should not assert it either.
#
# Testing the ROOT rather than the platform is what makes this safe in both directions. Git Bash
# resolves `/c/...`, `/tmp/...` and `/usr/...` perfectly well, so a missing leaf under one of those
# is genuinely missing and stays an ORPHAN; measured on this machine, `/home`, `/mnt`, `/var` and
# `/opt` are absent while `/c`, `/tmp` and `/usr` are present. On Linux `/home` exists, so the same
# code reports the same path as gone, which is the honest answer there.
_path_root() {
    local rest
    rest="${1#/}"
    printf '/%s' "${rest%%/*}"
}

# Prints why this container is an orphan, or returns 1 if it is not one. A reason beginning with
# `unverifiable` is a question rather than a verdict, and the caller prints it differently.
_orphan_reason() {
    local checkout="$1" labelled="${2:-}" root
    [ -z "$checkout" ] && return 1
    if [ ! -d "$checkout" ]; then
        if [ "${#checkout}" -gt "$PATH_LIMIT" ]; then
            printf 'unverifiable: %s characters, past what Windows can resolve' "${#checkout}"
            return 0
        fi
        case "$checkout" in
            /*)
                root="$(_path_root "$checkout")"
                if [ ! -d "$root" ]; then
                    printf 'unverifiable: %s is not a filesystem this shell can see' "$root"
                    return 0
                fi
                ;;
        esac
        printf 'checkout gone'
        return 0
    fi
    _is_ours "$checkout" "$labelled" || return 1
    if [ ! -e "$checkout/.git" ]; then
        printf 'worktree removed, directory left behind'
        return 0
    fi
    return 1
}

# `{{index .Config.Labels "x"}}` prints the literal `<no value>` for a label that is not set, and
# that string is truthy to every test below. It has to be spelled out of existence once, here.
_field() {
    case "$1" in '<no value>'|'<nil>') printf '' ;; *) printf '%s' "$1" ;; esac
}

#: One line per container, `recall.checkout|compose working_dir|name`. Named once because the
#: batching below and the parsing above have to agree about it, and a template that drifts between
#: them fails as a field that is silently always empty.
INSPECT_FORMAT='{{index .Config.Labels "recall.checkout"}}|{{index .Config.Labels "com.docker.compose.project.working_dir"}}|{{.Name}}'

#: How many ids go into one `docker inspect`. The round trips are what made this command slow: one
#: per container, measured at roughly 2.5s each against a loaded daemon, so about 45 seconds for
#: 17 containers, and `session-close.sh` runs it on every close. Chunked rather than unbounded
#: because a command line has a length limit, and a machine with hundreds of containers is exactly
#: where this matters.
INSPECT_BATCH=100

# ⚠️ The NAME is inside each line on purpose, and nothing here reads by position. A container can
# be removed between the `ps` that listed it and the `inspect` that reads it; docker then writes an
# error for that id and omits its line, so the reply is shorter than the argument list and every
# later id would be misattributed by an index. That is a rename of somebody else's container in the
# report, which is worse than the slowness this replaced.
_inspect_batch() {
    local -a batch=()
    local id
    for id in "$@"; do
        batch+=("$id")
        if [ "${#batch[@]}" -ge "$INSPECT_BATCH" ]; then
            docker inspect "${batch[@]}" --format "$INSPECT_FORMAT" 2>/dev/null || true
            batch=()
        fi
    done
    if [ "${#batch[@]}" -gt 0 ]; then
        docker inspect "${batch[@]}" --format "$INSPECT_FORMAT" 2>/dev/null || true
    fi
}

# Every candidate id, both populations, deduplicated. A container carrying both labels is this
# script's own container inside a checkout that also runs a compose stack, which is the normal
# case here, not an edge one.
_orphan_candidates() {
    local filter
    for filter in "label=${LABEL_KEY}" "label=com.docker.compose.project.working_dir"; do
        docker ps -aq --filter "$filter" 2>/dev/null || true
    done | awk 'NF && !seen[$0]++'
}

cmd_orphans() {
    local found=0 unsure=0 record line checkout compose_dir labelled reason name
    local -a ids=()

    # A daemon that does not answer produces an empty container list, and an empty list is
    # indistinguishable from a clean machine once it has been printed. That is the same false
    # green this command was fixed for, arriving by a different road, so it is refused rather
    # than reported. `session-close.sh` captures this output with `2>&1` and does not read the
    # exit status, so the refusal reaches its report as text either way.
    if ! command -v docker >/dev/null 2>&1; then
        echo "session-db: docker is not on PATH; cannot tell an orphan from a clean machine" >&2
        return 2
    fi
    if ! docker ps -q >/dev/null 2>&1; then
        echo "session-db: the docker daemon is not answering; refusing to report a clean machine" >&2
        return 2
    fi

    while read -r line; do
        [ -n "$line" ] && ids+=("$line")
    done < <(_orphan_candidates)

    if [ "${#ids[@]}" -gt 0 ]; then
        while read -r record; do
            [ -z "$record" ] && continue
            checkout="$(_field "${record%%|*}")"
            compose_dir="$(_field "$(printf '%s' "$record" | cut -d'|' -f2)")"
            name="$(printf '%s' "$record" | cut -d'|' -f3- | sed 's|^/||')"
            if [ -n "$checkout" ]; then
                labelled=session
            else
                labelled=compose
                checkout="$compose_dir"
            fi
            reason="$(_orphan_reason "$checkout" "$labelled")" || continue
            case "$reason" in
                unverifiable*)
                    printf 'CHECK  %-40s (%s: %s)\n' "${name:-?}" "$reason" "$checkout"
                    unsure=1
                    ;;
                *)
                    printf 'ORPHAN %-40s (%s: %s)\n' "${name:-?}" "$reason" "$checkout"
                    found=1
                    ;;
            esac
        done < <(_inspect_batch "${ids[@]}")
    fi

    if [ "$found" -eq 0 ]; then
        echo "session-db: no orphaned containers"
    else
        echo
        echo "Remove them with: docker rm -f <name>"
        echo "A compose stack is five containers; the whole project goes at once with:"
        echo "  docker rm -f \$(docker ps -aq --filter label=com.docker.compose.project=<project>)"
    fi
    if [ "$unsure" -ne 0 ]; then
        echo "session-db: a CHECK line is not a verdict. Look at the path yourself before removing."
    fi
}

cmd_id() { _session_id; }

case "${1:-}" in
    up)      cmd_up ;;
    id)      cmd_id ;;
    down)    cmd_down ;;
    status)  cmd_status ;;
    orphans) cmd_orphans ;;
    *)
        cat >&2 <<'USAGE'
usage: scripts/session-db.sh {up|down|status|orphans}

  up       start (or reuse) this checkout's container; prints an export line for `eval`
  down     remove this checkout's container; never touches another session's
  status   show whether this checkout's container is running
  orphans  list session AND compose containers whose checkout is gone, or whose
           directory survives without a .git entry (a removed worktree's remnant)
  id       print this checkout's session id (used by session-close.sh)

Typical use:

  eval "$(scripts/session-db.sh up)"
  python -m pytest tests/ -q
  scripts/session-db.sh down
USAGE
        exit 2
        ;;
esac
