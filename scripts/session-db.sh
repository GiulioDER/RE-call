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
# checkout has since been deleted.
#
# Usage:
#   eval "$(scripts/session-db.sh up)"   # start + export RECALL_TEST_DSN into this shell
#   scripts/session-db.sh status
#   scripts/session-db.sh down
#   scripts/session-db.sh orphans        # list session containers with no checkout left
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
# Two kinds are reported, because both have actually happened here:
#   - containers this script started, labelled `recall.checkout`
#   - containers `docker compose up` started from a worktree, which Compose labels with
#     `com.docker.compose.project.working_dir`. Deleting the worktree leaves these running, and
#     nothing else ever looks for them.
_report_orphan() {
    local id checkout
    id="$1"; checkout="$2"
    [ -z "$checkout" ] && return 1
    [ -d "$checkout" ] && return 1
    printf 'ORPHAN %-32s (checkout gone: %s)\n' \
        "$(docker inspect "$id" --format '{{.Name}}' | sed 's|^/||')" "$checkout"
    return 0
}

cmd_orphans() {
    local found=0 id checkout seen=""
    for filter in "label=${LABEL_KEY}" "label=com.docker.compose.project.working_dir"; do
        while read -r id; do
            [ -z "$id" ] && continue
            case " $seen " in *" $id "*) continue ;; esac
            seen="$seen $id"
            checkout="$(docker inspect "$id" --format \
                '{{with index .Config.Labels "recall.checkout"}}{{.}}{{else}}{{index .Config.Labels "com.docker.compose.project.working_dir"}}{{end}}' \
                2>/dev/null || true)"
            if _report_orphan "$id" "$checkout"; then
                found=1
            fi
        done < <(docker ps -aq --filter "$filter")
    done
    if [ "$found" -eq 0 ]; then
        echo "session-db: no orphaned containers"
    else
        echo
        echo "Remove them with: docker rm -f <name>"
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
  orphans  list session containers whose checkout has been deleted
  id       print this checkout's session id (used by session-close.sh)

Typical use:

  eval "$(scripts/session-db.sh up)"
  python -m pytest tests/ -q
  scripts/session-db.sh down
USAGE
        exit 2
        ;;
esac
