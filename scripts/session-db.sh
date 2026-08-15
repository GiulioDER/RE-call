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
_checkout_root() {
    git rev-parse --show-toplevel 2>/dev/null || pwd
}

_session_id() {
    local root
    root="$(_checkout_root)"
    # Lowercase hex, stable across runs, short enough to read in `docker ps`.
    printf '%s' "$root" | sha256sum | cut -c1-8
}

_container() { printf 'recall-sess-%s' "$(_session_id)"; }

# Deterministic starting point so a session reclaims its own port across restarts, then a linear
# scan because determinism alone does not guarantee the port is free.
_pick_port() {
    local sid start p
    sid="$(_session_id)"
    start=$(( 5400 + (0x${sid} % 400) ))
    for (( p = start; p < start + 120; p++ )); do
        if ! (echo > "/dev/tcp/127.0.0.1/$p") 2>/dev/null; then
            printf '%s' "$p"
            return 0
        fi
    done
    echo "session-db: no free port in ${start}..$((start + 120))" >&2
    return 1
}

_running_port() {
    docker inspect "$(_container)" \
        --format '{{range $p, $c := .NetworkSettings.Ports}}{{range $c}}{{.HostPort}}{{end}}{{end}}' \
        2>/dev/null || true
}

cmd_up() {
    local name port dsn
    name="$(_container)"

    port="$(_running_port)"
    if [ -n "$port" ]; then
        echo "session-db: reusing $name on port $port" >&2
    else
        # A stopped container with our name would make `docker run` fail on the name collision.
        docker rm -f "$name" >/dev/null 2>&1 || true
        port="$(_pick_port)"
        docker run -d \
            --name "$name" \
            --label "${LABEL_KEY}=$(_session_id)" \
            --label "recall.checkout=$(_checkout_root)" \
            -e POSTGRES_USER=recall \
            -e POSTGRES_PASSWORD=recall \
            -e POSTGRES_DB=recall \
            -p "127.0.0.1:${port}:5432" \
            "$IMAGE" >/dev/null
        echo "session-db: started $name on port $port" >&2
    fi

    # Readiness is checked against the container, not the port. The port answers as soon as
    # Docker binds it, which is before Postgres will accept a connection.
    local i
    for i in $(seq 1 60); do
        if docker exec "$name" pg_isready -U recall >/dev/null 2>&1; then
            break
        fi
        if [ "$i" -eq 60 ]; then
            echo "session-db: $name never became ready" >&2
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

case "${1:-}" in
    up)      cmd_up ;;
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

Typical use:

  eval "$(scripts/session-db.sh up)"
  python -m pytest tests/ -q
  scripts/session-db.sh down
USAGE
        exit 2
        ;;
esac
