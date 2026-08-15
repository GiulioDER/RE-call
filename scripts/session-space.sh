#!/usr/bin/env bash
# One session, one workspace. Claims this checkout for the current session and refuses shared ones.
#
# The problem this solves, measured in this repository on 2026-08-15: twenty worktrees, five of them
# holding uncommitted work, and **the main checkout itself dirty on a two-month-old feature branch**.
# Two sessions in one checkout share an index and a working tree, so one stages the other's files,
# a rebase moves HEAD under a session that is mid-edit, and `git log` looks fine afterwards because
# the subjects are identical. The recorded cost includes a force-push that erased three already
# pushed commits, and a session that re-did work another session had already merged.
#
# Two mechanisms, because prose has not been enough:
#
#   1. The main checkout is REFUSED outright. It is shared by definition: every worktree resolves
#      its objects through it, and it is the one directory every session can reach by habit.
#   2. A worktree carries a claim naming the session that holds it. A second session opening the
#      same worktree is told whose it is and how old the claim is, rather than discovering the
#      collision through a lost edit.
#
# The claim lives in this worktree's own git directory, so it is never committed, needs no ignore
# rule, and disappears with the worktree.
#
#   scripts/session-space.sh check           # is this an isolated, unclaimed space? exit 1 if not
#   scripts/session-space.sh claim           # take it, or refresh an existing claim of ours
#   scripts/session-space.sh release         # give it up (session-close does this)
#   scripts/session-space.sh new <name>      # create an isolated worktree off origin/master
#   scripts/session-space.sh whose           # print the current claim, if any

set -uo pipefail

#: A claim older than this with no live process is stale and may be taken over. Long enough to
#: survive a lunch break, short enough that an abandoned worktree does not stay locked for days.
STALE_HOURS="${RECALL_CLAIM_STALE_HOURS:-12}"

SESSION_ID="${CLAUDE_CODE_SESSION_ID:-${CLAUDE_SESSION_ID:-}}"
SESSION_PID="${CLAUDE_PID:-$$}"

_git_dir()        { git rev-parse --git-dir 2>/dev/null; }
_git_common_dir() { git rev-parse --git-common-dir 2>/dev/null; }
_claim_file()     { printf '%s/claude-session-claim' "$(_git_dir)"; }
_toplevel()       { git rev-parse --show-toplevel 2>/dev/null; }

# In the main checkout these two resolve to the same directory; in a linked worktree they differ.
# That is the whole test, and it does not depend on any path convention.
_is_main_checkout() {
    local d c
    d="$(cd "$(_git_dir)" 2>/dev/null && pwd -P)" || return 1
    c="$(cd "$(_git_common_dir)" 2>/dev/null && pwd -P)" || return 1
    [ "$d" = "$c" ]
}

_now()      { date -u +%Y-%m-%dT%H:%M:%SZ; }
_epoch()    { date -u +%s; }
_read_key() { sed -n "s/^$1=//p" "$(_claim_file)" 2>/dev/null | head -1; }

# Is this pid still running?
#
# `kill -0` alone is WRONG on this platform and wrong in the dangerous direction. `CLAUDE_PID` is a
# Windows pid, and Git Bash's kill only knows its own process table: measured 2026-08-15, `kill -0
# 7260` reported "No such process" while `tasklist` showed `claude.exe` running at 7260. Trusting
# it would age out a live session's claim and hand its worktree to somebody else.
#
# So: ask Windows first where Windows is the authority, fall back to kill -0 elsewhere, and treat
# "cannot tell" as ALIVE. The two errors are not symmetric. A false "alive" costs a session a
# `release --force`; a false "dead" costs a lost edit in a worktree somebody is mid-rebase in.
_process_alive() {
    local pid
    pid="$1"
    [ -z "$pid" ] && return 1
    case "$pid" in *[!0-9]*) return 0 ;; esac  # unparseable: assume alive
    if command -v tasklist >/dev/null 2>&1; then
        tasklist //FI "PID eq $pid" //NH 2>/dev/null | grep -q "$pid" && return 0
        return 1
    fi
    if command -v ps >/dev/null 2>&1; then
        ps -p "$pid" >/dev/null 2>&1 && return 0
        return 1
    fi
    kill -0 "$pid" 2>/dev/null
}

_claim_is_stale() {
    local when age
    when="$(_read_key claimed_epoch)"
    case "$when" in ''|*[!0-9]*) return 0 ;; esac  # unreadable claim is stale
    _process_alive "$(_read_key pid)" && return 1
    age=$(( $(_epoch) - when ))
    [ "$age" -gt $(( STALE_HOURS * 3600 )) ]
}

cmd_whose() {
    local f
    f="$(_claim_file)"
    if [ -f "$f" ]; then cat "$f"; else echo "unclaimed"; fi
}

cmd_check() {
    if _is_main_checkout; then
        cat >&2 <<EOF
session-space: REFUSED. This is the repository's MAIN checkout, which every session shares.

  $(_toplevel)
  branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null)

Work here collides with every other session by construction: one shared index, one shared working
tree. Right now this checkout is $( [ -n "$(git status --porcelain 2>/dev/null)" ] && echo "DIRTY" || echo "clean" ) and on the branch above.

Take your own space instead:

  scripts/session-space.sh new <short-name>

That creates a worktree off origin/master, on its own branch, claimed by this session.
EOF
        return 1
    fi

    local holder
    holder="$(_read_key session)"
    if [ -n "$holder" ] && [ "$holder" != "$SESSION_ID" ]; then
        if _claim_is_stale; then
            echo "session-space: stale claim by ${holder} (no live process, older than ${STALE_HOURS}h); taking over." >&2
        else
            cat >&2 <<EOF
session-space: this worktree is CLAIMED by another session.

  worktree: $(_toplevel)
  held by:  ${holder}
  since:    $(_read_key claimed_at)
  pid:      $(_read_key pid) $(_process_alive "$(_read_key pid)" && echo '(alive)' || echo '(no longer running)')

Do not edit here. That session may be mid-rebase or mid-commit, and a shared index loses one side
silently. Take your own space:

  scripts/session-space.sh new <short-name>

If you are certain that session is gone, release it deliberately:

  scripts/session-space.sh release --force
EOF
            return 1
        fi
    fi
    return 0
}

cmd_claim() {
    cmd_check || return 1
    local f
    f="$(_claim_file)"
    {
        printf 'session=%s\n' "${SESSION_ID:-unknown}"
        printf 'pid=%s\n' "$SESSION_PID"
        printf 'branch=%s\n' "$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
        printf 'worktree=%s\n' "$(_toplevel)"
        printf 'claimed_at=%s\n' "$(_now)"
        printf 'claimed_epoch=%s\n' "$(_epoch)"
    } > "$f"
    echo "session-space: claimed $(basename "$(_toplevel)") for this session" >&2
}

cmd_release() {
    local f holder
    f="$(_claim_file)"
    [ -f "$f" ] || { echo "session-space: nothing to release" >&2; return 0; }
    holder="$(_read_key session)"
    if [ "$holder" != "$SESSION_ID" ] && [ "${1:-}" != "--force" ]; then
        echo "session-space: claim belongs to ${holder}, not this session. Use --force if you are sure." >&2
        return 1
    fi
    rm -f "$f"
    echo "session-space: released" >&2
}

cmd_new() {
    local name root wt branch
    name="${1:-}"
    if [ -z "$name" ]; then
        echo "usage: scripts/session-space.sh new <short-name>" >&2
        return 2
    fi
    case "$name" in
        *[!A-Za-z0-9._-]*)
            echo "session-space: name may contain only letters, digits, dot, underscore and dash" >&2
            return 2
            ;;
    esac
    root="$(git rev-parse --show-toplevel)"
    # Anchor on the COMMON dir's parent, so `new` works the same from inside a worktree as from the
    # main checkout rather than nesting worktrees inside worktrees.
    root="$(cd "$(_git_common_dir)/.." && pwd -P)"
    wt="$root/.claude/worktrees/$name"
    branch="claude/$name"

    [ -e "$wt" ] && { echo "session-space: $wt already exists" >&2; return 1; }

    git -C "$root" fetch -q origin 2>/dev/null
    # Off origin/master, never the local ref: the local one goes stale here routinely, and a stale
    # base looks exactly like a regression once the work is under way.
    if ! git -C "$root" worktree add -q -b "$branch" "$wt" origin/master; then
        echo "session-space: could not create the worktree (branch $branch may already exist)" >&2
        return 1
    fi
    ( cd "$wt" && cmd_claim ) >/dev/null 2>&1
    cat >&2 <<EOF
session-space: created an isolated space

  worktree: $wt
  branch:   $branch  (off origin/master)
  claimed:  yes

Move into it:

  cd "$wt"
EOF
    printf '%s\n' "$wt"
}

case "${1:-}" in
    check)   cmd_check ;;
    claim)   cmd_claim ;;
    release) shift; cmd_release "${1:-}" ;;
    whose)   cmd_whose ;;
    new)     shift; cmd_new "${1:-}" ;;
    *)
        cat >&2 <<'USAGE'
usage: scripts/session-space.sh {check|claim|release|whose|new <name>}

  check    refuse the main checkout and any worktree another session holds
  claim    take this worktree for this session, or refresh our own claim
  release  give it up (session-close does this)
  whose    print the current claim
  new      create an isolated worktree off origin/master and claim it
USAGE
        exit 2
        ;;
esac
