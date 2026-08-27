#!/usr/bin/env bash
# Close THIS session's recall MCP servers on VPS2, and nothing else's.
#
# What a recall MCP server actually costs, measured on VPS2 on 2026-08-26:
#
#   18 live servers holding 14.67 GB, plus their 18 ssh transport wrappers at 0.39 GB,
#   on a 47 GB host that also runs the live trading services. Oldest 13.9 hours.
#
# 🔁 An earlier version of this header said "89 live `recall_mcp.server` processes, 21.5 GB". The
# memory was right and the COUNT was about double, because both a server and its ssh wrapper carry
# the string `python -m recall_mcp.server`: the server as the command it runs, the wrapper inside
# `--cmd=...`. The fleet count below now excludes wrappers, which is why the number it prints is
# roughly half what it printed yesterday on the same host.
#
# Each server is roughly 815 MB and it lives as long as its stdio transport does. A session that
# ends without closing the pipe leaves the whole thing running: ssh sets no keepalive, so a client
# that vanishes (a reboot, a closed laptop, a killed app) leaves a half-open connection the far
# side never notices. Nothing on either machine reports this, because a leaked server looks exactly
# like a working one.
#
# The mechanism, and why it is the LOCAL side
# -------------------------------------------
# `.mcp.json` launches each server as `ssh <host> '... exec python -m recall_mcp.server'`, so the
# server is the process ssh owns on the far side. Measured 2026-08-26 with a marked probe: killing
# the local ssh transport killed the remote server in **under 3 seconds**, by pid, confirmed by
# `ps -p`. So closing a session's server needs no remote kill, no pattern matching against another
# machine's process table, and no privileges there.
#
# ⛔ Ownership is decided by ANCESTRY, and never by the command line.
# On this workstation the same command line belongs to more than one agent: measured the same day,
# three live recall transports had `codex.exe` as their parent, not Claude. A `pkill -f
# recall_mcp.server` here, or a pattern sweep on VPS2, would have killed another agent's live
# servers mid-query. So a transport is closed only if its parent chain reaches THIS session's
# client process (`CLAUDE_PID`). With no session pid there is no positive identity, and this script
# then reports and kills nothing rather than guessing.
#
# The remote fleet is REPORTED and never swept. Age does not prove abandonment: a three-day-old
# server may belong to a session that is still open, and that is the same mistake as removing a
# container somebody is mid-run against.
#
#   scripts/session-mcp-close.sh              # report: this session's transports, and the fleet
#   scripts/session-mcp-close.sh close        # close this session's transports, verify, report
#   scripts/session-mcp-close.sh sweep        # what on the HOST has no client left (report only)
#   scripts/session-mcp-close.sh sweep --kill # close those, by positive identity only
#   scripts/session-mcp-close.sh close --dry-run    # name what would be closed, close nothing
#   scripts/session-mcp-close.sh --no-fleet   # skip the ssh that counts servers on the host

set -uo pipefail

HOST="${RECALL_VPS2_HOST:-vps2}"
#: What marks a process as a recall MCP transport. Overridable so a differently named server (a
#: fork, a renamed module) can be closed without editing this file.
PATTERN="${RECALL_MCP_PATTERN:-recall_mcp.server}"
#: Seconds to wait before asking the host whether the servers went with the transports. The probe
#: that established this measured under 3 seconds; 5 is that with room, and it is only ever paid
#: once, at the end of a session.
SETTLE="${RECALL_MCP_CLOSE_SETTLE:-5}"
FLEET_TIMEOUT="${RECALL_MCP_FLEET_TIMEOUT:-30}"

MODE="report"
SWEEP_ARGS=""
DRY_RUN=0
WANT_FLEET=1
for arg in "$@"; do
    case "$arg" in
        report|close|sweep) MODE="$arg" ;;
        --kill|--unmarked)  SWEEP_ARGS="$SWEEP_ARGS $arg" ;;
        --dry-run)    DRY_RUN=1 ;;
        --no-fleet)   WANT_FLEET=0 ;;
        *)
            echo "usage: scripts/session-mcp-close.sh [report|close|sweep]" >&2
            echo "       close: [--dry-run] [--no-fleet]   sweep: [--kill] [--unmarked]" >&2
            exit 2
            ;;
    esac
done

# --- the two platform-specific things, both overridable so the ownership logic can be tested ----
#
# `RECALL_MCP_PS` prints one line per process as `<pid> <ppid> <command line>`; `RECALL_MCP_KILL`
# takes a pid. The tests set both to fixtures, which is what lets the parent-chain walk (the part
# that decides what gets killed) be exercised on a machine with no ssh, no Claude and no VPS2.
_is_windows() {
    case "$(uname -s 2>/dev/null)" in
        MINGW*|MSYS*|CYGWIN*) return 0 ;;
        *) return 1 ;;
    esac
}

_ps_table() {
    if [ -n "${RECALL_MCP_PS:-}" ]; then
        eval "$RECALL_MCP_PS"
    elif _is_windows; then
        # String interpolation rather than Format-List: the formatter WRAPS long command lines at
        # the console width, and every MCP transport's command line is long enough to be wrapped,
        # which turns one process into several unparseable fragments.
        powershell -NoProfile -Command \
            "Get-CimInstance Win32_Process | ForEach-Object { \"\$(\$_.ProcessId) \$(\$_.ParentProcessId) \$(\$_.CommandLine)\" }" \
            2>/dev/null | tr -d '\r'
    else
        ps -eo pid=,ppid=,args= 2>/dev/null
    fi
}

_kill_pid() {
    if [ -n "${RECALL_MCP_KILL:-}" ]; then
        eval "$RECALL_MCP_KILL $1"
    elif _is_windows; then
        powershell -NoProfile -Command "Stop-Process -Id $1 -Force" >/dev/null 2>&1
    else
        kill "$1" 2>/dev/null
    fi
}

# This session's client process, and this script's own. Both are needed: the first is what makes a
# transport OURS, and the second is what keeps this script from killing the ssh it is itself using
# to count servers, whose command line necessarily contains the pattern it searches for.
SESSION_PID="${CLAUDE_PID:-}"
if _is_windows; then
    SELF_PID="$(cat "/proc/$$/winpid" 2>/dev/null || echo "$$")"
else
    SELF_PID="$$"
fi
#: Exported for one reason: the tests' process-table fixture has to name this pid to prove the
#: self-exclusion below works, and it cannot know it in advance. Nothing in production reads it.
export SELF_PID

# `sweep` is a different question from `close`: not "what did THIS session open" but "what on the
# host has no client left at all". It needs a marker to answer safely, and it lives in Python
# because the answer is a join between two process tables. One entry point, two questions.
if [ "$MODE" = "sweep" ]; then
    SWEEP_PY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/session_mcp_sweep.py"
    if [ ! -f "$SWEEP_PY" ]; then
        echo "session-mcp-close: missing $SWEEP_PY" >&2
        exit 2
    fi
    # shellcheck disable=SC2086 -- SWEEP_ARGS is a deliberate word-split list of flags.
    exec python "$SWEEP_PY" --host "$HOST" $SWEEP_ARGS
fi

TABLE="$(_ps_table)"

_parent_of() {
    printf '%s\n' "$TABLE" | awk -v p="$1" '$1 == p { print $2; exit }'
}

_cmd_of() {
    printf '%s\n' "$TABLE" | awk -v p="$1" '$1 == p { $1=""; $2=""; sub(/^  */, ""); print; exit }'
}

# Walk up from `pid` looking for `want`, at most 8 hops. Loops are possible in a stale table (a
# recycled pid can point at its own descendant), so the hop cap is a termination guarantee rather
# than a performance choice.
_descends_from() {
    local pid="$1" want="$2" hops=0 parent
    [ -n "$want" ] || return 1
    while [ "$hops" -lt 8 ]; do
        [ "$pid" = "$want" ] && return 0
        parent="$(_parent_of "$pid")"
        [ -n "$parent" ] || return 1
        [ "$parent" = "$pid" ] && return 1
        pid="$parent"
        hops=$((hops + 1))
    done
    return 1
}

_tenant_of() {
    printf '%s' "$1" | grep -o 'RECALL_TENANT=[^ ]*' | head -1 | cut -d= -f2
}

# The fleet on the host: how many servers exist, what they hold, and how old the oldest is. One
# ssh, read-only, and never fatal. It is REPORTING, not a target list.
_fleet() {
    timeout "$FLEET_TIMEOUT" ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" \
        "ps -eo rss,etimes,args | grep -F 'python -m $PATTERN' | grep -v grep | \
         grep -vE 'be-child ssh|sshd' | \
         awk '{s+=\$1; n++; if (\$2>m) m=\$2} END {printf \"%d %.1f %.1f\", n, s/1048576, m/3600}'" \
        2>/dev/null
}

_say_fleet() {
    local label="$1" out n gb hours
    [ "$WANT_FLEET" -eq 1 ] || return 0
    out="$(_fleet)"
    if [ -z "$out" ]; then
        printf 'FLEET %-7s unreachable, or no servers running on %s\n' "$label" "$HOST"
        return 0
    fi
    read -r n gb hours <<<"$out"
    printf 'FLEET %-7s %s server(s) on %s, %s GB resident, oldest %sh\n' "$label" "$n" "$HOST" "$gb" "$hours"
}

# --- what is ours -------------------------------------------------------------------------------

CANDIDATES="$(printf '%s\n' "$TABLE" | grep -F "$PATTERN" | awk '{print $1}')"

OURS=""
OTHERS=0
for pid in $CANDIDATES; do
    [ -n "$pid" ] || continue
    cmd="$(_cmd_of "$pid")"
    # This script's own ssh, and anything else it spawned. Checked FIRST: the fleet query's
    # command line contains the pattern, and it descends from this session too, so an ownership
    # test alone would select it.
    if _descends_from "$pid" "$SELF_PID"; then
        continue
    fi
    if [ -n "$SESSION_PID" ] && _descends_from "$pid" "$SESSION_PID"; then
        OURS="$OURS $pid"
    else
        OTHERS=$((OTHERS + 1))
    fi
done

if [ -z "$SESSION_PID" ]; then
    printf 'SESSION     unknown (CLAUDE_PID is not set)\n'
    printf '            Without it nothing here can prove a transport is this session'"'"'s, and on\n'
    printf '            this machine the same command line also belongs to other agents. Reporting only.\n'
else
    printf 'SESSION     client pid %s\n' "$SESSION_PID"
fi

n_ours=0
for pid in $OURS; do n_ours=$((n_ours + 1)); done
printf 'TRANSPORTS  %s this session, %s belonging to something else (left alone)\n' "$n_ours" "$OTHERS"
for pid in $OURS; do
    cmd="$(_cmd_of "$pid")"
    printf '  ours   pid %-7s tenant %s\n' "$pid" "$(_tenant_of "$cmd")"
done

_say_fleet "before"

if [ "$MODE" = "report" ]; then
    [ "$n_ours" -gt 0 ] && printf 'Close them with: scripts/session-mcp-close.sh close\n'
    exit 0
fi

if [ -z "$SESSION_PID" ]; then
    printf 'REFUSED     close needs CLAUDE_PID; nothing was killed.\n' >&2
    exit 3
fi

if [ "$n_ours" -eq 0 ]; then
    printf 'RESULT      nothing of this session'"'"'s to close.\n'
    exit 0
fi

if [ "$DRY_RUN" -eq 1 ]; then
    printf 'DRY-RUN     would close %s transport(s):%s\n' "$n_ours" "$OURS"
    exit 0
fi

closed=0
failed=0
for pid in $OURS; do
    if _kill_pid "$pid"; then
        printf 'CLOSED      pid %s\n' "$pid"
        closed=$((closed + 1))
    else
        printf 'FAILED      pid %s did not close\n' "$pid" >&2
        failed=$((failed + 1))
    fi
done

# A kill that returns 0 says a signal was delivered, not that 850 MB was freed on another machine.
# The fleet count before and after is the only evidence available here that the servers actually
# went with their transports, and it is why the count is taken twice.
if [ "$WANT_FLEET" -eq 1 ] && [ "$closed" -gt 0 ]; then
    sleep "$SETTLE"
    _say_fleet "after"
fi

printf 'RESULT      %s closed, %s failed, %s left alone as not ours\n' "$closed" "$failed" "$OTHERS"
[ "$failed" -eq 0 ] || exit 1
exit 0
