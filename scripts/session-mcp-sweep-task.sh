#!/usr/bin/env bash
# The body of the hourly `recall-mcp-sweep` task: close MCP servers on VPS2 whose client is gone.
#
# Why a schedule and not only the SessionEnd hook: the hook closes a session's servers when the
# session ENDS, and the servers that leak are the ones whose session never ended cleanly. Measured
# 2026-09-26: 33 servers on VPS2, 8 with a live transport, 23 left over from 24 and 25 September.
# Their transports died without the hop noticing (VPS3 still held 31 forwarded connections to
# VPS2's port 22 while this workstation held 13), so nothing on either host would ever end them.
#
# The task never runs a local copy of anything. It runs THIS file and the sweep straight out of
# `origin/master` of the checkout it is given, so a fix to either reaches the schedule when it
# merges, with nothing to redeploy and no copy to drift. The registered action is only:
#
#   git -C <repo> fetch -q origin master; git -C <repo> show origin/master:<this file> | bash -s -- <repo>
#
# Everything it may close is decided by `session_mcp_sweep.py`, by positive identity only, and it
# refuses outright when this machine's process table cannot be read. Register with
# scripts/register-mcp-sweep-task.ps1.
set -u

REPO="${1:-$HOME/Documents/recall}"
# Both overrides exist for a dry run of this exact body before a merge: a branch ref, and no
# `--kill`. The schedule sets neither.
REF="${RECALL_MCP_SWEEP_REF:-origin/master}"
SWEEP_ARGS="${RECALL_MCP_SWEEP_ARGS---kill}"
LOG_DIR="${RECALL_MCP_SWEEP_LOG_DIR:-$HOME/.claude/logs}"
LOG="$LOG_DIR/mcp-sweep.log"
mkdir -p "$LOG_DIR"

# One file, bounded: past 1 MB the old half is kept as .1 and a new file starts.
if [ -f "$LOG" ] && [ "$(wc -c < "$LOG")" -gt 1048576 ]; then
    mv -f "$LOG" "$LOG.1"
fi

{
    printf '=== %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    sweep="$(git -C "$REPO" show "$REF:scripts/session_mcp_sweep.py" 2>/dev/null)"
    if [ -z "$sweep" ]; then
        printf 'SKIPPED  %s:scripts/session_mcp_sweep.py is unreadable in %s\n' "$REF" "$REPO"
        exit 2
    fi
    # shellcheck disable=SC2086 -- SWEEP_ARGS is a deliberate word-split list of flags.
    printf '%s\n' "$sweep" | python - $SWEEP_ARGS
    printf 'exit %s\n' "$?"
} >> "$LOG" 2>&1
