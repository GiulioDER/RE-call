#!/usr/bin/env sh
# suite-preflight.sh — size the test suite's worker count from the machine's ACTUAL state,
# instead of assuming the box is idle.
#
# Why this exists: `make test` hard-coded `-n 4`, which is right on an idle 12 GB machine
# (measured 2026-08-23: serial 49:58, `-n 4` 14:05) and wrong on a loaded one, where workers are
# OOM-killed (`node down: Not properly terminated`, both `-n 6` runs that day) and the retries
# turn a 14 minute suite into a 40+ minute one that also starves every other session on the box.
# This machine routinely runs several Claude sessions, Docker, and other sessions' embedding or
# indexing processes at once; free memory at launch time is the only honest input.
#
# Usage:
#   scripts/suite-preflight.sh nworkers   # prints the worker count on stdout, reasoning on stderr
#   scripts/suite-preflight.sh report     # human-readable report only
#
# Overrides:
#   N=<n>                       take <n> verbatim (same override `make test N=8` always had)
#   RECALL_SUITE_AVAIL_MB=<mb>  bypass the memory probe (used by the tests, and as an escape
#                               hatch on a platform the probe cannot read)
#
# The thresholds are a HEURISTIC, not a measurement: what is measured (2026-08-23, 12 GB box) is
# that four workers fit an idle machine and six do not, and that workers die rather than slow
# down when memory runs out. The mapping below stays deliberately conservative because a killed
# worker costs a rerun of the whole suite, while a spare gigabyte costs nothing.

set -u

avail_mb() {
    # Set-but-EMPTY is honoured too: it simulates "every probe failed" for the tests, which
    # cannot portably remove /proc/meminfo from under the script.
    if [ -n "${RECALL_SUITE_AVAIL_MB+set}" ]; then
        echo "${RECALL_SUITE_AVAIL_MB}"
        return
    fi
    if [ -r /proc/meminfo ]; then
        mb=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
        # Git Bash EMULATES /proc/meminfo without a MemAvailable line (verified 2026-08-26 on
        # this workstation), so an empty answer here means "wrong probe", not "no memory":
        # fall through to the Windows probe rather than reporting unknown.
        if [ -n "${mb}" ]; then
            echo "${mb}"
            return
        fi
    fi
    if command -v powershell.exe >/dev/null 2>&1; then
        # AvailableMBytes matches Task Manager's "Available" (free + standby), which is what a
        # new process can actually claim; Win32_OperatingSystem.FreePhysicalMemory undercounts it.
        powershell.exe -NoProfile -Command \
            "(Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes" \
            2>/dev/null | tr -d '\r' | grep -E '^[0-9]+$' | head -1
        return
    fi
    if command -v vm_stat >/dev/null 2>&1; then
        vm_stat 2>/dev/null | awk '
            /Pages free|Pages inactive/ {gsub("\\.",""); pages+=$NF}
            END {if (pages) print int(pages*4096/1048576)}'
    fi
}

pick_n() {
    # $1 = available MB, possibly empty when the probe failed.
    mb="$1"
    if [ -z "${mb}" ]; then
        echo "preflight: cannot read available memory on this platform; assuming a loaded box." >&2
        echo 2
        return
    fi
    if [ "${mb}" -ge 6000 ]; then
        echo 4
    elif [ "${mb}" -ge 3000 ]; then
        echo 2
    else
        echo 1
    fi
}

# Best-effort: another suite (or a local embedding/indexing run) already on this machine shares
# the RAM this one is about to size itself against. "Cannot tell" stays silent rather than crying
# wolf; this is a warning, never a refusal, because the other run may be about to finish.
warn_concurrent() {
    others=""
    if command -v powershell.exe >/dev/null 2>&1; then
        others=$(powershell.exe -NoProfile -Command \
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'pytest|recall.cli.*index|fastembed' } | ForEach-Object { \$_.CommandLine.Substring(0, [Math]::Min(90, \$_.CommandLine.Length)) }" \
            2>/dev/null | tr -d '\r')
    elif command -v pgrep >/dev/null 2>&1; then
        others=$(pgrep -af 'pytest|recall\.cli.*index' 2>/dev/null | grep -v "suite-preflight" || true)
    fi
    if [ -n "${others}" ]; then
        echo "preflight: python processes already running that will compete with the suite:" >&2
        echo "${others}" | sed 's/^/    /' >&2
    fi
}

mb=$(avail_mb)
case "${1:-report}" in
    nworkers)
        n="${N:-}"
        if [ -n "${n}" ]; then
            echo "preflight: N=${n} taken verbatim (available memory ${mb:-unknown} MB)." >&2
        else
            n=$(pick_n "${mb}")
            echo "preflight: ${mb:-unknown} MB available -> ${n} worker(s). Override with N=<n>." >&2
        fi
        warn_concurrent
        echo "${n}"
        ;;
    report)
        echo "available memory : ${mb:-unknown} MB"
        echo "workers it picks : $(pick_n "${mb}" 2>/dev/null)"
        warn_concurrent
        ;;
    *)
        echo "usage: scripts/suite-preflight.sh {nworkers|report}" >&2
        exit 2
        ;;
esac
