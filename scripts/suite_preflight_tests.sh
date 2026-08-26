#!/usr/bin/env bash
# Regression tests for `scripts/suite-preflight.sh`.
#
# What these pin: the worker count `make test` launches with is decided by ACTUAL available
# memory, because the hard-coded `-n 4` it replaces was measured killing workers on a loaded
# 12 GB box (2026-08-23, `node down: Not properly terminated`), and a killed worker costs a rerun
# of the whole suite. The memory probe is bypassed with RECALL_SUITE_AVAIL_MB, deliberately:
# these tests are about the MAPPING, and a suite that needs a particular amount of free RAM to
# pass would fail on exactly the machines the guard exists for.
#
# Every boundary is tested one MB either side, because an off-by-one in a threshold is invisible
# in normal use until the box is at exactly that boundary, which is the moment it matters.
#
# Mutation-tested 2026-08-26, three ways (break, watch red, restore), re-runnable with the same
# three seds against this file's assertions:
#
#   `-ge 6000` -> `-ge 5000`         "5999 MB -> 2 workers" red        (4 picked where 2 fit)
#   `-ge 3000` -> `-ge 2000`         "2999 MB -> 1 worker" red         (2 picked where 1 fits)
#   unknown-memory fallback 2 -> 4   "unreadable memory -> 2" red      (no probe reads as idle)
set -uo pipefail

SP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/suite-preflight.sh"
pass=0; fail=0
ok() { pass=$((pass+1)); printf 'PASS  %s\n' "$1"; }
no() { fail=$((fail+1)); printf 'FAIL  %s\n     got: %s\n' "$1" "${2:-}"; }

# nworkers must print ONLY the number on stdout — `make test` substitutes it straight into
# `pytest -n`. Reasoning goes to stderr, which is asserted by discarding it and checking purity.
pick() { RECALL_SUITE_AVAIL_MB="$1" N="${2:-}" sh "$SP" nworkers 2>/dev/null; }

# 1. stdout purity: a single integer, nothing else, or `pytest -n $(...)` gets garbage.
out=$(pick 8000)
[[ "$out" =~ ^[0-9]+$ ]] && ok "stdout is a bare integer" || no "stdout is a bare integer" "$out"

# 2/3. the idle-box boundary: 6000 MB is the measured comfortable case for four workers.
[[ $(pick 6000) == 4 ]] && ok "6000 MB -> 4 workers" || no "6000 MB -> 4 workers" "$(pick 6000)"
[[ $(pick 5999) == 2 ]] && ok "5999 MB -> 2 workers" || no "5999 MB -> 2 workers" "$(pick 5999)"

# 4/5. the loaded-box boundary.
[[ $(pick 3000) == 2 ]] && ok "3000 MB -> 2 workers" || no "3000 MB -> 2 workers" "$(pick 3000)"
[[ $(pick 2999) == 1 ]] && ok "2999 MB -> 1 worker" || no "2999 MB -> 1 worker" "$(pick 2999)"

# 6. a probe that cannot answer must NOT read as an idle box. A set-but-EMPTY override is the
# documented simulation of "every probe failed" (the tests cannot portably remove /proc/meminfo
# from under the script), and it must land on the conservative 2, never the idle-box 4.
out=$(pick "")
[[ "$out" == 2 ]] && ok "unreadable memory -> conservative 2" || no "unreadable memory -> conservative 2" "$out"

# 7. N wins over everything: the operator's override is taken verbatim.
[[ $(pick 1000 8) == 8 ]] && ok "N=8 overrides a starved box" || no "N=8 overrides a starved box" "$(pick 1000 8)"

# 8. report mode never emits a bare integer stdout contract; it is for humans and must mention
# the memory figure it saw.
rep=$(RECALL_SUITE_AVAIL_MB=4321 sh "$SP" report 2>/dev/null)
grep -q "4321" <<<"$rep" && ok "report names the memory it saw" || no "report names the memory it saw" "$rep"

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
