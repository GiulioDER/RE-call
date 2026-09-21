#!/usr/bin/env bash
# Regression tests for `scripts/suite-preflight.sh`.
#
# What these pin: local `make test` uses exactly three pytest workers regardless of the current
# memory reading. The memory probe remains a report and warning input, while `N=<n>` remains an
# explicit operator override. The probe is bypassed with RECALL_SUITE_AVAIL_MB so these tests stay
# deterministic.
#
# Every boundary is tested one MB either side, because an off-by-one in a threshold is invisible
# in normal use until the box is at exactly that boundary, which is the moment it matters.
#
# The fixed count is intentionally tested at both former memory thresholds and with an unreadable
# probe so a later edit cannot silently restore adaptive scheduling.
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

# 2/3. former idle-box boundary values still use the fixed count.
[[ $(pick 6000) == 3 ]] && ok "6000 MB -> 3 workers" || no "6000 MB -> 3 workers" "$(pick 6000)"
[[ $(pick 5999) == 3 ]] && ok "5999 MB -> 3 workers" || no "5999 MB -> 3 workers" "$(pick 5999)"

# 4/5. the loaded-box boundary.
[[ $(pick 3000) == 3 ]] && ok "3000 MB -> 3 workers" || no "3000 MB -> 3 workers" "$(pick 3000)"
[[ $(pick 2999) == 3 ]] && ok "2999 MB -> 3 workers" || no "2999 MB -> 3 workers" "$(pick 2999)"

# 6. a probe that cannot answer still follows the fixed rule. A set-but-EMPTY override is the
# documented simulation of "every probe failed".
out=$(pick "")
[[ "$out" == 3 ]] && ok "unreadable memory -> fixed 3" || no "unreadable memory -> fixed 3" "$out"

# 7. N wins over everything: the operator's override is taken verbatim.
[[ $(pick 1000 8) == 8 ]] && ok "N=8 overrides a starved box" || no "N=8 overrides a starved box" "$(pick 1000 8)"

# 8. report mode never emits a bare integer stdout contract; it is for humans and must mention
# the memory figure it saw.
rep=$(RECALL_SUITE_AVAIL_MB=4321 sh "$SP" report 2>/dev/null)
grep -q "4321" <<<"$rep" && ok "report names the memory it saw" || no "report names the memory it saw" "$rep"

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
