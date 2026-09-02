#!/usr/bin/env bash
# Tests for `prereg-pending.sh`. Fixtures only: no repository files are read, so this needs no
# database, no network and no particular state of docs/preregistrations.
#
# Mutation-tested. Each case names the mutation it dies to, because a guard nobody has watched fail
# has not been tested, and this guard's whole job is to be quiet about 91 records and loud about 7.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/prereg-pending.sh"
PASS=0
FAIL=0

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

check() {  # check <name> <expected pending basenames, space separated>
    local name="$1" expected="$2" got
    got="$(bash "$SCRIPT" "$work/docs" | sed 's|.*/||' | sort | tr '\n' ' ' | sed 's/ $//')"
    expected="$(printf '%s' "$expected" | tr ' ' '\n' | sort | tr '\n' ' ' | sed 's/ $//')"
    if [ "$got" = "$expected" ]; then
        PASS=$((PASS + 1))
        printf '  ok    %s\n' "$name"
    else
        FAIL=$((FAIL + 1))
        printf '  FAIL  %s\n         expected [%s]\n         got      [%s]\n' "$name" "$expected" "$got"
    fi
}

reset() { rm -rf "$work/docs"; mkdir -p "$work/docs"; }
write() { printf '%s\n' "$2" > "$work/docs/$1"; }

printf 'prereg-pending.sh\n'

# 1. The case the old grep got wrong, and the reason this script exists. A completed record KEEPS
#    its prediction header verbatim, because the project's rule is that a prediction is never
#    edited. Dies to: reverting to `grep -rl "predicted, not yet measured"`.
reset
write done.md '**Status:** predicted, not yet measured

## Observed results

**Status:** measured 2026-08-29. The prediction was falsified.'
check "a measured record keeps its prediction header and is NOT pending" ""

# 2. The genuinely pending case, which must stay loud.
reset
write open.md '**Status:** predicted, not yet measured

## Observed results

Not yet measured.'
check "a record with no resolving status IS pending" "open.md"

# 3. "still predicted, not yet measured" is a restatement, not a resolution. Dies to: treating any
#    second Status line as resolving.
reset
write still.md '**Status:** predicted, not yet measured

## Observed results

Status: still predicted, not yet measured -- the harness is not built.'
check "a restated prediction is still pending" "still.md"

# 4. A result need not use the word "measured". Dies to: matching on `Status:.*measured`.
reset
write partial.md '**Status:** predicted, not yet measured

Status: partially confirmed, with the interesting half falsified.'
check "a result worded differently still resolves" ""

# 5. The script itself offers "abandoned" as an outcome, so it must accept one.
reset
write abandoned.md '**Status:** predicted, not yet measured

Status: abandoned, the lane closed before the harness was built.'
check "an abandoned record resolves" ""

# 6. A qualified result resolves. Dies to: requiring the status line to be exactly "measured".
reset
write qualified.md '**Status:** predicted, not yet measured

**Status:** measured (LoCoMo arm still running)'
check "a qualified result resolves" ""

# 7. A file with no prediction header at all is not a pre-registration in progress.
reset
write notes.md '# Some design note

No status line here at all.'
check "a file with no prediction header is not pending" ""

# 8. Mixed directory: the whole point is picking the few out of the many.
reset
write a-open.md '**Status:** predicted, not yet measured'
write b-done.md '**Status:** predicted, not yet measured
**Status:** measured'
write c-open.md '**Status:** predicted, not yet measured'
check "reports only the unresolved ones from a mixed directory" "a-open.md c-open.md"

# 9. An empty or absent directory is silence, not an error. Dies to: an unguarded glob.
reset
check "an empty directory reports nothing" ""
rm -rf "$work/docs"
if out="$(bash "$SCRIPT" "$work/docs" 2>&1)" && [ -z "$out" ]; then
    PASS=$((PASS + 1)); printf '  ok    a missing directory exits quietly\n'
else
    FAIL=$((FAIL + 1)); printf '  FAIL  a missing directory should exit 0 and print nothing, got [%s]\n' "$out"
fi

printf '\n%d passed, %d failed\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
