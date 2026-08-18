#!/usr/bin/env bash
# Regression tests for scripts/session-space.sh claim atomicity and staleness.
set -uo pipefail

SPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/session-space.sh"
BASE="${TMPDIR:-/tmp}/recall-spacetests"
pass=0; fail=0
ok()  { pass=$((pass+1)); printf 'PASS  %s\n' "$1"; }
no()  { fail=$((fail+1)); printf 'FAIL  %s\n     %s\n' "$1" "${2:-}"; }

rm -rf "$BASE"; mkdir -p "$BASE"; cd "$BASE"
git init -q .; git config user.email t@t; git config user.name t
echo x > a.txt; git add a.txt; git commit -qm init
git worktree add -q -b wt ./wt HEAD
WT="$BASE/wt"
CLAIM="$BASE/.git/worktrees/wt/claude-session-claim"

# --- 1. main checkout is refused -------------------------------------------
if CLAUDE_CODE_SESSION_ID=S1 bash "$SPACE" check >/dev/null 2>&1; then
    no "main checkout is refused" "check passed in the main checkout"
else
    ok "main checkout is refused"
fi

# --- 2. a live pid with NO epoch must NOT be stale --------------------------
# A REAL Windows pid, because $$ from Git Bash is NOT one and tasklist cannot see
# it: using $$ made this test pass for the wrong reason once already.
#
# ANY live Windows process will do, not specifically claude.exe. Requiring
# claude.exe meant the suite exited "SKIP" wherever one was not running, which is
# every CI runner, so the test that proves a live claim is protected would never
# have executed in CI at all.
# ⚠️ DIGITS ONLY, and that is the entire point of this function.
#
# `tasklist //FI ...` with no match prints "INFO: No tasks are running which
# match the specified criteria." on STDOUT and exits 0, so `awk '{print $2}'`
# harvests the literal string **"No"**. That is not a rejected value: it flows
# into the claim file as `pid=No`, and `_process_alive` treats an unparseable pid
# as ALIVE, so the test still passes. Mutation-tested: with a real pid, breaking
# session-space.sh's liveness check turns test 2 RED; with `pid=No` the same
# broken code passes. Every CI runner is a machine with no claude.exe, so this
# made the suite run in CI without testing anything there.
_live_pid() {
    local p
    p=$(tasklist //FI "IMAGENAME eq claude.exe" //NH 2>/dev/null \
        | awk 'NF>1 && $2 ~ /^[0-9]+$/ {print $2; exit}')
    if [ -z "$p" ]; then
        p=$(tasklist //NH 2>/dev/null | awk 'NF>1 && $2 ~ /^[0-9]+$/ {print $2; exit}')
    fi
    printf '%s' "$p"
}
# `|| true`: awk exits early, tasklist takes SIGPIPE, and under `-eo pipefail`
# the substitution's status is 141, which would kill the suite at this line.
LIVE_PID=$(_live_pid) || true
case "$LIVE_PID" in
    ''|*[!0-9]*) echo "FAIL: no numeric live pid available (got '${LIVE_PID}')" >&2; exit 1 ;;
esac
printf 'session=OTHER\npid=%s\n' "$LIVE_PID" > "$CLAIM"
cd "$WT"
if CLAUDE_CODE_SESSION_ID=ME bash "$SPACE" check >/dev/null 2>&1; then
    no "live pid + no epoch is refused" "check PASSED; the live claim was treated as stale"
else
    ok "live pid + no epoch is refused"
fi

# --- 2b. control: dead pid + old epoch IS takeoverable ----------------------
printf 'session=OTHER\npid=999999999\nclaimed_epoch=1\n' > "$CLAIM"
if CLAUDE_CODE_SESSION_ID=ME bash "$SPACE" check >/dev/null 2>&1; then
    ok "control: dead pid + old epoch is takeoverable"
else
    no "control: dead pid + old epoch is takeoverable" "refused, so test 2 proves nothing"
fi

# --- 3. concurrent claim on an unclaimed worktree: one winner ---------------
# REPEATED, because one round is a coin toss. Measured against a deliberately
# broken pid: a single round of 4 racers caught the defect only 2 times in 5, so
# a one-shot version of this test would miss a real regression 60% of the time.
# Five rounds take it to ~92%, and the fixed code is green in every round, so the
# repetition buys sensitivity without buying flaky reds.
ROUNDS="${RECALL_RACE_ROUNDS:-5}"
race_bad=0
for round in $(seq 1 "$ROUNDS"); do
rm -f "$CLAIM"
outdir="$BASE/out"; mkdir -p "$outdir"
# ⚠️ `env -u CLAUDE_PID`, deliberately, and the whole test depends on it.
#
# With CLAUDE_PID exported, all four racers record the SAME live Windows pid, so
# no racer ever judges another's lock dead and the test passes for a reason that
# has nothing to do with the locking. Without it, each records its own pid, which
# is the real contended case AND the documented manual workflow. That difference
# is not academic: it is exactly why this passed here and produced TWO winners on
# a clean CI runner. A test that only exercises the easy path when the ambient
# environment happens to be rich is a test of the environment.
for s in A B C D; do
    ( env -u CLAUDE_PID CLAUDE_CODE_SESSION_ID="$s" bash "$SPACE" claim \
        >"$outdir/$s.log" 2>&1; echo $? > "$outdir/$s.rc" ) &
done
wait
winners=0
for s in A B C D; do
    [ "$(cat "$outdir/$s.rc")" = "0" ] && winners=$((winners+1))
done
[ "$winners" -eq 1 ] || { race_bad=$((race_bad+1)); race_detail="round $round: winners=$winners"; }
done
if [ "$race_bad" -eq 0 ]; then
    ok "concurrent claim has exactly one winner (n=4, ${ROUNDS} rounds)"
else
    no "concurrent claim has exactly one winner (n=4, ${ROUNDS} rounds)" \
       "$race_bad of $ROUNDS rounds wrong; last: ${race_detail:-}"
fi

# --- 4. the claim file is never torn ---------------------------------------
lines=$(wc -l < "$CLAIM")
if [ "$lines" -eq 6 ]; then
    ok "the winning claim file is complete (6 lines)"
else
    no "the winning claim file is complete (6 lines)" "got $lines lines"
fi

# --- 5. re-claiming by the holder is idempotent -----------------------------
holder=$(sed -n 's/^session=//p' "$CLAIM" | head -1)
if CLAUDE_CODE_SESSION_ID="$holder" bash "$SPACE" claim >/dev/null 2>&1; then
    ok "the holder can refresh its own claim"
else
    no "the holder can refresh its own claim" "refresh returned non-zero"
fi

# --- 6. no temp files left behind ------------------------------------------
strays=$(ls "$BASE/.git/worktrees/wt/"*.tmp 2>/dev/null | wc -l)
if [ "$strays" -eq 0 ]; then
    ok "no .tmp strays left in the git dir"
else
    no "no .tmp strays left in the git dir" "$strays found"
fi

# --- 7. an orphaned lock is reclaimable -------------------------------------
# AUDIT-2 BUG-003: the first version of the lock could not be reclaimed. A holder
# killed without running its trap blocked every future claim in that worktree
# forever, `release --force` did not clear it, and `whose` reported "unclaimed"
# at the same time. The SessionStart hook creates exactly that state on its own
# timeout path, which uses `taskkill /T /F` and runs no trap.
rm -f "$CLAIM"
mkdir -p "$CLAIM.lock"
printf '999999999\n' > "$CLAIM.lock/pid"        # a pid that is certainly dead
if CLAUDE_CODE_SESSION_ID=ORPHAN bash "$SPACE" claim >/dev/null 2>&1; then
    ok "a lock held by a dead pid is broken, not waited for"
else
    no "a lock held by a dead pid is broken, not waited for" "claim still refused"
fi
[ -d "$CLAIM.lock" ] && no "the lock is released after a successful claim" "lock still present" \
                     || ok "the lock is released after a successful claim"

# --- 8. a lock held by a LIVE pid is respected -------------------------------
# The declared control for test 7: if test 7 passed because the lock is ALWAYS
# broken, this must fail.
#
# ⚠️ The claim must be removed first. Test 7 leaves the worktree claimed by ORPHAN
# with a fresh epoch, so without this the refusal comes from the existing claim
# and the control passes with no lock present at all: verified by deleting the
# mkdir below and still getting 11/11. A control that cannot fail is not a
# control.
rm -f "$CLAIM"
LIVE_PID2=$(_live_pid) || true
mkdir -p "$CLAIM.lock"; printf '%s\n' "$LIVE_PID2" > "$CLAIM.lock/pid"
out8=$(RECALL_CLAIM_LOCK_WAIT=3 CLAUDE_CODE_SESSION_ID=OTHER2 bash "$SPACE" claim 2>&1)
if [ $? -eq 0 ]; then
    no "a lock held by a LIVE pid is respected" "claim succeeded; test 7 proves nothing"
elif printf '%s' "$out8" | grep -qi "claiming this worktree right now"; then
    ok "a lock held by a LIVE pid is respected (refused BY THE LOCK)"
else
    no "a lock held by a LIVE pid is respected" "refused, but not by the lock: $out8"
fi

# --- 9. release --force clears a stranded lock -------------------------------
if CLAUDE_CODE_SESSION_ID=ANY bash "$SPACE" release --force >/dev/null 2>&1; [ ! -d "$CLAIM.lock" ]; then
    ok "release --force clears a stranded lock"
else
    no "release --force clears a stranded lock" "lock survived the documented recovery"
fi

printf '\n%d/%d passed\n' "$pass" "$((pass+fail))"
[ "$fail" -eq 0 ]
