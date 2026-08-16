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
# A REAL Windows pid, because $$ from Git Bash is not one and tasklist cannot
# see it. Using $$ here made this test pass for the wrong reason.
LIVE_PID=$(tasklist //FI "IMAGENAME eq claude.exe" //NH 2>/dev/null | awk 'NF{print $2; exit}')
[ -z "$LIVE_PID" ] && { echo "SKIP: no live claude.exe to test against"; exit 2; }
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
rm -f "$CLAIM"
outdir="$BASE/out"; mkdir -p "$outdir"
for s in A B C D; do
    ( CLAUDE_CODE_SESSION_ID="$s" bash "$SPACE" claim >"$outdir/$s.log" 2>&1; echo $? > "$outdir/$s.rc" ) &
done
wait
winners=0
for s in A B C D; do
    [ "$(cat "$outdir/$s.rc")" = "0" ] && winners=$((winners+1))
done
if [ "$winners" -eq 1 ]; then
    ok "concurrent claim has exactly one winner (n=4)"
else
    no "concurrent claim has exactly one winner (n=4)" "winners=$winners"
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
# The control: if test 7 passed because the lock is always broken, this fails.
LIVE_PID2=$(tasklist //FI "IMAGENAME eq claude.exe" //NH 2>/dev/null | awk 'NF{print $2; exit}')
mkdir -p "$CLAIM.lock"; printf '%s\n' "$LIVE_PID2" > "$CLAIM.lock/pid"
if RECALL_CLAIM_LOCK_WAIT=3 CLAUDE_CODE_SESSION_ID=OTHER2 bash "$SPACE" claim >/dev/null 2>&1; then
    no "a lock held by a LIVE pid is respected" "claim succeeded; test 7 proves nothing"
else
    ok "a lock held by a LIVE pid is respected"
fi

# --- 9. release --force clears a stranded lock -------------------------------
if CLAUDE_CODE_SESSION_ID=ANY bash "$SPACE" release --force >/dev/null 2>&1; [ ! -d "$CLAIM.lock" ]; then
    ok "release --force clears a stranded lock"
else
    no "release --force clears a stranded lock" "lock survived the documented recovery"
fi

printf '\n%d/%d passed\n' "$pass" "$((pass+fail))"
[ "$fail" -eq 0 ]
