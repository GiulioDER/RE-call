#!/usr/bin/env bash
# Tests for `scripts/session-corpus.sh`, the corpus report a session opens with.
#
# Why this file exists, written 2026-08-31
# ---------------------------------------
# It did not, and the script had just grown a control-flow change: the "NOT PROVED: that the server
# starts" boundary moved from two lines at the end into a function called before EVERY exit. The
# first draft put those lines at the end and commented them "printed always". They were not: two
# earlier `exit 0` paths returned first, and those are exactly the states a session most needs the
# pointer from. Five auditors found that independently, and none of them could have been a test,
# because there was no test.
#
# That is the same failure this script is itself about. `session-corpus.sh` exists because a report
# said `.mcp.json present (2 servers)` and was read as a statement about a corpus; its new footer
# exists because the report was read as a statement about a process. A fix for that class,
# shipped with nothing that can observe it fail, is the third instance rather than the cure.
#
# ssh is stubbed, so nothing here reaches VPS2, a database, or a network.
#
# Mutation-tested 2026-08-31, four ways, all four killed. Measured:
#   `not_proved` call deleted from the UNREACHABLE branch        1 red
#   `not_proved` call deleted from the NO-ACTIVE-GENERATION path 2 red
#   `not_proved` call deleted from the summary path              3, 4 red
#   `grep -cF '| CERTIFIED '` -> `grep -vc CERTIFIED`            3 red
#
# 🔁 That last row was written as `4 red` before the mutation was run, and the run said 3. The
# prediction was the intuitive one and it was wrong: an inverted match makes the ALL-CERTIFIED
# case report `0 of 3`, which test 3 catches immediately, so test 4 never gets to be the witness.
# Corrected to the measured value rather than deleted, because which test actually catches a bug
# is the part worth knowing, and a header full of guesses that were never run is how a mutation
# record becomes decoration.
set -uo pipefail

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/session-corpus.sh"
# Unique per run, for the reason session_serving_tests.sh now carries: a fixed shared fixture makes
# two concurrent runs report each other's timing rather than the code under test.
BASE="${TMPDIR:-/tmp}/recall-corpustests-$$"
# Removed only on a GREEN run, matching session_serving_tests.sh: the isolation this trap gives is
# about concurrent runs, and deleting the fixture a reviewer needs is the wrong half of it.
trap '[ "$fail" -eq 0 ] && rm -rf "$BASE"' EXIT
rm -rf "$BASE"; mkdir -p "$BASE/bin"

pass=0; fail=0
ok() { pass=$((pass+1)); printf 'PASS  %s\n' "$1"; }
no() { fail=$((fail+1)); printf 'FAIL  %s\n     %s\n' "$1" "${2:-}"; }

# The stub answers as the real `ssh <host> "...psql..."` does: whatever FAKE_ROWS holds, with
# FAKE_SSH_RC as the exit status. It swallows its arguments the way the real one would use them.
cat > "$BASE/bin/ssh" <<'STUB'
#!/usr/bin/env bash
[ -n "${FAKE_ROWS:-}" ] && printf '%s\n' "$FAKE_ROWS"
[ "${FAKE_SSH_RC:-0}" -ne 0 ] && printf 'ssh: connect to host vps2 port 22: Connection refused\n' >&2
exit "${FAKE_SSH_RC:-0}"
STUB
chmod +x "$BASE/bin/ssh"

OUT=""; RC=0
# `run <ssh-rc> <rows>` rather than a word-split KEY=VALUE string: the rows are multi-line and
# contain `|`, which any split-on-space scheme mangles into an `env: '|': No such file` error.
run() {
    OUT="$(PATH="$BASE/bin:$PATH" RECALL_VPS2_HOST=stub-host            FAKE_SSH_RC="${1:-0}" FAKE_ROWS="${2:-}" bash "$SCRIPT" status 2>&1)"
    RC=$?
}

CERT3='memory | gen gen_aaaa | CERTIFIED thr=0.509 sep=0.974 n=50/28
re-call-code-gen | gen gen_bbbb | CERTIFIED thr=0.662 sep=0.988 n=22/26
re-call-docs | gen gen_cccc | CERTIFIED thr=0.637 sep=0.976 n=40/40'

MIXED='memory | gen gen_aaaa | CERTIFIED thr=0.509 sep=0.974 n=50/28
re-call-code-gen | gen gen_bbbb | NOT-CERTIFIED
re-call-docs | gen gen_cccc | STALE (fingerprint moved)'

# --- 1. the host is unreachable ----------------------------------------------------------------
# The branch where the pointer matters most: "ssh is down" and "the server will not start" are
# different failures, and this path used to assert the second from evidence of only the first.
run 255 ""
if [ "$RC" -eq 0 ] && printf '%s' "$OUT" | grep -q 'UNREACHABLE' \
   && printf '%s' "$OUT" | grep -q 'NOT PROVED: that the server starts'; then
    ok "1  an unreachable host still prints the boundary and the command that crosses it"
else
    no "1  an unreachable host still prints the boundary and the command that crosses it" "rc=$RC $OUT"
fi

# --- 2. reachable, but nothing promoted --------------------------------------------------------
run 0 ""
if [ "$RC" -eq 0 ] && printf '%s' "$OUT" | grep -q 'NO ACTIVE GENERATION' \
   && printf '%s' "$OUT" | grep -q 'NOT PROVED: that the server starts'; then
    ok "2  a corpus with no active generation still prints the boundary"
else
    no "2  a corpus with no active generation still prints the boundary" "rc=$RC $OUT"
fi

# --- 3. every tenant certified -----------------------------------------------------------------
# The green path. The claim must stay subjunctive: this asked the corpus, not the process.
run 0 "$CERT3"
if [ "$RC" -eq 0 ] && printf '%s' "$OUT" | grep -q 'all 3 active tenants certified' \
   && printf '%s' "$OUT" | grep -q 'WOULD serve these as trusted' \
   && printf '%s' "$OUT" | grep -q 'NOT PROVED: that the server starts'; then
    ok "3  an all-certified corpus reports it subjunctively and still prints the boundary"
else
    no "3  an all-certified corpus reports it subjunctively and still prints the boundary" "rc=$RC $OUT"
fi

# --- 4. a genuinely uncertified tenant is counted POSITIVELY -----------------------------------
# The control for 3, and the regression this script's own comment records as having shipped once:
# the failure state used to render as "UNCERTIFIED", which CONTAINS "CERTIFIED", so a `grep -v`
# selected nothing and an uncertified corpus was summarised as "all certified". A summary line that
# can only ever say "fine" is worse than no summary.
run 0 "$MIXED"
if [ "$RC" -eq 0 ] && printf '%s' "$OUT" | grep -q '1 of 3 active tenants certified' \
   && printf '%s' "$OUT" | grep -q 'NOT PROVED: that the server starts'; then
    ok "4  a mixed corpus is counted positively, not by an inverted match"
else
    no "4  a mixed corpus is counted positively, not by an inverted match" "rc=$RC $OUT"
fi

printf '\n%s passed, %s failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ] || exit 1
