#!/usr/bin/env bash
# Tests for `scripts/session_serving_remote.sh`, the half that moves a live deployment.
#
# Everything expensive is stubbed: a fake origin and a fake serving clone made with real git, and
# a `python` stub that answers the four calls the script makes. Nothing here needs VPS2, a
# database, or a network, because a guard that can only be exercised against the one host it
# protects is a guard nobody runs before changing it.
#
# What each test defends, in one line: this script fast-forwards a deployment and can roll it
# back, so the failure modes worth pinning are "moved when it should have refused" and "left the
# deployment broken". Every refusal test therefore has a CONTROL that must still sync, or a
# blanket refusal would pass the whole file.
#
# Mutation-tested 2026-08-26, seven ways. The guard was broken on purpose and the tests below
# watched to go red. These are the measured results, not the predicted ones:
#
#   dirty refusal -> `if false`                           5 red
#   ancestor refusal -> `if false`                        8 red
#   migration gate -> `if false`                          6 red
#   rollback `git reset --hard "$HEAD_SHA"` -> `true`     9, 11, 13, 14 red
#   `_verify_schema` drops the `compatible: yes` grep     11 red
#   `_verify_all` ignores the skip flag                   10 red
#   `if ! flock -n 8` -> `if false`                       19 red   (watched ON VPS2, see below)
#   the wrapper's content comparison -> `false`           21 red   (the squash false alarm returns)
#   the wrapper's 127 case removed                        23 red
#
# Mutation-tested again 2026-08-31, FIFTEEN ways against this file, all fifteen killed, on an
# isolated fixture (see the BASE note below). Measured, never predicted:
#
#   `_verify_all 0` -> `_verify_all "$NO_VERIFY"`         28 red
#   the verify failure path exits 0 instead of 4          27, 33 red
#   the DIRTY report dropped from verify                  26 red
#   the verify block moved BELOW the fetch                25 red
#   the wrapper drops `verify` from its mode list         29, 34, 35 red
#   the wrapper prints LOCAL on a verify                  29 red
#   the verify short-circuit DELETED (falls into sync)    24, 25, 26, 28, 32 red
#   the remote stops refusing a second mode word          30, 31 red
#   the wrapper stops refusing a second mode word         35 red
#   the wrapper stops skipping the value after `--to`     36 red
#   the missing-env-file guard removed                    33 red
#   RESULT stops naming the tenant and the gaps           24, 32 red
#   the stub ssh stops logging argv                       34, 36 red
#   the `--help` sentinel markers removed                 37 red
#   `_remote_arg` always %q-quotes (kills tilde support)  38 red
#
# ⚠️ THE SCORE ABOVE REPLACES A SEVEN-MUTATION SCORE THAT WAS NOT TRUSTWORTHY, and the reason is
# the fixture rather than the mutations. `BASE` used to be a fixed shared path that this file
# `rm -rf`s at start, so two runs at once delete each other's stubs. Reproduced the same day: one
# run alone is 27/0; two launched concurrently from one shell, same commit, both report 9/18.
# Five reviewers running this suite while each other ran it got 27/0, 21/6, 20/7, 16/11 and 12/15
# on identical code. A mutation score collected on a fixture another process can delete is not a
# score, so the whole set was re-run after `BASE` became per-pid.
#
# ⚠️ An EARLIER harness reported 0 of 7 killed, which was an artefact rather than a result: on
# Windows `subprocess.run(["bash", ...])` resolves to the WSL relay, not Git Bash, so the suite
# never ran, stdout was empty, and "no FAIL lines" scored as "the guard held". A mutation harness
# that cannot tell a surviving guard from a suite that did not start is the same defect this
# file's subject is about, one level up. Assert the run produced verdicts before scoring it.
#
# 🔑 Two mutations exist only because earlier rounds left a test green that should have died.
# The short-circuit deletion was added when the first six all left test 24 alive, and 24 is the
# only test asserting `verify` did not MOVE a checkout that was behind. The argv-logging mutation
# was added when a reviewer proved that NO wrapper test could see a wrapper that recognises
# `verify` and then forwards nothing: the stub ssh discarded its arguments, the remote half
# defaults to `status`, and every assertion still passed. That is a guard that cannot fail, which
# is the failure mode this whole file exists to prevent. Test 34 closes it.
#
# ⚠️ Tests 19 and 20 need `flock` and therefore SKIP on Windows, which is where this file is most
# often run by hand. They were watched on VPS2 instead (`scp` the two scripts, run the suite under
# a scratch TMPDIR): 20 passed there, and the flock mutation above turned 19 red and left 20 green.
# CI runs the suite on Linux for the same reason. A SKIP is printed rather than nothing, because
# silence would read as a pass.
#
# ⚠️ BASE stays SHORT for the same reason session_db_tests.sh says so: Windows resolves no path
# past 260 characters, and a git clone under a deep temp directory fails in ways that read as
# logic errors.
set -uo pipefail

REMOTE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/session_serving_remote.sh"
# `-$$` is what stops two concurrent runs from deleting each other's fixtures. It stays SHORT
# for the Windows 260-character reason above: a pid is a few digits, a mktemp -d name is not.
BASE="${TMPDIR:-/tmp}/recall-servtests-$$"
# Removed only on a GREEN run. A red run leaves its fixture for inspection, which is the whole
# point of having one; the isolation this trap provides is about concurrent runs, not about tidiness.
trap '[ "$fail" -eq 0 ] && rm -rf "$BASE"' EXIT
pass=0; fail=0
ok() { pass=$((pass+1)); printf 'PASS  %s\n' "$1"; }
no() { fail=$((fail+1)); printf 'FAIL  %s\n     %s\n' "$1" "${2:-}"; }

rm -rf "$BASE"; mkdir -p "$BASE/bin"

# --- the python stub --------------------------------------------------------------------------
# One file, four behaviours, each keyed on how the script invokes it. Every call is logged so a
# test can assert that a check ran, which is the only way to prove `--no-verify` skips the
# handshake rather than merely tolerating it.
cat > "$BASE/bin/python" <<'STUB'
#!/usr/bin/env bash
set -uo pipefail
log() { [ -n "${FAKE_PY_LOG:-}" ] && printf '%s\n' "$1" >> "$FAKE_PY_LOG"; return 0; }
case "${1:-}" in
    -)
        log handshake
        cat >/dev/null
        printf 'tools=%s\n' "${FAKE_TOOLS:-18}"
        exit "${FAKE_HANDSHAKE_RC:-0}"
        ;;
    -c)
        log import
        exit "${FAKE_IMPORT_RC:-0}"
        ;;
esac
case "$*" in
    *"recall.cli schema"*status*)
        log schema
        printf 'table: chunks\ncurrent: 0016\nrequired: 0016\ncompatible: %s\n' "${FAKE_SCHEMA_COMPAT:-yes}"
        exit "${FAKE_SCHEMA_RC:-0}"
        ;;
    *"recall.cli schema"*apply*)
        log apply
        exit "${FAKE_APPLY_RC:-0}"
        ;;
    *"pip install"*)
        log pip
        exit "${FAKE_PIP_RC:-0}"
        ;;
esac
printf 'stub python: unhandled call: %s\n' "$*" >&2
exit 99
STUB
chmod +x "$BASE/bin/python"
printf '# fake env for the tests\nRECALL_DSN=postgresql://nobody@127.0.0.1:1/none\n' > "$BASE/fake.env"

ORIGIN=""
SERVING=""
WORK=""
SCENARIO=0

_git() { git -C "$1" -c user.email=t@t -c user.name=t -c commit.gpgsign=false "${@:2}"; }

# A fresh origin/serving pair for every test, in a NUMBERED directory rather than a reused one.
# State from one scenario leaking into the next is the classic way a rollback test passes because
# nothing had moved in the first place, and `rm -rf` on a git repository is unreliable on Windows:
# the first draft of this file deleted the tree between tests and git answered "destination path
# already exists and is not an empty directory" on the very next clone.
fresh() {
    SCENARIO=$((SCENARIO + 1))
    ORIGIN="$BASE/s$SCENARIO/origin"
    SERVING="$BASE/s$SCENARIO/serving"
    WORK="$BASE/s$SCENARIO/work"
    mkdir -p "$BASE/s$SCENARIO"
    git init -q --bare -b master "$ORIGIN"
    git init -q -b master "$WORK"
    mkdir -p "$WORK/recall/migrations/sql" "$WORK/recall_mcp"
    printf '__version__ = "0.10.0"\n' > "$WORK/recall/__init__.py"
    printf 'name = "recall-rag"\n' > "$WORK/pyproject.toml"
    printf '\n' > "$WORK/recall/migrations/sql/0016_semantic_graph_foundation.sql"
    _git "$WORK" add recall pyproject.toml >/dev/null
    _git "$WORK" commit -qm "base" >/dev/null
    _git "$WORK" remote add origin "$ORIGIN"
    _git "$WORK" push -q origin master
    git clone -q "$ORIGIN" "$SERVING"
    _git "$SERVING" config user.email t@t
    _git "$SERVING" config user.name t
}

# A commit on origin that serving does not have. `--migration` adds a migration file, which is the
# one class of update that must not ride along silently.
land() {
    local kind="${1:-plain}"
    case "$kind" in
        migration) printf '\n' > "$WORK/recall/migrations/sql/0017_new_thing.sql"
                   _git "$WORK" add recall/migrations/sql/0017_new_thing.sql >/dev/null ;;
        deps)      printf 'name = "recall-rag"\ndependencies = ["new-thing"]\n' > "$WORK/pyproject.toml"
                   _git "$WORK" add pyproject.toml >/dev/null ;;
        *)         printf 'x = 1\n' > "$WORK/recall/new_module.py"
                   _git "$WORK" add recall/new_module.py >/dev/null ;;
    esac
    _git "$WORK" commit -qm "landed $kind" >/dev/null
    _git "$WORK" push -q origin master
}

head_of() { git -C "$1" rev-parse HEAD 2>/dev/null; }
origin_head() { git -C "$ORIGIN" rev-parse master 2>/dev/null; }

RC=0
OUT=""
# `run_env "FAKE_X=1 FAKE_Y=2" sync --dry-run`. The extra assignments go to `env` rather than as a
# prefix on the function call: bash keeps a prefix assignment on a FUNCTION in scope afterwards
# under some settings, and a FAKE_HANDSHAKE_RC leaking into the next test would turn an unrelated
# run red, or worse, a red one green.
run_env() {
    local envs="$1"; shift
    # shellcheck disable=SC2086 -- $envs is a deliberate word-split list of KEY=VALUE for env.
    OUT="$(env PATH="$BASE/bin:$PATH" \
        RECALL_SERVING_PATH="$SERVING" \
        RECALL_SERVING_VENV="$BASE/bin/python" \
        RECALL_SERVING_ENV="$BASE/fake.env" \
        RECALL_SERVING_LOCKS="$BASE/locks" \
        RECALL_SERVING_NO_FLOCK=1 \
        FAKE_PY_LOG="$BASE/py.log" \
        $envs \
        bash "$REMOTE" "$@" 2>&1)"
    RC=$?
}
run() { run_env "" "$@"; }
pylog() { cat "$BASE/py.log" 2>/dev/null; }
reset_log() { : > "$BASE/py.log"; }

# --- 1. status on a current checkout ------------------------------------------------------------
fresh; reset_log
run status
if [ "$RC" -eq 0 ] && printf '%s' "$OUT" | grep -q 'STATUS      current'; then
    ok "1  status reports a current checkout"
else
    no "1  status reports a current checkout" "rc=$RC $OUT"
fi

# --- 2. status reports drift and moves nothing --------------------------------------------------
fresh; land plain; before="$(head_of "$SERVING")"; reset_log
run status
if [ "$RC" -eq 0 ] && printf '%s' "$OUT" | grep -q 'STATUS      behind by 1' \
   && [ "$(head_of "$SERVING")" = "$before" ]; then
    ok "2  status reports drift and changes nothing"
else
    no "2  status reports drift and changes nothing" "rc=$RC $OUT"
fi

# --- 3. sync fast-forwards and verifies ---------------------------------------------------------
fresh; land plain; reset_log
run sync
if [ "$RC" -eq 0 ] && [ "$(head_of "$SERVING")" = "$(origin_head)" ] \
   && printf '%s' "$OUT" | grep -q 'RESULT      serving updated' \
   && pylog | grep -q handshake; then
    ok "3  sync fast-forwards, verifies, and drives the handshake"
else
    no "3  sync fast-forwards, verifies, and drives the handshake" "rc=$RC $OUT $(pylog | tr '\n' ',')"
fi

# --- 4. dry-run moves nothing -------------------------------------------------------------------
fresh; land plain; before="$(head_of "$SERVING")"; reset_log
run sync --dry-run
if [ "$RC" -eq 0 ] && [ "$(head_of "$SERVING")" = "$before" ] \
   && printf '%s' "$OUT" | grep -q 'DRY-RUN     would fast-forward'; then
    ok "4  --dry-run reports the move and makes none"
else
    no "4  --dry-run reports the move and makes none" "rc=$RC $OUT"
fi

# --- 5. a dirty serving tree is refused ---------------------------------------------------------
fresh; land plain; printf 'hot patch\n' >> "$SERVING/recall/__init__.py"
before="$(head_of "$SERVING")"; reset_log
run sync
if [ "$RC" -eq 3 ] && [ "$(head_of "$SERVING")" = "$before" ] \
   && printf '%s' "$OUT" | grep -q 'uncommitted changes'; then
    ok "5  a dirty serving checkout is refused, not reset"
else
    no "5  a dirty serving checkout is refused, not reset" "rc=$RC $OUT"
fi

# --- 6. a migration in the update is refused without the flag -----------------------------------
fresh; land migration; before="$(head_of "$SERVING")"; reset_log
run sync
if [ "$RC" -eq 3 ] && [ "$(head_of "$SERVING")" = "$before" ] \
   && printf '%s' "$OUT" | grep -q 'changes migrations'; then
    ok "6  an update carrying a migration is refused without --with-migrations"
else
    no "6  an update carrying a migration is refused without --with-migrations" "rc=$RC $OUT"
fi

# --- 7. ... and applied with it (the control for 6) ---------------------------------------------
fresh; land migration; reset_log
run sync --with-migrations
if [ "$RC" -eq 0 ] && [ "$(head_of "$SERVING")" = "$(origin_head)" ] \
   && pylog | grep -q apply && printf '%s' "$OUT" | grep -q 'SCHEMA      migrations applied'; then
    ok "7  --with-migrations moves and applies"
else
    no "7  --with-migrations moves and applies" "rc=$RC $OUT $(pylog | tr '\n' ',')"
fi

# --- 8. a diverged serving checkout is refused --------------------------------------------------
fresh; land plain
printf 'local fix\n' > "$SERVING/local_fix.py"
_git "$SERVING" add local_fix.py >/dev/null
_git "$SERVING" commit -qm "local commit on the host" >/dev/null
before="$(head_of "$SERVING")"; reset_log
run sync
if [ "$RC" -eq 3 ] && [ "$(head_of "$SERVING")" = "$before" ] \
   && printf '%s' "$OUT" | grep -q 'not an ancestor'; then
    ok "8  a diverged serving checkout is refused"
else
    no "8  a diverged serving checkout is refused" "rc=$RC $OUT"
fi

# --- 9. a failed handshake rolls the deployment back --------------------------------------------
fresh; land plain; before="$(head_of "$SERVING")"; reset_log
run_env "FAKE_HANDSHAKE_RC=1" sync
if [ "$RC" -eq 4 ] && [ "$(head_of "$SERVING")" = "$before" ] \
   && printf '%s' "$OUT" | grep -q 'ROLLED-BACK'; then
    ok "9  a server that will not answer rolls the checkout back"
else
    no "9  a server that will not answer rolls the checkout back" "rc=$RC $OUT"
fi

# --- 10. --no-verify skips the handshake and only the handshake ---------------------------------
fresh; land plain; reset_log
run sync --no-verify
if [ "$RC" -eq 0 ] && [ "$(head_of "$SERVING")" = "$(origin_head)" ] \
   && ! pylog | grep -q handshake && pylog | grep -q schema; then
    ok "10 --no-verify skips the handshake and keeps the cheap checks"
else
    no "10 --no-verify skips the handshake and keeps the cheap checks" "rc=$RC $(pylog | tr '\n' ',')"
fi

# --- 11. schema status exiting 0 while INCOMPATIBLE is still a failure ---------------------------
# The quiet half of the check. `recall schema status` prints a ledger and exits 0 in states a
# server still refuses to start against, so the exit code alone would call this a green sync.
fresh; land plain; before="$(head_of "$SERVING")"; reset_log
run_env "FAKE_SCHEMA_COMPAT=no" sync
if [ "$RC" -eq 4 ] && [ "$(head_of "$SERVING")" = "$before" ]; then
    ok "11 'compatible: no' with exit 0 fails verification and rolls back"
else
    no "11 'compatible: no' with exit 0 fails verification and rolls back" "rc=$RC $OUT"
fi

# --- 12. a detached serving HEAD is refused -----------------------------------------------------
fresh; land plain
_git "$SERVING" checkout -q --detach HEAD
before="$(head_of "$SERVING")"; reset_log
run sync
if [ "$RC" -eq 3 ] && [ "$(head_of "$SERVING")" = "$before" ] \
   && printf '%s' "$OUT" | grep -q 'detached'; then
    ok "12 a detached serving HEAD is refused"
else
    no "12 a detached serving HEAD is refused" "rc=$RC $OUT"
fi

# --- 13. a broken import rolls back before the expensive check ----------------------------------
fresh; land plain; before="$(head_of "$SERVING")"; reset_log
run_env "FAKE_IMPORT_RC=1" sync
if [ "$RC" -eq 4 ] && [ "$(head_of "$SERVING")" = "$before" ] \
   && ! pylog | grep -q handshake; then
    ok "13 an unimportable checkout rolls back without paying for a handshake"
else
    no "13 an unimportable checkout rolls back without paying for a handshake" "rc=$RC $(pylog | tr '\n' ',')"
fi

# --- 14. a failed schema apply rolls the code back ----------------------------------------------
fresh; land migration; before="$(head_of "$SERVING")"; reset_log
run_env "FAKE_APPLY_RC=1" sync --with-migrations
if [ "$RC" -eq 4 ] && [ "$(head_of "$SERVING")" = "$before" ]; then
    ok "14 a failed schema apply rolls the code back to the database's level"
else
    no "14 a failed schema apply rolls the code back to the database's level" "rc=$RC $OUT"
fi

# --- 15. a serving path that is not a checkout is an environment error, not a sync ---------------
fresh; rm -rf "$SERVING"; mkdir -p "$SERVING"; reset_log
run sync
if [ "$RC" -eq 2 ] && printf '%s' "$OUT" | grep -q 'not a git checkout'; then
    ok "15 a serving path that is not a checkout exits 2"
else
    no "15 a serving path that is not a checkout exits 2" "rc=$RC $OUT"
fi

# --- 16. verification failing with nothing moved is reported, not blamed on this run -------------
fresh; reset_log
run_env "FAKE_HANDSHAKE_RC=1" sync
if [ "$RC" -eq 4 ] && printf '%s' "$OUT" | grep -q 'pre-existing' \
   && ! printf '%s' "$OUT" | grep -q 'ROLLED-BACK'; then
    ok "16 a pre-existing failure is named as one and nothing is rolled back"
else
    no "16 a pre-existing failure is named as one and nothing is rolled back" "rc=$RC $OUT"
fi

# --- 17. deps moving without --with-deps is reported and does not block --------------------------
fresh; land deps; reset_log
run sync
if [ "$RC" -eq 0 ] && [ "$(head_of "$SERVING")" = "$(origin_head)" ] \
   && printf '%s' "$OUT" | grep -q 'DEPS        pyproject.toml moved' && ! pylog | grep -q pip; then
    ok "17 a dependency change is reported and left to the import check"
else
    no "17 a dependency change is reported and left to the import check" "rc=$RC $OUT"
fi

# --- 18. ... and reinstalled with the flag (the control for 17) ----------------------------------
fresh; land deps; reset_log
run sync --with-deps
if [ "$RC" -eq 0 ] && pylog | grep -q pip && printf '%s' "$OUT" | grep -q 'DEPS        reinstalled'; then
    ok "18 --with-deps reinstalls the editable package"
else
    no "18 --with-deps reinstalls the editable package" "rc=$RC $OUT $(pylog | tr '\n' ',')"
fi

# --- 19. an embedding run holding embed.lock stops the sync -------------------------------------
# Skipped where flock does not exist, which is every Git Bash runner. Reported as SKIP rather than
# passed: a guard nobody has watched refuse has not been tested, and silence would read as a pass.
if command -v flock >/dev/null 2>&1; then
    fresh; land plain; before="$(head_of "$SERVING")"; reset_log
    mkdir -p "$BASE/locks"; : > "$BASE/locks/embed.lock"
    flock "$BASE/locks/embed.lock" sleep 12 &
    holder=$!
    sleep 1
    run_env "RECALL_SERVING_NO_FLOCK=0" sync
    if [ "$RC" -eq 3 ] && [ "$(head_of "$SERVING")" = "$before" ] \
       && printf '%s' "$OUT" | grep -q 'embed.lock'; then
        ok "19 a live indexer's embed.lock stops the sync"
    else
        no "19 a live indexer's embed.lock stops the sync" "rc=$RC $OUT"
    fi
    kill "$holder" 2>/dev/null
    wait "$holder" 2>/dev/null

    # --- 20. ... and does NOT stop a status read ------------------------------------------------
    # The control for 19, and the reason `status` takes no locks at all: session-close.sh asks for
    # this report on every close, including a close that happens while an indexer is running, and
    # a report that refuses under load is a report nobody can rely on being there.
    fresh; land plain; reset_log
    mkdir -p "$BASE/locks"; : > "$BASE/locks/embed.lock"
    flock "$BASE/locks/embed.lock" sleep 12 &
    holder=$!
    sleep 1
    run_env "RECALL_SERVING_NO_FLOCK=0" status
    if [ "$RC" -eq 0 ] && printf '%s' "$OUT" | grep -q 'STATUS      behind by 1'; then
        ok "20 a held embed.lock does not stop a status read"
    else
        no "20 a held embed.lock does not stop a status read" "rc=$RC $OUT"
    fi
    kill "$holder" 2>/dev/null
    wait "$holder" 2>/dev/null
else
    printf 'SKIP  19 embed.lock exclusion (no flock on this host; covered on Linux CI)\n'
    printf 'SKIP  20 status is not blocked by embed.lock (same reason)\n'
fi

# --- the wrapper's own reporting ----------------------------------------------------------------
# `session-serving.sh` decides nothing about the deployment, but it does tell a session whether its
# work is on master, and both tests below pin a line that was WRONG in a way that reads as fine.
# ssh is stubbed, so nothing here reaches a host.
WRAPPER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/session-serving.sh"

# The argv is LOGGED, and that is not incidental. Before it was, the stub discarded "$@", so no
# wrapper test could tell a wrapper that forwards `verify` from one that recognises the word and
# then sends nothing: the remote half defaults to `status`, the run exits 0, and every assertion
# still passes. That is a guard that cannot fail, which is the failure mode this whole file is
# about. Test 34 is the assertion that closes it.
cat > "$BASE/bin/ssh" <<'STUB'
#!/usr/bin/env bash
[ -n "${FAKE_SSH_LOG:-}" ] && printf '%s\n' "$*" >> "$FAKE_SSH_LOG"
cat >/dev/null           # swallow the piped remote script, as the real ssh does
printf 'SERVING     /home/x/serving\nSTATUS      current\n'
exit "${FAKE_SSH_RC:-0}"
STUB
chmod +x "$BASE/bin/ssh"

wrapper() {
    local repo="$1"; shift
    : > "$BASE/ssh.log"
    OUT="$(cd "$repo" && env PATH="$BASE/bin:$PATH" FAKE_SSH_RC="${FAKE_SSH_RC:-0}" \
        FAKE_SSH_LOG="$BASE/ssh.log" \
        RECALL_VPS2_HOST=stub-host bash "$WRAPPER" "$@" 2>&1)"
    RC=$?
}

# A branch that was SQUASHED onto master: ahead by sha, identical in content. `master` refuses
# merge commits in this repository, so this is the state of every branch after its PR lands, and
# the wrapper used to tell all of them that a sync would not ship their work.
fresh
printf 'squashed change\n' > "$WORK/recall/squashed.py"
_git "$WORK" checkout -q -b feature
_git "$WORK" add recall/squashed.py >/dev/null
_git "$WORK" commit -qm "the branch's own commit" >/dev/null
_git "$WORK" checkout -q master
printf 'squashed change\n' > "$WORK/recall/squashed.py"
_git "$WORK" add recall/squashed.py >/dev/null
_git "$WORK" commit -qm "the same change, squashed onto master" >/dev/null
_git "$WORK" push -q origin master
_git "$WORK" checkout -q feature

wrapper "$WORK" status
if [ "$RC" -eq 0 ] && printf '%s' "$OUT" | grep -q 'identical in content' \
   && ! printf '%s' "$OUT" | grep -q 'will NOT ship them'; then
    ok "21 a squash-merged branch is not reported as unshipped work"
else
    no "21 a squash-merged branch is not reported as unshipped work" "rc=$RC $OUT"
fi

# The control. Real unmerged work must still be called out, or the fix above would have replaced a
# false alarm with silence, which is worse: this line is what tells a session the sync ships
# somebody else's work rather than its own.
printf 'genuinely new\n' > "$WORK/recall/unmerged.py"
_git "$WORK" add recall/unmerged.py >/dev/null
_git "$WORK" commit -qm "work that never landed" >/dev/null
wrapper "$WORK" status
if [ "$RC" -eq 0 ] && printf '%s' "$OUT" | grep -q 'will NOT ship them'; then
    ok "22 genuinely unmerged work is still reported as unshipped"
else
    no "22 genuinely unmerged work is still reported as unshipped" "rc=$RC $OUT"
fi

# 127 arrived once, in real use, with no output and no explanation, and an unexplained exit code on
# a tool that moves a deployment is indistinguishable from one that moved something quietly.
FAKE_SSH_RC=127 wrapper "$WORK" sync
if [ "$RC" -eq 127 ] && printf '%s' "$OUT" | grep -q 'NOTHING moved'; then
    ok "23 exit 127 says the remote never started and nothing moved"
else
    no "23 exit 127 says the remote never started and nothing moved" "rc=$RC $OUT"
fi
unset FAKE_SSH_RC

# --- `verify`: the handshake on its own ---------------------------------------------------------
#
# Numbered after the wrapper tests rather than beside the other remote-half tests on purpose. The
# mutation results in this file's header are recorded against test NUMBERS, and renumbering would
# quietly invalidate a measurement to tidy an ordering. Of these, 24 to 28 exercise the remote
# half and 29 the wrapper.
#
# What `verify` has to be, for it to close the gap it was added for: it must run the handshake, it
# must move nothing, and it must stay answerable in the states `sync` correctly refuses. Every one
# of those is a way this could be written to look right and be useless.

# --- 24. verify drives the handshake and moves nothing, even when behind ------------------------
# The RESULT wording this pins was changed deliberately by the fix for the audit finding that
# `verify` drives ONE server while reporting an unqualified green; test 32 owns the new wording and
# its NOT PROVED lines. This test keeps the invariant it was written for: the handshake ran, and a
# checkout that was BEHIND did not move.
fresh; land plain; before="$(head_of "$SERVING")"; reset_log
run verify
if [ "$RC" -eq 0 ] && [ "$(head_of "$SERVING")" = "$before" ] \
   && pylog | grep -q handshake && printf '%s' "$OUT" | grep -q '^RESULT      this checkout starts'; then
    ok "24 verify drives the handshake and leaves a behind checkout where it was"
else
    no "24 verify drives the handshake and leaves a behind checkout where it was" "rc=$RC $OUT $(pylog | tr '\n' ',')"
fi

# --- 25. verify never fetches (the control pair that proves it) ---------------------------------
# A verify that fetches is a verify the network can fail, and the question it answers (does the
# process on this host still start?) has nothing to do with a remote. Proved by breaking the
# remote and watching `status` exit 2 on the same checkout that `verify` reports on.
fresh; _git "$SERVING" remote set-url origin "$BASE/no-such-origin"; reset_log
run status
status_rc="$RC"
reset_log
run verify
if [ "$status_rc" -eq 2 ] && [ "$RC" -eq 0 ] && pylog | grep -q handshake; then
    ok "25 verify answers with the remote unreachable, where status cannot"
else
    no "25 verify answers with the remote unreachable, where status cannot" "status=$status_rc verify=$RC $OUT"
fi

# --- 26. a dirty tree is verified and named, not refused (the counterpart of test 5) ------------
# `sync` refuses a hot-patched tree because it would reset it. `verify` writes nothing, and a
# hot-patched server is the one people most need to ask this question about, but the report must
# say so, or a green line would describe a deployment nobody can reproduce from master.
fresh; printf 'hot patch\n' >> "$SERVING/recall/__init__.py"; reset_log
run verify
if [ "$RC" -eq 0 ] && pylog | grep -q handshake \
   && printf '%s' "$OUT" | grep -q 'DIRTY       verifying a hot-patched tree'; then
    ok "26 verify reports a dirty tree and still verifies it"
else
    no "26 verify reports a dirty tree and still verifies it" "rc=$RC $OUT"
fi

# --- 27. a server that will not answer is exit 4, with nothing rolled back ----------------------
fresh; before="$(head_of "$SERVING")"; reset_log
run_env "FAKE_HANDSHAKE_RC=1" verify
if [ "$RC" -eq 4 ] && [ "$(head_of "$SERVING")" = "$before" ] \
   && printf '%s' "$OUT" | grep -q 'verification FAILED' \
   && ! printf '%s' "$OUT" | grep -q 'ROLLED-BACK'; then
    ok "27 a failing verify exits 4 and rolls nothing back"
else
    no "27 a failing verify exits 4 and rolls nothing back" "rc=$RC $OUT"
fi

# --- 28. verify ignores --no-verify -------------------------------------------------------------
# The flag exists so a sync can skip the 21s handshake and keep the cheap checks. Honouring it here
# would produce `VERIFY ok: ... handshake skipped` from a command whose entire purpose is the
# handshake: a green line for a check that did not run, which is the failure these scripts exist
# to remove rather than reproduce.
fresh; reset_log
run verify --no-verify
if [ "$RC" -eq 0 ] && pylog | grep -q handshake \
   && ! printf '%s' "$OUT" | grep -q 'handshake skipped'; then
    ok "28 verify runs the handshake even when told --no-verify"
else
    no "28 verify runs the handshake even when told --no-verify" "rc=$RC $OUT $(pylog | tr '\n' ',')"
fi

# --- 29. the wrapper accepts verify and does not ask about unmerged work ------------------------
# The LOCAL line answers "is my work on master", which is a question about a move. On a verify it
# is noise, and noise above a one-line answer is how a report stops being read.
wrapper "$WORK" verify
if [ "$RC" -eq 0 ] && ! printf '%s' "$OUT" | grep -q 'LOCAL'; then
    ok "29 the wrapper passes verify through without the LOCAL commit report"
else
    no "29 the wrapper passes verify through without the LOCAL commit report" "rc=$RC $OUT"
fi

# --- audit fixes: one mode word, and a report that names its own boundary -----------------------
#
# 30 and 31 are the red->green pair for the P1. Before the fix, measured on this harness:
#   argv [verify sync] -> rc=0 and the serving HEAD MOVED, behind a read-only verb;
#   argv [sync verify] -> rc=0, nothing shipped, printing the verify's green RESULT.
# Both are the same root cause: a mode word was recognised in any position and the last one won.

# --- 30. `verify sync` is refused, and moves nothing --------------------------------------------
fresh; land plain; before="$(head_of "$SERVING")"; reset_log
run verify sync
if [ "$RC" -eq 2 ] && [ "$(head_of "$SERVING")" = "$before" ] \
   && printf '%s' "$OUT" | grep -q 'say ONE of status, verify or sync' \
   && ! pylog | grep -q handshake; then
    ok "30 a second mode word is refused, and the checkout does not move"
else
    no "30 a second mode word is refused, and the checkout does not move" "rc=$RC $OUT"
fi

# --- 31. ... and so is the reverse order, which used to be a silent no-op ----------------------
# The mirror matters on its own: `sync verify` exiting 0 having shipped nothing is a session that
# asked to update the live server, was told everything was fine, and left it stale.
fresh; land plain; before="$(head_of "$SERVING")"; reset_log
run sync verify
if [ "$RC" -eq 2 ] && [ "$(head_of "$SERVING")" = "$before" ]; then
    ok "31 'sync verify' is refused rather than silently verifying"
else
    no "31 'sync verify' is refused rather than silently verifying" "rc=$RC $OUT"
fi

# --- 32. verify names the tenant it drove and what it did not prove ----------------------------
fresh; land plain; reset_log
run verify
if [ "$RC" -eq 0 ] && printf '%s' "$OUT" | grep -q "driving tenant 'memory'" \
   && printf '%s' "$OUT" | grep -q 'NOT PROVED  any other tenant' \
   && printf '%s' "$OUT" | grep -q 'NOT PROVED  that this checkout is CURRENT'; then
    ok "32 verify names the tenant it drove and the two things it did not prove"
else
    no "32 verify names the tenant it drove and the two things it did not prove" "rc=$RC $OUT"
fi

# --- 33. a missing env file is named, not swallowed --------------------------------------------
# Before the fix this failed with the complete diagnosis `verification FAILED: schema: `, an
# empty string after the colon, on the command a session runs precisely when it suspects the
# server is broken.
fresh; reset_log
run_env "RECALL_SERVING_ENV=$BASE/no-such.env" verify
if [ "$RC" -eq 4 ] && printf '%s' "$OUT" | grep -q 'env file .* is missing or unreadable'; then
    ok "33 a missing env file is named instead of producing an empty diagnosis"
else
    no "33 a missing env file is named instead of producing an empty diagnosis" "rc=$RC $OUT"
fi

# --- 34. the wrapper actually FORWARDS the mode, not merely recognises it -----------------------
# The gap this closes: every wrapper assertion above passes against a wrapper that matches `verify`
# and then sends no mode at all, because the remote half defaults to status and exits 0. Nothing
# read the argv until the stub started logging it.
fresh
wrapper "$WORK" verify
if [ "$RC" -eq 0 ] && grep -q ' verify' "$BASE/ssh.log"; then
    ok "34 the wrapper forwards the mode word to ssh, not just recognises it"
else
    no "34 the wrapper forwards the mode word to ssh, not just recognises it" "rc=$RC log=$(cat "$BASE/ssh.log" 2>/dev/null)"
fi

# --- 35. the wrapper refuses two mode words before it ever reaches ssh -------------------------
fresh
wrapper "$WORK" verify sync
if [ "$RC" -eq 2 ] && ! grep -q . "$BASE/ssh.log" 2>/dev/null; then
    ok "35 the wrapper refuses two mode words without contacting the host"
else
    no "35 the wrapper refuses two mode words without contacting the host" "rc=$RC $OUT"
fi

# --- 36. `--to`'s value is not read as a mode --------------------------------------------------
# `sync --to verify` used to leave the WRAPPER thinking MODE=verify while the remote ran a real
# sync, so a rollback would have been reported to the operator as "nothing was moved".
fresh
wrapper "$WORK" sync --to verify
if [ "$RC" -eq 0 ] && grep -q -- '--to verify' "$BASE/ssh.log"; then
    ok "36 the value after --to is forwarded, not mistaken for a mode"
else
    no "36 the value after --to is forwarded, not mistaken for a mode" "rc=$RC log=$(cat "$BASE/ssh.log" 2>/dev/null)"
fi

# --- 37. --help renders the usage block and stops there ----------------------------------------
# The range used to be a hand-counted `sed -n '2,20p'`, which had ALREADY drifted: on HEAD it
# printed three lines past the usage block and cut a sentence mid-word. Retuning the number would
# have fixed today and left the next header edit to break it silently, so the block is delimited
# now and this test is what notices if the markers go.
OUT="$(bash "$WRAPPER" --help 2>&1)"; RC=$?
if [ "$RC" -eq 0 ] && printf '%s' "$OUT" | grep -q 'session-serving.sh verify' \
   && ! printf '%s' "$OUT" | grep -q 'This file is the wrapper'; then
    ok "37 --help shows every mode and stops at the end of the usage block"
else
    no "37 --help shows every mode and stops at the end of the usage block" "rc=$RC $OUT"
fi

# --- 38. a `~/` override is expanded by the REMOTE shell, not quoted into a literal ------------
# `printf '%q'` escapes a tilde, and `~/recall-repos/.env` is the documented form for these paths.
# Quoting it would hand the remote half a path that cannot exist, and the readability guard added
# for the missing-env-file finding would then report it as missing with total confidence. That is a
# fix manufacturing the exact failure the other fix was written to make honest.
fresh
RECALL_VPS2_ENV='~/recall-repos/.env' wrapper "$WORK" verify
if [ "$RC" -eq 0 ] && grep -q 'RECALL_SERVING_ENV=\$HOME/recall-repos/.env' "$BASE/ssh.log" \
   && ! grep -q 'RECALL_SERVING_ENV=.\?~' "$BASE/ssh.log"; then
    ok "38 a ~/ override is passed for the remote shell to expand, not quoted literally"
else
    no "38 a ~/ override is passed for the remote shell to expand, not quoted literally" \
       "rc=$RC log=$(cat "$BASE/ssh.log" 2>/dev/null)"
fi
unset RECALL_VPS2_ENV

printf '\n%s passed, %s failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ] || exit 1
