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
BASE="${TMPDIR:-/tmp}/recall-servtests"
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

printf '\n%s passed, %s failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ] || exit 1
