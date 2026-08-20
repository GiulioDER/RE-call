#!/usr/bin/env bash
# Regression tests for `scripts/session-db.sh orphans`.
#
# The defect these pin, measured 2026-08-20: the command printed "no orphaned containers" on a
# machine holding 21 exited ones, five of which belonged to a worktree that had been removed and
# whose empty directory was left behind. `[ -d "$checkout" ]` cannot tell that from a live
# checkout, so every remnant passed.
#
# Docker is stubbed, deliberately. These tests are about the CLASSIFICATION, and a suite that
# needs a live daemon would not run in CI, which is where the guard has to keep working. The stub
# answers the two calls `cmd_orphans` makes and nothing else; if the command ever grows a third,
# the stub fails loudly rather than returning an empty string that reads as "no containers".
#
# Every ORPHAN assertion has a control that must stay silent, because a classifier that reports
# everything would otherwise pass this whole file. Mutation-tested 2026-08-20, three ways:
#
#   `[ ! -e "$checkout/.git" ]` -> `false`   4, 6, 7 red   (remnants stop being reported)
#   `[ ! -e "$checkout/.git" ]` -> `true`    3, 7, 8, 9 red (every live checkout reported)
#   `_is_ours` -> `return 0`                 5 red          (other projects' stacks reported)
#
# ⚠️ BASE must stay SHORT. Windows resolves no path past 260 characters, and `[ -d ]` cannot tell
# "not there" from "too long to look at", so running this suite from a deep temp directory made
# every live worktree report `checkout gone` and three tests failed with the guard untouched. That
# is where test 10 came from: the guard now says CHECK rather than ORPHAN near the limit.
set -uo pipefail

DB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/session-db.sh"
BASE="${TMPDIR:-/tmp}/recall-dbtests"
pass=0; fail=0
ok() { pass=$((pass+1)); printf 'PASS  %s\n' "$1"; }
no() { fail=$((fail+1)); printf 'FAIL  %s\n     %s\n' "$1" "${2:-}"; }

rm -rf "$BASE"; mkdir -p "$BASE/state"
export FAKE_DOCKER_STATE="$BASE/state"

# --- the stub ---------------------------------------------------------------
# One file per container id: line 1 the `recall.checkout` label, line 2 the compose working_dir,
# line 3 the name. An empty line is a label that is not set, which docker renders as the literal
# `<no value>`; the stub emits that spelling on purpose, because handling it is part of what is
# under test.
mkdir -p "$BASE/bin"
cat > "$BASE/bin/docker" <<'STUB'
#!/usr/bin/env bash
set -uo pipefail
_line() { sed -n "${2}p" "$FAKE_DOCKER_STATE/$1"; }
_val()  { local v; v="$(_line "$1" "$2")"; [ -z "$v" ] && printf '<no value>' || printf '%s' "$v"; }
case "${1:-}" in
    ps)
        for f in "$FAKE_DOCKER_STATE"/*; do
            [ -e "$f" ] || continue
            id="$(basename "$f")"
            case "$*" in
                *label=recall.session*)
                    [ -n "$(_line "$id" 1)" ] && echo "$id" ;;
                *label=com.docker.compose.project.working_dir*)
                    [ -n "$(_line "$id" 2)" ] && echo "$id" ;;
            esac
        done
        ;;
    inspect)
        id="$2"
        [ -e "$FAKE_DOCKER_STATE/$id" ] || exit 1
        printf '%s|%s|/%s\n' "$(_val "$id" 1)" "$(_val "$id" 2)" "$(_line "$id" 3)"
        ;;
    *)
        echo "stub: unexpected docker call: $*" >&2
        exit 97
        ;;
esac
STUB
chmod +x "$BASE/bin/docker"
export PATH="$BASE/bin:$PATH"

# `container <id> <recall.checkout> <compose_dir> <name>`
container() {
    printf '%s\n%s\n%s\n' "$2" "$3" "$4" > "$FAKE_DOCKER_STATE/$1"
}

# --- the fixture repository -------------------------------------------------
mkdir -p "$BASE/repo"
cd "$BASE/repo"
git init -q .
git config user.email t@t
git config user.name t
echo x > a.txt
git add a.txt
git commit -qm init
MAIN="$(git rev-parse --show-toplevel)"
mkdir -p "$MAIN/.claude/worktrees"
git worktree add -q -b live "$MAIN/.claude/worktrees/live" HEAD
LIVE="$MAIN/.claude/worktrees/live"

# A worktree removed with its directory left behind, which is the case that was invisible.
REMNANT="$MAIN/.claude/worktrees/remnant"
git worktree add -q -b gone "$REMNANT" HEAD
git worktree remove --force "$REMNANT"
mkdir -p "$REMNANT"

# A directory that is not a checkout and has nothing to do with this repository.
OTHER="$BASE/not-our-project"
mkdir -p "$OTHER"

run() { cd "$MAIN" && bash "$DB" orphans 2>&1; }
saw() { printf '%s' "$1" | grep -q "ORPHAN.*$2"; }

# --- 1. a session container whose checkout is gone --------------------------
rm -f "$FAKE_DOCKER_STATE"/*
container c1 "$BASE/deleted-checkout" "" sess-gone
out="$(run)"
if saw "$out" "sess-gone"; then ok "session container with a deleted checkout is reported"
else no "session container with a deleted checkout is reported" "$out"; fi

# --- 2. a compose container whose worktree is gone --------------------------
rm -f "$FAKE_DOCKER_STATE"/*
container c2 "" "$BASE/deleted-checkout" stack-gone-db-1
out="$(run)"
if saw "$out" "stack-gone-db-1"; then ok "compose container with a deleted worktree is reported"
else no "compose container with a deleted worktree is reported" "$out"; fi

# --- 3. CONTROL: a compose container on a LIVE worktree ---------------------
rm -f "$FAKE_DOCKER_STATE"/*
container c3 "" "$LIVE" live-db-1
out="$(run)"
if saw "$out" "live-db-1"; then no "a live worktree is left alone" "reported it: $out"
else ok "a live worktree is left alone"; fi

# --- 4. THE DEFECT: directory survives, worktree does not -------------------
rm -f "$FAKE_DOCKER_STATE"/*
container c4 "" "$REMNANT" remnant-db-1
out="$(run)"
if saw "$out" "remnant-db-1"; then ok "a removed worktree's leftover directory is reported"
else no "a removed worktree's leftover directory is reported" "$out"; fi

# --- 5. CONTROL: someone else's compose project -----------------------------
# The remnant test must not fire outside this repository. A false ORPHAN here reads as an
# instruction to remove a stack that another project is using.
rm -f "$FAKE_DOCKER_STATE"/*
container c5 "" "$OTHER" someone-elses-db-1
out="$(run)"
if saw "$out" "someone-elses-db-1"; then no "an unrelated compose project is left alone" "reported it: $out"
else ok "an unrelated compose project is left alone"; fi

# --- 6. a session container's remnant is reported wherever it lives ---------
# `recall.checkout` is written by this script and by nothing else, so the scoping that protects
# other projects does not apply to it.
rm -f "$FAKE_DOCKER_STATE"/*
container c6 "$OTHER" "" sess-remnant
out="$(run)"
if saw "$out" "sess-remnant"; then ok "a session container's remnant is reported outside the repo"
else no "a session container's remnant is reported outside the repo" "$out"; fi

# --- 7. a Windows path spelled with backslashes -----------------------------
# Compose records `C:\Users\...` while this script's own label records `C:/Users/...`. The two
# describe the same directory and must classify the same way.
rm -f "$FAKE_DOCKER_STATE"/*
if command -v cygpath >/dev/null 2>&1; then
    WIN_REMNANT="$(cygpath -w "$REMNANT")"
    WIN_LIVE="$(cygpath -w "$LIVE")"
    container c7 "" "$WIN_REMNANT" win-remnant-db-1
    container c8 "" "$WIN_LIVE" win-live-db-1
    out="$(run)"
    if saw "$out" "win-remnant-db-1" && ! saw "$out" "win-live-db-1"; then
        ok "a backslash-spelled path classifies like its forward-slash twin"
    else
        no "a backslash-spelled path classifies like its forward-slash twin" "$out"
    fi
else
    ok "backslash paths (skipped: no cygpath, not a Windows shell)"
fi

# --- 8. CONTROL: the main checkout itself -----------------------------------
# `.git` is a directory here and a file in a worktree. Testing for a file would report the main
# checkout, and `recall-db-1` from the root compose stack carries exactly this path.
rm -f "$FAKE_DOCKER_STATE"/*
container c9 "" "$MAIN" recall-db-1
out="$(run)"
if saw "$out" "recall-db-1"; then no "the main checkout is not an orphan" "reported it: $out"
else ok "the main checkout is not an orphan"; fi

# --- 9. a clean machine still says so ---------------------------------------
rm -f "$FAKE_DOCKER_STATE"/*
container c10 "" "$LIVE" live-again-db-1
out="$(run)"
if printf '%s' "$out" | grep -q "no orphaned containers"; then ok "a clean machine reports itself clean"
else no "a clean machine reports itself clean" "$out"; fi

# --- 10. a path too long for Windows is a question, not a verdict -----------
# `[ -d ]` answers false for a path past the limit exactly as it does for one that is gone. The
# guard must not turn that into an ORPHAN line, because ORPHAN reads as "safe to remove".
rm -f "$FAKE_DOCKER_STATE"/*
LONG="$BASE/$(printf 'd%.0s' $(seq 1 250))"
container c11 "" "$LONG" too-long-db-1
out="$(run)"
if printf '%s' "$out" | grep -q "CHECK.*too-long-db-1" && ! saw "$out" "too-long-db-1"; then
    ok "a path past the Windows limit is reported as CHECK, not ORPHAN"
else
    no "a path past the Windows limit is reported as CHECK, not ORPHAN" "$out"
fi

# --- 11. CONTROL: a short missing path is still a verdict -------------------
# Without this, test 10 would also pass if the guard reported everything as CHECK, which would
# make the command useless while looking careful.
rm -f "$FAKE_DOCKER_STATE"/*
container c12 "" "$BASE/short-and-gone" short-gone-db-1
out="$(run)"
if saw "$out" "short-gone-db-1"; then ok "a short missing path is still ORPHAN"
else no "a short missing path is still ORPHAN" "$out"; fi

# --- 12. a daemon that does not answer is not a clean machine ---------------
# The bug this whole file exists for was a false "no orphaned containers". An unreachable daemon
# produces an empty list, and an empty list prints exactly the same sentence.
rm -f "$FAKE_DOCKER_STATE"/*
mkdir -p "$BASE/deadbin"
printf '#!/usr/bin/env bash\nexit 1\n' > "$BASE/deadbin/docker"
chmod +x "$BASE/deadbin/docker"
out="$(cd "$MAIN" && PATH="$BASE/deadbin:$PATH" bash "$DB" orphans 2>&1)"
rc=$?
if [ "$rc" -ne 0 ] && ! printf '%s' "$out" | grep -q "no orphaned containers"; then
    ok "an unreachable daemon is refused, not reported clean"
else
    no "an unreachable daemon is refused, not reported clean" "rc=$rc out=$out"
fi

# --- 13. a WSL path seen from Git Bash --------------------------------------
# A container started inside WSL records `/home/...`, which does not exist from Git Bash. On
# Linux the same path really is missing and ORPHAN is the honest answer, so the expectation
# flips with the shell rather than being skipped: a test that skips on the platform it matters
# on tests nothing.
rm -f "$FAKE_DOCKER_STATE"/*
container c13 "" "/home/someone/project" wsl-db-1
out="$(run)"
case "$(uname -s 2>/dev/null)" in
    MINGW*|MSYS*|CYGWIN*)
        if printf '%s' "$out" | grep -q "CHECK.*wsl-db-1"; then
            ok "a POSIX path from a Windows shell is CHECK, not ORPHAN"
        else
            no "a POSIX path from a Windows shell is CHECK, not ORPHAN" "$out"
        fi
        ;;
    *)
        if saw "$out" "wsl-db-1"; then
            ok "a missing POSIX path on Linux is still ORPHAN"
        else
            no "a missing POSIX path on Linux is still ORPHAN" "$out"
        fi
        ;;
esac

printf '\n%d/%d passed\n' "$pass" "$((pass+fail))"
[ "$fail" -eq 0 ]
