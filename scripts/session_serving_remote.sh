#!/usr/bin/env bash
# Bring the checkout VPS2 SERVES from up to what is merged, and prove it still serves.
#
# This is the half that runs ON VPS2. `scripts/session-serving.sh` is the workstation wrapper that
# pipes it over ssh; the split exists so this logic can be tested against a fake checkout with a
# stubbed python (`scripts/session_serving_tests.sh`) rather than only against the live host.
#
# Why a session has to do this at all
# -----------------------------------
# The `recall-memory` and `recall-code` MCP servers do not run from the checkout you are editing.
# They run on VPS2, out of `~/recall-repos/serving`, which is a git clone tracking origin/master,
# and the venv there holds an EDITABLE install pointing at it. Measured 2026-08-26:
#
#   .venv/lib/python3.12/site-packages/_editable_impl_recall_rag.pth
#     -> /home/sentiment/recall-repos/serving          (a symlink to serving-master)
#   python -c "import recall; print(recall.__file__)"
#     -> /home/sentiment/recall-repos/serving-master/recall/__init__.py
#
# So a fast-forward in that clone changes the code every future server session runs, with no
# reinstall and no restart. The consequence in the other direction is the reason this exists: a
# session that merges a retrieval fix to master and stops there leaves every search answered by
# the code that was current whenever somebody last remembered to pull. Nothing reports the gap,
# because a stale server is a WORKING server.
#
# What it refuses to do, and why each refusal is not caution for its own sake
# --------------------------------------------------------------------------
#   * A dirty serving tree is never moved. Somebody hot-patching a live server is the one case
#     where `reset --hard` destroys the only copy of a fix.
#   * A diverged serving tree is never moved. If HEAD is not an ancestor of the target, local
#     commits exist there and a fast-forward is not what anybody means by "update".
#   * New or edited migrations do not ride along silently. `recall schema status` is what the
#     server calls at startup, so code carrying migration 0017 against a database at 0016 raises
#     `SchemaTooOld` and the MCP client renders that as a server with NO TOOLS, which is also the
#     symptom of a missing file, an unapproved server and an unreachable host. Applying migrations
#     to a live corpus is a one-way door, so it takes an explicit `--with-migrations`.
#   * A sync never runs while an embedding or indexing run holds `embed.lock`. Swapping modules
#     under a live indexer is the kind of breakage that surfaces hours later as a partial corpus.
#
# And what it proves afterwards. A row count or a green `git merge` says something was written; it
# does not say the server still starts. Verification drives the real thing: `recall schema status`
# for the migration level, an import of `recall_mcp.server`, then a full JSON-RPC handshake against
# the server launched exactly as `.mcp.json` launches it. Measured 2026-08-26: 18 tools, 21.0s.
# If any of that fails the checkout is reset to where it was and the failure is named.
#
#   session_serving_remote.sh status                 # read-only: where serving is, what it lacks
#   session_serving_remote.sh sync                   # fast-forward to origin/master and verify
#   session_serving_remote.sh sync --dry-run         # every check, no move
#   session_serving_remote.sh sync --with-migrations # allow `recall schema apply` after the move
#   session_serving_remote.sh sync --with-deps       # allow `pip install -e .` when deps moved
#   session_serving_remote.sh sync --no-verify       # skip the 21s handshake (keeps cheap checks)
#
# Exit codes are part of the interface, because the wrapper and the session report read them:
#   0 up to date and verified   2 usage or environment   3 refused, nothing moved
#   4 verification failed, rolled back                   5 rollback itself failed (loud)

set -uo pipefail

SERVING="${RECALL_SERVING_PATH:-$HOME/recall-repos/serving}"
PY="${RECALL_SERVING_VENV:-$HOME/recall-repos/.venv/bin/python}"
ENV_FILE="${RECALL_SERVING_ENV:-$HOME/recall-repos/.env}"
LOCK_DIR="${RECALL_SERVING_LOCKS:-$HOME/recall-repos/.locks}"
#: All three served tenants are 1024 dimensions and the schema check validates against a width, so
#: this is not a free parameter. It is overridable rather than hardcoded because the day a tenant
#: moves to another width, a script that cannot be told is a script that has to be edited on a host.
DIM="${RECALL_SERVING_DIM:-1024}"
#: Which server the handshake drives. Defaults to the pair `.mcp.json` writes; the wrapper reads
#: the real values out of that file when it exists, so the two cannot drift apart unnoticed.
TENANT="${RECALL_SERVING_TENANT:-memory}"
EMBEDDER="${RECALL_SERVING_EMBEDDER:-voyage:voyage-4}"
LOCK_WAIT="${RECALL_SERVING_LOCK_WAIT:-20}"
FETCH_TIMEOUT="${RECALL_SERVING_FETCH_TIMEOUT:-120}"
CHECK_TIMEOUT="${RECALL_SERVING_CHECK_TIMEOUT:-120}"

MODE=""
TO="${RECALL_SERVING_REF:-origin/master}"
WITH_MIGRATIONS=0
WITH_DEPS=0
NO_VERIFY=0
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        status|sync)       MODE="$1" ;;
        --to)              shift; TO="${1:-}" ;;
        --with-migrations) WITH_MIGRATIONS=1 ;;
        --with-deps)       WITH_DEPS=1 ;;
        --no-verify)       NO_VERIFY=1 ;;
        --dry-run)         DRY_RUN=1 ;;
        *)
            printf 'session-serving: unknown argument %s\n' "$1" >&2
            exit 2
            ;;
    esac
    shift
done
[ -n "$MODE" ] || MODE="status"
[ -n "$TO" ] || { printf 'session-serving: --to needs a ref\n' >&2; exit 2; }

say()  { printf '%s\n' "$*"; }
warn() { printf '%s\n' "$*" >&2; }

# --- preconditions ----------------------------------------------------------------------------

if [ ! -d "$SERVING/.git" ] && [ ! -f "$SERVING/.git" ]; then
    say "SERVING     $SERVING"
    warn "session-serving: $SERVING is not a git checkout, so nothing here can update it."
    warn "session-serving: on VPS2 that symlink must point at a clone of the repository."
    exit 2
fi
cd "$SERVING" || exit 2

# `flock` is what makes "one sync at a time" a fact rather than a hope, and an absent guard must
# never read as a passing one: a checkout old enough or a host odd enough to lack flock is exactly
# where two sessions collide. The tests set RECALL_SERVING_NO_FLOCK=1 deliberately, and say so.
#
# Only `sync` takes them. `status` reads, and a read that refuses to answer while an indexer is
# running is a report nobody can trust to be there when they need it; `session-close.sh` calls
# status on every close, including closes that happen mid-index.
LOCKING=1
[ "$MODE" = "sync" ] || LOCKING=0
if [ "$LOCKING" -eq 1 ] && ! command -v flock >/dev/null 2>&1; then
    if [ "${RECALL_SERVING_NO_FLOCK:-0}" = "1" ]; then
        LOCKING=0
        warn "session-serving: flock absent, locks DISABLED by RECALL_SERVING_NO_FLOCK=1."
    else
        warn "session-serving: flock is not available on this host; refusing to sync unlocked."
        exit 2
    fi
fi

if [ "$LOCKING" -eq 1 ]; then
    mkdir -p "$LOCK_DIR" 2>/dev/null
    exec 9>"$LOCK_DIR/serving.lock" || { warn "session-serving: cannot open serving.lock"; exit 2; }
    if ! flock -w "$LOCK_WAIT" 9; then
        warn "session-serving: another sync holds serving.lock (waited ${LOCK_WAIT}s). Nothing moved."
        exit 3
    fi
    # Non-blocking on purpose. An indexing run takes minutes and queueing behind it would turn a
    # session-close step into a stall; the useful answer is "not now, an indexer is live".
    exec 8>"$LOCK_DIR/embed.lock" || { warn "session-serving: cannot open embed.lock"; exit 2; }
    if ! flock -n 8; then
        warn "session-serving: an embedding or indexing run holds embed.lock. Nothing moved."
        warn "session-serving: swapping modules under a live indexer corrupts a run silently."
        exit 3
    fi
fi

# --- where we are, and where the merged code is -----------------------------------------------

BRANCH="$(git symbolic-ref --quiet --short HEAD 2>/dev/null)"
HEAD_SHA="$(git rev-parse HEAD 2>/dev/null)"

if ! timeout "$FETCH_TIMEOUT" git fetch --quiet origin 2>/dev/null; then
    warn "session-serving: git fetch failed on the serving checkout. Nothing moved."
    exit 2
fi

TARGET_SHA="$(git rev-parse --verify --quiet "$TO^{commit}" 2>/dev/null)"
if [ -z "$TARGET_SHA" ]; then
    warn "session-serving: '$TO' does not resolve in the serving checkout. Nothing moved."
    exit 2
fi

DIRTY="$(git status --porcelain 2>/dev/null | head -20)"
BEHIND="$(git rev-list --count "HEAD..$TARGET_SHA" 2>/dev/null || echo '?')"
AHEAD="$(git rev-list --count "$TARGET_SHA..HEAD" 2>/dev/null || echo '?')"

say "SERVING     $SERVING"
say "BRANCH      ${BRANCH:-<detached>}"
say "HEAD        $(git log --oneline -1 2>/dev/null)"
say "TARGET      $TO $(git log --oneline -1 "$TARGET_SHA" 2>/dev/null)"
say "DISTANCE    $BEHIND behind, $AHEAD ahead"

# Reported before any refusal, because a session that has to fix something wants the whole list at
# once rather than one blocker per run.
MIG_CHANGES=""
DEP_CHANGES=""
if [ "$HEAD_SHA" != "$TARGET_SHA" ]; then
    MIG_CHANGES="$(git diff --name-only "$HEAD_SHA" "$TARGET_SHA" -- recall/migrations/ 2>/dev/null)"
    DEP_CHANGES="$(git diff --name-only "$HEAD_SHA" "$TARGET_SHA" -- pyproject.toml 2>/dev/null)"
    [ -n "$MIG_CHANGES" ] && say "MIGRATIONS  $(printf '%s' "$MIG_CHANGES" | tr '\n' ' ')"
    [ -n "$DEP_CHANGES" ] && say "DEPS        pyproject.toml changed between HEAD and target"
fi

if [ -n "$DIRTY" ]; then
    say "DIRTY"
    printf '%s\n' "$DIRTY" | sed 's/^/  /'
fi

# --- status stops here ------------------------------------------------------------------------

if [ "$MODE" = "status" ]; then
    if [ "$HEAD_SHA" = "$TARGET_SHA" ]; then
        say "STATUS      current"
    else
        say "STATUS      behind by $BEHIND commit(s)"
        git log --oneline "HEAD..$TARGET_SHA" 2>/dev/null | head -20 | sed 's/^/  /'
    fi
    exit 0
fi

# --- refusals ---------------------------------------------------------------------------------

if [ -n "$DIRTY" ]; then
    warn "session-serving: the serving checkout has uncommitted changes. Nothing moved."
    warn "session-serving: somebody may be hot-patching a live server; a fast-forward would"
    warn "session-serving: refuse anyway and a reset would destroy the only copy. Resolve by hand."
    exit 3
fi

if [ -z "$BRANCH" ]; then
    warn "session-serving: the serving checkout is on a detached HEAD. Nothing moved."
    warn "session-serving: check out the branch it should serve first (normally master)."
    exit 3
fi

if [ "$HEAD_SHA" != "$TARGET_SHA" ] && ! git merge-base --is-ancestor HEAD "$TARGET_SHA" 2>/dev/null; then
    warn "session-serving: serving HEAD is not an ancestor of $TO ($AHEAD commit(s) ahead)."
    warn "session-serving: that is a diverged deployment, not a stale one, and only a human knows"
    warn "session-serving: whether those commits matter. Nothing moved."
    exit 3
fi

if [ -n "$MIG_CHANGES" ] && [ "$WITH_MIGRATIONS" -ne 1 ]; then
    warn "session-serving: $TO changes migrations, so the update needs a schema apply."
    warn "session-serving: code that knows migration N against a database at N-1 raises"
    warn "session-serving: SchemaTooOld at server startup, which a client shows as no tools."
    warn "session-serving: applying to a live corpus is a one-way door, so re-run with:"
    warn "session-serving:   scripts/session-serving.sh sync --with-migrations"
    exit 3
fi

if [ "$DRY_RUN" -eq 1 ]; then
    if [ "$HEAD_SHA" = "$TARGET_SHA" ]; then
        say "DRY-RUN     already at $TO, a sync would only verify"
    else
        say "DRY-RUN     would fast-forward $BEHIND commit(s) to $TARGET_SHA"
        [ -n "$MIG_CHANGES" ] && say "DRY-RUN     would then run: recall schema --dim $DIM apply"
        [ -n "$DEP_CHANGES" ] && [ "$WITH_DEPS" -eq 1 ] && say "DRY-RUN     would then run: pip install -e ."
    fi
    exit 0
fi

# --- verification, used both after a move and to confirm a rollback ----------------------------

VERIFY_DETAIL=""

_verify_import() {
    local out
    out="$(cd "$SERVING" && timeout "$CHECK_TIMEOUT" "$PY" -c 'import recall.cli, recall_mcp.server' 2>&1)"
    if [ $? -ne 0 ]; then
        VERIFY_DETAIL="import failed: $(printf '%s' "$out" | tail -3 | tr '\n' ' ')"
        return 1
    fi
    return 0
}

_verify_schema() {
    local out
    out="$(cd "$SERVING" && set -a && . "$ENV_FILE" 2>/dev/null && set +a && \
           timeout "$CHECK_TIMEOUT" "$PY" -m recall.cli schema --dim "$DIM" status 2>&1)"
    # Both halves are checked. A non-zero exit is the loud failure; `compatible: yes` is the
    # quiet one, because `schema status` prints a full ledger and exits 0 on states a server
    # would still refuse to start against.
    if [ $? -ne 0 ] || ! printf '%s' "$out" | grep -q 'compatible: yes'; then
        VERIFY_DETAIL="schema: $(printf '%s' "$out" | grep -E 'current:|required:|compatible:|Error|error' | tr '\n' ' ')"
        return 1
    fi
    VERIFY_DETAIL="$(printf '%s' "$out" | grep -E 'current:|required:' | tr '\n' ' ')"
    return 0
}

# The one check that answers the question a session actually has: will the client get tools?
# Launched exactly as `.mcp.json` launches it, including `RECALL_ENV=production` (without it the
# server reads the legacy store and every search reports `generation=None`) and with
# RECALL_TRUST_MODE unset so the strict default stands.
_verify_handshake() {
    local out rc
    out="$(cd "$SERVING" && set -a && . "$ENV_FILE" 2>/dev/null && set +a && \
        timeout "$CHECK_TIMEOUT" env -u RECALL_TRUST_MODE \
            RECALL_TENANT="$TENANT" RECALL_EMBEDDER="$EMBEDDER" RECALL_ENV=production \
            "$PY" - <<'PY' 2>&1
import json, subprocess, sys, tempfile

# The server's own diagnosis goes to stderr, and it is the only place the useful sentence appears:
# `SchemaTooOld: ... run 'recall schema apply'` is what distinguishes a migration gap from an
# unreachable database. It is captured to a FILE rather than a pipe, because reading a pipe while
# the child is still writing to it is how this deadlocks.
errlog = tempfile.NamedTemporaryFile(mode="w", suffix=".err", delete=False)
proc = subprocess.Popen(
    [sys.executable, "-m", "recall_mcp.server"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errlog, text=True,
)


def send(payload):
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()


def die(reason):
    proc.terminate()
    proc.wait(timeout=10)
    errlog.close()
    with open(errlog.name, encoding="utf-8", errors="replace") as fh:
        tail = [ln.strip() for ln in fh.readlines()[-3:] if ln.strip()]
    print(reason, "|", " ".join(tail))
    raise SystemExit(1)


try:
    send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {},
        "clientInfo": {"name": "session-serving", "version": "1"}}})
    if not proc.stdout.readline():
        die("server exited before answering initialize")
    send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
    send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    line = proc.stdout.readline()
    try:
        tools = json.loads(line)["result"]["tools"]
    except Exception as exc:
        die(f"tools/list did not answer ({exc})")
    if not tools:
        die("server started but exposes NO tools")
    print(f"tools={len(tools)}")
finally:
    proc.terminate()
PY
)"
    rc=$?
    if [ $rc -ne 0 ]; then
        VERIFY_DETAIL="handshake: $(printf '%s' "$out" | tail -3 | tr '\n' ' ')"
        return 1
    fi
    VERIFY_DETAIL="handshake $(printf '%s' "$out" | grep -o 'tools=[0-9]*' | tail -1)"
    return 0
}

_verify_all() {
    local skip_handshake="${1:-0}"
    _verify_import || return 1
    local imported="import ok"
    _verify_schema || return 1
    local schema="$VERIFY_DETAIL"
    if [ "$skip_handshake" -eq 1 ]; then
        VERIFY_DETAIL="$imported; $schema; handshake skipped"
        return 0
    fi
    _verify_handshake || return 1
    VERIFY_DETAIL="$imported; $schema; $VERIFY_DETAIL"
    return 0
}

# --- the move -----------------------------------------------------------------------------------

MOVED=0
if [ "$HEAD_SHA" = "$TARGET_SHA" ]; then
    say "MOVE        none, already at $TO"
else
    LANDED="$(git log --oneline "HEAD..$TARGET_SHA" 2>/dev/null | head -20)"
    if ! git merge --ff-only "$TARGET_SHA" >/dev/null 2>&1; then
        warn "session-serving: fast-forward to $TO failed. Nothing moved."
        exit 3
    fi
    MOVED=1
    say "MOVE        $(printf '%s' "$HEAD_SHA" | cut -c1-8) -> $(printf '%s' "$TARGET_SHA" | cut -c1-8) ($BEHIND commit(s))"
    printf '%s\n' "$LANDED" | sed 's/^/  /'
fi

if [ "$MOVED" -eq 1 ] && [ -n "$DEP_CHANGES" ]; then
    if [ "$WITH_DEPS" -eq 1 ]; then
        if (cd "$SERVING" && timeout "$CHECK_TIMEOUT" "$PY" -m pip install -e . --quiet >/dev/null 2>&1); then
            say "DEPS        reinstalled (pip install -e .)"
        else
            say "DEPS        pip install -e . FAILED; verification below decides what happens"
        fi
    else
        say "DEPS        pyproject.toml moved and --with-deps was not given."
        say "DEPS        the editable install still imports the new code; only NEW third-party"
        say "DEPS        requirements would be missing, and the import check below catches those."
    fi
fi

if [ "$MOVED" -eq 1 ] && [ -n "$MIG_CHANGES" ]; then
    # Reached only with --with-migrations: the refusal above is the gate.
    if (cd "$SERVING" && set -a && . "$ENV_FILE" 2>/dev/null && set +a && \
        timeout "$CHECK_TIMEOUT" "$PY" -m recall.cli schema --dim "$DIM" apply >/dev/null 2>&1); then
        say "SCHEMA      migrations applied"
    else
        warn "session-serving: 'recall schema apply' failed after the fast-forward."
        warn "session-serving: the code is new and the database is not. Rolling back the code."
        if git reset --hard "$HEAD_SHA" >/dev/null 2>&1; then
            warn "session-serving: rolled back to $HEAD_SHA. The database was not changed by a"
            warn "session-serving: failed apply, but check 'recall schema status' before retrying."
            exit 4
        fi
        warn "session-serving: ROLLBACK FAILED. The serving checkout is at $TARGET_SHA with a"
        warn "session-serving: database that has not been migrated. Fix this by hand, now."
        exit 5
    fi
fi

# --- verify, and undo if it does not hold -------------------------------------------------------

if _verify_all "$NO_VERIFY"; then
    say "VERIFY      ok: $VERIFY_DETAIL"
    if [ "$MOVED" -eq 1 ]; then
        say "RESULT      serving updated to $TO and verified"
    else
        say "RESULT      serving already at $TO and verified"
    fi
    exit 0
fi

FAILURE="$VERIFY_DETAIL"
warn "session-serving: verification FAILED: $FAILURE"

if [ "$MOVED" -eq 0 ]; then
    # Nothing this run did caused it, so there is nothing to undo and hiding it would be worse.
    warn "session-serving: the checkout did not move, so this is pre-existing. Nothing rolled back."
    exit 4
fi

if ! git reset --hard "$HEAD_SHA" >/dev/null 2>&1; then
    warn "session-serving: ROLLBACK FAILED. Serving is at $TARGET_SHA and failing verification."
    warn "session-serving: fix this by hand before the next session: the MCP servers are down."
    exit 5
fi

say "ROLLED-BACK $(printf '%s' "$TARGET_SHA" | cut -c1-8) -> $(printf '%s' "$HEAD_SHA" | cut -c1-8)"
if _verify_all "$NO_VERIFY"; then
    say "VERIFY      ok after rollback: $VERIFY_DETAIL"
    say "RESULT      serving left where it was; $TO does not serve here yet"
else
    warn "session-serving: the rolled-back checkout ALSO fails verification: $VERIFY_DETAIL"
    warn "session-serving: this was broken before the sync. Fix the host, not the branch."
fi
exit 4
