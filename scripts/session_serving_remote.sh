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
# 🔁 Corrected 2026-08-31, both lines above. `serving` now resolves to
# `serving-master/master-live`, a WORKTREE of the serving-master clone whose checked-out branch is
# `serving-live`, and `import recall` resolves through the symlink to
# `serving-master/master-live/recall/__init__.py`. The symlink was last moved 2026-08-29. The
# 2026-08-26 measurement is left as written: it is dated, and what it teaches is that this
# arrangement moves under the notes that describe it. Re-measure:
#   ssh vps2 'ls -ld ~/recall-repos/serving; git -C ~/recall-repos/serving branch --show-current'
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
# 🔁 Corrected 2026-08-31: the same host now answers with 20 tools at migration 0017. The
# 2026-08-26 pair is left exactly as written, because a dated claim is allowed to age and the
# useful fact is that this number MOVES. Nothing asserts a count at runtime: the handshake checks
# `if not tools`, never equality, so no behaviour depends on either figure. Re-measure with
# `bash scripts/session-serving.sh verify`.
#
# `verify` is that same handshake with nothing else attached, added 2026-08-31 to close a gap that
# had been named and left open: the handshake was reachable ONLY as the tail of a sync, so the one
# question a session actually has (will my recall tools work?) could not be asked without a
# command that moves a live deployment. `session-corpus.sh` was the other half of the gap; it
# proves the CORPUS is certified and says nothing about whether the process starts.
#
#   session_serving_remote.sh status                 # read-only: where serving is, what it lacks
#   session_serving_remote.sh verify                 # read-only: does it still START and serve?
#   session_serving_remote.sh sync                   # fast-forward to origin/master and verify
#   session_serving_remote.sh sync --dry-run         # every check, no move
#   session_serving_remote.sh sync --with-migrations # allow `recall schema apply` after the move
#   session_serving_remote.sh sync --with-deps       # allow `pip install -e .` when deps moved
#   session_serving_remote.sh sync --no-verify       # skip the handshake (keeps cheap checks)
#
# Exit codes are part of the interface, because the wrapper and the session report read them:
#   0 up to date and verified   2 usage or environment   3 refused, nothing moved
#   4 verification failed, rolled back                   5 rollback itself failed (loud)
# `verify` answers with 0 or 4, and returns 2 for an environment it cannot even look at (a
# serving path that is not a checkout). It can never return 3 or 5: it moves nothing to roll
# back, and the preconditions above it run before the mode is dispatched.

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
        status|sync|verify)
            # Refused rather than last-wins. Measured on the fixture harness before this guard
            # existed: `verify sync` fast-forwarded the serving checkout and reported
            # "serving updated", and `sync verify` shipped nothing while reporting the verify
            # green. A mode word is a verb, and two verbs in one sentence is a typo whose two
            # possible readings differ by whether a live deployment moves.
            if [ -n "$MODE" ]; then
                printf 'session-serving: say ONE of status, verify or sync (got %s then %s).\n' \
                    "$MODE" "$1" >&2
                exit 2
            fi
            MODE="$1"
            ;;
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
# Only for the modes that resolve a target. `verify` never reads TO, and refusing it over an empty
# ref would be a refusal from the one mode documented as refusing nothing.
if [ "$MODE" != "verify" ] && [ -z "$TO" ]; then
    printf 'session-serving: --to needs a ref\n' >&2
    exit 2
fi

say()  { printf '%s\n' "$*"; }
warn() { printf '%s\n' "$*" >&2; }

# --- verification, used after a move, to confirm a rollback, and on its own -------------------
#
# Defined above the flow rather than beside its first caller so that `verify` can run WITHOUT
# the fetch, the target arithmetic and the refusals below. Those exist to protect a move, and
# `verify` moves nothing; a read-only check that refuses because the network is down or because
# somebody hot-patched the tree has answered a question nobody asked.

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
    if [ ! -r "$ENV_FILE" ]; then
        VERIFY_DETAIL="env file $ENV_FILE is missing or unreadable, so nothing can reach the database"
        return 1
    fi
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
    if [ ! -r "$ENV_FILE" ]; then
        VERIFY_DETAIL="env file $ENV_FILE is missing or unreadable, so nothing can reach the database"
        return 1
    fi
    out="$(cd "$SERVING" && set -a && . "$ENV_FILE" 2>/dev/null && set +a && \
        timeout "$CHECK_TIMEOUT" env -u RECALL_TRUST_MODE \
            RECALL_TENANT="$TENANT" RECALL_EMBEDDER="$EMBEDDER" RECALL_ENV=production \
            "$PY" - <<'PY' 2>&1
import json, os, signal, subprocess, sys, tempfile

# The server's own diagnosis goes to stderr, and it is the only place the useful sentence appears:
# `SchemaTooOld: ... run 'recall schema apply'` is what distinguishes a migration gap from an
# unreachable database. It is captured to a FILE rather than a pipe, because reading a pipe while
# the child is still writing to it is how this deadlocks.
errlog = tempfile.NamedTemporaryFile(mode="w", suffix=".err", delete=False)
proc = subprocess.Popen(
    [sys.executable, "-m", "recall_mcp.server"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errlog, text=True,
    # A hand-traceable mark, and NOTHING MORE. Read the next paragraph before relying on it.
    #
    # ⛔ `scripts/session-mcp-close.sh sweep` CANNOT see this, and an earlier version of this
    # comment claimed it could. Two independent reasons, both in `scripts/session_mcp_sweep.py`:
    # its probe takes the mark from the PARENT process's argv (`m = MARK.search(parent)`), and an
    # environment variable handed to `Popen(env=...)` is in no argv anywhere; and the classifier
    # additionally requires the `<host>-<checkout id>` format before a server reaches the
    # closeable bucket. `session-mcp.sh` works because it puts the variable on the ssh COMMAND
    # LINE. No parent-argv scheme could rescue the case this exists for anyway, since an orphan
    # reparents to init and loses its parent entirely.
    #
    # 🔑 So what actually prevents the leak is the SIGTERM handler and reap() below, which is
    # measured: on VPS2 (coreutils 9.4) the spawned server was alive one second after the outer
    # `timeout` fired without them, and dead with them. This mark is worth keeping only because it
    # makes a survivor identifiable BY HAND, which is otherwise impossible on a host where 16 of
    # 18 servers belong to another agent:
    #     ssh vps2 'tr "\0" "\n" < /proc/<pid>/environ | grep RECALL_MCP_CLIENT'
    #
    # The comment this replaces asserted the sweep would collect it. That is the same defect this
    # very change set added `scripts/session_corpus_tests.sh` to catch: a comment stating an
    # invariant the code does not have. Caught here by two reviewers, not by a test, because no
    # test can observe a claim that only a comment makes.
    env={**os.environ, "RECALL_MCP_CLIENT": f"session-serving-verify-{os.getpid()}"},
)


def reap():
    """Terminate, then insist.

    A server wedged on its database does not die politely, and `proc.terminate()` alone returns
    without waiting, so the parent can exit first and leave it. One of these costs ~815 MB on a
    host that also runs the live trading services.
    """
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def scrub():
    # `os._exit` below skips the `finally`, so the temp file has to be removed here too or the
    # timeout path keeps leaking the small half of what it used to leak.
    try:
        errlog.close()
    except Exception:
        pass
    try:
        os.unlink(errlog.name)
    except OSError:
        pass


def on_sigterm(*_):
    # `timeout` signals THIS process only, and CPython's default SIGTERM action exits without
    # unwinding, so without this handler the `finally` below never runs and the server is
    # orphaned to init. Measured on VPS2 2026-08-31, coreutils 9.4: grandchild alive after the
    # outer timeout fired. The timeout path is the one `verify` exists to diagnose, so it is
    # exactly the path that must not leak.
    reap()
    scrub()
    os._exit(1)


signal.signal(signal.SIGTERM, on_sigterm)


def send(payload):
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()


def die(reason):
    reap()
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
    reap()
    # The file is read by die() before this point when there is anything to read; nothing needs it
    # to survive the process, and every run used to leave one behind on the host. Only the
    # LIFETIME changes here: the capture mechanism stays a real file on purpose, because reading a
    # pipe while the child is still writing to it is how this deadlocks.
    try:
        errlog.close()
    except Exception:
        pass
    try:
        os.unlink(errlog.name)
    except OSError:
        pass
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
# Only `sync` takes them. `status` and `verify` read, and a read that refuses to answer while an
# indexer is running is a report nobody can trust to be there when they need it; `session-close.sh`
# calls status on every close, including closes that happen mid-index, and `verify` is what
# somebody runs DURING an incident, which is the worst possible moment to be told to wait.
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

# --- verify stops before the fetch -------------------------------------------------------------
#
# The one question neither `status` nor `scripts/session-corpus.sh` answers: does the server this
# checkout holds actually START and hand a client a tool list? `status` compares shas.
# `session-corpus.sh` asks the database whether a certified calibration is bound to the active
# generation. Both are true and cheap, and both stay green against a serving checkout whose code
# knows a migration the database does not, which a client renders as a server with NO tools.
#
# So this fetches nothing, resolves no target, takes no locks and refuses nothing, because it moves
# nothing. A dirty or detached serving tree is precisely the state somebody most wants to verify.
#
# `--no-verify` is ignored here rather than honoured. A verify that skips the handshake is not a
# cheaper verify, it is a green line for a check that did not run, which is the whole failure mode
# these scripts exist to remove.
if [ "$MODE" = "verify" ]; then
    say "SERVING     $SERVING"
    say "BRANCH      ${BRANCH:-<detached>}"
    say "HEAD        $(git log --oneline -1 2>/dev/null)"
    VERIFY_DIRTY="$(git status --porcelain 2>/dev/null | head -20)"
    if [ -n "$VERIFY_DIRTY" ]; then
        # Said out loud rather than refused: what is verified here is what is RUNNING, and on a
        # hot-patched tree that is not what is merged. A green line without this one would be a
        # true statement about a deployment nobody can reproduce.
        say "DIRTY       verifying a hot-patched tree; this is what RUNS, not what is merged"
        printf '%s\n' "$VERIFY_DIRTY" | sed 's/^/  /'
    fi
    # Free, offline, and moves nothing: origin/master is a ref already on disk, so reading the
    # distance costs no fetch and keeps the "resolves no target" property intact. Without it a
    # green verify is exit 0 for a checkout arbitrarily far behind, which is this file's own
    # opening sentence: a stale server is a working server.
    VERIFY_BEHIND="$(git rev-list --count HEAD..origin/master 2>/dev/null || echo '?')"
    if _verify_all 0; then
        say "VERIFY      ok: $VERIFY_DETAIL"
        say "RESULT      this checkout starts and answers with tools, driving tenant '$TENANT'."
        # Named rather than left to be inferred, for the same reason session-corpus.sh names its
        # own boundary: an unqualified green over a partial check is the substitution both of
        # these scripts exist to remove. Each server in .mcp.json carries its OWN tenant and
        # embedder (all 1024 dimensions, all different models), and only one is driven here.
        say "NOT PROVED  any other tenant. Server-wide faults (a failed import, a schema the"
        say "            server refuses) are checkout-wide and WOULD have shown up above; a"
        say "            tenant-specific one, an unpromoted generation or a drifted embedder,"
        say "            would not. -> scripts/session-corpus.sh status"
        say "NOT PROVED  that this checkout is CURRENT: verify fetches nothing. As of the last"
        say "            fetch it is $VERIFY_BEHIND behind origin/master."
        say "            -> scripts/session-serving.sh status"
        exit 0
    fi
    warn "session-serving: verification FAILED: $VERIFY_DETAIL"
    warn "session-serving: nothing was moved; this is the state the host was already in. Until it"
    warn "session-serving: is fixed a client shows the recall servers with NO tools, which is also"
    warn "session-serving: how a missing config, an unapproved server and an unreachable host look."
    exit 4
fi

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
