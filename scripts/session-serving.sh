#!/usr/bin/env bash
# ---8<--- help
# Keep the checkout VPS2 serves from at what has actually been merged, and prove it still serves.
#
# The MCP servers a session gets its memory and code answers from do not run out of this checkout.
# They run on VPS2, out of `~/recall-repos/serving`, whose venv holds an editable install pointing
# at that clone. So merging a fix to master changes nothing anyone can search until that clone is
# fast-forwarded, and the failure is silent in the worst way available: a stale server is a working
# server. It answers every query, confidently, with the code that was current whenever somebody
# last remembered to pull.
#
#   scripts/session-serving.sh                      # status: where serving is, what it lacks
#   scripts/session-serving.sh verify               # read-only: does the server still START?
#   scripts/session-serving.sh sync                 # fast-forward to origin/master and verify
#   scripts/session-serving.sh sync --dry-run       # every check, nothing moved
#   scripts/session-serving.sh sync --with-migrations   # allow a schema apply after the move
#   scripts/session-serving.sh sync --with-deps         # allow pip install -e . after the move
#   scripts/session-serving.sh sync --no-verify         # skip the handshake
# --->8--- help
#
# This file is the wrapper. Every decision lives in `scripts/session_serving_remote.sh`, which is
# piped to VPS2 over ssh and is tested against a fake checkout by `scripts/session_serving_tests.sh`.
# The split is deliberate: logic that only ever runs on a host nobody can stub is logic nobody
# tests, and this one can roll a deployment back.
#
# `verify` is that handshake on its own, and it exists because it was previously reachable only
# as the tail of a `sync`. `scripts/session-corpus.sh` asks whether the CORPUS is certified, in
# one SQL query; this asks whether the SERVER starts, in about thirty seconds. Neither substitutes
# for the other, and each can be green while the other is not: a certified corpus behind a
# serving checkout at the wrong migration level is a client with no tools at all.
#
# The cost is four samples on 2026-08-31, all against the live host: 29.2, 29.7, 34.2 and 72.3
# seconds. The last was taken while ten agents were driving the same host, and it is quoted rather
# than discarded because it is the honest top of the range, not an outlier to hide. Budget for
# thirty seconds and do not be surprised by a minute. Re-measure:
#   time bash scripts/session-serving.sh verify
#
# Exit codes come straight from the remote half:
#   0 up to date and verified   2 usage or environment   3 refused, nothing moved
#   4 verification failed, rolled back                   5 rollback itself failed
# `verify` uses the same 0 and 4, and can never return 3 or 5: it moves nothing to roll back.

set -uo pipefail

HOST="${RECALL_VPS2_HOST:-vps2}"
SSH_TIMEOUT="${RECALL_SERVING_SSH_TIMEOUT:-300}"
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
REMOTE_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/session_serving_remote.sh"

if [ ! -f "$REMOTE_SCRIPT" ]; then
    echo "session-serving: missing $REMOTE_SCRIPT" >&2
    exit 2
fi

MODE=""
ARGS=()
# `--to` takes a VALUE, and that value must not be read as a mode. Before this flag existed the
# loop matched a bare word anywhere, so `sync --to verify` set the wrapper's MODE to verify while
# the remote half (which parses --to correctly) ran a real sync, and the rc=4 branch below then
# reported a rollback that had happened as "nothing was moved".
expect_value=0
for arg in "$@"; do
    if [ "$expect_value" -eq 1 ]; then
        expect_value=0
        ARGS+=("$arg")
        continue
    fi
    case "$arg" in
        status|sync|verify)
            # Same refusal as the remote half, for the same measured reason: `verify sync` used
            # to move a live deployment behind a read-only verb.
            if [ -n "$MODE" ]; then
                echo "session-serving: say ONE of status, verify or sync (got $MODE then $arg)." >&2
                exit 2
            fi
            MODE="$arg"
            ARGS+=("$arg")
            ;;
        --to)
            expect_value=1
            ARGS+=("$arg")
            ;;
        --help|-h)
            sed -n '/^# ---8<--- help$/,/^# --->8--- help$/p' "${BASH_SOURCE[0]}" \
                | grep -v -- '---8<--- help\|--->8--- help' | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            # Flags. Passed through rather than validated twice: the remote half refuses what it
            # does not recognise, so a typo cannot become a sync to something nobody meant.
            #
            # 🔁 This comment used to say "Flags, and the value after `--to`", which was wrong and
            # wrong in the direction that hid a defect: the value after `--to` never reached this
            # arm, because `status|sync|verify` matched it first. `expect_value` above is what
            # makes the old comment's claim true.
            ARGS+=("$arg")
            ;;
    esac
done
if [ -z "$MODE" ]; then
    # A bare invocation reports. A flags-only invocation is refused rather than defaulted, because
    # `session-serving.sh --dry-run` means "sync, but tell me first" to everyone who types it, and
    # answering it with a status report would be a green line for a question nobody asked.
    if [ ${#ARGS[@]} -gt 0 ]; then
        echo "session-serving: say 'status', 'verify' or 'sync' before any flag." >&2
        echo "usage: scripts/session-serving.sh [status|verify|sync] [--to REF] [--with-migrations]" >&2
        echo "                                  [--with-deps] [--no-verify] [--dry-run]" >&2
        exit 2
    fi
    MODE="status"
    ARGS=("status")
fi

# Which server the remote handshake drives. Read out of THIS checkout's `.mcp.json` when it exists,
# rather than restated here: the tenants are all 1024 dimensions and all different models, so a
# tenant/embedder pair that has drifted from the generated config would verify a server nobody
# runs while pgvector returns a confidently ranked list that means nothing.
TENANT=""
EMBEDDER=""
if [ -f "$ROOT/.mcp.json" ] && command -v python >/dev/null 2>&1; then
    read -r TENANT EMBEDDER <<<"$(MCP="$ROOT/.mcp.json" python - <<'PY' 2>/dev/null
import json, os, re
try:
    cfg = json.load(open(os.environ["MCP"], encoding="utf-8"))["mcpServers"]["recall-memory"]
    inner = " ".join(cfg.get("args", []))
except Exception:
    raise SystemExit(0)
tenant = re.search(r"RECALL_TENANT=(\S+)", inner)
embedder = re.search(r"RECALL_EMBEDDER=(\S+)", inner)
if tenant and embedder:
    print(tenant.group(1).strip("'\""), embedder.group(1).strip("'\""))
PY
)"
fi

# No banner. `session-close.sh` prints its own heading above this output, and two headings for one
# section reads as two sections. The `LOCAL` and `SERVING` lines identify what is being reported
# well enough for a standalone run.
#
# What this session has that master does not. The sync ships MERGED code by design, so a branch
# with unpushed or unmerged commits is not shipped by it, and saying so here is the difference
# between "the server is current" and "the server is current with somebody else's work".
if [ "$MODE" != "verify" ] && git -C "$ROOT" rev-parse --verify --quiet origin/master >/dev/null 2>&1; then
    local_branch="$(git -C "$ROOT" symbolic-ref --quiet --short HEAD 2>/dev/null || echo '<detached>')"
    unmerged="$(git -C "$ROOT" rev-list --count origin/master..HEAD 2>/dev/null || echo 0)"
    if [ "${unmerged:-0}" = "0" ]; then
        printf 'LOCAL       %s is contained in origin/master.\n' "$local_branch"
    elif git -C "$ROOT" diff --quiet origin/master HEAD 2>/dev/null; then
        # A SQUASH merge leaves the branch ahead by sha while its content is already on master,
        # and `master` here refuses merge commits, so squashing is the ONLY way work lands. The
        # count alone therefore raises a false alarm on every branch this repository merges: the
        # first run of this script after its own PR landed said "3 commit(s) not on origin/master;
        # a sync will NOT ship them" about work the sync had just shipped. Ask the trees, not the
        # shas: identical content means nothing of yours is missing from what was synced.
        printf 'LOCAL       %s is %s commit(s) ahead by sha, but identical in content to\n' \
            "$local_branch" "$unmerged"
        printf '            origin/master (a squash merge). A sync ships your work.\n'
    else
        printf 'LOCAL       %s has %s commit(s) not on origin/master; a sync will NOT ship them.\n' \
            "$local_branch" "$unmerged"
    fi
fi

# A leading `~/` has to be expanded by the REMOTE shell, and `printf '%q'` would quote it into a
# literal tilde instead. That matters because `~/recall-repos/.env` is the DOCUMENTED default form
# for these paths, and `scripts/session-mcp.sh` interpolates the same variables unquoted so the far
# side expands them. Without this, setting `RECALL_VPS2_ENV=~/...` would make the readability guard
# in the remote half report a confident, wrong "env file is missing or unreadable".
_remote_arg() {
    case "$1" in
        "~/"*) printf '$HOME/%s' "$(printf '%q' "${1#\~/}")" ;;
        *)     printf '%q' "$1" ;;
    esac
}

remote_env=""
[ -n "$TENANT" ]   && remote_env="$remote_env RECALL_SERVING_TENANT=$(printf '%q' "$TENANT")"
[ -n "$EMBEDDER" ] && remote_env="$remote_env RECALL_SERVING_EMBEDDER=$(printf '%q' "$EMBEDDER")"
[ -n "${RECALL_VPS2_CHECKOUT:-}" ] && \
    remote_env="$remote_env RECALL_SERVING_PATH=$(_remote_arg "$RECALL_VPS2_CHECKOUT")"
# `verify` claims the server is launched exactly as `.mcp.json` launches it, and these are the two
# overrides that can make that false without saying so: `scripts/session-mcp.sh` reads
# RECALL_VPS2_PYTHON and RECALL_VPS2_ENV for the interpreter and the env file, while the remote
# half reads RECALL_SERVING_VENV and RECALL_SERVING_ENV for the same two roles. The defaults
# agree, so an override moves one side only, which is the silent kind of wrong. The checkout path
# above was already bridged; these two were not.
[ -n "${RECALL_VPS2_PYTHON:-}" ] && \
    remote_env="$remote_env RECALL_SERVING_VENV=$(_remote_arg "$RECALL_VPS2_PYTHON")"
[ -n "${RECALL_VPS2_ENV:-}" ] && \
    remote_env="$remote_env RECALL_SERVING_ENV=$(_remote_arg "$RECALL_VPS2_ENV")"

quoted=""
for arg in "${ARGS[@]}"; do
    [ -z "$arg" ] && continue
    quoted="$quoted $(printf '%q' "$arg")"
done

timeout "$SSH_TIMEOUT" ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" \
    "${remote_env# } bash -s --$quoted" < "$REMOTE_SCRIPT"
rc=$?

case "$rc" in
    0) ;;
    3) printf 'session-serving: REFUSED, and nothing on %s was moved. The reason is above.\n' "$HOST" >&2 ;;
    4) if [ "$MODE" = "verify" ]; then
           printf 'session-serving: the serving checkout does NOT serve, and nothing was moved.\n' >&2
           printf 'session-serving: the recall tools will be missing until it is fixed.\n' >&2
       else
           printf 'session-serving: verification failed; the serving checkout was put back.\n' >&2
       fi ;;
    5) printf 'session-serving: rollback FAILED on %s. The MCP servers are down until this is fixed.\n' "$HOST" >&2 ;;
    124) if [ "$MODE" = "verify" ]; then
             # The one thing a verify always knows. Saying "nothing is known about what moved"
             # here would manufacture uncertainty about a state that is certain, which is the
             # opposite of what this mode is for.
             printf 'session-serving: verify timed out after %ss. It moves nothing, so the\n' "$SSH_TIMEOUT" >&2
             printf 'session-serving: serving checkout is exactly as it was. Raise the budget with\n' >&2
             printf 'session-serving: RECALL_SERVING_SSH_TIMEOUT if the host is merely slow.\n' >&2
         else
             printf 'session-serving: timed out after %ss. Nothing is known about what moved.\n' "$SSH_TIMEOUT" >&2
         fi ;;
    # 127 is the remote shell saying it could not run the command at all, and it is the one exit
    # code that used to arrive with no explanation and no output. Observed once, unreproduced,
    # immediately after a merge. Nothing had moved, and that is knowable rather than hopeful: the
    # remote half prints SERVING, BRANCH and HEAD before it touches anything, so an empty report
    # means it never started.
    127) printf 'session-serving: the remote command never started (exit 127) and NOTHING moved.\n' >&2
         printf 'session-serving: the remote half prints SERVING/BRANCH/HEAD before it acts, so an\n' >&2
         printf 'session-serving: empty report above means it did not run. Re-run it.\n' >&2 ;;
    255) printf 'session-serving: ssh to %s failed. The serving checkout was not touched.\n' "$HOST" >&2 ;;
esac
exit $rc
