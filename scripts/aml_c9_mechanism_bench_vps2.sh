#!/usr/bin/env bash
# Run scripts/aml_c9_mechanism_bench.py against the official C9 unit on VPS2, then read its journal.
#
#   scripts/aml_c9_mechanism_bench_vps2.sh            # refuses while other traffic is live
#   scripts/aml_c9_mechanism_bench_vps2.sh --force    # run even so (the user said go)
#
# The key never leaves VPS2: the benchmark runs there as root, reads RECALL_AML_API_KEY and the port
# from C9's own env file inside a subshell, and talks to 127.0.0.1. The public hostname is checked
# separately with /health and /version, which need no key.
#
# Guard: the official Smoke and Full runs share this unit. If the journal shows any Add or Search
# completed in the last GUARD_S seconds, the run is refused, because this benchmark would then
# load the same process as a platform run. Its own traffic is isolated per user either way.
set -euo pipefail

HOST="${C9_ROOT_HOST:-root@100.91.148.25}"
KEY="${C9_SSH_KEY:-$HOME/.ssh/contabo_sentiment}"
UNIT="recall-aml-c9-official"
ENV_FILE="/etc/recall-aml/c9-official.env"
PUBLIC="${C9_PUBLIC_URL:-https://memory.pred-markets.com}"
GUARD_S="${GUARD_S:-120}"
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

HERE="$(cd "$(dirname "$0")" && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REMOTE_PY="/root/c9-mech-bench-${STAMP}.py"
REMOTE_OUT="/root/c9-mech-bench-${STAMP}.json"
LOCAL_OUT="${C9_BENCH_OUT:-$HERE/../docs/results/2026-09-24-c9-mechanism-bench-${STAMP}.json}"
ssh_root() { ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=15 "$HOST" "$@"; }

echo "== preflight"
ssh_root "systemctl is-active $UNIT; grep -E '^RECALL_AML_PORT=' $ENV_FILE"
recent=$(ssh_root "journalctl -u $UNIT --since '-${GUARD_S}s' -o cat | grep -cE 'POST /v1/(add|search)' || true")
if [ "${recent:-0}" -gt 0 ] && [ "$FORCE" -ne 1 ]; then
  echo "REFUSED: $recent Add/Search log lines in the last ${GUARD_S}s; a platform run may be live."
  echo "Re-run with --force once the smoke is finished."
  exit 3
fi

echo "== benchmark (on VPS2, 127.0.0.1)"
scp -q -i "$KEY" -o BatchMode=yes "$HERE/aml_c9_mechanism_bench.py" "$HOST:$REMOTE_PY"
since="$(ssh_root 'date -u "+%Y-%m-%d %H:%M:%S"')"
set +e
# Only the two variables the benchmark needs are read: the unit's env file is systemd syntax, not
# shell, and nothing else in it belongs in this process.
ssh_root bash -s -- "$ENV_FILE" "$REMOTE_PY" "$REMOTE_OUT" "/root/c9-mech-bench-${STAMP}.err" <<'REMOTE'
val() { grep -E "^$1=" "$ENV_FILE" | tail -1 | cut -d= -f2- | sed -e 's/^["'\'']//' -e 's/["'\'']$//'; }
ENV_FILE="$1"; PY="$2"; OUT="$3"; ERR="$4"
RECALL_AML_API_KEY="$(val RECALL_AML_API_KEY)" python3 "$PY" \
  --base-url "http://127.0.0.1:$(val RECALL_AML_PORT)" --out "$OUT" >/dev/null 2>"$ERR"
echo "exit=$?"; tail -5 "$ERR"
REMOTE
set -e
scp -q -i "$KEY" -o BatchMode=yes "$HOST:$REMOTE_OUT" "$LOCAL_OUT" || echo "no report produced"
ssh_root "rm -f $REMOTE_PY $REMOTE_OUT /root/c9-mech-bench-${STAMP}.err"

echo "== journal since $since UTC"
ssh_root "journalctl -u $UNIT --since '$since' -o cat \
  | grep -oE 'hosted_[a-z_]+|compiler_[a-z_]+|Traceback|error_class\": ?\"[A-Za-z]+\"|compiler_fallback\": ?true|graph_fallback\": ?true|atomic_rescue_fallback\": ?true|\" [0-9]{3} ' \
  | sort | uniq -c | sort -rn; \
  systemctl show $UNIT -p NRestarts -p MemoryPeak -p MemoryCurrent"

echo "== public hostname"
curl -s -o /dev/null -w "health %{http_code}\n" --max-time 20 "$PUBLIC/health"
curl -s --max-time 20 "$PUBLIC/version" | python -c "import json,sys;d=json.load(sys.stdin);print('version', d['git_commit'][:8], d['variant'], 'graph', d['graph_sidecar'], d['atomic_rescue'].get('mode'))"

echo "== report: $LOCAL_OUT"
[ -f "$LOCAL_OUT" ] && python -c "import json,sys;r=json.load(open(sys.argv[1]));print('passed', r['passed'], 'failed', r['failed_checks']);print(json.dumps(r['summary'], indent=2))" "$LOCAL_OUT"
