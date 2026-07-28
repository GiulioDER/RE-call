#!/usr/bin/env bash
# Three LOCOMO arms, serially, in ONE process.
#
# Written after a run produced a WRONG published number: two copies of an earlier version of this
# script ran concurrently into the same tables, every tenant held its corpus twice (11,764 rows
# against a correct 5,882), and every depth came in ~0.05 low without anything erroring.
#
# Two defences, in order of preference:
#   1. run_conversation now REFUSES to index over an existing tenant (recall/eval/locomo.py).
#      That makes a concurrent double-write fail loudly instead of silently corrupting.
#   2. This script verifies the ARTIFACT after each arm — row count, not "did the process exit 0".
#      Verifying the process is what failed last time.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

OUT=results/locomo_rerank
DSN=${RECALL_DSN:-postgresql://recall:recall@localhost:5432/recall}
COMMON="--data locomo10.json --k 5 --k-curve 1,3,5,10,20 --candidate-k 20"
EXPECTED_ROWS=5882   # measured on a clean run that reproduces postfix_pool20.json exactly

mkdir -p "$OUT"

# A single lock: a second copy of this script exits instead of racing the first.
LOCK=$OUT/.arms.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "FATAL: $LOCK exists — another run is in progress (or died; rmdir it to clear)."
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

verify_rows() {  # $1 = table, $2 = arm name
  local n
  n=$(python -c "
import psycopg,sys
c=psycopg.connect('$DSN')
try: print(c.execute('select count(*) from $1').fetchone()[0])
except Exception: print(-1)
")
  if [ "$n" != "$EXPECTED_ROWS" ]; then
    echo "FATAL: $2 indexed $n rows, expected $EXPECTED_ROWS — result NOT trustworthy"
    return 1
  fi
  echo "  verified: $2 has $n rows (expected $EXPECTED_ROWS)"
}

run_arm() {  # $1 = name, $2 = table, $3.. = extra flags
  local name=$1 table=$2; shift 2
  echo "=== $name ==="
  python -c "
import psycopg
c=psycopg.connect('$DSN', autocommit=True); c.execute('drop table if exists $table cascade')"
  python -m recall.eval.locomo $COMMON --dsn "$DSN" --table "$table" \
      --out "$OUT/$name.json" "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  [ $rc -ne 0 ] && { echo "FATAL: $name exited $rc — see $OUT/$name.log"; return 1; }
  verify_rows "$table" "$name" || return 1
  echo "  $name OK"
}

run_arm baseline        locomo_arm_base    || exit 1
run_arm rerank_shipped  locomo_arm_shipped --rerank || exit 1
run_arm rerank_modern   locomo_arm_modern  --rerank \
    --reranker-model BAAI/bge-reranker-base \
    --reranker-revision 2cfc18c9415c912f9d8155881c133215df768a70 || exit 1

echo "=== ALL THREE ARMS COMPLETE AND VERIFIED ==="
