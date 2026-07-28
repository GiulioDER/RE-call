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

# The DSN and the table reach Python through the ENVIRONMENT, never pasted into the body of a
# `python -c` source string. Two reasons, and the first is a live defect this replaces:
#
#   1. `psycopg.connect('$DSN')` placed the DSN inside a single-quoted Python literal. An
#      apostrophe in the password — legal in a Postgres URI — closes that literal and the rest of
#      the DSN is parsed as Python SOURCE. That is arbitrary code execution in a script that runs
#      as root.
#   2. The generated source, and `--dsn "$DSN"`, both put the password on a command line where any
#      local user can read it from /proc/<pid>/cmdline for the hours an arm takes. The library
#      already holds the opposite standard: `recall.store.redacted_dsn` exists so that a
#      connection failure never writes a plaintext password to a log.
#
# The table name still cannot be parameterised — identifiers are not values in SQL — so it is
# validated as a bare identifier inside Python before interpolation.
_py_sql() {  # $1 = table, $2 = statement template using {t}
  RECALL_ARM_TABLE="$1" RECALL_ARM_SQL="$2" RECALL_DSN="$DSN" python -c '
import os, re, sys

import psycopg

table = os.environ["RECALL_ARM_TABLE"]
if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
    sys.exit(f"refusing to interpolate {table!r} into SQL: not a bare identifier")
conn = psycopg.connect(os.environ["RECALL_DSN"], autocommit=True)
cur = conn.execute(os.environ["RECALL_ARM_SQL"].format(t=table))
# `cur.description` is None for a statement that returns no rows (DDL such as DROP TABLE).
# Calling fetchone() there raises ProgrammingError, which would make every `drop table` exit
# non-zero and abort the arm it was about to run.
print(cur.fetchone()[0] if cur.description else "")
'
}

verify_rows() {  # $1 = table, $2 = arm name
  local n
  n=$(_py_sql "$1" "select count(*) from {t}" 2>/dev/null) || n=-1
  if [ "$n" != "$EXPECTED_ROWS" ]; then
    echo "FATAL: $2 indexed $n rows, expected $EXPECTED_ROWS — result NOT trustworthy"
    return 1
  fi
  echo "  verified: $2 has $n rows (expected $EXPECTED_ROWS)"
}

run_arm() {  # $1 = name, $2 = table, $3.. = extra flags
  local name=$1 table=$2; shift 2
  echo "=== $name ==="
  _py_sql "$table" "drop table if exists {t} cascade" >/dev/null || return 1
  RECALL_DSN="$DSN" python -m recall.eval.locomo $COMMON --table "$table" \
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
