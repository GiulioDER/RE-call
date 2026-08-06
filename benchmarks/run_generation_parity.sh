#!/bin/bash
# Four index stages CONCURRENTLY, two cores each, then one compare stage over the tables they left.
#
# Prior work: NONE FOUND, and the search is recorded once for this experiment rather than twice.
# `docs_search(source_type="memory")` for "generation parity raw content hash identical across
# context modes RE-call" on 2026-08-06 returned gap_warning TRUE (top-3 cosine 0.485/0.480/0.479,
# all under the 0.50 floor), so the hits are noise. See the same note in the docstring of
# `check_generation_parity.py`, which this script is only the driver for.
#
# The arms are independent: each writes its own table and shares nothing but the database server.
# Run sequentially in one process this pays the SUM of four arms (measured: ~4h20m). Split across
# the same eight cores it pays the slowest one.
#
# Cores 0-7 of 12, leaving 8-11 free, everything at nice 19 so the unrelated live production on
# this host preempts all of it. This host carries a permanent load from that production and is a
# documented standing blocker for this program; the core budget here is the same one the
# 2026-08-06 campaign used.
#
# ⚠️ `rc=$?` is captured IMMEDIATELY after the command. Writing `echo "$(date -Is) rc=$?"` reports
# the exit status of DATE, because the command substitution runs first and resets `$?`. That is not
# hypothetical: it is why every arm of the 2026-08-06 campaign reported exit=0, including one that
# died after 3h11m having written nothing.
set -u

REPO=/var/tmp/recall-parity
VENV=/var/tmp/migvenv/bin/python
OUT=/var/tmp/parity-full
CORPUS=/opt/peps-corpus/peps
DSN="postgresql:///recall_campaign?options=-c%20role%3Drecall_migrator"

export PYTHONPATH=$REPO
export RECALL_EMBEDDER=fastembed
export RECALL_MODEL_CACHE=/var/tmp/campaign/tree
export RECALL_MODEL_SHA256=9a443d711e063427f62cf559a38863122ee5ed107fdd7920de882fd66dbc919c
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# These stop numpy opening a second pool underneath. They are NOT the cap: OMP_NUM_THREADS does
# not bound onnxruntime here (measured on this host at 24 threads / 6.2 cores despite it).
# `taskset` is the cap.
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2

cd "$REPO" || exit 1
command -v taskset >/dev/null || { echo "FATAL: taskset absent; refusing to run unpinned"; exit 2; }

idx() {
  local prof=$1 cpus=$2 rc
  echo "=== $(date -Is) START $prof cpus=$cpus ==="
  taskset -c "$cpus" nice -n 19 "$VENV" benchmarks/check_generation_parity.py \
    --dsn "$DSN" --corpus-dir "$CORPUS" --glob '**/*.rst' \
    --table-prefix pfull --out /dev/null \
    --stage index --profile "$prof" > "$OUT/idx-$prof.log" 2>&1
  rc=$?
  echo "=== $(date -Is) DONE $prof rc=$rc ==="
}

# STAGGERED, not simultaneous. `ensure_schema()` takes a process-wide advisory lock and
# `apply_migrations` refuses rather than waiting, so four arms started together race and three die
# with ConcurrentMigrator. A stagger is a race made unlikely, not a race removed.
idx bge-small-symmetric-v1 0,1 &
sleep 90
idx bge-small-context-document-v1 2,3 &
sleep 90
idx bge-small-context-section-v1 4,5 &
sleep 90
idx bge-small-context-neighbor-v1 6,7 &
wait
echo "=== $(date -Is) ALL INDEX STAGES DONE ==="

taskset -c 0-7 nice -n 19 "$VENV" benchmarks/check_generation_parity.py \
  --dsn "$DSN" --corpus-dir "$CORPUS" --glob '**/*.rst' \
  --table-prefix pfull --out "$OUT/generation-parity.json" \
  --stage compare --control-corpus "$OUT/ctl" --control-files 24 > "$OUT/compare.log" 2>&1
rc=$?
echo "=== $(date -Is) COMPARE rc=$rc ==="
exit $rc
