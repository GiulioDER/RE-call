#!/bin/bash
# Four index stages CONCURRENTLY, two cores each, then one compare stage over the tables they left.
#
# Prior work: NONE FOUND, and the search is recorded once for this experiment rather than twice.
# `docs_search(source_type="memory")` for "generation parity raw content hash identical across
# context modes RE-call" on 2026-08-06 returned gap_warning TRUE (top-3 cosine 0.485/0.480/0.479,
# all under the 0.50 floor), so the hits are noise. See the same note in the docstring of
# `check_generation_parity.py`, which this script is only the driver for.
#
# The arms write SEPARATE TABLES but share the database server, the migration advisory lock and
# the schema ledger, which is why the stagger below exists. (An earlier version of this comment
# said they "share nothing but the database server", which the stagger comment 50 lines down
# directly contradicts.)
#
# Run sequentially in one process this pays the SUM of four arms, PROJECTED at ~4h20m from the
# measured per-arm rate. That number is a projection, not an observed sequential wall clock: no
# sequential four-arm run was ever completed. Split across the same eight cores it pays the
# slowest arm instead of the sum.
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
# These stop numpy opening a second pool underneath. They are NOT the cap: `OMP_NUM_THREADS`, set
# just below, does not bound onnxruntime here (measured on this host at 24 threads / 6.2 cores
# despite it). `taskset` is the cap. An earlier version of this comment reasoned about
# OMP_NUM_THREADS while the script never set it, so it named a setting it did not apply.
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OMP_NUM_THREADS=2
# ⚠️ `RECALL_EMBED_THREADS` deliberately NOT set here. It is a LIVE knob (`recall/embeddings.py`
# reads it into the fastembed thread budget), so setting it would change every arm's thread count,
# which contradicts this header's claim that the core budget is the one the 2026-08-06 campaign
# used. The finding this block answers was a FALSE COMMENT, and the fix for a false comment is a
# true comment, not a new runtime knob.

cd "$REPO" || exit 1
command -v taskset >/dev/null || { echo "FATAL: taskset absent; refusing to run unpinned"; exit 2; }
[ -x "$VENV" ] || { echo "FATAL: no interpreter at $VENV"; exit 2; }
[ -d "$CORPUS" ] || { echo "FATAL: no corpus at $CORPUS"; exit 2; }
# Every redirection below writes into $OUT. Without this the arms fail at the redirect, before
# `taskset` ever runs, and the banners still print as though they had started.
mkdir -p "$OUT" || { echo "FATAL: cannot create $OUT"; exit 2; }
# `$(nproc)` expands EMPTY if nproc is absent or errors, and `[ "" -ge 8 ]` exits 2 with
# "integer expected", so the bare form refuses a 12-core host on a missing binary. Default to 0
# and test a variable that is always an integer.
cores=$(nproc 2>/dev/null || echo 0)
[ "$cores" -ge 8 ] || { echo "FATAL: $cores cores visible; the pins below do not exist"; exit 2; }

idx() {
  local prof=$1 cpus=$2 rc
  echo "=== $(date -Is) START $prof cpus=$cpus ==="
  taskset -c "$cpus" nice -n 19 "$VENV" benchmarks/check_generation_parity.py \
    --dsn "$DSN" --corpus-dir "$CORPUS" --glob '**/*.rst' \
    --table-prefix pfull --out /dev/null \
    --stage index --profile "$prof" > "$OUT/idx-$prof.log" 2>&1
  rc=$?
  echo "=== $(date -Is) DONE $prof rc=$rc ==="
  # ⚠️ RETURN it. An earlier version captured `rc` correctly, under a four-line comment explaining
  # why the capture must be immediate, and then only echoed it: the function's exit status was its
  # trailing `echo`'s, a bare `wait` always returns 0, and the compare stage ran regardless. That
  # is the same class of defect as the campaign's `exit=$?` bug this header warns about, one level
  # further out. Capturing a status and discarding it is not checking it.
  return $rc
}

# STAGGERED, not simultaneous. `ensure_schema()` takes a process-wide advisory lock and
# `apply_migrations` refuses rather than waiting, so four arms started together race and three die
# with ConcurrentMigrator. A stagger is a race made unlikely, not a race removed.
pids=()
profiles=()
idx bge-small-symmetric-v1 0,1 &         pids+=($!); profiles+=(bge-small-symmetric-v1)
sleep 90
idx bge-small-context-document-v1 2,3 &  pids+=($!); profiles+=(bge-small-context-document-v1)
sleep 90
idx bge-small-context-section-v1 4,5 &   pids+=($!); profiles+=(bge-small-context-section-v1)
sleep 90
idx bge-small-context-neighbor-v1 6,7 &  pids+=($!); profiles+=(bge-small-context-neighbor-v1)

# `wait` with no operand returns 0 no matter how its children died. Waiting on each PID is what
# turns four captured statuses into one verdict.
failed=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "FAILED arm: ${profiles[$i]} (see $OUT/idx-${profiles[$i]}.log)"
    failed=$((failed + 1))
  fi
done
echo "=== $(date -Is) ALL INDEX STAGES DONE, failed=$failed ==="

# Refuse rather than compare an incomplete set. With a dead arm the compare stage would either
# refuse on an empty table or, worse, compare three complete generations and publish a verdict
# over a set that is missing one candidate.
if [ "$failed" -ne 0 ]; then
  echo "FATAL: $failed index arm(s) failed; refusing to run the compare stage."
  exit 3
fi

# ⚠️ `--drop-generations` is deliberately NOT passed, so the four `pfull_*` tables SURVIVE this
# run. That is the point: they cost about an hour of embedding each, and keeping them makes a
# re-run of the compare stage free. They are the operator's to remove, and they must be removed
# through `PgVectorStore.drop_table()`, NEVER through psql: `drop_table` deletes the matching
# `recall_schema_migrations` rows in the same transaction, and a bare `DROP TABLE` leaves them
# behind, after which the next run skips creation and fails validating a table that is not there.
taskset -c 0-7 nice -n 19 "$VENV" benchmarks/check_generation_parity.py \
  --dsn "$DSN" --corpus-dir "$CORPUS" --glob '**/*.rst' \
  --table-prefix pfull --out "$OUT/generation-parity.json" \
  --stage compare --control-corpus "$OUT/ctl" --control-files 24 > "$OUT/compare.log" 2>&1
rc=$?
echo "=== $(date -Is) COMPARE rc=$rc ==="
exit $rc
