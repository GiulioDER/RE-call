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
# ⚠️ DO NOT COUNT CORES HERE, and the reason is worth the paragraph.
#
# This guard went through three shapes before it was right. `[ "$(nproc)" -ge 8 ]` refused a
# 12-core host whenever `nproc` was missing, because the empty expansion made `[ "" -ge 8 ]` exit
# with "integer expected". Validating the value fixed that and was still wrong, because **`nproc`
# honours `OMP_NUM_THREADS`**, which THIS SCRIPT exports as 2 about fifteen lines above. So the
# count read 2 on a twelve-core machine and refused the run outright.
#
# That was an interaction between two fixes from the SAME batch: one set `OMP_NUM_THREADS` to make
# a false comment true, the other counted cores to fail closed. Three review rounds missed it
# because each read the two hunks separately and the script was never run end to end between them.
#
# The repair is not a better count. A count is a PROXY for the question the pins below actually
# ask, and every shape of the proxy was wrong in a different way. Ask the kernel the real question
# instead: can this process take the mask it is about to use? That is immune to OMP_NUM_THREADS,
# to a missing binary, to cpuset restrictions and to affinity inherited from the caller, because it
# is not a measurement of anything. It is the operation itself.
# Probe EACH mask this script uses, not the union. A partially valid mask SUCCEEDS: `taskset -c
# 0-7` on a four-core host sets affinity to 0-3 and returns 0, so probing the union would pass on a
# host where the `4,5` and `6,7` arms cannot run at all. Only a mask containing NO valid CPU is
# refused (measured: `taskset -c 50-99 true` fails "Invalid argument"), so the probe has to be at
# the granularity of the masks actually used.
for mask in 0,1 2,3 4,5 6,7 0-7; do
  taskset -c "$mask" true 2>/dev/null || {
    echo "FATAL: cannot pin to cores '$mask' on this host (nproc --all reports $(nproc --all 2>&1));"
    echo "       refusing to run unpinned."
    exit 2
  }
done

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
ARMS=(
  bge-small-symmetric-v1
  bge-small-context-document-v1
  bge-small-context-section-v1
  bge-small-context-neighbor-v1
)
CPUS=(0,1 2,3 4,5 6,7)
failed=0

if [ -n "${PARITY_SEQUENTIAL:-}" ]; then
  # ⚠️ MEMORY is why this mode exists, not CPU. Four concurrent arms held roughly 18 GB and drove
  # a 47 GB host to 3 GB available on 2026-08-06, while it was also running live production.
  # `taskset` and `nice` bound CPU and do NOTHING about resident memory, so the pins gave no
  # protection at all. One arm at a time keeps ONE embedding model resident, and it gets all eight
  # cores because nothing else of ours is running.
  #
  # This mode is also how a RESUME is meant to run: the fingerprint guard skips files already
  # indexed at the same content, so an interrupted arm continues instead of restarting.
  echo "=== $(date -Is) SEQUENTIAL mode: one arm at a time on cores 0-7 ==="
  for arm in "${ARMS[@]}"; do
    if ! idx "$arm" 0-7; then
      echo "FAILED arm: $arm (see $OUT/idx-$arm.log)"
      failed=$((failed + 1))
    fi
  done
else
  pids=()
  profiles=()
  for i in "${!ARMS[@]}"; do
    idx "${ARMS[$i]}" "${CPUS[$i]}" &
    pids+=($!)
    profiles+=("${ARMS[$i]}")
    # Stagger every arm but the last; see the advisory-lock note above.
    [ "$i" -lt 3 ] && sleep 90
  done

  # `wait` with no operand returns 0 no matter how its children died. Waiting on each PID is what
  # turns four captured statuses into one verdict.
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "FAILED arm: ${profiles[$i]} (see $OUT/idx-${profiles[$i]}.log)"
      failed=$((failed + 1))
    fi
  done
fi
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
# re-run of the compare stage free.
#
# To remove them afterwards, re-run this same compare invocation WITH `--drop-generations`. Do not
# reach for psql: `drop_table()` deletes the matching `recall_schema_migrations` rows in the same
# transaction, and a bare `DROP TABLE` leaves them behind, after which the next run skips creation
# and fails validating a table that is not there. That trap was hit for real on 2026-08-06.
taskset -c 0-7 nice -n 19 "$VENV" benchmarks/check_generation_parity.py \
  --dsn "$DSN" --corpus-dir "$CORPUS" --glob '**/*.rst' \
  --table-prefix pfull --out "$OUT/generation-parity.json" \
  --stage compare --control-corpus "$OUT/ctl" --control-files 24 > "$OUT/compare.log" 2>&1
rc=$?
echo "=== $(date -Is) COMPARE rc=$rc ==="
exit $rc
