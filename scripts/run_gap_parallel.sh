#!/usr/bin/env bash
# One worker per corpus, few threads each.
#
# Two measurements shaped this, both on an EPYC 7763 with a ~61-CPU cgroup quota:
#
# 1. ONNX intra-op parallelism on bge-small scales at ~7% efficiency — 2.2 / 5.7 / 9.0 docs/s at
#    1 / 8 / 32 threads, i.e. 2.2 docs/s PER THREAD at one thread against 0.28 at 32. Small-model
#    CPU inference wants many processes with few threads, not one process with many.
#
# 2. Wall clock is set by the slowest WORKER QUEUE, not by total work. Seven workers holding two
#    or three corpora each finish in the sum of their queue; one worker per corpus finishes in the
#    time of the single largest corpus. Same total compute, several times less wall clock.
#
# RECALL_EMBED_THREADS is what actually reaches fastembed. OMP_NUM_THREADS does NOT: fastembed
# sizes its pool from os.cpu_count(), which in an unprivileged container reports the HOST's cores
# (256 here) rather than the quota — so without this each worker spawned ~135 threads and seven of
# them thrashed at ~945 threads against 61 CPUs, halving aggregate throughput.
set -u
cd /root/recall || exit 1

PY=./.venv/bin/python

# Fail CLOSED on the two preconditions. Without these the loop backgrounds regardless: a missing
# interpreter or an unset VOYAGE_API_KEY still printed a pid per worker and "LAUNCHED 16 workers",
# while every one of them died at exec or failed its first corpus. The launcher could not report
# a failed launch — it could only report having tried.
[ -x "$PY" ] || { echo "FATAL: no $PY — is the venv built?" >&2; exit 1; }
if [ -r /root/.voyage_env ]; then . /root/.voyage_env; fi
: "${VOYAGE_API_KEY:?FATAL: set VOYAGE_API_KEY (or create /root/.voyage_env) before launching}"

# Honour an operator-supplied DSN instead of silently replacing it, matching run_locomo_arms.sh.
export RECALL_DSN=${RECALL_DSN:-postgresql://recall:recall@127.0.0.1:5432/recall}
export RECALL_EMBED_THREADS=${RECALL_EMBED_THREADS:-4}
export TOKENIZERS_PARALLELISM=false

# Derived from the module, never restated. The hand-maintained copy that used to live here had
# ALREADY drifted: it listed 16 corpora against the preregistered 17, silently dropping nfcorpus.
# That list sets `n`, and `n` is what the preregistered power floor gates — so a typo here decides
# whether a null result reads as "no effect" or "underpowered".
DATASETS=$("$PY" -c 'from recall.eval.gap_run import DATASETS; print(" ".join(DATASETS))') || {
  echo "FATAL: could not read DATASETS from recall.eval.gap_run" >&2; exit 1; }

mkdir -p /root/recall/results/gap /root/logs
i=0
pids=""
for d in $DATASETS; do
  # gap_run skips a corpus that already has a result, so relaunching is safe and cheap.
  # Each worker gets a SUBSET via --datasets, and gap_run therefore refuses to write the shared
  # summary.json — otherwise all sixteen would race to overwrite it with a summary of their own
  # single corpus, which is how the committed summary came to read `usable: 1` beside eighteen
  # corpus records. The summary is written once, below, over the full roster.
  nohup "$PY" -m recall.eval.gap_run \
      --beir-root ./beir --out ./results/gap \
      --work /root/work/w$i --datasets "$d" \
      > /root/logs/$d.log 2>&1 &
  pids="$pids $!"
  echo "worker $i -> $d (pid $!)"
  i=$((i+1))
done

# Verify the workers are actually ALIVE rather than trusting that backgrounding succeeded.
sleep 5
alive=0
for p in $pids; do kill -0 "$p" 2>/dev/null && alive=$((alive+1)); done
echo "LAUNCHED $i workers at ${RECALL_EMBED_THREADS} threads each; $alive still alive after 5s"
if [ "$alive" -ne "$i" ]; then
  echo "FATAL: $((i - alive)) worker(s) died immediately — check /root/logs/*.log" >&2
  exit 1
fi

echo
echo "When every worker has finished, write the study-wide summary with:"
echo "  $PY -m recall.eval.gap_run --out ./results/gap --summarise"
