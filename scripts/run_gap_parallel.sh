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
. /root/.voyage_env
export RECALL_DSN=postgresql://recall:recall@127.0.0.1:5432/recall
export RECALL_EMBED_THREADS=${RECALL_EMBED_THREADS:-4}
export TOKENIZERS_PARALLELISM=false

DATASETS="scifact scidocs fiqa arguana \
cqadupstack-android cqadupstack-english cqadupstack-gaming cqadupstack-gis \
cqadupstack-mathematica cqadupstack-physics cqadupstack-programmers cqadupstack-stats \
cqadupstack-tex cqadupstack-unix cqadupstack-webmasters cqadupstack-wordpress"

mkdir -p /root/recall/results/gap /root/logs
i=0
for d in $DATASETS; do
  # gap_run skips a corpus that already has a result, so relaunching is safe and cheap.
  nohup ./.venv/bin/python -m recall.eval.gap_run \
      --beir-root ./beir --out ./results/gap \
      --work /root/work/w$i --datasets "$d" \
      > /root/logs/$d.log 2>&1 &
  echo "worker $i -> $d (pid $!)"
  i=$((i+1))
done
echo "LAUNCHED $i workers at ${RECALL_EMBED_THREADS} threads each"
