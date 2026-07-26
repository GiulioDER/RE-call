#!/usr/bin/env bash
# 7 workers x 8 threads = 56 threads, under the ~61-CPU cgroup quota.
# Process parallelism, not thread parallelism: ONNX intra-op scaling on bge-small is ~7%
# efficient, so one 56-thread process is far slower than seven 8-thread ones.
cd /root/recall || exit 1
. /root/.voyage_env
export RECALL_DSN=postgresql://recall:recall@127.0.0.1:5432/recall
export RECALL_EMBED_THREADS=8 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false

W0="scifact cqadupstack-android cqadupstack-tex"
W1="scidocs cqadupstack-english cqadupstack-unix"
W2="fiqa cqadupstack-gaming cqadupstack-webmasters"
W3="arguana cqadupstack-gis cqadupstack-wordpress"
W4="cqadupstack-mathematica cqadupstack-stats"
W5="cqadupstack-physics"
W6="cqadupstack-programmers"

mkdir -p /root/recall/results/gap /root/logs
i=0
for w in "$W0" "$W1" "$W2" "$W3" "$W4" "$W5" "$W6"; do
  nohup ./.venv/bin/python -m recall.eval.gap_run \
      --beir-root ./beir --out ./results/gap \
      --work /root/work/w$i --datasets $w \
      > /root/logs/w$i.log 2>&1 &
  echo "worker $i -> $w (pid $!)"
  i=$((i+1))
done
echo "LAUNCHED $i workers"
