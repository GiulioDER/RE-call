#!/usr/bin/env bash
# Run every pending algorithmic-judge pass back to back, so a rented box is never idle waiting
# for me to notice one finished.
#
# Each pass is BertScore over roberta-large: ~8 min for 842 rows on a 5090, against ~26 h on
# VPS2's contended CPU. That ratio is the whole reason a GPU is rented at all.
#
# Takes no credentials. The LLM-judge half runs on VPS2 where the key lives; this half needs only
# the scoring files. Same split as `scripts/score_pairs.py`.
#
# Usage, after gpu_bootstrap.sh and after uploading the *.scoring.jsonl files to /root:
#     bash /root/gpu_score_queue.sh
#
# Skips any input that is absent and any output already complete, so it is safe to re-run and safe
# to start before every generation has landed: upload what exists, run it, upload the rest, run
# again.
set -uo pipefail

cd /root
mkdir -p out

# ⚠️ `run_algorithmic.py` calls os.makedirs(os.path.dirname(output)), so the output MUST carry a
# directory component. A bare filename raises FileNotFoundError on the empty dirname.
run_one() {
    local name="$1"
    local input="/root/${name}.scoring.jsonl"
    local output="/root/out/${name}.algorithmic.jsonl"

    if [ ! -f "$input" ]; then
        echo "SKIP  ${name}: no input yet"
        return 0
    fi
    # A complete output has one row per input row. A short one is a crashed pass, not a done one.
    if [ -f "$output" ] && [ "$(wc -l < "$output")" -eq "$(wc -l < "$input")" ]; then
        echo "SKIP  ${name}: already complete ($(wc -l < "$output") rows)"
        return 0
    fi

    echo "START ${name} at $(date -u +%H:%M:%S)"
    local began=$SECONDS
    if python3 run_algorithmic.py --evaluators config.yaml --input "$input" --output "$output" \
            > "/root/algo_${name}.log" 2>&1; then
        echo "DONE  ${name}: $(wc -l < "$output") rows in $((SECONDS - began))s"
    else
        # Keep going. One failed pass must not strand the others on a box that bills by the hour.
        echo "FAIL  ${name}: see /root/algo_${name}.log"
        tail -3 "/root/algo_${name}.log"
    fi
}

for name in taskb_official taskc_recall taskc_benchmark_official taskc_recall_official; do
    run_one "$name"
done

echo
echo "== outputs =="
ls -la /root/out/ 2>/dev/null | tail -n +2
echo
echo "Pull them with:  scp -P <port> root@<host>:/root/out/'*.algorithmic.jsonl' ."
