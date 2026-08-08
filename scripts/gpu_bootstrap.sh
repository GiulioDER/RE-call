#!/usr/bin/env bash
# Bring a rented GPU box up for the two jobs that need one. No credentials, ever.
#
# Both jobs here take files in and give files back. Neither touches a database, a DSN, or an API
# key: that is deliberate, and it is why a rented box a stranger provisioned is an acceptable place
# to run them. `scripts/score_pairs.py` says it plainly -- "the only stage that wants rented
# hardware is also the only stage that runs on a machine nobody trusts".
#
#   JOB 1  algorithmic judges (BertScore over roberta-large)
#          842 rows in ~4 min on a 5090, against ~26 h on VPS2's 12 contended CPU cores.
#          Needs: run_algorithmic.py, config.yaml, <task>.scoring.jsonl
#
#   JOB 2  MiniLM cross-encoder pair scoring
#          Needs the dump's queries/docs/pairs. ⛔ The ORDERING stays in RE-call; only
#          `model.predict` moves, which is what makes the offload exact rather than approximate.
#          `rerank_offload validate` afterwards is NOT optional.
#
# Usage on a fresh instance:
#     scp -P <port> scripts/gpu_bootstrap.sh root@<host>:/root/
#     ssh -p <port> root@<host> 'bash /root/gpu_bootstrap.sh'
#
# CUDA note: cu128 is pinned because a 5090 is Blackwell (sm_120) and older CUDA wheels ship no
# kernels for it -- torch imports fine and then fails at the first kernel launch.
set -euo pipefail

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 -V

echo "== torch (cu128) =="
pip install -q --index-url https://download.pytorch.org/whl/cu128 torch
python3 - <<'PY'
import torch
assert torch.cuda.is_available(), "torch cannot see the GPU"
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))
PY

echo "== job 1 deps: algorithmic judges =="
# `evaluate==0.4.3` calls `hf_api.HfFolder`, removed in huggingface-hub 1.x, so the hub is pinned
# below 1.0 and transformers pinned to a version that agrees with it. Learned the hard way.
pip install -q "huggingface-hub==0.34.4" "transformers==4.55.2" "evaluate==0.4.3" \
    bert_score rouge-score pandas beautifulsoup4 lxml pyyaml tqdm

echo "== job 2 deps: cross-encoder scoring =="
pip install -q sentence-transformers

mkdir -p /root/out
echo
echo "READY. Both jobs, once their inputs are uploaded:"
echo
echo "  JOB 1  python3 run_algorithmic.py --evaluators config.yaml \\"
echo "             --input <task>.scoring.jsonl --output /root/out/<task>.algorithmic.jsonl"
echo "         (output MUST have a directory component; the script calls os.makedirs(dirname))"
echo
echo "  JOB 2  python3 score_pairs.py --queries queries.jsonl --docs docs.jsonl \\"
echo "             --pairs pairs.jsonl --output /root/out/scores.jsonl --batch-size 256"
echo "         (model + revision default to the pin and are recorded in the output header)"
