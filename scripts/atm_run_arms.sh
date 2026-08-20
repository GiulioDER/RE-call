#!/usr/bin/env bash
# Four preregistered arms over the fixed 300 question subset.
#
# Retrieval is NOT rerun: each arm's directory is seeded with the retrieval.jsonl produced by the
# completed Voyage run, so every arm reads byte-identical evidence and the comparison isolates the
# answer stage. It also spends nothing on Voyage.
#
# WORKERS defaults to 2. Four concurrent requests made the provider close connections on every one
# of them ("Response ended prematurely") while a direct probe of the same model, the same ceiling
# and an 8k prompt answered in 4.6 seconds. Two was verified on a six question smoke. The parallel
# path itself is newer than the run that produced 68.92, which used one worker.
set -euo pipefail

# Paths come from the environment with defaults, the way the neighbouring benchmark scripts in this
# directory already do it. This file is committed to a PUBLIC repository, and CLAUDE.md is explicit
# that a host and path inventory is disclosure on its own, separately from any credential.
ATM_RUN_ROOT="${ATM_RUN_ROOT:-$HOME/atm-bench-run}"
ATM_PYTHON="${ATM_PYTHON:-python3}"
# Space separated list of env files to source for provider keys. Empty by default: nothing is read
# unless the operator names it.
ATM_ENV_FILES="${ATM_ENV_FILES:-}"

cd "$ATM_RUN_ROOT"

set -a
for env_file in $ATM_ENV_FILES; do
  [ -r "$env_file" ] && source <(grep -E "^[A-Za-z_][A-Za-z0-9_]*=" "$env_file")
done
set +a

PY="$ATM_PYTHON"
SRC=source-arms
QA="${QA_FILE:-data/atm-bench/atm-bench-subset300.json}"
ROOT="${ARMS_ROOT:-results/arms300}"
SEED=results/voyage4-deepseek-full/retrieval.jsonl

run_arm () {
  local name="$1" packer="$2" policy="$3"
  local out="$ROOT/$name"
  mkdir -p "$out"
  # -n so a resume never clobbers a checkpoint.
  cp -n "$SEED" "$out/retrieval.jsonl" || true
  echo "=== $(date -Is) $name packer=$packer policy=$policy ==="
  "$PY" "$SRC/benchmarks/atm_full_run.py" \
    --qa-file "$QA" \
    --image-file data/image/batch_results.json \
    --video-file data/video/batch_results.json \
    --email-file data/emails.json \
    --out-dir "$out" \
    --evidence-packer "$packer" \
    --answer-policy "$policy" \
    --max-output-tokens 8192 \
    --answer-workers "${WORKERS:-2}" \
    --reuse-index
  echo "=== $(date -Is) $name finished ==="
}

for spec in "${ARMS:-A-greedy-baseline:greedy:baseline B-allocated-baseline:allocated:baseline C-allocated-disposition:allocated:disposition D-allocated-selection:allocated:selection}"; do
  for one in $spec; do
    IFS=: read -r name packer policy <<< "$one"
    run_arm "$name" "$packer" "$policy"
  done
done
echo "ALL_ARMS_DONE $(date -Is)"
