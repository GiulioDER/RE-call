#!/usr/bin/env bash
set -euo pipefail

ROOT="${RECALL_ROOT:-$PWD}"
cd "$ROOT"

source .venv/bin/activate
set -a
[[ ! -f .env ]] || source ./.env
set +a

export PYTHONUNBUFFERED=1
mkdir -p results/enterprise_rag logs

if [[ ! -s results/enterprise_rag/vast_top_splade_full.index.done ]]; then
  ./scripts/enterprise_rag_vast_index.sh
fi

python -m benchmarks.enterprise_rag \
  --questions .benchdata/enterprise-rag-v1.0.0/questions.jsonl \
  --documents .benchdata/enterprise-rag-v1.0.0/all_documents.zip \
  --out results/enterprise_rag/vast_top_splade_full.answers.jsonl \
  --table bench_enterprise_rag_top_full \
  --tenant enterprise-rag-top-full \
  --pool-size 4 \
  --embedder voyage:voyage-4-large \
  --sparse-backend both \
  --sparse-device cuda \
  --reranker voyage:rerank-2.5 \
  --answer-mode openrouter \
  --model openai/gpt-4o \
  --k 8 \
  --candidate-k 200 \
  --gap-threshold 0.5 \
  --max-context-chars 12000 \
  --skip-index \
  --resume
