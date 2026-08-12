#!/usr/bin/env bash
set -euo pipefail

ROOT="${RECALL_ROOT:-/workspace/RE-call}"
TABLE="${ENTERPRISE_RAG_DENSE_TABLE:-ber_voy_lex_12k_full}"
TENANT="${ENTERPRISE_RAG_TENANT:-enterprise-rag-voyage-lexical-chunk12k-full}"
OUT="${ENTERPRISE_RAG_SPLADE_OUT:-results/enterprise_rag/vast_voyage_lexical_splade_from_vps2.answers.jsonl}"
BACKFILL_MARKER="${ENTERPRISE_RAG_SPLADE_MARKER:-results/enterprise_rag/vast_splade_from_import.done}"

cd "$ROOT"
source .venv/bin/activate
set -a
[[ ! -f .env ]] || source ./.env
set +a

export PYTHONUNBUFFERED=1
mkdir -p results/enterprise_rag logs

if [[ ! -s "$BACKFILL_MARKER" ]]; then
  python -m benchmarks.enterprise_rag \
    --questions .benchdata/enterprise-rag-v1.0.0/questions.jsonl \
    --documents .benchdata/enterprise-rag-v1.0.0/all_documents.zip \
    --out results/enterprise_rag/vast_splade_from_import.index_only.answers.jsonl \
    --table "$TABLE" \
    --tenant "$TENANT" \
    --pool-size 4 \
    --embedder voyage:voyage-4-large \
    --sparse-backend both \
    --backfill-splade \
    --sparse-device cuda \
    --reranker voyage:rerank-2.5 \
    --rerank-document-chars 4000 \
    --answer-mode openrouter \
    --model openai/gpt-4o \
    --k 8 \
    --candidate-k 200 \
    --gap-threshold 0.5 \
    --max-context-chars 12000 \
    --skip-index \
    --index-only \
    --overwrite
  date -Is > "$BACKFILL_MARKER"
fi

python -m benchmarks.enterprise_rag \
  --questions .benchdata/enterprise-rag-v1.0.0/questions.jsonl \
  --documents .benchdata/enterprise-rag-v1.0.0/all_documents.zip \
  --out "$OUT" \
  --table "$TABLE" \
  --tenant "$TENANT" \
  --pool-size 4 \
  --embedder voyage:voyage-4-large \
  --sparse-backend both \
  --sparse-device cuda \
  --reranker voyage:rerank-2.5 \
  --rerank-document-chars 4000 \
  --answer-mode openrouter \
  --model openai/gpt-4o \
  --k 8 \
  --candidate-k 200 \
  --gap-threshold 0.5 \
  --max-context-chars 12000 \
  --skip-index \
  --resume
