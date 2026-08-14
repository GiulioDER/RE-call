#!/usr/bin/env bash
set -euo pipefail

BENCH_DIR="${ENTERPRISE_RAG_BENCH_DIR:-/home/sentiment/enterprise-rag-run/EnterpriseRAG-Bench}"
RECALL_DIR="${RECALL_DIR:-/home/sentiment/enterprise-rag-run/RE-call}"
ANSWERS_FILE="${ENTERPRISE_RAG_ANSWERS_FILE:-$BENCH_DIR/answer_evaluation/re_call_voyage_splade_gpt4o.answers.jsonl}"
RESULTS_FILE="${ENTERPRISE_RAG_RESULTS_FILE:-$BENCH_DIR/answer_evaluation/re_call_voyage_splade_gpt4o.judge_gpt54_medium.default_results.json}"
UPDATED_QUESTIONS_FILE="${ENTERPRISE_RAG_UPDATED_QUESTIONS_FILE:-$BENCH_DIR/answer_evaluation/re_call_voyage_splade_gpt4o.judge_gpt54_medium.default_questions_updated.jsonl}"
UUID_INDEX_FILE="${ENTERPRISE_RAG_UUID_INDEX_FILE:-$BENCH_DIR/generated_data/uuid_index.json}"
PARALLELISM="${ENTERPRISE_RAG_SCORE_PARALLELISM:-4}"
MODEL="${ENTERPRISE_RAG_JUDGE_MODEL:-openai/gpt-5.4}"
JUDGE_REASONING="${ENTERPRISE_RAG_JUDGE_REASONING:-medium}"

cd "$BENCH_DIR"
source .venv/bin/activate

set -a
[[ ! -f "$RECALL_DIR/.env" ]] || source "$RECALL_DIR/.env"
set +a

export PYTHONUNBUFFERED=1
export LLM_PROVIDER=openai
export LLM_API_KEY="${LLM_API_KEY:-${OPENROUTER_API_KEY:?OPENROUTER_API_KEY or LLM_API_KEY is required}}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}"
export LLM_MODEL_NAME="$MODEL"

if [[ "$JUDGE_REASONING" == "off" || "$JUDGE_REASONING" == "none" ]]; then
  python - "$ANSWERS_FILE" "$RESULTS_FILE" "$UPDATED_QUESTIONS_FILE" "$UUID_INDEX_FILE" "$PARALLELISM" <<'PY'
import sys

import src.llm.factory as factory
import src.scripts.answer_evaluation.metrics_based_eval as evaluator

original_get_llm = factory.get_llm


def get_llm_reasoning_off(
    tools=None,
    quiet=False,
    reasoning_level="medium",
    model=None,
):
    return original_get_llm(
        tools=tools,
        quiet=quiet,
        reasoning_level=None,
        model=model,
    )


factory.get_llm = get_llm_reasoning_off
evaluator.get_llm = get_llm_reasoning_off

sys.argv = [
    "metrics_based_eval",
    "--answers-file",
    sys.argv[1],
    "--questions-file",
    "questions.jsonl",
    "--results-file",
    sys.argv[2],
    "--updated-questions-file",
    sys.argv[3],
    "--uuid-index-cache-file",
    sys.argv[4],
    "--parallelism",
    sys.argv[5],
    "--resume",
]
evaluator.main()
PY
else
  python -m src.scripts.answer_evaluation.metrics_based_eval \
    --answers-file "$ANSWERS_FILE" \
    --questions-file questions.jsonl \
    --results-file "$RESULTS_FILE" \
    --updated-questions-file "$UPDATED_QUESTIONS_FILE" \
    --uuid-index-cache-file "$UUID_INDEX_FILE" \
    --parallelism "$PARALLELISM" \
    --resume
fi
