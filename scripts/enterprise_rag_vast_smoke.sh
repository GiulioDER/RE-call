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

python - <<'PY'
import json
import zipfile
from pathlib import Path

base = Path(".benchdata/enterprise-rag-v1.0.0")
question = json.loads((base / "questions.jsonl").read_text(encoding="utf-8").splitlines()[0])
target = question["expected_doc_ids"][0]
source = zipfile.ZipFile(base / "all_documents.zip")
target_names = [name for name in source.namelist() if target in name]
if len(target_names) != 1:
    raise SystemExit(f"expected one target doc for {target}, found {len(target_names)}")
names = target_names + [
    name for name in source.namelist() if name.endswith(".txt") and name not in target_names
][:4]
out = base / "vast_smoke_docs_with_gold.zip"
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as dest:
    for name in names:
        dest.writestr(name, source.read(name))
print(out)
for name in names:
    print(name)
PY

python -m benchmarks.enterprise_rag \
  --questions .benchdata/enterprise-rag-v1.0.0/questions.jsonl \
  --documents .benchdata/enterprise-rag-v1.0.0/vast_smoke_docs_with_gold.zip \
  --out results/enterprise_rag/vast_top_splade_smoke.answers.jsonl \
  --table bench_enterprise_rag_vast_smoke \
  --tenant enterprise-rag-vast-smoke \
  --pool-size 4 \
  --embedder voyage:voyage-4-large \
  --sparse-backend both \
  --backfill-splade \
  --sparse-device cuda \
  --reranker voyage:rerank-2.5 \
  --answer-mode openrouter \
  --model openai/gpt-4o \
  --k 8 \
  --candidate-k 200 \
  --gap-threshold 0.5 \
  --max-context-chars 12000 \
  --splade-batch-size "${SPLADE_BATCH_SIZE:-32}" \
  --limit-questions 1 \
  --reset-index \
  --overwrite
